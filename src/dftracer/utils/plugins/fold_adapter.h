#ifndef DFTRACER_UTILS_PLUGINS_FOLD_ADAPTER_H
#define DFTRACER_UTILS_PLUGINS_FOLD_ADAPTER_H

#include <dftracer/utils/core/common/hash/constants.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/monoid.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/query/query.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/utilities/fileio/parallel/parallel_writer.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>
#endif

#include <cstddef>
#include <cstdint>
#include <deque>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// Adapts a loaded C-ABI plugin to the internal fold interface so N plugins ride
// one fused scan. A non-null dftu_task from on_batch/on_finalize is a lazy
// CoroTask<void> the fuse worker awaits.
namespace dftracer::utils::plugins {

// Host-owned parallel writer handed to the plugin as an opaque dftu_writer*;
// num_workers/gzip are captured at writer_create for a later writer_open.
struct PluginWriter {
    std::unique_ptr<utilities::fileio::parallel::ParallelWriter> writer;
    std::string path;
    std::size_t num_workers = 1;
    bool gzip = false;
};

// Host-owned trace writer handed to the plugin as an opaque dftu_trace_writer*.
// trace_close compresses `buffer` into gzip members as a re-indexable .pfw.gz.
struct PluginTraceWriter {
    std::string path;
    std::string buffer;
    std::uint64_t next_id = 0;
};

// Per-slice mergeable map: a key tuple -> a product of scalar Monoids,
// materialized to a columnar Arrow table at finalize. key_n, key_types, and
// value_kinds are fixed at map_new and never change.
struct MapAccum {
    struct KeyHash {
        static std::size_t hash_range(const std::int64_t* p, std::size_t n) {
            std::uint64_t h = dftracer::utils::hash::FNV1A_OFFSET_BASIS_LEGACY;
            for (std::size_t i = 0; i < n; ++i) {
                h ^= static_cast<std::uint64_t>(p[i]);
                h *= dftracer::utils::hash::FNV1A_PRIME;
            }
            return static_cast<std::size_t>(h);
        }
        std::size_t operator()(const std::vector<std::int64_t>& k) const {
            return hash_range(k.data(), k.size());
        }
    };

    using EntriesMap =
        std::unordered_map<std::vector<std::int64_t>,
                           std::vector<MonoidAccumulator>, KeyHash>;

    std::string name;
    std::uint32_t key_n = 0;
    std::vector<dftu_type> key_types;
    std::vector<dftu_monoid_kind> value_kinds;
    // Radix-partitioned storage: K = 1u << part_bits maps routed by
    // partition_of(key). part_bits=0 => K=1 => byte-identical to unpartitioned;
    // K>1 lets spill drain one partition at a time.
    std::uint32_t part_bits = 0;
    std::vector<EntriesMap> parts;
    // When set, rows materialize sorted by key; ORed across merged slices.
    bool ordered = false;

    // Spill state, empty unless spill is enabled. runs[p] hold partition p's
    // spilled partial aggregates as sorted deltas (each spill clears the live
    // partition; reload merges every run plus the live residual by key).
    // part_bytes[p] and footprint are coarse live-byte estimates gating spill.
    static constexpr std::size_t KEY_OVERHEAD_BYTES = 32;
    std::vector<std::vector<std::string>> runs;
    std::vector<std::size_t> part_bytes;
    std::size_t footprint = 0;
    std::size_t value_base_bytes = 0;
    std::size_t adds_since_check = 0;

    // Nested-preserved map: a concat-key flat map keyed (outer.., inner..);
    // nested_inner_n counts the trailing inner components, grouped by outer
    // prefix into one list<struct> row per outer key at materialize.
    std::uint32_t nested_inner_n = 0;
    std::vector<dftu_type> inner_key_types;

    // Payload row column types for an ARGMIN_ROW/ARGMAX_ROW value, whose one
    // value component materializes to payload_types.size() columns.
    std::vector<dftu_type> payload_types;

    // Requested quantiles for a SKETCH value, whose one value component
    // materializes to a count column followed by one column per quantile.
    std::vector<double> quantile_qs;

    // Per-component result-table names for a fused map; non-empty means each
    // component materializes as its own named table rather than one product.
    std::vector<std::string> fused_out_names;

    std::uint32_t partitions() const { return 1u << part_bits; }

    void set_part_bits(std::uint32_t bits) {
        part_bits = bits;
        parts.assign(partitions(), EntriesMap{});
        runs.assign(partitions(), {});
        part_bytes.assign(partitions(), 0);
        footprint = 0;
        adds_since_check = 0;
    }

    bool has_runs() const {
        for (const auto& r : runs)
            if (!r.empty()) return true;
        return false;
    }

    // Nested maps hash only the outer-key prefix so an outer key's inner rows
    // co-locate in one partition, keeping the materialize-time grouping intact.
    std::uint32_t partition_of(const std::vector<std::int64_t>& key) const {
        if (part_bits == 0) return 0;
        const std::size_t n = nested_inner_n ? key_n - nested_inner_n : key_n;
        const std::uint64_t h = KeyHash::hash_range(key.data(), n);
        return static_cast<std::uint32_t>(h >> (64 - part_bits));
    }

    std::size_t total_entries() const {
        std::size_t n = 0;
        for (const EntriesMap& p : parts) n += p.size();
        return n;
    }

    std::vector<MonoidAccumulator> make_value() const {
        std::vector<MonoidAccumulator> v;
        v.reserve(value_kinds.size());
        for (dftu_monoid_kind k : value_kinds) v.emplace_back(k);
        return v;
    }

    std::vector<MonoidAccumulator>& touch(const std::vector<std::int64_t>& key,
                                          bool& inserted) {
        if (parts.empty()) parts.resize(partitions());
        EntriesMap& e = parts[partition_of(key)];
        auto it = e.find(key);
        inserted = it == e.end();
        if (inserted) it = e.emplace(key, make_value()).first;
        return it->second;
    }

    std::vector<MonoidAccumulator>& touch(
        const std::vector<std::int64_t>& key) {
        bool ignored = false;
        return touch(key, ignored);
    }
};

// Host definition of the opaque dftu_op handle: a node in a plugin's compose
// graph. Leaf nodes carry the plugin callback + owned state; combinator nodes
// carry their children. Arena-owned by PluginFold (scan-lifetime).
struct ComposeOp {
    enum class Kind { LEAF, THEN, WHEN_ALL, WHEN_ANY };
    Kind kind = Kind::LEAF;
    dftu_op_fn fn = nullptr;
    void* state = nullptr;
    void (*free_state)(void*) = nullptr;
    dftu_type in_ty = DFTU_T_VOID;
    dftu_type out_ty = DFTU_T_VOID;
    std::uint32_t in_size = 0;
    std::uint32_t out_size = 0;
    std::vector<ComposeOp*> children;

    ComposeOp() = default;
    ComposeOp(const ComposeOp&) = delete;
    ComposeOp& operator=(const ComposeOp&) = delete;
    ~ComposeOp() {
        if (free_state) free_state(state);
    }
};

// dft.ext.agg accumulator; defined in the .cpp to keep agg.h (and its Arrow
// tangle) out of this header.
struct AggAccum;

class PluginFold : public trace::views::detail::Fold {
    using Fold = trace::views::detail::Fold;
    using FoldBatch = trace::views::detail::FoldBatch;
    using ScanShape = trace::views::detail::ScanShape;
    using ScanUnit = trace::views::detail::ScanUnit;
    using CoverageSet = trace::views::detail::CoverageSet;
    using FoldPortBus = trace::views::detail::FoldPortBus;

   public:
    PluginFold(const dftu_plugin* plugin, dftracer::utils::StringIntern& intern,
               SharedResultRegistry* results = nullptr,
               NamedResultRegistry* named_results = nullptr,
               std::string plugin_name = {});
    ~PluginFold() override;

    PluginFold(const PluginFold&) = delete;
    PluginFold& operator=(const PluginFold&) = delete;

    bool accepts(const ScanShape&) const override { return true; }
    bool needs_args() const override {
        return (plugin_->needs(plugin_->self) & DFTU_NEED_ARGS) != 0;
    }

    std::unique_ptr<Fold> slice() const override {
        return std::make_unique<PluginFold>(plugin_, *intern_, results_,
                                            named_results_, plugin_name_);
    }

    /// Name this plugin logs under (its load-path stem); empty if unknown.
    const std::string& plugin_name() const { return plugin_name_; }

    void step(const FoldBatch& batch) override;
    void seal_unit(const ScanUnit&) override {}
    void drop_unit(const ScanUnit&) override {}
    void merge(Fold& other) override;
    coro::CoroTask<bool> finalize(const CoverageSet& covered) override;

    // The fuse worker awaits this after step(); consumed once, then null.
    ::dftu_task* take_pending() override {
        ::dftu_task* t = pending_;
        pending_ = nullptr;
        return t;
    }

    void bind_port_bus(FoldPortBus* bus) override { port_bus_ = bus; }

    // No-ops / NULL when no bus is bound (e.g. a unit test without the driver).
    void port_publish(std::uint64_t key, const void* data, std::uint32_t len) {
        if (port_bus_) port_bus_->publish(key, data, len);
    }
    const void* port_consume(std::uint64_t key, std::uint32_t* out_len) const {
        if (port_bus_) return port_bus_->consume(key, out_len);
        if (out_len) *out_len = 0;
        return nullptr;
    }

    // Get-or-create this slice's named monoid handle; null on allocation
    // failure or a null cap_id. The address is stable for the slice, and a cap
    // seen before keeps its creation-time kind.
    MonoidAccumulator* handle_get(const char* cap_id, dftu_monoid_kind kind);
    // Fill *out from the shared registry's merged value; -1 if no handle there.
    int handle_result(const char* cap_id, dftu_monoid_value* out) const;

    // Emit a named result into the host registry; no-ops when none is bound.
    void result_emit(const char* name, const void* data, std::uint64_t len);
    int result_emit_arrow(const char* name, ::ArrowArray* a, ::ArrowSchema* s);
    // Emit a native dataframe result; takes ownership of the handle. -1 when no
    // named-result registry is bound.
    int result_emit_frame(const char* name, dftu_dataframe* df);
    // Emit a deferred (lazyframe) result; takes ownership of the handle. -1
    // when no named-result registry is bound.
    int result_emit_lazyframe(const char* name, dftu_lazyframe* lf);

    // Get-or-create this slice's named map; null if a key type is not I64/STR
    // or any value monoid has no scalar result. The address is stable for the
    // slice. map_get is single-component, map_get_product a value_n product.
    MapAccum* map_get(const char* name, const dftu_type* key_types,
                      std::uint32_t key_n, dftu_monoid_kind value);
    MapAccum* map_get_product(const char* name, const dftu_type* key_types,
                              std::uint32_t key_n,
                              const dftu_monoid_kind* values,
                              std::uint32_t value_n);
    // Get-or-create a nested-preserved map (see MapAccum::nested_inner_n); null
    // if any key type is unsupported or a value monoid is not a materializable
    // scalar/product.
    MapAccum* map_get_nested(const char* name, const dftu_type* outer_key_types,
                             std::uint32_t outer_key_n,
                             const dftu_type* inner_key_types,
                             std::uint32_t inner_key_n,
                             const dftu_monoid_kind* values,
                             std::uint32_t value_n);
    void map_add_nested_u64(MapAccum* m, const std::int64_t* outer_key,
                            const std::int64_t* inner_key, std::uint32_t comp,
                            std::uint64_t v);
    void map_add_nested_f64(MapAccum* m, const std::int64_t* outer_key,
                            const std::int64_t* inner_key, std::uint32_t comp,
                            double v);
    void map_add_u64(MapAccum* m, const std::int64_t* key, std::uint64_t v);
    void map_add_f64(MapAccum* m, const std::int64_t* key, double v);
    void map_add_u64_at(MapAccum* m, const std::int64_t* key,
                        std::uint32_t comp, std::uint64_t v);
    void map_add_f64_at(MapAccum* m, const std::int64_t* key,
                        std::uint32_t comp, double v);
    void map_add_ordered_at(MapAccum* m, const std::int64_t* key,
                            std::uint32_t comp, std::int64_t order_key,
                            std::uint64_t element);
    void map_add_argby_at(MapAccum* m, const std::int64_t* key,
                          std::uint32_t comp, double by, std::int64_t payload);
    // Contribute (x, y) to a co-moment value component (CORR/COVAR_*/REGR_*);
    // x is the independent variable, y the dependent.
    void map_add_xy_at(MapAccum* m, const std::int64_t* key, std::uint32_t comp,
                       double x, double y);
    // Get-or-create an argmin-row/argmax-row map; null if any key or payload
    // type is unsupported. map_add_argrow contributes (by, payload-row) at key.
    MapAccum* map_get_argrow(const char* name, const dftu_type* key_types,
                             std::uint32_t key_n, int is_max,
                             const dftu_type* payload_types,
                             std::uint32_t payload_n);
    void map_add_argrow(MapAccum* m, const std::int64_t* key, double by,
                        const std::int64_t* payload, std::uint32_t payload_n);
    // Get-or-create a DDSketch quantile map over the quantiles in qs (each in
    // [0,1]); null if any key type is unsupported or nq is 0. Fed via
    // map_add_f64.
    MapAccum* map_get_sketch(const char* name, const dftu_type* key_types,
                             std::uint32_t key_n, const double* qs,
                             std::uint32_t nq);
    // Get-or-create a fused map: a product whose components each materialize as
    // their own named table (out_names). Null if a key/value type is
    // unsupported or a component is not a fused-eligible scalar monoid.
    MapAccum* map_get_fused(const char* name, const dftu_type* key_types,
                            std::uint32_t key_n, const char* const* out_names,
                            const dftu_monoid_kind* values,
                            std::uint32_t value_n);
    // Apply n row contributions at key with a single hash lookup.
    void map_add_row(MapAccum* m, const std::int64_t* key,
                     const ::dftu_row_val* vals, std::uint32_t n);
    void map_add_topk_at(MapAccum* m, const std::int64_t* key,
                         std::uint32_t comp, std::uint32_t k, double by,
                         std::int64_t payload);
    void map_add_approx_topk_at(MapAccum* m, const std::int64_t* key,
                                std::uint32_t comp, std::uint32_t k,
                                std::int64_t value);
    void map_add_sample_at(MapAccum* m, const std::int64_t* key,
                           std::uint32_t comp, std::uint32_t k,
                           std::int64_t item);
    void map_set_ordered(MapAccum* m, int ordered);
    // Record a map-to-map join to run at finalize; carried across merge and
    // deduped by out_name.
    void declare_join(const char* out_name, const char* left_name,
                      const char* right_name, dftu_join_type type);

    // Get-or-create this slice's named dft.ext.agg accumulator; null on a bad
    // op code, a missing output name, or an allocation failure. A name seen
    // before keeps its creation-time specs. agg_accumulate skips a batch
    // missing any referenced column.
    dftu_agg* agg_new(const char* name, const char* const* key_names,
                      std::uint32_t key_n, const dftu_agg_col* specs,
                      std::uint32_t spec_n);
    void agg_accumulate(dftu_agg* a, const dftu_dataframe* df);

    // Seed part_bits (K = 1u << bits) on maps created after this call.
    // Internal, not on the ABI. Default 0 => K=1.
    void set_map_part_bits(std::uint32_t bits) { map_part_bits_ = bits; }

    // Enable out-of-core spill on maps created after this call: `share_bytes`
    // is this fold's budget share (0 disables), `dir` the spill root (empty
    // keeps the env/$TMPDIR default). Internal, host/runtime controlled, never
    // a plugin knob. Enabling forces K=256 partitions.
    void set_map_spill(std::size_t share_bytes, const std::string& dir);

    std::size_t map_spill_count() const { return map_spill_count_; }

    // Materialize partition by partition and surface the result as a sequence
    // of same-schema batches. `chunk_rows` caps rows per batch on the
    // ordered/nested global path (0 => default). Internal, off by default.
    void set_map_stream(bool enabled, std::size_t chunk_rows = 0);

    std::size_t map_stream_batch_count() const { return map_stream_batches_; }
    std::size_t map_stream_max_resident_partitions() const {
        return map_stream_max_resident_parts_;
    }

    // Allocate a lazy CoroTask into this step's arena, returning its stable
    // address as an opaque dftu_task*; null if allocation throws.
    ::dftu_task* emplace_task(coro::CoroTask<void>&& task);

    // dftu_ext_compose backing. Ops are arena-owned (scan-lifetime); the
    // combinators validate value sizes and return null on a mismatch.
    ::dftu_op* compose_make(dftu_op_fn fn, void* state,
                            void (*free_state)(void*), dftu_type in_ty,
                            std::uint32_t in_size, dftu_type out_ty,
                            std::uint32_t out_size);
    ::dftu_op* compose_then(::dftu_op* a, ::dftu_op* b);
    ::dftu_op* compose_when_all(::dftu_op* const* ops, std::uint32_t n);
    ::dftu_op* compose_when_any(::dftu_op* const* ops, std::uint32_t n);
    ::dftu_task* compose_run(::dftu_op* op, const void* in, void* out, int* rc);
    // A compose leaf wrapping registered host utility `util_id` (single-value
    // only); null for a stream-only or unknown id. Lets then()/when_all pipe
    // host utilities through the dftu_op engine.
    ::dftu_op* compose_util_op(std::uint32_t util_id);

    dftracer::utils::StringIntern& intern_table() { return *intern_; }
    ::dftu_writer* create_writer(const char* path, std::uint32_t num_workers,
                                 int gzip);
    ::dftu_trace_writer* create_trace_writer(const char* path);
    int close_trace_writer(::dftu_trace_writer* w);

    // The fully wired host service table.
    ::dftu_host& host() { return host_; }

    // Valid until this PluginFold is destroyed; null on parse error.
    ::dftu_query* compile_query(const char* src, std::uint32_t len);
    int match_query(const query::Query& q, const dftu_event& e);

   private:
    using Query = query::Query;
    using ValueMap = query::ValueMap;
    using FoldEvent = trace::views::detail::FoldEvent;

    bool passes_query(const FoldEvent& ev);
    // Materialize this batch's events into a DataFrame and drive the plugin's
    // on_batch_columns seam (the vectorized-fold path).
    void step_columns(const FoldBatch& batch);
    // Build each merged map's Arrow table and emit it under its name; reloads
    // spilled runs first, so a spilled map materializes identically.
    void materialize_maps();
    // Finalize each merged aggregation accumulator to an Arrow table and emit
    // it under its name.
    void materialize_aggs();
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    // max_rows_per_batch == 0 builds exactly one batch; > 0 flushes every
    // that-many rows so an ordered/nested global sort streams in chunks.
    // only_comp >= 0 emits just that one value component as a single "value"
    // column (used to split a fused map into one table per component).
    void materialize_map(
        MapAccum& m, std::size_t max_rows_per_batch,
        std::vector<utilities::common::arrow::ArrowExportResult>& out,
        int only_comp = -1);
    void collect_map_batches_streaming(
        MapAccum& m,
        std::vector<utilities::common::arrow::ArrowExportResult>& out);
#endif

    // Spill machinery. account_add bumps the footprint on a new key or a
    // growing variable monoid; note_and_maybe_spill drains the largest
    // partitions to sorted runs; reload_runs merges them back before
    // materialize. A failed write sets map_spill_failed_ and is logged, never
    // silently swallowed. None of these throw across the C-ABI add path.
    void account_add(MapAccum& m, const std::vector<std::int64_t>& key,
                     std::uint32_t comp, bool inserted);
    void note_and_maybe_spill(MapAccum& m, bool inserted);
    void spill_partition(MapAccum& m, std::uint32_t p);
    void reload_runs(MapAccum& m);
    const std::string& ensure_spill_dir();
    void remove_spill_dirs();

    const dftu_plugin* plugin_;
    std::string plugin_name_;
    dftracer::utils::StringIntern* intern_;
    void* slice_;
    dftu_host host_{};
    ::dftu_task* pending_ = nullptr;
    FoldPortBus* port_bus_ = nullptr;
    SharedResultRegistry* results_ = nullptr;
    NamedResultRegistry* named_results_ = nullptr;

    // Per-slice named handles; the deque keeps each MonoidAccumulator address
    // stable so a dftu_handle* handed to a plugin never dangles across a
    // rehash.
    std::deque<MonoidAccumulator> handles_;
    std::unordered_map<std::uint64_t, std::size_t> handle_index_;

    // Per-slice named maps; the deque keeps each MapAccum address stable so a
    // dftu_map* handed to a plugin never dangles across a rehash.
    std::deque<MapAccum> maps_;
    std::unordered_map<std::uint64_t, std::size_t> map_index_;
    std::uint32_t map_part_bits_ = 0;

    // dft.ext.agg accumulators; unique_ptr keeps each handed-out dftu_agg*
    // stable and lets the header forward-declare AggAccum.
    std::deque<std::unique_ptr<AggAccum>> aggs_;
    std::unordered_map<std::uint64_t, std::size_t> agg_index_;

    struct DeclaredJoin {
        std::string out_name, left_name, right_name;
        dftu_join_type type;
    };
    std::vector<DeclaredJoin> joins_;

    // map_spill_dirs_ are the run-scoped temp dirs this fold created or adopted
    // from merged workers; removed after materialize and again in the
    // destructor so an exceptional unwind still cleans up.
    bool map_spill_enabled_ = false;
    bool map_spill_failed_ = false;
    std::size_t map_spill_share_ = 0;
    std::string map_spill_dir_root_;
    std::string map_spill_cur_dir_;
    std::vector<std::string> map_spill_dirs_;
    std::uint64_t map_spill_run_seq_ = 0;
    std::size_t map_spill_count_ = 0;

    bool map_stream_enabled_ = false;
    std::size_t map_stream_chunk_rows_ = 0;
    std::size_t map_stream_batches_ = 0;
    std::size_t map_stream_max_resident_parts_ = 0;

    // Backs the dftu_batch passed to a lazy async on_batch; must outlive
    // pending_, so it lives here not on step()'s stack.
    dftu_batch cbatch_{};

    // unique_ptr keeps each CoroTask address stable so when_all/then can
    // reference them; cleared once the awaited root completes.
    std::vector<std::unique_ptr<coro::CoroTask<void>>> task_arena_;
    // deque keeps op node addresses stable as combinators reference children.
    std::deque<ComposeOp> compose_ops_;
    std::vector<std::unique_ptr<PluginWriter>> writers_;
    std::vector<std::unique_ptr<PluginTraceWriter>> trace_writers_;

    // Unset means deliver every event.
    std::optional<Query> query_;
    ValueMap qmap_;

    std::deque<Query> compiled_queries_;
    ValueMap match_qmap_;

    std::vector<std::size_t> kept_scratch_;
    std::vector<dftu_event> event_scratch_;
    std::vector<dftu_arg> arg_scratch_;
    std::vector<FoldEvent> col_scratch_;  // vectorized-fold materialization
};

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_FOLD_ADAPTER_H
