#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_H

#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/utilities/common/query/query.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>
#include <dftracer/utils/utilities/composites/dft/trace_config.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::indexing {
class BloomFilterCache;
}

namespace dftracer::utils::utilities::common::json {
class JsonValue;
}

namespace dftracer::utils::utilities::composites::dft::views {

using common::query::Query;

/// `ph="X"` events, `ph="C"` counters, or both.
enum class Phase { Events, Counters, Any };

/// One column of a (possibly composite) group-by key. `Arg` groups on an
/// args-map entry named by `arg`; the rest group on the like-named field.
struct GroupKey {
    // Fhash/Hhash group on the args file/host hash; IoCat is the dfanalyzer
    // I/O category derived from the function name; AccPat is the access pattern
    // (0 today). FilePath/HostName group on the same hash as Fhash/Hhash but
    // the group key is relabeled to the resolved name after aggregation (a
    // bijection, so grain is identical), keeping the fold hash-only.
    enum class Kind {
        Name,
        Cat,
        Pid,
        Tid,
        Fhash,
        Hhash,
        IoCat,
        AccPat,
        FilePath,
        FileName,
        HostName,
        Arg
    };
    /// Value transform applied to the resolved group value, before the
    /// merge key is built. Coarsens the grain (many values fold to one), so
    /// it must be part of any cache identity derived from the plan.
    enum class Transform { None, Dirname, Basename, Lower, Bucket };

    Kind kind = Kind::Name;
    std::string arg;
    Transform transform = Transform::None;
    /// Bucket substrings, in priority order; the first one contained in the
    /// value wins and values matching none fold to empty. Bucket only.
    std::vector<std::string> transform_args{};

    static GroupKey name() { return {Kind::Name, {}}; }
    static GroupKey cat() { return {Kind::Cat, {}}; }
    static GroupKey pid() { return {Kind::Pid, {}}; }
    static GroupKey tid() { return {Kind::Tid, {}}; }
    static GroupKey fhash() { return {Kind::Fhash, {}}; }
    static GroupKey hhash() { return {Kind::Hhash, {}}; }
    static GroupKey io_cat() { return {Kind::IoCat, {}}; }
    static GroupKey acc_pat() { return {Kind::AccPat, {}}; }
    static GroupKey file_path() { return {Kind::FilePath, {}}; }
    static GroupKey file_name() { return {Kind::FileName, {}}; }
    static GroupKey host_name() { return {Kind::HostName, {}}; }
    static GroupKey of_arg(std::string key) {
        return {Kind::Arg, std::move(key)};
    }
};

/// Var/Std are population variance / standard deviation of `field`, over the
/// same N as Mean (the group's event count). Pct is the `q`-quantile of `field`
/// from a DDSketch (q in (0,1)).
enum class AggOp {
    Count,
    Sum,
    Min,
    Max,
    Mean,
    Var,
    Std,
    ArgMax,
    SumSq,
    Pct,
    Skew,     // population skewness of `field`
    Kurt,     // population excess kurtosis of `field`
    Hist,     // raw histogram of `field` from a DDSketch
              // (list<struct<lo,hi,count>>)
    SetUnion  // distinct string values of `field`, emitted as a delimiter-
              // joined text column (sorted)
};

/// `field` is the field to reduce (ignored for `Count()`; for `ArgMax` it is
/// the value returned, e.g. "name"). `by` is the numeric field maximized over
/// for `ArgMax` (e.g. "dur"). `q` is the quantile for `Pct` (e.g. 0.99).
/// `out_name` names the output column.
struct AggSpec {
    AggOp op = AggOp::Count;
    std::string field;
    std::string out_name;
    std::string by;  // ArgMax only: the field whose max selects `field`
    double q = 0.0;  // Pct only: the quantile in (0, 1)

    AggSpec() = default;
    AggSpec(AggOp o, std::string f = "", std::string out = "",
            std::string b = "", double quant = 0.0)
        : op(o),
          field(std::move(f)),
          out_name(std::move(out)),
          by(std::move(b)),
          q(quant) {}
};

/// A file and its index. Sizes are optional; zero means read them from the
/// index.
struct ViewFile {
    std::string file_path;
    std::string index_path;
    std::uint64_t uncompressed_size = 0;
    std::size_t num_checkpoints = 0;
    std::size_t checkpoint_size = 0;  // 0 = indexer default
};

struct ExportStats {
    std::uint64_t events_matched = 0;
    std::uint64_t events_scanned = 0;
    std::uint64_t chunks_scanned = 0;
    std::uint64_t chunks_skipped = 0;
    /// Chunks read end to end. Below chunks_scanned when the scan stopped
    /// early, so a chunk was touched but not wholly seen.
    std::uint64_t chunks_covered = 0;
    /// Every index artifact riding along on this scan persisted. False when
    /// none was attached, or when one declined (nothing to record, over
    /// budget, a prerequisite artifact missing).
    bool artifacts_committed = false;
    bool truncated = false;  // scan stopped early on a limit
    /// This read was served from a materialized view instead of the base
    /// traces (the scan ran over a subsuming filtered-trace MV).
    bool served_from_mv = false;
};

/// Options for export_trace(): write the View's events as a new re-indexable
/// multi-member trace through the parallel (PFS-aware) writer. `member_size` is
/// the target uncompressed bytes per gzip member (0 = default); `num_workers`
/// caps the writer fan-out. The writer layout (striped/sharded/padded) and the
/// exact worker count are chosen from the target filesystem.
struct TraceWriteOptions {
    std::string output_path;
    std::size_t member_size = 0;
    std::size_t num_workers = 1;
    bool compress = true;
    int level = 6;
    /// Build the member+bloom+stats index during the write (no re-inflate of
    /// the output). `index_path` is the index DB directory; empty falls back to
    /// the indexer default location for `output_path`. When set, the write uses
    /// a single in-order member stream (scan/decompress stays parallel) so the
    /// index needs no post-hoc offset remap.
    bool build_index = false;
    std::string index_path;
    /// Roll to a new `part-<k>` file once a part reaches this many uncompressed
    /// bytes, so one large result splits into several indexed part files
    /// sharing `index_path`. 0 = one file. Only honored on the indexed write
    /// path.
    std::size_t part_size = 0;
};

/// Streaming output target. The executor serializes each `write` call, so an
/// implementation only needs to handle one call at a time.
class ExportSink {
   public:
    virtual ~ExportSink() = default;
    virtual void write(std::string_view data) = 0;
};

/// `keys` aligns to `group_columns`, `values` to `value_columns` (numeric
/// aggs), `texts` to `text_columns` (string aggs like ArgMax).
struct ResultRow {
    std::vector<std::string> keys;
    std::vector<double> values;
    std::vector<std::string> texts;
    // Aligned to ResultTable::hist_columns: one raw histogram per Hist agg.
    std::vector<std::vector<common::statistics::HistogramBin>> hists;
};

struct ResultTable {
    std::vector<std::string> group_columns;
    std::vector<std::string> value_columns;
    std::vector<std::string> text_columns;
    std::vector<std::string> hist_columns;
    std::vector<ResultRow> rows;
};

/// The three record families of one aggregation index, read in a single pass:
/// `regular` (ph="X" events), `aggregated` (aggregated records carrying
/// arbitrary extra-key dims), and `counters` (ph="C" counters incl. system).
/// Returned by collect_typed().
struct TypedResult {
    ResultTable regular;
    ResultTable aggregated;
    ResultTable counters;
};

/// Progress callback for a shard-range read: called with (done, total) shard
/// units as the scan advances through the range.
using ProgressFn = std::function<void(std::size_t, std::size_t)>;

/// Result of map_batches(): the reduced fold `value` plus the `stats` of the
/// scan that produced it.
template <class P>
struct BatchResult {
    P value;
    ExportStats stats;
};

namespace detail {
struct ViewPlan;
class PartialSource;  // view_aggregate.h (internal)

/// Drive the plan's index-pruned parallel scan; invoke `on_batch(slot, events)`
/// for each decoded batch on worker slots in [0, num_slots). Each slot runs
/// single-threaded, so per-slot state needs no locking. Stops early once
/// `limit` (0 = unlimited) events are produced. Defined in view_executor.cpp;
/// declared here so the map_batches template can reach it. Fold logic lives in
/// the caller, not here, so map_batches stays type-erasure-free.
coro::CoroTask<ExportStats> run_scan_batches(
    const std::shared_ptr<const ViewPlan>& plan, std::size_t num_slots,
    std::uint64_t limit,
    const std::function<void(std::size_t,
                             const std::vector<std::string_view>&)>& on_batch);

// One branch of a fused partition: a predicate selecting events, a per-event
// `consume` (given the slot, the parsed event, and its raw JSON), and a
// `finalize` that reduces the branch's per-slot partials into its result. Plain
// std::functions so branch state stays private to whoever builds the hooks.
struct BranchHooks {
    std::optional<common::query::Query> predicate;  // nullopt = match all
    std::function<void(std::size_t, const common::json::JsonValue&,
                       std::string_view)>
        consume;
    std::function<void()> finalize;
};

struct PartitionState;
std::shared_ptr<PartitionState> make_partition_state(
    std::shared_ptr<const ViewPlan> plan, std::size_t num_slots,
    std::uint64_t limit);
void add_branch(PartitionState& state, BranchHooks hooks);
coro::CoroTask<ExportStats> run_partition(
    std::shared_ptr<PartitionState> state);

// Built-in branch terminals (defined in view_executor.cpp, where the fold
// engine lives). The caller sets `predicate` on the returned hooks.
BranchHooks make_collect_branch(std::vector<GroupKey> group_by,
                                std::vector<AggSpec> agg,
                                std::shared_ptr<ResultTable> out,
                                std::size_t num_slots);
BranchHooks make_export_branch(ExportSink& sink,
                               std::shared_ptr<ExportStats> out);
}  // namespace detail

/// A fused multi-branch run over a single scan of a base View. Attach a
/// terminal to each predicate-branch, then execute() scans the base once and
/// routes each event (parsed once) to every branch whose predicate matches. The
/// base filter is applied in the scan; only the per-branch predicates run per
/// event. Branch results are read from the returned typed handles after
/// execute() completes.
class PartitionRun {
   public:
    /// Aggregate the branch's matching events (group_by + agg) into a table.
    std::shared_ptr<ResultTable> collect(Query predicate,
                                         std::vector<GroupKey> group_by,
                                         std::vector<AggSpec> agg);

    /// Stream the branch's matching events verbatim to `sink`.
    std::shared_ptr<ExportStats> export_json(Query predicate, ExportSink& sink);

    /// Fold the branch's matching events into a caller partial `P`, reduced
    /// across slots by `combine`. The fold gets the parsed event (no re-parse)
    /// and its raw JSON bytes (for verbatim passthrough). `P` must be
    /// default-constructible and a default `P` the identity of combine.
    template <class P>
    std::shared_ptr<P> fold(
        Query predicate,
        std::function<void(P&, const common::json::JsonValue&,
                           std::string_view)>
            f,
        std::function<P(P&&, P&&)> combine) {
        auto out = std::make_shared<P>();
        auto partials = std::make_shared<std::vector<P>>(num_slots_);
        detail::BranchHooks h;
        h.predicate = std::move(predicate);
        h.consume = [f = std::move(f), partials](
                        std::size_t slot, const common::json::JsonValue& jv,
                        std::string_view raw) {
            f((*partials)[slot], jv, raw);
        };
        h.finalize = [partials, out, combine = std::move(combine)]() {
            P acc = std::move((*partials)[0]);
            for (std::size_t i = 1; i < partials->size(); ++i)
                acc = combine(std::move(acc), std::move((*partials)[i]));
            *out = std::move(acc);
        };
        detail::add_branch(*state_, std::move(h));
        return out;
    }

    /// Scan the base once and run every attached branch.
    coro::CoroTask<ExportStats> execute();

   private:
    friend class View;
    PartitionRun(std::shared_ptr<const detail::ViewPlan> plan,
                 std::size_t num_slots, std::uint64_t limit)
        : num_slots_(num_slots ? num_slots : 1),
          state_(detail::make_partition_state(std::move(plan), num_slots_,
                                              limit)) {}
    std::size_t num_slots_;
    std::shared_ptr<detail::PartitionState> state_;
};

class AggregatedView;

/// A composable, lazy, relational view over trace events - the pandas/SQL-table
/// of a trace. Builder ops return a new `View` and do no I/O; terminals execute
/// the plan once, pushing filters into the index (bloom + chunk pruning) and
/// scanning candidates in parallel. The only public entry point; the
/// planner/scanner/aggregator engines it drives are internal.
class View {
   public:
    /// An empty view (no files); collects to an empty result. Also the
    /// placeholder a `coro::CoroTask<View>` needs before co_return.
    View();

    static View from_files(std::vector<ViewFile> files,
                           indexing::BloomFilterCache* bloom_cache = nullptr);
    static View from_file(std::string file_path, std::string index_path = "");

    /// Scan `dir` recursively for .pfw.gz files via the parallel directory
    /// scanner and build a View over them (sorted for determinism). Each file
    /// gets its own index, resolved under `index_path` (empty = the sidecar
    /// beside each trace, built on first touch). Self-contained; co_await it.
    static coro::CoroTask<View> from_directory(std::string dir,
                                               std::string index_path = "");

    // Builder ops (lazy; each returns a new View).
    View filter(Query q) const;
    View query(const std::string& dsl) const;
    View phase(Phase p) const;
    View time_range(double begin, double end) const;
    View time_bucket(std::uint64_t interval_us) const;
    /// Normalize ts/dur/te by `ns_ratio` = source_ns_per_unit /
    /// target_ns_per_unit (1.0 = none), applied before time_bucket. Callers
    /// resolve the trace's native unit (read_time_metric) and the target.
    View time_scale(double ns_ratio) const;
    /// group_by/agg/agg_numeric_args return an AggregatedView: the same lazy
    /// plan, but the type now advertises an aggregation, which is what unlocks
    /// the cache terminals (persist/reconstruct). A plain View cannot reach
    /// them, so a raw-event plan can never be persisted by mistake.
    AggregatedView group_by(std::vector<GroupKey> keys) const;
    AggregatedView agg(std::vector<AggSpec> specs) const;
    /// Aggregate every numeric args.* field as a per-group mean, one value
    /// column per discovered field (metadata and pre-aggregated *_sum/_min/_max
    /// args are skipped). Lets counters aggregate without naming the fields up
    /// front.
    AggregatedView agg_numeric_args() const;
    /// Out-of-core aggregation budget shared by collect(), export_counters(),
    /// and aggregate_partial(): cap each worker's in-memory group map at
    /// `bytes`, spilling to sorted temp runs that are k-way merged at the end,
    /// so the scan/merge peak stays bounded no matter the group cardinality.
    /// 0 = pure in-memory. Note collect() still materializes the final table,
    /// so this bounds intermediate memory, not the returned result.
    View memory_budget(std::uint64_t bytes) const;
    /// Convenience for memory_budget(): spill at ~1/3 of the machine's
    /// available memory, so aggregation stays bounded without a hand-picked
    /// number.
    View auto_spill() const;
    View select(std::vector<std::string> cols) const;
    /// Pagination: cap the output to `n` rows/events (0 = unlimited) after
    /// skipping `offset()` leading ones. On collect() this slices the result
    /// rows; on export it caps the streamed events.
    View limit(std::uint64_t n) const;
    View offset(std::uint64_t n) const;
    /// Opt in to persisting this query's result as a materialized view so a
    /// later matching query reads it back instead of rescanning. A performance
    /// lever for repeated, stable query shapes (e.g. dfanalyzer); a plain
    /// collect()/read never persists. For a row query, run() writes a filtered
    /// trace split into `part_size`-byte files, each indexed at
    /// `checkpoint_size` granularity (0 = engine defaults); for an aggregation
    /// it persists a rollup.
    View materialize(std::uint64_t checkpoint_size = 0,
                     std::uint64_t part_size = 0) const;
    /// Drop `ph="M"` metadata from streamed output (default: keep).
    View metadata(bool include) const;
    /// Harvest every hash-metadata (FH/HH/SH) record even when no scanned event
    /// references it (default: off). For whole-trace metadata collection.
    View emit_all_metadata(bool v) const;
    /// Use a materialized-aggregate source for collect() when the
    /// aggregation is reducible (covered chunks answered without decode).
    View with_partial_source(const detail::PartialSource* source) const;
    /// Override the root directory for the aggregation cache (persist/
    /// reconstruct); empty derives it from the files' index location.
    View rollup_root(std::string dir) const;
    /// Override the root directory for materialized filtered-trace views; empty
    /// derives `<parent-of-index>/.dftindex-views`.
    View views_root(std::string dir) const;
    /// Cooperative cancellation: every terminal polls `pred` at its scan/plan
    /// loop boundaries and bails out when it returns true. Set once here rather
    /// than per terminal. The predicate owns its own semantics (e.g. a server
    /// request token that also probes the client socket).
    View cancel_when(std::function<bool()> pred) const;

    /// True when export_json() / collect() on this view would take the raw-gzip
    /// bootstrap: answer the query and build the index in one pass over an
    /// unindexed first-touch input. A caller that would otherwise pre-build the
    /// index eagerly can skip that when this is true and let the bootstrap do
    /// it. Applies to the plain export_json / collect() paths respectively.
    bool export_would_bootstrap() const;
    bool collect_would_bootstrap() const;

    /// Stream matching events to `sink` as newline-delimited JSON; the emitted
    /// lines are verbatim trace events, so a file/gzip sink yields a normal
    /// re-indexable dftracer trace.
    coro::CoroTask<ExportStats> export_json(ExportSink& sink) const;

    /// Write matching events out as a new multi-member trace via the parallel
    /// writer, choosing striped/sharded/padded layout for the target
    /// filesystem. Each gzip member holds whole lines (self-contained), so
    /// members are valid at any offset and the result re-indexes normally. This
    /// is the PFS-aware path behind `dftracer_view --merge`/`--compress`.
    coro::CoroTask<ExportStats> export_trace(TraceWriteOptions opts) const;

    /// The trace self-description (one entry per process, from each file's
    /// "end" event). Probes each file's last gzip member first (no decode of
    /// the rest); for a file whose end is scattered elsewhere (reorganized
    /// trace), falls back to a bloom-pruned scan when the file is indexed.
    /// Ignores the view's filters/agg.
    std::vector<TraceConfig> config() const;

    /// Run group_by + agg. No agg counts per group; no group_by folds the whole
    /// set into one row.
    coro::CoroTask<ResultTable> collect() const;

    /// Build-only terminal: run the query for its side effect - materialize the
    /// rollup - and return scan stats, with no ResultTable. The prewarm form of
    /// materialize(); idempotent when the view is already materialized.
    coro::CoroTask<ExportStats> run(const ProgressFn* progress = nullptr) const;

    /// Distributed row-MV materialize. The coordinator (a View over the FULL
    /// file set) calls materialize_dir() to create and get the shared MV
    /// directory; each rank exports its files' filtered events into a subdir of
    /// it (export_trace with build_index, e.g. `<dir>/shard-<r>/part.pfw.gz`);
    /// then the coordinator calls register_materialized(dir) to write the
    /// manifest describing the full base set. A later matching read gathers the
    /// parts across all subdirs. Empty dir when there is no views anchor.
    std::string materialize_dir() const;
    void register_materialized(const std::string& dir) const;

    /// Observability: the materialized-view trace file(s) that would serve this
    /// query (a subsuming MV), or empty if a read would scan the base. Lets a
    /// caller report or assert MV reuse without running the query.
    std::vector<std::string> mv_source() const;

    /// Read the aggregation index's regular/aggregated/counters families in one
    /// pass over shard range [shard_begin, shard_end). Distributed callers fan
    /// disjoint ranges across workers and concatenate; counters are read only
    /// when the range starts at shard 0. Empty with no aggregation tier.
    coro::CoroTask<TypedResult> collect_typed(
        int shard_begin = 0, int shard_end = 4096,
        const ProgressFn* progress = nullptr) const;

    /// Aggregate (as collect()) and emit each row as a dftracer ph="C" counter
    /// event. With a time_bucket this is the aggregator's counter-trace format,
    /// so the output is a normal re-indexable counter trace.
    coro::CoroTask<ExportStats> export_counters(ExportSink& sink) const;

    /// Distributed aggregation. Each rank aggregates its shard of files with
    /// the view's group_by/agg and returns an opaque serialized partial;
    /// combine partials from all ranks with merge_partials_to_table()
    /// (materialized collect) or merge_counter_partials() (streamed ph="C"
    /// counters). The transport between ranks (MPI etc.) is the caller's - the
    /// View core carries no MPI dependency.
    coro::CoroTask<std::string> aggregate_partial() const;
    ResultTable merge_partials_to_table(
        const std::vector<std::string_view>& partials) const;
    ExportStats merge_counter_partials(
        const std::vector<std::string_view>& partials, ExportSink& sink) const;

    /// Escape hatch for aggregations the built-in agg vocabulary cannot express
    /// (custom bucketing, representative values, side outputs). `fold` runs
    /// over each scanned batch on one of `num_slots` worker slots, mutating
    /// that slot's partial; `combine` reduces the per-slot partials into one.
    /// The fold keeps its own logic and runs over the view's index-pruned
    /// parallel scan, so a caller consolidates onto View without forcing its
    /// shape into group_by/agg. `P` must be default-constructible and a default
    /// `P` must be the identity for `combine` (unused slots fold nothing).
    /// `limit` (0 = unlimited) caps produced events; the result's
    /// stats.truncated reports it.
    template <class P>
    coro::CoroTask<BatchResult<P>> map_batches(
        std::function<void(P&, const std::vector<std::string_view>&)> fold,
        std::function<P(P&&, P&&)> combine, std::size_t num_slots,
        std::uint64_t limit = 0) const {
        if (num_slots == 0) num_slots = 1;
        std::vector<P> partials(num_slots);
        ExportStats stats = co_await detail::run_scan_batches(
            plan_, num_slots, limit,
            [&](std::size_t slot, const std::vector<std::string_view>& events) {
                fold(partials[slot], events);
            });
        P acc = std::move(partials[0]);
        for (std::size_t i = 1; i < partials.size(); ++i)
            acc = combine(std::move(acc), std::move(partials[i]));
        co_return BatchResult<P>{std::move(acc), stats};
    }

    /// Lowest-level scan terminal: invoke `on_batch(slot, events)` for each
    /// decoded batch on worker slots in [0, num_slots), stopping at `limit`
    /// (0 = unlimited). The caller owns all per-slot state and merges after -
    /// use this when a fold keeps a big shared/per-slot accumulator that does
    /// not fit map_batches' per-slot partial + combine model. map_batches and
    /// partition build on this.
    coro::CoroTask<ExportStats> for_each_batch(
        std::function<void(std::size_t, const std::vector<std::string_view>&)>
            on_batch,
        std::size_t num_slots, std::uint64_t limit = 0) const;

    /// Begin a fused multi-branch run over one scan of this view (see
    /// PartitionRun). `num_slots` bounds the parallel worker slots; `limit`
    /// (0 = unlimited) caps produced events.
    PartitionRun partition(std::size_t num_slots,
                           std::uint64_t limit = 0) const;

   protected:
    /// Rollup terminals, reachable only through AggregatedView (which promotes
    /// them to public). materialize_partials builds the rollup from distributed
    /// aggregate_partial output (each rank scans its shard, the coordinator
    /// reduces and writes); reconstruct_if_cached returns a subsuming rollup as
    /// a table, or nullopt on a miss, so a distributed caller picks read vs
    /// recompute without a scan.
    coro::CoroTask<void> materialize_partials(
        const std::vector<std::string_view>& partials) const;
    std::optional<ResultTable> reconstruct_if_cached() const;

    std::shared_ptr<const detail::ViewPlan> plan_;

   private:
    explicit View(std::shared_ptr<const detail::ViewPlan> plan);
};

/// A View whose type advertises a group_by/agg. Structurally identical to View
/// (same plan, same terminals), but it additionally exposes the distributed
/// rollup terminals materialize_partials()/reconstruct_if_cached(). Reached
/// only via View::group_by/agg/agg_numeric_args so a raw-event plan can never
/// call them. Builder ops are re-exposed to return AggregatedView, keeping the
/// type through a chain. A plain collect() (inherited) already reads a
/// subsuming rollup when one exists, so there is no separate cache flag:
/// materialize() builds, collect() reads.
class AggregatedView : public View {
   public:
    using View::materialize_partials;
    using View::reconstruct_if_cached;

    AggregatedView filter(Query q) const {
        return {View::filter(std::move(q))};
    }
    AggregatedView query(const std::string& dsl) const {
        return {View::query(dsl)};
    }
    AggregatedView phase(Phase p) const { return {View::phase(p)}; }
    AggregatedView time_range(double begin, double end) const {
        return {View::time_range(begin, end)};
    }
    AggregatedView time_bucket(std::uint64_t interval_us) const {
        return {View::time_bucket(interval_us)};
    }
    AggregatedView time_scale(double ns_ratio) const {
        return {View::time_scale(ns_ratio)};
    }
    AggregatedView group_by(std::vector<GroupKey> keys) const {
        return View::group_by(std::move(keys));
    }
    AggregatedView agg(std::vector<AggSpec> specs) const {
        return View::agg(std::move(specs));
    }
    AggregatedView agg_numeric_args() const { return View::agg_numeric_args(); }
    AggregatedView memory_budget(std::uint64_t bytes) const {
        return {View::memory_budget(bytes)};
    }
    AggregatedView auto_spill() const { return {View::auto_spill()}; }
    AggregatedView select(std::vector<std::string> cols) const {
        return {View::select(std::move(cols))};
    }
    AggregatedView limit(std::uint64_t n) const { return {View::limit(n)}; }
    AggregatedView offset(std::uint64_t n) const { return {View::offset(n)}; }
    AggregatedView materialize(std::uint64_t checkpoint_size = 0,
                               std::uint64_t part_size = 0) const {
        return {View::materialize(checkpoint_size, part_size)};
    }
    AggregatedView metadata(bool include) const {
        return {View::metadata(include)};
    }
    AggregatedView emit_all_metadata(bool v) const {
        return {View::emit_all_metadata(v)};
    }
    AggregatedView with_partial_source(
        const detail::PartialSource* source) const {
        return {View::with_partial_source(source)};
    }
    AggregatedView rollup_root(std::string dir) const {
        return {View::rollup_root(std::move(dir))};
    }
    AggregatedView views_root(std::string dir) const {
        return {View::views_root(std::move(dir))};
    }
    AggregatedView cancel_when(std::function<bool()> pred) const {
        return {View::cancel_when(std::move(pred))};
    }

   private:
    friend class View;
    AggregatedView(View v) : View(std::move(v)) {}
};

}  // namespace dftracer::utils::utilities::composites::dft::views

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_H
