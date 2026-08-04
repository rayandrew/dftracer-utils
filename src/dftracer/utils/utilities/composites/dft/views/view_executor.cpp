#include <dftracer/utils/core/common/byte_view.h>
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/rocksdb/column_families.h>
#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/utilities/common/serialization/binary_codec.h>
#include <dftracer/utils/utilities/composites/dft/views/aggfold.h>
#include <dftracer/utils/utilities/composites/dft/views/bloom_fold.h>
#include <dftracer/utils/utilities/composites/dft/views/coverage.h>
#include <dftracer/utils/utilities/composites/dft/views/dict_fold.h>
#include <dftracer/utils/utilities/composites/dft/views/fold.h>
#include <dftracer/utils/utilities/composites/dft/views/index_fold_driver.h>
#include <dftracer/utils/utilities/composites/dft/views/mv_store.h>
#include <dftracer/utils/utilities/composites/dft/views/rollup_store.h>
#include <dftracer/utils/utilities/composites/dft/views/view_agg_tier.h>
#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>
#include <dftracer/utils/utilities/composites/dft/views/view_counter_format.h>
#include <dftracer/utils/utilities/composites/dft/views/view_definition.h>
#include <dftracer/utils/utilities/composites/dft/views/view_executor.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scan.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scanner_utility.h>
#include <dftracer/utils/utilities/composites/dft/views/view_spill.h>
#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>
#include <dftracer/utils/utilities/fileio/parallel/merge.h>
#include <dftracer/utils/utilities/fileio/parallel/parallel_writer.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/index_database_writer_context.h>
#include <dftracer/utils/utilities/indexer/internal/gzip/gzip_indexer.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>
#include <dftracer/utils/utilities/indexer/internal/indexer.h>
#include <rocksdb/sst_file_writer.h>
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

// The View terminals: each lowers a ViewPlan onto the module pieces and runs a
// pruned parallel scan. The scan Source lives in view_scan (gather_units +
// make_vdef); the group/agg fold in view_aggregate; ph="C" emission in
// view_counter_format; out-of-core spill in view_spill. A terminal is therefore
// "gather units -> fan out a scan -> feed each batch to its own body (write /
// fold / spill / write_chunk / branch-route)". The bodies differ on purpose;
// only the shared scan setup is factored out.
namespace dftracer::utils::utilities::composites::dft::views::detail {

// Every file is a clean first touch: it has an index path that does not exist
// yet, so a bootstrap can build it (and there is nothing to read instead).
static bool all_files_fresh(const ViewPlan& plan) {
    for (const auto& f : plan.files)
        if (f.index_path.empty() || fs::exists(f.index_path)) return false;
    return true;
}

bool export_bootstrap_eligible(const ViewPlan& plan) {
    if (plan.query || plan.time_range || plan.offset || plan.limit ||
        plan.phase != Phase::Any || !plan.include_metadata ||
        plan.files.empty())
        return false;
    return all_files_fresh(plan);
}

bool collect_bootstrap_eligible(const ViewPlan& plan) {
    if (plan.files.empty() || plan.time_range) return false;
    if (plan.query && !query_evaluable_by_fold(*plan.query)) return false;
    // A partial source (chunk-stats fast path) answers only from an existing
    // index, so it contributes nothing once all_files_fresh holds - no need to
    // exclude it here.
    return all_files_fresh(plan);
}

namespace {

// Write the index a bootstrap pass produced: members from the artifacts, bloom
// + hash from the folds. Caps claim MEMBERS/FILE_SUMMARY/INDEXING_COMPLETE
// here; BloomFold adds BLOOM.
void persist_bootstrap_index(
    const std::string& index_path, const std::string& file_path,
    const utilities::indexer::internal::gzip::GzipBuildArtifacts& arts,
    BloomFold& bloom, DictFold& dict) {
    namespace idx = utilities::indexer;
    idx::IndexDatabase db(index_path);
    auto logical = idx::internal::get_logical_path(file_path);
    const auto hash = idx::internal::calculate_file_hash(file_path);
    const auto mtime = static_cast<std::uint64_t>(
        idx::internal::get_file_modification_time(file_path));
    const auto bytes = idx::internal::file_size_bytes(file_path);
    auto w = db.begin_write();
    int fid = w->get_or_create_file_info(
        logical, hash,
        idx::IndexFileEntryCapability::INDEXING_COMPLETE |
            idx::IndexFileEntryCapability::MEMBERS |
            idx::IndexFileEntryCapability::FILE_SUMMARY,
        mtime, bytes);
    for (const auto& m : arts.members) w->insert_gzip_member(fid, m);
    w->insert_file_metadata(fid, arts.checkpoint_size, arts.total_lines,
                            arts.total_uc_size);
    bloom.write_to_sink(*w, fid);
    dict.write_to_sink(*w);
    w->add_file_capability(fid, idx::IndexFileEntryCapability::BLOOM);
    w->commit();
}

// An IndexVisitor that streams a member's plaintext lines straight to the
// export sink. Carries the partial last line across on_chunk calls so an event
// straddling an inflate boundary is written whole.
class SinkWriterVisitor : public utilities::indexer::IndexVisitor {
   public:
    explicit SinkWriterVisitor(ExportSink& sink) : sink_(&sink) {}
    void begin(std::size_t) override {}
    coro::CoroTask<void> on_checkpoint(std::size_t) override { co_return; }
    coro::CoroTask<void> on_chunk(const char* data, std::size_t len,
                                  std::size_t) override {
        buf_.append(data, len);
        const std::size_t last_nl = buf_.rfind('\n');
        if (last_nl != std::string::npos) {
            sink_->write(std::string_view(buf_.data(), last_nl + 1));
            buf_.erase(0, last_nl + 1);
        }
        co_return;
    }
    void finalize(utilities::indexer::IndexDatabaseWriterContext&,
                  int) override {}

   private:
    ExportSink* sink_;
    std::string buf_;
};

// First-touch full export: for an unindexed, unfiltered whole-trace dump, one
// raw-gzip pass per file streams every line to the sink AND builds the index.
// Returns false unless every file is a clean first touch and the export is a
// plain full dump (no filter, window, pagination, or phase/metadata selection),
// so anything else falls through to the streaming scanner path.
coro::CoroTask<bool> try_export_bootstrap(const ViewPlan& plan,
                                          ExportSink& sink) {
    namespace idx = utilities::indexer;
    namespace gzi = utilities::indexer::internal::gzip;
    if (!export_bootstrap_eligible(plan)) co_return false;

    for (const auto& f : plan.files) {
        dftracer::utils::StringIntern intern;
        BloomFold bloom(intern);
        DictFold dict(intern);
        std::array<Fold*, 2> folds{&bloom, &dict};
        IndexFoldDriver driver(intern, folds, f.file_path, f.index_path);
        SinkWriterVisitor writer(sink);

        gzi::GzipBuildArtifacts arts;
        bool ok = false;
        co_await dftracer::utils::run_coro_scope(
            [&](dftracer::utils::CoroScope& scope) -> coro::CoroTask<void> {
                idx::internal::Indexer::VisitorList vl;
                vl.emplace_back(driver);
                vl.emplace_back(writer);
                auto a = co_await gzi::build_gzip_index_artifacts(
                    f.file_path,
                    idx::internal::Indexer::DEFAULT_CHECKPOINT_SIZE, vl,
                    &scope);
                if (a) {
                    arts = std::move(*a);
                    ok = true;
                }
                co_return;
            });
        if (!ok) co_return false;
        driver.seal();
        persist_bootstrap_index(f.index_path, f.file_path, arts, bloom, dict);
    }
    co_return true;
}

}  // namespace

namespace {

// Re-point a read at a materialized filtered-trace view that subsumes it, if
// one exists. The predicate/window/phase are kept so the scan re-applies them
// on the (smaller) MV rows - the compensation that makes reuse correct even
// when the MV is broader than the query.
ViewPlan redirect_reads(const ViewPlan& plan, bool* redirected = nullptr) {
    ViewPlan eff = plan;
    if (auto mv = find_subsuming_view(plan)) {
        eff.files = std::move(*mv);
        if (redirected) *redirected = true;
    }
    return eff;
}

}  // namespace

coro::CoroTask<ExportStats> run_export(const ViewPlan& base_plan,
                                       ExportSink& sink) {
    bool from_mv = false;
    const ViewPlan plan = redirect_reads(base_plan, &from_mv);
    // First touch: one raw-gzip pass dumps the trace and builds its index.
    if (co_await try_export_bootstrap(plan, sink)) co_return ExportStats{};

    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/false);
    std::mutex sink_mtx;
    // Pagination: skip `offset` events then write `limit`. The scan cap stops
    // the fan-out once offset+limit events are produced; the offset/limit
    // window is applied under the sink lock, so it is scan-order (not a stable
    // page).
    const std::uint64_t cap = plan.limit ? plan.offset + plan.limit : 0;
    std::uint64_t seen = 0, written = 0;
    // Build whatever index artifacts this scan can establish along the way.
    // Nothing attaches unless the scan already satisfies it, so this cannot
    // make the read slower than not having it.
    dftracer::utils::StringIntern intern;
    auto folds = select_index_folds(plan, vdef, intern);
    std::vector<Fold*> fold_ptrs;
    for (auto& f : folds) fold_ptrs.push_back(f.get());
    // Stream matched events verbatim; the scan driver owns the gather +
    // fan-out.
    ExportStats stats = co_await for_each_scanned_batch(
        plan, vdef, /*num_slots=*/0, cap,
        [&](std::size_t, const std::vector<std::string_view>& events) {
            std::lock_guard<std::mutex> lk(sink_mtx);
            for (const auto& ev : events) {
                if (seen++ < plan.offset) continue;
                if (plan.limit && written >= plan.limit) break;
                sink.write(ev);
                sink.write("\n");
                ++written;
            }
        },
        fold_ptrs, &intern);
    stats.served_from_mv = from_mv;
    co_return stats;
}

coro::CoroTask<ExportStats> run_export_trace(const ViewPlan& plan,
                                             const TraceWriteOptions& opts) {
    namespace pfw = utilities::fileio::parallel;
    namespace cmp = utilities::fileio::compress;
    // Member size defaults to the checkpoint granularity (a member == a chunk).
    constexpr std::size_t DEFAULT_FLUSH_BYTES =
        constants::indexer::DEFAULT_CHECKPOINT_SIZE;
    constexpr std::size_t BUFFER_HEADROOM_BYTES = 1 * 1024 * 1024;

    if (opts.build_index)
        co_return co_await run_export_trace_indexed(plan, opts);

    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/false);

    std::uint64_t skipped = 0;
    auto units = co_await gather_units(plan, vdef, skipped);

    const std::size_t baseline = std::max<std::size_t>(1, opts.num_workers);
    auto cw = pfw::make_writer_for_path(
        {opts.output_path, baseline,
         opts.member_size ? opts.member_size : DEFAULT_FLUSH_BYTES,
         BUFFER_HEADROOM_BYTES, opts.compress});
    const std::size_t num_workers = cw.sizing.num_workers;
    const std::size_t member_size = cw.sizing.flush_threshold;
    auto writer = std::move(cw.writer);

    std::vector<std::uint64_t> matched_v(num_workers, 0);
    std::vector<std::uint64_t> scanned_v(num_workers, 0);
    bool ok = true;

    // One scope owns the writer's internal packer (padded layout) plus the
    // per-worker producers; close() runs inside it so the packer is drained
    // before the scope joins.
    co_await run_coro_scope([&](CoroScope& scope) -> coro::CoroTask<void> {
        if (co_await writer->open(opts.output_path, num_workers, opts.compress,
                                  &scope) != 0) {
            ok = false;
            co_return;
        }

        co_await scope.scope([&](CoroScope& child) -> coro::CoroTask<void> {
            for (std::size_t w = 0; w < num_workers; ++w) {
                child.spawn([&, w](CoroScope&) -> coro::CoroTask<void> {
                    cmp::GzipMemberCompressor comp(opts.level);
                    std::string buf;
                    std::vector<std::uint8_t> scratch;

                    // Emit buf[0, cut) as one self-contained member (whole
                    // lines, so it is valid at any offset). Returns false on
                    // write failure.
                    auto emit = [&](std::size_t cut) -> coro::CoroTask<bool> {
                        ByteView chunk;
                        if (opts.compress) {
                            if (!comp.compress_member_into(scratch, buf.data(),
                                                           cut)) {
                                co_return false;
                            }
                            chunk = ByteView(
                                reinterpret_cast<const char*>(scratch.data()),
                                scratch.size());
                        } else {
                            chunk = ByteView(buf.data(), cut);
                        }
                        bool wrote =
                            co_await writer->write_chunk(w, chunk) == 0;
                        buf.erase(0, cut);
                        co_return wrote;
                    };

                    for (std::size_t i = w; i < units.size();
                         i += num_workers) {
                        if (is_cancelled(plan)) co_return;
                        ViewScannerInput sin =
                            make_scanner_input(units[i], vdef, vdef.query);
                        ViewScannerUtility scanner;
                        auto gen = scanner.process(sin);
                        while (auto b = co_await gen.next()) {
                            if (is_cancelled(plan)) co_return;
                            matched_v[w] += b->events_matched;
                            scanned_v[w] += b->events_scanned;
                            for (const auto& ev : b->events) {
                                buf.append(ev);
                                buf.push_back('\n');
                            }
                            while (buf.size() >= member_size) {
                                std::size_t cut = member_size;
                                while (cut < buf.size() && buf[cut] != '\n')
                                    ++cut;
                                if (cut >= buf.size()) break;  // no newline yet
                                ++cut;  // include it: member holds whole lines
                                if (!co_await emit(cut)) {
                                    ok = false;
                                    co_return;
                                }
                            }
                        }
                    }
                    if (!buf.empty() && !co_await emit(buf.size())) ok = false;
                    co_return;
                });
            }
            co_return;
        });

        if (co_await writer->close() != 0) ok = false;
    });

    if (ok && cw.layout.layout == pfw::FileLayout::SHARDED) {
        auto shards = writer->output_paths();
        if (co_await pfw::merge_shards(opts.output_path, shards) != 0)
            ok = false;
    }

    if (!ok) {
        throw DFTUtilsException(
            ErrorCode::IO, "export_trace failed writing " + opts.output_path);
    }

    ExportStats st;
    st.chunks_skipped = skipped;
    st.chunks_scanned = units.size();
    for (auto m : matched_v) st.events_matched += m;
    for (auto s : scanned_v) st.events_scanned += s;
    co_return st;
}

// Aggregate the plan through the fused fold and invoke `on_group` once per
// folded group (order unspecified). memory_budget bounds peak via AggFold
// spill.
static coro::CoroTask<ExportStats> fused_aggregate(
    const ViewPlan& plan, const ViewDefinition& vdef,
    const std::function<void(const std::string&, const AggAccum&)>& on_group) {
    ensure_schema(plan);  // AggFold reads plan.schema; fuse does not build it
    dftracer::utils::StringIntern intern;
    AggFold agg(plan, intern);
    std::array<Fold*, 1> folds{&agg};
    auto stats = co_await fuse(plan, vdef, folds, intern);
    agg.for_each_sorted_group(on_group);  // streaming, bounded memory
    co_return stats;
}

namespace {

// First-touch fast path: for a plan whose files have no index yet, one raw-gzip
// pass per file both aggregates it and builds its full index (members + bloom +
// hash) as a byproduct, so the query never triggers a separate eager index
// build. Returns false (caller falls through to the indexed path) unless every
// file is a clean first touch and nothing materialized can answer instead.
coro::CoroTask<bool> try_collect_bootstrap(const ViewPlan& plan,
                                           GroupMap& merged) {
    namespace idx = utilities::indexer;
    namespace gzi = utilities::indexer::internal::gzip;
    // A time window still needs the indexed path (the bootstrap has no ts
    // pruning), and a predicate is only applied here when every field it names
    // is POD-evaluable; otherwise defer to the scan, which filters properly.
    if (!collect_bootstrap_eligible(plan)) co_return false;

    for (const auto& f : plan.files) {
        dftracer::utils::StringIntern intern;
        AggFold agg(plan, intern, /*apply_query=*/true);
        BloomFold bloom(intern);
        DictFold dict(intern);
        std::array<Fold*, 3> folds{&agg, &bloom, &dict};
        IndexFoldDriver driver(intern, folds, f.file_path, f.index_path);

        gzi::GzipBuildArtifacts arts;
        bool ok = false;
        co_await dftracer::utils::run_coro_scope(
            [&](dftracer::utils::CoroScope& scope) -> coro::CoroTask<void> {
                idx::internal::Indexer::VisitorList vl;
                vl.emplace_back(driver);
                auto a = co_await gzi::build_gzip_index_artifacts(
                    f.file_path,
                    idx::internal::Indexer::DEFAULT_CHECKPOINT_SIZE, vl,
                    &scope);
                if (a) {
                    arts = std::move(*a);
                    ok = true;
                }
                co_return;
            });
        if (!ok) co_return false;
        driver.seal();
        persist_bootstrap_index(f.index_path, f.file_path, arts, bloom, dict);
        for (auto& [k, a] : agg.finish_map()) {
            if (auto it = merged.find(k); it != merged.end())
                merge_accum(it->second, a, plan);
            else
                merged.emplace(k, std::move(a));
        }
    }
    co_return true;
}

}  // namespace

coro::CoroTask<ExportStats> run_materialize(const ViewPlan& plan,
                                            const ProgressFn* progress) {
    ensure_schema(plan);

    // Row query (no aggregation): materialize a filtered trace + index into a
    // slug directory under .dftindex-views, then record its manifest so a later
    // subsuming query reads it instead of rescanning the base.
    if (plan.group_by.empty() && plan.agg.empty() &&
        !plan.auto_numeric_metrics) {
        if (view_is_fresh(plan)) co_return ExportStats{};  // already built
        const std::string dir = materialize_view_dir(plan);
        if (dir.empty()) co_return ExportStats{};
        // Single-flight: one builder per slug dir. A loser skips - the winner
        // produces the MV. Re-check freshness under the lock in case the winner
        // just finished.
        const int lock = lock_view_dir(dir);
        if (lock < 0) co_return ExportStats{};
        if (view_is_fresh(plan)) {
            unlock_view_dir(lock);
            co_return ExportStats{};
        }
        constexpr std::uint64_t DEFAULT_PART_SIZE = 128ull * 1024 * 1024;
        TraceWriteOptions opts;
        opts.output_path = (fs::path(dir) / "part.pfw.gz").string();
        opts.build_index = true;
        opts.compress = true;
        // One gzip member per index checkpoint, so the user's checkpoint_size
        // is the member size to write at.
        opts.member_size = plan.mv_checkpoint_size;
        opts.part_size =
            plan.mv_part_size ? plan.mv_part_size : DEFAULT_PART_SIZE;
        ExportStats stats;
        try {
            stats = co_await run_export_trace_indexed(plan, opts, progress);
            register_view(dir, plan);
        } catch (...) {
            unlock_view_dir(lock);
            throw;
        }
        unlock_view_dir(lock);
        co_return stats;
    }

    const std::string rdir = rollup_index_path(plan);
    // Idempotent prewarm: skip if this view is already materialized.
    if (!rdir.empty() && fs::exists(fs::path(rdir) / "CURRENT")) {
        try {
            auto db = open_rollup_db(
                rdir, rocksdb::RocksDatabase::OpenMode::ReadOnly);
            if (db && rollup_exists(*db, plan_signature(plan)))
                co_return ExportStats{};
        } catch (const std::exception& e) {
            DFTRACER_UTILS_LOG_WARN("rollup check skipped: %s", e.what());
        }
    }

    // A build wants full coverage, so scan through the fused fold with no
    // tier/agg_source fast path, then persist the complete raw-keyed result.
    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);
    dftracer::utils::StringIntern intern;
    AggFold agg(plan, intern);
    std::array<Fold*, 1> folds{&agg};
    ExportStats stats = co_await fuse(plan, vdef, folds, intern, nullptr);

    GroupMap merged = agg.finish_map();
    if (!rdir.empty()) {
        try {
            auto db = open_rollup_db(
                rdir, rocksdb::RocksDatabase::OpenMode::ReadWrite);
            if (db)
                persist_rollup(*db, plan_signature(plan), rest_signature(plan),
                               plan.time_bucket_us, plan.group_by, merged);
        } catch (const std::exception& e) {
            DFTRACER_UTILS_LOG_WARN("rollup materialize skipped: %s", e.what());
        }
    }
    co_return stats;
}

coro::CoroTask<ResultTable> run_collect(const ViewPlan& plan) {
    ensure_schema(plan);

    // Materialized-view fast path: a stored rollup that subsumes this query
    // answers with no scan. A materialize() query opens ReadWrite so a miss
    // reuses the same handle to persist below (avoids a RO-then-RW conflict).
    {
        const std::string rdir = rollup_index_path(plan);
        if (!rdir.empty() && fs::exists(fs::path(rdir) / "CURRENT")) {
            const auto mode = plan.materialize
                                  ? rocksdb::RocksDatabase::OpenMode::ReadWrite
                                  : rocksdb::RocksDatabase::OpenMode::ReadOnly;
            try {
                auto db = open_rollup_db(rdir, mode);
                if (db) {
                    if (auto r = find_subsuming_rollup(*db, plan)) {
                        resolve_group_keys(*r, plan);
                        ResultTable t = to_result_table(*r, plan);
                        project_columns(t, plan.select);
                        co_return t;
                    }
                }
            } catch (const std::exception& e) {
                // A locked/unreadable rollup just falls through to a normal
                // compute; never fail the query over the cache.
                DFTRACER_UTILS_LOG_WARN("rollup read skipped: %s", e.what());
            }
        }
    }

    // First touch: one raw-gzip pass answers the aggregation and builds the
    // index, so no separate eager build pass is needed.
    {
        GroupMap merged;
        if (co_await try_collect_bootstrap(plan, merged)) {
            resolve_group_keys(merged, plan);
            ResultTable t = to_result_table(merged, plan);
            project_columns(t, plan.select);
            co_return t;
        }
    }

    // Whole-result fast path: the aggregation tier answers the entire grouping
    // from its pre-folded MetricStats, no event scan.
    {
        GroupMap tier;
        if (agg_tier_collect(plan, tier)) {
            resolve_group_keys(tier, plan);
            ResultTable t = to_result_table(tier, plan);
            project_columns(t, plan.select);
            co_return t;
        }
    }

    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);

    GroupMap merged;
    CoverageSet covered;

    // Answer covered chunks from a materialized aggregate (per-chunk stats,
    // summary, ...); scan only what it leaves behind. The source emits AggAccum
    // partials, merged exactly like scanned groups.
    if (plan.agg_source) {
        auto [field, single_field] = source_agg_field(plan);
        // Materialized chunk aggregates cover ALL events in a chunk, so they
        // are only valid when the query constrains nothing but ts (the window,
        // handled by coverage). Any other predicate needs a real scan.
        bool ts_only = true;
        if (plan.query)
            for (auto f : plan.query->fields())
                if (f != "ts") {
                    ts_only = false;
                    break;
                }
        // Var/Std need a sum-of-squares and Pct a DDSketch the chunk aggregates
        // do not carry, so a partial coverage would be wrong; full scan
        // instead.
        bool has_variance = false;
        for (const auto& s : plan.agg)
            if (s.op == AggOp::Var || s.op == AggOp::Std ||
                s.op == AggOp::Pct || s.op == AggOp::Skew ||
                s.op == AggOp::Kurt)
                has_variance = true;
        // A field-less (Count(*)) aggregation is a row count the dur-sketch
        // a source cannot answer; only field-based aggregations.
        if (ts_only && !has_variance && single_field && !field.empty()) {
            PartialRequest req;
            req.files = plan.files;
            req.schema = plan.schema.get();
            req.group_by = plan.group_by;
            req.time_bucket_us = plan.time_bucket_us;
            req.agg_field = field;
            req.has_window = plan.time_range.has_value();
            if (plan.time_range) {
                req.begin = plan.time_range->first;
                req.end = plan.time_range->second;
            }
            if (const AggSpec* am = find_argmax(plan)) {
                req.needs_argmax = true;
                req.argmax_value = am->field;
                req.argmax_by = am->by;
            }
            std::string keybuf;
            auto res = plan.agg_source->lookup(req, [&](AggAccum&& a) {
                keybuf.clear();
                for (const auto& k : a.keys) {
                    keybuf += k;
                    keybuf += GROUP_SEP;
                }
                merge_accum(merged[keybuf], a, plan);
            });
            if (res.handled)
                for (const auto& [path, ckpt] : res.covered_chunks)
                    covered.add(path, ckpt);
        }
    }

    // Scan what the fast-path left uncovered through the fused fold, then merge
    // its groups with any agg_source partials already in `merged`. When no
    // rollup/tier answered and we must scan, aggregate over a subsuming
    // filtered-trace MV instead of the base (the fold re-applies the predicate,
    // so it is correct). Skipped for the agg_source path, whose covered chunks
    // are keyed to base paths.
    ViewPlan scan_plan = plan;
    if (!plan.agg_source) {
        if (auto mv = find_subsuming_view(plan))
            scan_plan.files = std::move(*mv);
    }
    dftracer::utils::StringIntern intern;
    AggFold agg(plan, intern);
    std::array<Fold*, 1> folds{&agg};
    co_await fuse(scan_plan, vdef, folds, intern, &covered);
    for (auto& [k, a] : agg.finish_map()) {
        if (auto it = merged.find(k); it != merged.end())
            merge_accum(it->second, a, plan);
        else
            merged.emplace(k, std::move(a));
    }

    // materialize() persists this query's result as a rollup (opt-in), so a
    // later matching query hits the fast path. Raw-keyed; a read-back
    // re-resolves. Skipped for a paginated (partial) result.
    if (plan.materialize && !plan.limit && !plan.offset) {
        const std::string rdir = rollup_index_path(plan);
        if (!rdir.empty()) {
            try {
                auto db = open_rollup_db(
                    rdir, rocksdb::RocksDatabase::OpenMode::ReadWrite);
                if (db)
                    persist_rollup(*db, plan_signature(plan),
                                   rest_signature(plan), plan.time_bucket_us,
                                   plan.group_by, merged);
            } catch (const std::exception& e) {
                // A locked or read-only index persists nothing rather than
                // failing the query; single-flight coordination comes later.
                DFTRACER_UTILS_LOG_WARN("rollup materialize skipped: %s",
                                        e.what());
            }
        }
    }

    resolve_group_keys(merged, plan);
    ResultTable table = to_result_table(merged, plan);
    if (plan.offset || plan.limit) {
        auto& rows = table.rows;
        const std::size_t off = std::min<std::size_t>(plan.offset, rows.size());
        rows.erase(rows.begin(),
                   rows.begin() + static_cast<std::ptrdiff_t>(off));
        if (plan.limit && rows.size() > plan.limit) rows.resize(plan.limit);
    }
    project_columns(table, plan.select);
    co_return table;
}

namespace {
// Partition a unified events+profiles table (leading "type" column) into the
// regular and aggregated tables, dropping the discriminator column.
void split_by_type(const ResultTable& unified, ResultTable& regular,
                   ResultTable& aggregated) {
    std::vector<std::string> gcols(
        unified.group_columns.begin() + (unified.group_columns.empty() ? 0 : 1),
        unified.group_columns.end());
    regular.group_columns = gcols;
    regular.value_columns = unified.value_columns;
    regular.text_columns = unified.text_columns;
    regular.hist_columns = unified.hist_columns;
    aggregated.group_columns = gcols;
    aggregated.value_columns = unified.value_columns;
    aggregated.text_columns = unified.text_columns;
    aggregated.hist_columns = unified.hist_columns;
    for (const auto& row : unified.rows) {
        const bool is_agg = !row.keys.empty() && row.keys.front() == "profile";
        ResultRow r = row;
        if (!r.keys.empty()) r.keys.erase(r.keys.begin());  // drop type
        (is_agg ? aggregated : regular).rows.push_back(std::move(r));
    }
}
}  // namespace

coro::CoroTask<TypedResult> run_collect_typed(const ViewPlan& plan,
                                              int shard_begin, int shard_end,
                                              const ProgressFn* progress) {
    ensure_schema(plan);
    TypedResult out;
    ResultTable unified;
    if (events_profiles_collect(plan, unified, shard_begin, shard_end,
                                progress))
        split_by_type(unified, out.regular, out.aggregated);
    system_collect(plan, out.counters, shard_begin, shard_end, progress);
    co_return out;
}

static GroupMap merge_partials_into_map(
    const ViewPlan& plan, const std::vector<std::string_view>& partials);

// Distributed materialize: reduce rank-local partials (from aggregate_partial)
// app-side into one GroupMap and persist it as the rollup - the distributed
// analog of run_materialize, with no re-scan.
coro::CoroTask<void> run_materialize_partials(
    const ViewPlan& plan, const std::vector<std::string_view>& partials) {
    ensure_schema(plan);
    GroupMap merged = merge_partials_into_map(plan, partials);
    const std::string rdir = rollup_index_path(plan);
    if (!rdir.empty()) {
        try {
            auto db = open_rollup_db(
                rdir, rocksdb::RocksDatabase::OpenMode::ReadWrite);
            if (db)
                persist_rollup(*db, plan_signature(plan), rest_signature(plan),
                               plan.time_bucket_us, plan.group_by, merged);
        } catch (const std::exception& e) {
            DFTRACER_UTILS_LOG_WARN("rollup materialize (partials) skipped: %s",
                                    e.what());
        }
    }
    co_return;
}

std::optional<ResultTable> run_reconstruct_if_cached(const ViewPlan& plan) {
    if (plan.group_by.empty() && plan.agg.empty()) return std::nullopt;
    ensure_schema(plan);
    const std::string rdir = rollup_index_path(plan);
    if (rdir.empty() || !fs::exists(fs::path(rdir) / "CURRENT"))
        return std::nullopt;
    try {
        auto db =
            open_rollup_db(rdir, rocksdb::RocksDatabase::OpenMode::ReadOnly);
        if (db) {
            if (auto r = find_subsuming_rollup(*db, plan)) {
                resolve_group_keys(*r, plan);
                return to_result_table(*r, plan);
            }
        }
    } catch (const std::exception& e) {
        DFTRACER_UTILS_LOG_WARN("rollup reconstruct skipped: %s", e.what());
    }
    return std::nullopt;
}

coro::CoroTask<ExportStats> run_scan_batches(
    const std::shared_ptr<const ViewPlan>& plan_ptr, std::size_t num_slots,
    std::uint64_t limit,
    const std::function<void(std::size_t,
                             const std::vector<std::string_view>&)>& on_batch) {
    bool from_mv = false;
    const ViewPlan plan = redirect_reads(*plan_ptr, &from_mv);
    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/false);
    // map_batches wants a fixed slot count; keep its historical 0 -> 1.
    ExportStats stats = co_await for_each_scanned_batch(
        plan, vdef, num_slots ? num_slots : 1, limit, on_batch);
    stats.served_from_mv = from_mv;
    co_return stats;
}

// ---- fused partition ------------------------------------------------------

struct PartitionState {
    std::shared_ptr<const ViewPlan> plan;
    std::size_t num_slots = 1;
    std::uint64_t limit = 0;
    std::vector<BranchHooks> branches;
};

std::shared_ptr<PartitionState> make_partition_state(
    std::shared_ptr<const ViewPlan> plan, std::size_t num_slots,
    std::uint64_t limit) {
    auto st = std::make_shared<PartitionState>();
    st->plan = std::move(plan);
    st->num_slots = num_slots ? num_slots : 1;
    st->limit = limit;
    return st;
}

void add_branch(PartitionState& state, BranchHooks hooks) {
    state.branches.push_back(std::move(hooks));
}

BranchHooks make_collect_branch(std::vector<GroupKey> group_by,
                                std::vector<AggSpec> agg,
                                std::shared_ptr<ResultTable> out,
                                std::size_t num_slots) {
    auto plan = std::make_shared<ViewPlan>();
    plan->group_by = std::move(group_by);
    plan->agg = std::move(agg);
    ensure_schema(*plan);  // before the per-slot consume callbacks fold
    const std::size_t slots = num_slots ? num_slots : 1;
    auto partials = std::make_shared<std::vector<GroupMap>>(slots);
    // Per-slot key buffers: each slot folds single-threaded, so no sharing.
    auto keybufs = std::make_shared<std::vector<std::string>>(slots);

    BranchHooks h;
    h.consume = [plan, partials, keybufs](std::size_t slot,
                                          const common::json::JsonValue& jv,
                                          std::string_view) {
        fold_event((*partials)[slot], jv.element(), *plan, (*keybufs)[slot]);
    };
    h.finalize = [plan, partials, out]() {
        GroupMap merged;
        for (const auto& p : *partials) merge_maps(merged, p, *plan);
        resolve_group_keys(merged, *plan);
        *out = to_result_table(merged, *plan);
    };
    return h;
}

BranchHooks make_export_branch(ExportSink& sink,
                               std::shared_ptr<ExportStats> out) {
    auto mtx = std::make_shared<std::mutex>();
    auto count = std::make_shared<std::uint64_t>(0);

    BranchHooks h;
    h.consume = [&sink, mtx, count](std::size_t, const common::json::JsonValue&,
                                    std::string_view raw) {
        std::lock_guard<std::mutex> lk(*mtx);
        sink.write(raw);
        sink.write("\n");
        ++*count;
    };
    h.finalize = [out, count]() { out->events_matched = *count; };
    return h;
}

coro::CoroTask<ExportStats> run_partition(
    std::shared_ptr<PartitionState> state) {
    const ViewPlan& plan = *state->plan;
    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/false);

    std::uint64_t skipped = 0;
    auto units = co_await gather_units(plan, vdef, skipped);

    const std::size_t num_slots = state->num_slots ? state->num_slots : 1;
    const std::uint64_t cap = state->limit > 0
                                  ? state->limit
                                  : std::numeric_limits<std::uint64_t>::max();
    std::atomic<std::uint64_t> produced{0};
    std::atomic<std::size_t> next_unit{0};
    std::vector<std::uint64_t> matched_v(num_slots, 0);
    std::vector<std::uint64_t> scanned_v(num_slots, 0);
    // Frame-owned per-slot parsers: a slot is single-threaded, but a
    // thread_local parser is not reliable across pool-thread resumes.
    std::vector<simdjson::dom::parser> parsers(num_slots);
    std::vector<std::string> bufs(num_slots);
    const auto& branches = state->branches;
    const std::size_t nworkers = std::min(num_slots, units.size());

    co_await run_coro_scope([&](CoroScope& scope) -> coro::CoroTask<void> {
        for (std::size_t w = 0; w < nworkers; ++w) {
            scope.spawn([&, w](CoroScope&) -> coro::CoroTask<void> {
                for (;;) {
                    if (produced.load(std::memory_order_relaxed) >= cap) break;
                    if (is_cancelled(plan)) break;
                    std::size_t i =
                        next_unit.fetch_add(1, std::memory_order_relaxed);
                    if (i >= units.size()) break;
                    ViewScannerInput sin =
                        make_scanner_input(units[i], vdef, vdef.query);
                    ViewScannerUtility scanner;
                    auto gen = scanner.process(sin);
                    while (auto b = co_await gen.next()) {
                        if (b->events.empty()) continue;
                        if (produced.load(std::memory_order_relaxed) >= cap)
                            break;
                        if (is_cancelled(plan)) break;
                        matched_v[w] += b->events_matched;
                        scanned_v[w] += b->events_scanned;
                        for (auto ev : b->events) {
                            bufs[w].assign(ev);
                            auto res = parsers[w].parse(bufs[w]);
                            if (res.error()) continue;
                            auto root = res.value_unsafe();
                            if (!root.is_object()) continue;
                            common::json::JsonValue jv(root);
                            for (const auto& br : branches) {
                                if (!br.predicate || br.predicate->evaluate(jv))
                                    br.consume(w, jv, ev);
                            }
                        }
                        produced.fetch_add(b->events.size(),
                                           std::memory_order_relaxed);
                    }
                }
                co_return;
            });
        }
        co_return;
    });

    for (const auto& br : branches)
        if (br.finalize) br.finalize();

    ExportStats st;
    st.chunks_skipped = skipped;
    st.chunks_scanned = units.size();
    for (auto m : matched_v) st.events_matched += m;
    for (auto s : scanned_v) st.events_scanned += s;
    st.truncated = produced.load(std::memory_order_relaxed) >= cap;
    co_return st;
}

// export_counters with bounded memory: workers aggregate their share of chunks
// into in-memory maps, spilling to sorted runs when over budget; the runs are
// k-way merged and streamed to the sink, so peak memory stays bounded no matter
// the group cardinality. memory_budget == 0 keeps it purely in-memory.
coro::CoroTask<ExportStats> run_export_counters(const ViewPlan& plan,
                                                ExportSink& sink) {
    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);
    co_return co_await fused_aggregate(
        plan, vdef, [&](const std::string& k, const AggAccum& a) {
            emit_group_counter(k, a, plan, sink);
        });
}

// Rank-local counter aggregation for distributed runs: aggregate this view's
// (shard of) files in memory and serialize the groups into an opaque partial
// buffer. The transport (MPI etc.) lives in the caller; only bytes cross ranks.
coro::CoroTask<std::string> run_aggregate_partial(const ViewPlan& plan) {
    // Index-only fast path: when the aggregation tier answers the whole plan,
    // read its pre-folded accumulators instead of scanning the trace files and
    // serialize them into the same partial format a scan produces. This lets a
    // sharded/distributed reader merge shards straight from their indexes, with
    // no trace read. Restricted to plain event aggregations: the tier is
    // EVENT-only, so counter and dynamic-numeric-args plans still scan (their
    // values are not in the tier).
    if (plan.phase != Phase::Counters && !plan.auto_numeric_metrics) {
        GroupMap tier;
        if (agg_tier_collect(plan, tier)) {
            std::string out;
            for (const auto& [k, a] : tier) serialize_accum(out, k, a);
            co_return out;
        }
    }

    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);
    std::string out;
    co_await fused_aggregate(plan, vdef,
                             [&](const std::string& k, const AggAccum& a) {
                                 serialize_accum(out, k, a);
                             });
    co_return out;
}

// Deserialize and combine rank-local partials (from run_aggregate_partial) back
// into one group map. The plan supplies the group/agg shape.
static GroupMap merge_partials_into_map(
    const ViewPlan& plan, const std::vector<std::string_view>& partials) {
    ensure_schema(plan);
    GroupMap merged;
    for (auto p : partials) {
        codec::BinaryReader br(p);
        while (br.has_remaining()) {
            std::string key;
            AggAccum a;
            deserialize_accum(br, key, a);
            merge_accum(merged[key], a, plan);
        }
    }
    return merged;
}

// Merge partials and emit each combined group as a ph="C" counter event.
ExportStats merge_counters_partials(
    const ViewPlan& plan, const std::vector<std::string_view>& partials,
    ExportSink& sink) {
    GroupMap merged = merge_partials_into_map(plan, partials);
    ExportStats st;
    for (const auto& [k, a] : merged) {
        emit_group_counter(k, a, plan, sink);
        ++st.events_matched;
    }
    return st;
}

// Merge partials into a materialized ResultTable (distributed collect()).
ResultTable merge_partials_to_table(
    const ViewPlan& plan, const std::vector<std::string_view>& partials) {
    GroupMap merged = merge_partials_into_map(plan, partials);
    resolve_group_keys(merged, plan);
    return to_result_table(merged, plan);
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
