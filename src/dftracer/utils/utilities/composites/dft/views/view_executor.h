#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_EXECUTOR_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_EXECUTOR_H

#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/utilities/composites/dft/views/view.h>
#include <dftracer/utils/utilities/composites/dft/views/view_plan.h>

namespace dftracer::utils::utilities::composites::dft::views::detail {

// Both prune per file (bloom + chunk pruning), then scan surviving candidates
// in parallel on the ambient executor. run_export writes matching events;
// run_collect folds them into a grouped aggregate.
// True when a first-touch query would take the raw-gzip bootstrap (answer the
// query and build the index in one pass), so a caller can skip a separate eager
// index build and let the bootstrap fire. These are the exact gates run_export
// / run_collect apply internally, exposed so the CLI does not duplicate them.
bool export_bootstrap_eligible(const ViewPlan& plan);
bool collect_bootstrap_eligible(const ViewPlan& plan);

coro::CoroTask<ExportStats> run_export(const ViewPlan& plan, ExportSink& sink);

// Write matching events as a new multi-member trace through the parallel writer
// (PFS-aware layout). Each worker frames its events into self-contained gzip
// members (whole lines) and writes them via ParallelWriter; sharded output is
// concatenated on close.
coro::CoroTask<ExportStats> run_export_trace(const ViewPlan& plan,
                                             const TraceWriteOptions& opts);

// Fused variant of run_export_trace: build the member+bloom+stats index during
// the write, avoiding a re-inflate of the output. Scan/decompress stays
// parallel (producers); a single in-order consumer frames + writes members and
// feeds them to the index visitors, so member_idx == checkpoint_idx and no
// offset remap is needed. Reached when TraceWriteOptions::build_index is set.
coro::CoroTask<ExportStats> run_export_trace_indexed(
    const ViewPlan& plan, const TraceWriteOptions& opts,
    const ProgressFn* progress = nullptr);

coro::CoroTask<ResultTable> run_collect(const ViewPlan& plan);

// Build-only terminal: run the query for its side effect - materialize the
// filtered-trace MV (row query) or the rollup (aggregation) - and return scan
// stats without a ResultTable. A no-op when the MV already exists, so it is an
// idempotent prewarm. `progress`, if set, is called with (done, total) scan
// units as the build advances.
coro::CoroTask<ExportStats> run_materialize(
    const ViewPlan& plan, const ProgressFn* progress = nullptr);

// Distributed materialize: reduce rank-local partials (from aggregate_partial)
// app-side and persist the rollup, so each rank scans its shard once and the
// coordinator writes the merged view. No re-scan.
coro::CoroTask<void> run_materialize_partials(
    const ViewPlan& plan, const std::vector<std::string_view>& partials);

// Read the aggregation index's three record families (regular events,
// aggregated records, counters) from shard range [shard_begin, shard_end).
// Empty tables when the index has no aggregation tier.
coro::CoroTask<TypedResult> run_collect_typed(
    const ViewPlan& plan, int shard_begin = 0, int shard_end = 4096,
    const ProgressFn* progress = nullptr);

// Return a subsuming rollup as a table, or nullopt on a miss, so a distributed
// caller decides read vs recompute without a scan.
std::optional<ResultTable> run_reconstruct_if_cached(const ViewPlan& plan);

// Emit each aggregate row as a dftracer ph="C" counter event. Plain function so
// callers await run_collect at the same depth as collect() (an extra coroutine
// Streaming, out-of-core counter aggregation: aggregate with a per-worker
// memory budget (plan.memory_budget), spilling to sorted temp runs, then k-way
// merge and emit each group as a ph="C" event. Bounded peak memory regardless
// of group cardinality; budget 0 stays purely in-memory.
coro::CoroTask<ExportStats> run_export_counters(const ViewPlan& plan,
                                                ExportSink& sink);

// Distributed building blocks: rank-local aggregate -> opaque serialized
// partial, and merge partials -> emit. The MPI (or other) transport lives in
// the caller so the View core stays transport-free.
coro::CoroTask<std::string> run_aggregate_partial(const ViewPlan& plan);
ExportStats merge_counters_partials(
    const ViewPlan& plan, const std::vector<std::string_view>& partials,
    ExportSink& sink);
ResultTable merge_partials_to_table(
    const ViewPlan& plan, const std::vector<std::string_view>& partials);

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_EXECUTOR_H
