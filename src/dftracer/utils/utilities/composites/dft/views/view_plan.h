#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_PLAN_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_PLAN_H

#include <dftracer/utils/utilities/common/query/query.h>
#include <dftracer/utils/utilities/composites/dft/views/view.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::indexing {
class BloomFilterCache;
}

namespace dftracer::utils::utilities::composites::dft::views::detail {

/// Query-derived fold schema (distinct agg fields + per-spec mapping), a pure
/// function of the plan. Cached lazily on the plan so the per-event fold builds
/// it once, not per event. Defined in view_aggregate.h.
struct AggSchema;

/// Materialized-aggregate source consulted by collect(). Defined in
/// view_aggregate.h.
class PartialSource;

/// Index-backed name maps for resolved-name group keys. Defined in
/// view_resolver.h.
class GroupResolver;

/// The logical plan a `View` carries. The ops form a linear pipeline, so a flat
/// struct captures it; each builder copies and mutates one field.
struct ViewPlan {
    std::vector<ViewFile> files;
    indexing::BloomFilterCache* bloom_cache = nullptr;

    std::optional<common::query::Query> query;
    std::optional<std::pair<double, double>> time_range;
    // Default spans all phases so a naive View sees every event; callers that
    // want only ph="X" events (or only ph="C") say so with .phase().
    Phase phase = Phase::Any;
    bool include_metadata = true;
    bool emit_all_metadata = false;  // harvest every hash-metadata record

    std::uint64_t time_bucket_us = 0;
    /// Multiply ts/dur/te by this to normalize from the trace's native time
    /// unit to a target (source_ns / target_ns; 1.0 = no scaling). Applied
    /// before time_bucket so bucketing is in the target unit.
    double time_scale = 1.0;
    std::vector<GroupKey> group_by;
    std::vector<AggSpec> agg;
    /// Aggregate every numeric args.* field as a dynamic per-group mean metric
    /// (skipping metadata/pre-aggregated args). Adds one value column per
    /// discovered arg. This is how ph="C" counters aggregate without naming the
    /// fields up front.
    bool auto_numeric_metrics = false;
    /// Out-of-core aggregation budget shared by collect / export_counters /
    /// aggregate_partial: when a worker's in-memory group map grows past this
    /// many bytes it spills to a sorted temp run; the runs are k-way merged at
    /// the end, so the scan/merge peak stays bounded regardless of group
    /// cardinality. 0 = pure in-memory (never spill). collect() still
    /// materializes the final table, so this bounds intermediate memory only.
    std::uint64_t memory_budget = 0;
    std::vector<std::string> select;

    /// Pagination. `limit` caps the output (0 = unlimited); `offset` skips that
    /// many leading rows/events first. Applied to collect()'s materialized rows
    /// and to the streamed event output. Under a parallel scan the export order
    /// is scan-order, so a stable page needs a deterministic sort first.
    std::uint64_t limit = 0;
    std::uint64_t offset = 0;

    /// Root for the materialized-view cache (persist/reconstruct). Empty =
    /// derive from the files' index location. The cache dir is
    /// `<root>/<plan-signature>`, so a changed grain (files, filter, group_by,
    /// agg, bucket, range) lands in a different dir and recomputes.
    std::string rollup_root;

    /// Directory holding materialized filtered-trace views. Empty = derive as
    /// `<parent-of-index>/.dftindex-views`. Set to relocate MVs (e.g. a scratch
    /// dir) away from the dataset.
    std::string views_root;

    /// Persist this query's result as a materialized view (rollup) as a
    /// byproduct of answering it, so a later matching query reads it back.
    /// Opt-in via View::materialize() - a plain collect() never persists.
    bool materialize = false;

    /// Row-MV materialize granularity (bytes): `mv_checkpoint_size` is the
    /// index checkpoint size (one gzip member per checkpoint, so it drives
    /// intra-file read parallelism); `mv_part_size` is the uncompressed size at
    /// which the write rolls to a new part file. 0 = engine defaults.
    std::uint64_t mv_checkpoint_size = 0;
    std::uint64_t mv_part_size = 0;

    const PartialSource* agg_source = nullptr;

    /// Cooperative cancellation, polled by every terminal at its loop
    /// boundaries; empty = never cancelled.
    std::function<bool()> cancelled;

    /// Memoized fold schema (ensure_schema builds it once before the parallel
    /// fold; workers read it race-free). Mutable so a const plan can cache it;
    /// shared_ptr so plan copies share one build.
    mutable std::shared_ptr<const AggSchema> schema;

    /// Index-backed name maps, built lazily by ensure_resolver when the plan
    /// has a FilePath/HostName group key; used by the post-aggregation re-key
    /// pass.
    mutable std::shared_ptr<const GroupResolver> resolver;
};

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_PLAN_H
