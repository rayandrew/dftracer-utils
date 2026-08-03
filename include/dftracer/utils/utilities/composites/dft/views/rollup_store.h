#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_ROLLUP_STORE_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_ROLLUP_STORE_H

#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::views::detail {

// Rollup CF key layout. A leading tag keeps descriptors and rows in disjoint
// key ranges, and the 8-byte big-endian signature groups one view's rows
// contiguously so a prefix scan reconstructs it:
//   row:  0x01 | sig | group_key
//   desc: 0x00 | sig
std::string rollup_row_key(std::uint64_t sig, std::string_view group_key);
std::string rollup_desc_key(std::uint64_t sig);

// Combine `sa` into `da` positionally (no plan): rows under one signature share
// a schema, so field/argmax/sketch vectors align by index; an empty `da` adopts
// `sa`. Reduces distributed partials app-side before they are written.
void merge_accum_free(AggAccum& da, const AggAccum& sa);

// Open (or reuse) the index DB at `index_path` for rollup access. The rollup CF
// has no merge operator, so the index's standard open serves it.
std::shared_ptr<dftracer::utils::rocksdb::RocksDatabase> open_rollup_db(
    const std::string& index_path,
    dftracer::utils::rocksdb::RocksDatabase::OpenMode mode);

// Persist `map` as the rollup identified by `sig`: one Put per group, plus a
// descriptor recording the view's shape (`rest_sig` = the plan hash excluding
// group_by, `time_bucket_us` = the rollup's time grain for coarsening, and
// `group_by` itself) so the planner can test subsumption. `map` must already be
// the complete, merged result - distributed callers reduce partials with
// merge_accum_free first.
void persist_rollup(dftracer::utils::rocksdb::RocksDatabase& db,
                    std::uint64_t sig, std::uint64_t rest_sig,
                    std::uint64_t time_bucket_us,
                    const std::vector<GroupKey>& group_by, const GroupMap& map);

// True if a rollup for `sig` has been persisted.
bool rollup_exists(const dftracer::utils::rocksdb::RocksDatabase& db,
                   std::uint64_t sig);

// Read the rollup for `sig` back into a GroupMap (empty if absent).
GroupMap read_rollup(const dftracer::utils::rocksdb::RocksDatabase& db,
                     std::uint64_t sig);

// A materialized view's identity: a stable hash over the plan's file set and
// query shape. Per-rank scans of the same plan share it, so their partials land
// under the same rollup key.
std::uint64_t plan_signature(const ViewPlan& plan);

// The plan hash EXCLUDING group_by AND time_bucket: two views with the same
// rest_signature differ only in grouping and/or time grain, so one can serve
// the other by re-aggregating (and re-bucketing to a coarser grain).
std::uint64_t rest_signature(const ViewPlan& plan);

// Find a stored rollup that subsumes `plan` and return its rows re-aggregated
// to `plan`'s grouping, or nullopt. A rollup R subsumes plan Q when they share
// a rest_signature (same files/filter/agg/window) and Q's group keys are a
// subset of R's - then Q is R rolled up over the dropped dimensions. Exact
// match is the identity case. This is the materialized-view query rewrite.
std::optional<GroupMap> find_subsuming_rollup(
    const dftracer::utils::rocksdb::RocksDatabase& db, const ViewPlan& plan);

// The index that anchors `plan`'s rollup: its shared aggregation index (all
// files must share one index_path, as the tier requires). The rollup lives in
// this index's ROLLUP column family. Empty if there is no single index.
std::string rollup_index_path(const ViewPlan& plan);

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_ROLLUP_STORE_H
