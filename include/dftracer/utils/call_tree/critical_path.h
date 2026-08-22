#ifndef DFTRACER_UTILS_CALL_TREE_CRITICAL_PATH_H
#define DFTRACER_UTILS_CALL_TREE_CRITICAL_PATH_H

#include <dftracer/utils/call_tree/call_tree.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dftracer::utils::call_tree {

/// One operation name aggregated across every root's critical path.
struct CriticalPathOp {
    std::string name;
    std::uint64_t total_exclusive_us;  ///< summed on-path exclusive time
    std::uint64_t on_path_root_count;  ///< roots whose path includes this name
};

struct CriticalPathProfile {
    std::vector<CriticalPathOp>
        operations;  ///< total_exclusive_us desc, name asc
    std::uint64_t total_roots;
    double p50_us;
    double p90_us;
    double p95_us;
    double p99_us;
    utilities::common::statistics::DDSketch duration_sketch;

    CriticalPathProfile()
        : total_roots(0), p50_us(0), p90_us(0), p95_us(0), p99_us(0) {}
};

/// Critical path per root: at each node follow the LATEST-ending child (ties ->
/// larger duration, then smaller id); a node's exclusive time is
/// duration - critical_child.duration (full duration at a leaf). The exclusive
/// times along one path sum to the root's duration. Aggregates per-name
/// exclusive time and feeds each root duration into a DDSketch for quantiles.
CriticalPathProfile aggregate_critical_paths(const CallTree& tree);

}  // namespace dftracer::utils::call_tree

#endif  // DFTRACER_UTILS_CALL_TREE_CRITICAL_PATH_H
