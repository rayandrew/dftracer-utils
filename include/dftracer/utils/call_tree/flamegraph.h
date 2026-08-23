#ifndef DFTRACER_UTILS_CALL_TREE_FLAMEGRAPH_H
#define DFTRACER_UTILS_CALL_TREE_FLAMEGRAPH_H

#include <dftracer/utils/call_tree/call_tree.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dftracer::utils::call_tree {

/// A distinct root-to-node name-path. total_weight sums duration over every
/// call-tree node with this path; self_weight is total minus child totals.
struct FlamegraphNode {
    std::uint64_t id;
    std::uint64_t parent_id;  ///< 0 == root
    std::string name;
    std::string path;         ///< ancestor names joined by '/'
    int depth;
    std::uint64_t total_weight;
    std::uint64_t self_weight;
};

struct FlamegraphProfile {
    std::vector<FlamegraphNode>
        nodes;  ///< roots first, parents before children
    std::uint64_t total;

    FlamegraphProfile() : total(0) {}
};

struct FlamegraphDiff {
    std::string path;
    std::uint64_t a_weight;
    std::uint64_t b_weight;
    std::int64_t delta;  ///< b_weight - a_weight
};

/// Fold by name-path (weight = duration_us): total = summed duration, self =
/// total - child totals. Emitted depth asc, total_weight desc, path asc.
FlamegraphProfile aggregate_flamegraph(const CallTree& tree);

/// Per-path weight delta (b - a), one row per path in either, |delta| desc.
std::vector<FlamegraphDiff> diff_flamegraphs(const FlamegraphProfile& a,
                                             const FlamegraphProfile& b);

}  // namespace dftracer::utils::call_tree

#endif  // DFTRACER_UTILS_CALL_TREE_FLAMEGRAPH_H
