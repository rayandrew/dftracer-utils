#ifndef DFTRACER_UTILS_PLUGINS_MAP_GROUPING_H
#define DFTRACER_UTILS_PLUGINS_MAP_GROUPING_H

#include <dftracer/utils/plugins/map_join.h>

#include <cstdint>
#include <vector>

// GROUPING SETS / CUBE / ROLLUP over a MapAccum, composed from regroup_map. No
// Arrow dependency, like map_join.h.
namespace dftracer::utils::plugins {

// 2^n subsets; cube_sets refuses beyond this (2^20 is already unusable).
inline constexpr std::uint32_t CUBE_MAX_KEY_N = 20;

// GROUPING SETS: one regrouped MapAccum per keep-set (each via regroup_map).
// The full key reproduces the base grouping; the empty set {} is the grand
// total (all rows collapse to one entry). An out-of-range keep index yields
// that set's empty (key_n==0) MapAccum, as regroup_map does.
// Precondition: m fully resident; a spilled input yields an empty vector.
std::vector<MapAccum> grouping_sets(
    const MapAccum& m,
    const std::vector<std::vector<std::uint32_t>>& keep_sets);

// CUBE over the full key: all 2^key_n subsets, ordered by bitmask ascending
// with each subset's components ascending. Empty if key_n > CUBE_MAX_KEY_N.
std::vector<std::vector<std::uint32_t>> cube_sets(std::uint32_t key_n);

// ROLLUP over the full key: the key_n+1 prefixes {0..n-1}, {0..n-2}, ..., {0},
// {}.
std::vector<std::vector<std::uint32_t>> rollup_sets(std::uint32_t key_n);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_MAP_GROUPING_H
