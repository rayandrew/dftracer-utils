#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_RADIX_DEDUP_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_RADIX_DEDUP_H

#include <ankerl/unordered_dense.h>
#include <dftracer/utils/dataframe/parallel.h>

#include <cstdint>
#include <string>
#include <vector>

// Shared radix-partitioned dedup helpers for the EAGER batch ops, mirroring
// the pattern already proven in the lazy cursors (lazyframe.cpp
// UniqueCursor/IsDupCursor): partition rows by row-key hash into disjoint
// buckets, each probed by exactly one worker with its own hash set/map (no
// locks), then combine. Header-only so eager ops in batch_ops.cpp/select.cpp
// can call it without a new translation unit.
namespace dftracer::utils::dataframe {

/// Fan-out width for the partitioned dedup below.
constexpr std::size_t RADIX_DEDUP_PARTITIONS = 32;

/// Below this row count, the partition/fan-out bookkeeping costs more than the
/// serial hash-set/map it would replace.
constexpr std::int64_t RADIX_DEDUP_MIN_ROWS = 1 << 14;

/// Radix-partition `keys` by hash into `RADIX_DEDUP_PARTITIONS` buckets, each
/// with its own hash set, and mark keep[i] the first time each key is seen
/// (order-preserving): buckets never overlap, so each is probed by exactly one
/// worker with no lock. Falls back to one serial set when no parallel backend
/// is installed or `n` is below the fan-out threshold.
inline std::vector<std::uint8_t> radix_first_seen_mask(
    const std::vector<std::string>& keys, std::int64_t n) {
    std::vector<std::uint8_t> keep(static_cast<std::size_t>(n), 0);
    if (!parallel_backend_installed() || n < RADIX_DEDUP_MIN_ROWS) {
        ankerl::unordered_dense::set<std::string> seen;
        seen.reserve(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            if (seen.insert(keys[static_cast<std::size_t>(i)]).second)
                keep[static_cast<std::size_t>(i)] = 1;
        return keep;
    }
    const std::size_t p = RADIX_DEDUP_PARTITIONS;
    std::vector<std::vector<std::int64_t>> buckets(p);
    for (std::int64_t i = 0; i < n; ++i)
        buckets[std::hash<std::string>{}(keys[static_cast<std::size_t>(i)]) % p]
            .push_back(i);
    parallel_for(
        static_cast<std::int64_t>(p), 1, [&](std::int64_t pb, std::int64_t pe) {
            for (std::int64_t g = pb; g < pe; ++g) {
                ankerl::unordered_dense::set<std::string> seen;
                for (std::int64_t i : buckets[static_cast<std::size_t>(g)])
                    if (seen.insert(keys[static_cast<std::size_t>(i)]).second)
                        keep[static_cast<std::size_t>(i)] = 1;
            }
        });
    return keep;
}

/// Radix-partition `keys` by hash into `RADIX_DEDUP_PARTITIONS` buckets, each
/// with its own count map, and return the occurrence count of each row's key.
/// Falls back to one serial map below the fan-out threshold.
inline std::vector<std::int64_t> radix_key_counts(
    const std::vector<std::string>& keys, std::int64_t n) {
    std::vector<std::int64_t> counts(static_cast<std::size_t>(n), 0);
    if (!parallel_backend_installed() || n < RADIX_DEDUP_MIN_ROWS) {
        ankerl::unordered_dense::map<std::string, std::int64_t> cnt;
        cnt.reserve(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            ++cnt[keys[static_cast<std::size_t>(i)]];
        for (std::int64_t i = 0; i < n; ++i)
            counts[static_cast<std::size_t>(i)] =
                cnt[keys[static_cast<std::size_t>(i)]];
        return counts;
    }
    const std::size_t p = RADIX_DEDUP_PARTITIONS;
    std::vector<std::vector<std::int64_t>> buckets(p);
    for (std::int64_t i = 0; i < n; ++i)
        buckets[std::hash<std::string>{}(keys[static_cast<std::size_t>(i)]) % p]
            .push_back(i);
    parallel_for(
        static_cast<std::int64_t>(p), 1, [&](std::int64_t pb, std::int64_t pe) {
            for (std::int64_t g = pb; g < pe; ++g) {
                ankerl::unordered_dense::map<std::string, std::int64_t> cnt;
                for (std::int64_t i : buckets[static_cast<std::size_t>(g)])
                    ++cnt[keys[static_cast<std::size_t>(i)]];
                for (std::int64_t i : buckets[static_cast<std::size_t>(g)])
                    counts[static_cast<std::size_t>(i)] =
                        cnt[keys[static_cast<std::size_t>(i)]];
            }
        });
    return counts;
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_RADIX_DEDUP_H
