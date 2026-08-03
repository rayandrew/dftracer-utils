#ifndef DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_STATISTICS_H
#define DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_STATISTICS_H

#include <dftracer/utils/utilities/composites/dft/indexing/chunk_dimension_stats.h>
#include <dftracer/utils/utilities/composites/dft/indexing/chunk_statistics.h>
#include <dftracer/utils/utilities/composites/dft/indexing/queries/queries.h>

#include <cstdint>

namespace dftracer::utils::utilities::indexer {

using ChunkStatistics = composites::dft::indexing::ChunkStatistics;
using ChunkDimensionStats = composites::dft::indexing::ChunkDimensionStats;
using ChunkStatisticsResult =
    composites::dft::indexing::queries::ChunkStatisticsResult;
using ChunkDimensionStatsResult =
    composites::dft::indexing::ChunkDimensionStatsResult;

struct MergedStatisticsResult {
    ChunkStatistics stats;
    std::uint64_t num_chunks = 0;
};

struct RootStatisticsResult {
    ChunkStatistics stats;
    std::uint64_t num_chunks = 0;
    std::uint64_t num_files = 0;
    std::uint64_t total_lines = 0;
    std::uint64_t total_uncompressed_bytes = 0;
};

struct NameSummaryResult {
    StringViewMap<std::uint64_t> counts;
    std::uint64_t other_count = 0;
    std::uint64_t unique_count = 0;
};

}  // namespace dftracer::utils::utilities::indexer

#endif  // DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_STATISTICS_H
