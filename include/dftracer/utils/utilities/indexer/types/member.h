#ifndef DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_MEMBER_H
#define DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_MEMBER_H

#include <dftracer/utils/utilities/composites/dft/indexing/queries/queries.h>
#include <dftracer/utils/utilities/indexer/internal/gzip_member_record.h>

#include <cstdint>

namespace dftracer::utils::utilities::indexer {

using GzipMemberRecord = internal::GzipMemberRecord;
using TimeBounds = composites::dft::indexing::queries::TimeBounds;

/// Uncompressed extent and line range of one pruner chunk. The authoritative
/// chunk -> position mapping for every reader: derived from the gzip member
/// table.
struct ChunkSpan {
    std::uint64_t uc_offset = 0;
    std::uint64_t uc_size = 0;
    std::uint64_t first_line_num = 0;
    std::uint64_t last_line_num = 0;
};

}  // namespace dftracer::utils::utilities::indexer

#endif  // DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_MEMBER_H
