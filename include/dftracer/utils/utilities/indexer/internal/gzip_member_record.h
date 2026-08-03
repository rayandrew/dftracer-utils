#ifndef DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_GZIP_MEMBER_RECORD_H
#define DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_GZIP_MEMBER_RECORD_H

#include <cstdint>

namespace dftracer::utils::utilities::indexer::internal {

/// One gzip member: the unit of random access and one-shot decode.
/// Boundaries come from the inflater, not a header scan, so they are exact
/// rather than candidates. The table is an optimization - a reader must
/// tolerate its absence and fall back to
/// `enumerate_gzip_member_candidates`.
struct GzipMemberRecord {
    std::uint64_t member_idx;
    std::uint64_t c_offset;
    std::uint64_t c_size;
    std::uint64_t uc_offset;
    std::uint64_t uc_size;
    std::uint64_t first_line_num;
    std::uint64_t last_line_num;
};

}  // namespace dftracer::utils::utilities::indexer::internal

#endif  // DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_GZIP_MEMBER_RECORD_H
