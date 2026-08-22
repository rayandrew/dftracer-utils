#ifndef DFTRACER_UTILS_TRACE_INTERNAL_CHUNK_SPEC_H
#define DFTRACER_UTILS_TRACE_INTERNAL_CHUNK_SPEC_H

#include <dftracer/utils/utilities/fileio/types/chunk_spec.h>

namespace dftracer::utils::trace::internal {

/** @brief utilities::fileio::ChunkSpec plus 1-based line-range tracking. */
struct DFTracerChunkSpec : public utilities::fileio::ChunkSpec {
    std::size_t start_line;  ///< 1-based, inclusive.
    std::size_t end_line;    ///< 1-based, inclusive.

    DFTracerChunkSpec()
        : utilities::fileio::ChunkSpec(), start_line(0), end_line(0) {}

    DFTracerChunkSpec(std::string path, std::string idx, double mb,
                      std::size_t start_byte_offset,
                      std::size_t end_byte_offset, std::size_t start_ln,
                      std::size_t end_ln)
        : utilities::fileio::ChunkSpec(std::move(path), std::move(idx), mb,
                                       start_byte_offset, end_byte_offset),
          start_line(start_ln),
          end_line(end_ln) {}

    std::size_t num_lines() const {
        return (end_line >= start_line && start_line > 0)
                   ? (end_line - start_line + 1)
                   : 0;
    }

    bool has_line_info() const { return start_line > 0 && end_line > 0; }

    bool operator==(const DFTracerChunkSpec& other) const {
        return utilities::fileio::ChunkSpec::operator==(other) &&
               start_line == other.start_line && end_line == other.end_line;
    }

    bool operator!=(const DFTracerChunkSpec& other) const {
        return !(*this == other);
    }
};

}  // namespace dftracer::utils::trace::internal

#endif  // DFTRACER_UTILS_TRACE_INTERNAL_CHUNK_SPEC_H
