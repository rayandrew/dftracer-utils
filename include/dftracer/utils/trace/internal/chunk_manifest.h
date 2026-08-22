#ifndef DFTRACER_UTILS_TRACE_INTERNAL_CHUNK_MANIFEST_H
#define DFTRACER_UTILS_TRACE_INTERNAL_CHUNK_MANIFEST_H

#include <dftracer/utils/trace/internal/chunk_spec.h>

#include <vector>

namespace dftracer::utils::trace::internal {

/** @brief DFTracer-specific chunk manifest with per-chunk line tracking. */
struct DFTracerChunkManifest {
    std::vector<DFTracerChunkSpec> specs;
    double total_size_mb;

    DFTracerChunkManifest() : total_size_mb(0.0) {}

    void add_spec(DFTracerChunkSpec spec) {
        total_size_mb += spec.size_mb;
        specs.push_back(std::move(spec));
    }

    std::size_t total_lines() const {
        std::size_t total = 0;
        for (const auto& spec : specs) {
            total += spec.num_lines();
        }
        return total;
    }

    bool operator==(const DFTracerChunkManifest& other) const {
        return specs == other.specs && total_size_mb == other.total_size_mb;
    }

    bool operator!=(const DFTracerChunkManifest& other) const {
        return !(*this == other);
    }
};

}  // namespace dftracer::utils::trace::internal

#endif  // DFTRACER_UTILS_TRACE_INTERNAL_CHUNK_MANIFEST_H
