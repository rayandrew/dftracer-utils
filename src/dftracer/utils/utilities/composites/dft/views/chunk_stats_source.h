#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_CHUNK_STATS_SOURCE_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_CHUNK_STATS_SOURCE_H

#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>

#include <functional>

namespace dftracer::utils::utilities::composites::dft::views {

/// Answers name/cat/pid duration aggregations from the per-chunk DDSketches in
/// the index: a chunk whose whole time span sits inside the window contributes
/// its stored count/min/max/sum with no decode. Declines (handled=false) for
/// time-bucketed, multi-column, or non-duration requests, which fall back to a
/// scan.
class ChunkStatsSource : public detail::PartialSource {
   public:
    detail::PartialSource::Result lookup(
        const detail::PartialRequest& req,
        const std::function<void(detail::AggAccum&&)>& emit) const override;
};

}  // namespace dftracer::utils::utilities::composites::dft::views

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_CHUNK_STATS_SOURCE_H
