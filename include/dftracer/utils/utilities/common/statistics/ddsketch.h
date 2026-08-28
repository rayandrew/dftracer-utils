#ifndef DFTRACER_UTILS_UTILITIES_COMMON_STATISTICS_DDSKETCH_H
#define DFTRACER_UTILS_UTILITIES_COMMON_STATISTICS_DDSKETCH_H

// The one mergeable DDSketch now lives at the dataframe layer so the columnar
// aggregation engine can feed it SIMD-computed bucket keys. This header keeps
// the historical include path and namespace as thin aliases for the trace /
// View / plugin code that predates the move.

#include <dftracer/utils/dataframe/sketch.h>

namespace dftracer::utils::utilities::common::statistics {

using DDSketch = dftracer::utils::dataframe::DDSketch;
using HistogramBin = dftracer::utils::dataframe::HistogramBin;

}  // namespace dftracer::utils::utilities::common::statistics

#endif  // DFTRACER_UTILS_UTILITIES_COMMON_STATISTICS_DDSKETCH_H
