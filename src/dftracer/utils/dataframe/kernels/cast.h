#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_CAST_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_CAST_H

#include <dftracer/utils/dataframe/dataframe.h>

namespace dftracer::utils::dataframe {

/// Convert a FLAT numeric column to `target` (numeric), a new FLAT column
/// carrying the source validity. Invalid (empty) for non-numeric inputs.
Series cast(const Series& v, TypeId target);

/// SIMD fast path for the numeric casts Highway maps onto one ConvertTo /
/// Promote / Demote (float <-> same-width int, float widen/narrow). Writes `n`
/// converted values to `dv` and returns true; returns false for pairs it does
/// not vectorize, leaving the scalar path to handle them.
bool cast_simd(std::int32_t src, std::int32_t dst, const void* sv, void* dv,
               std::size_t n);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_CAST_H
