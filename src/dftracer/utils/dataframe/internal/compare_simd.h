#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_COMPARE_SIMD_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_COMPARE_SIMD_H

#include <dftracer/utils/dataframe/abi.h>

#include <cstdint>

struct dftu_series;

namespace dftracer::utils::dataframe {

/// Compare each value of `v` against `rhs` (a DFTU_CMP_* op), packing the
/// result into the bit-packed bool bitmap `out` (which the caller zeroed) using
/// a vectorized compare + block mask store. Applies to a 4/8-byte numeric
/// column; returns true if handled, false to fall back to the scalar path.
bool compare(const dftu_series& v, std::int32_t op, dftu_scalar rhs,
             std::uint8_t* out);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_COMPARE_SIMD_H
