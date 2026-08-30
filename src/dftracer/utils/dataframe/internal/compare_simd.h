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

/// Pack `n` per-row byte flags (nonzero = set) into the bit-packed bool bitmap
/// `out` (which the caller zeroed), 64 bits at a time via a vectorized
/// nonzero-mask store.
void pack_flags(const char* flags, std::int64_t n, std::uint8_t* out);

/// Monotonicity of a FLAT numeric column (integer or float), checked a vector
/// of adjacent pairs at a time. Sets `*out` and returns true if handled;
/// returns false (leaving `*out` untouched) for non-numeric, null-carrying, or
/// non-flat columns so the caller takes the scalar path.
bool is_sorted_numeric(const dftu_series& v, bool descending, bool* out);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_COMPARE_SIMD_H
