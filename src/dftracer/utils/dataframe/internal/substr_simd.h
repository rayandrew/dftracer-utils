#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_SUBSTR_SIMD_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_SUBSTR_SIMD_H

#include <cstdint>

namespace dftracer::utils::dataframe {

/// Byte index of the first occurrence of `needle` (length `needle_len`) in
/// `hay` (length `hay_len`), or -1 if absent. Vectorized (Highway dynamic
/// dispatch): the needle's first byte is broadcast across a u8 vector to find
/// candidate offsets, each verified with memcmp. An empty needle matches at 0,
/// mirroring std::string_view::find (so results are byte-identical to the
/// scalar search).
std::int64_t substr_find(const char* hay, std::int64_t hay_len,
                         const char* needle, std::int64_t needle_len);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_SUBSTR_SIMD_H
