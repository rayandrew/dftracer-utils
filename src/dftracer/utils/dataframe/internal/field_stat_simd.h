#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_FIELD_STAT_SIMD_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_FIELD_STAT_SIMD_H

#include <cstdint>

namespace dftracer::utils::dataframe {

// Target-independent reduction result filled by the per-target Highway kernels
// in kernels/field_stat.cpp. Include-guarded so foreach_target's repeated
// inclusion of the translation unit does not redefine it.
struct FsRaw {
    double sum, sumsq, m3, m4, min, max;
    std::int64_t esum, emin, emax;
};

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_FIELD_STAT_SIMD_H
