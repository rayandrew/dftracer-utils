#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_CAST_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_CAST_H

#include <dftracer/utils/dataframe/dataframe.h>

namespace dftracer::utils::dataframe {

/// Convert a FLAT numeric column to `target` (numeric), a new FLAT column
/// carrying the source validity. Invalid (empty) for non-numeric inputs.
Series cast(const Series& v, TypeId target);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_CAST_H
