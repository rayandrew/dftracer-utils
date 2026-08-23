#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_SORT_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_SORT_H

#include <dftracer/utils/dataframe/dataframe.h>

namespace dftracer::utils::dataframe {

/// Stable argsort: an Int64 column of row indices ordering `v` ascending, or
/// descending when `descending`. Null rows sort last in both directions.
Series argsort(const Series& v, bool descending);

/// An Int64 column of the k row indices of the k largest (or smallest) values,
/// ordered best-first (partial sort). k is clamped to [0, length].
Series topk_indices(const Series& v, std::int64_t k, bool largest);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_SORT_H
