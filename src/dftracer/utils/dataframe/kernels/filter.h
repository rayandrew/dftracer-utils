#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_FILTER_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_FILTER_H

#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/scalar.h>

#include <cstdint>
#include <vector>

namespace dftracer::utils::dataframe {

/// Rows of `v` whose value is greater than `threshold`, returned as a SELECTION
/// column over `v` (an index list; no value data is copied). `threshold` may be
/// any numeric type; it is converted to the column's element type, exactly for
/// 64-bit integers.
template <class T>
inline Series filter_gt(const Series& v, T threshold) {
    return Series{dftu_series_filter_gt(v.handle(), to_scalar(threshold))};
}

/// Rows of `v` where the bit-packed Bool `mask` (same length) is true, as a
/// SELECTION column over `v`. Composes with the comparison/string kernels that
/// produce Bool masks. Invalid (empty) if `mask` is not a matching Bool column.
Series filter(const Series& v, const Series& mask);

/// Resolve any encoding to a new FLAT column (gather through the indirection).
Series materialize(const Series& v);

/// Gather the rows of `v` at `indices` into a new FLAT column. Handles every
/// column type (nested List/Struct included) and propagates validity.
Series take(const Series& v, const std::vector<std::int64_t>& indices);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_FILTER_H
