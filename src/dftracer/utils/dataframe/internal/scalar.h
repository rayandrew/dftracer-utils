#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_SCALAR_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_SCALAR_H

#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/types.h>

namespace dftracer::utils::dataframe {

/// Read a dftu_scalar as the column type T (converting from its tagged
/// domain). The inverse of to_scalar; used inside the kernels. An unknown tag
/// is read as F64 (its widest domain).
template <class T>
inline T scalar_as(const dftu_scalar& s) {
    if (s.kind < static_cast<std::int32_t>(ScalarTag::I64) ||
        s.kind > static_cast<std::int32_t>(ScalarTag::F64))
        return static_cast<T>(s.value.d);
    switch (static_cast<ScalarTag>(s.kind)) {
        case ScalarTag::I64:
            return static_cast<T>(s.value.i);
        case ScalarTag::U64:
            return static_cast<T>(s.value.u);
        case ScalarTag::F64:
            return static_cast<T>(s.value.d);
    }
    return static_cast<T>(s.value.d);
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_SCALAR_H
