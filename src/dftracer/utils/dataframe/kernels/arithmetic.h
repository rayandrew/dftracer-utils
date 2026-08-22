#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_ARITHMETIC_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_ARITHMETIC_H

#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/scalar.h>

namespace dftracer::utils::dataframe {

/// Elementwise a + b / a - b / a * b. Inputs must share a numeric type and
/// length; the result is a FLAT column of that type. Invalid (empty) when the
/// type is unsupported or the inputs mismatch. Supported types: the fixed-width
/// integers and floats (not bool/string/binary).
Series add(const Series& a, const Series& b);
Series sub(const Series& a, const Series& b);
Series mul(const Series& a, const Series& b);

/// Elementwise a / b. Floats divide with SIMD; integers use a scalar loop with
/// a divide-by-zero guard (result 0). Same type and length as the inputs.
Series div(const Series& a, const Series& b);

/// Elementwise arithmetic against a broadcast scalar: a + value / a - value /
/// a * value. FLAT numeric input, carries validity. `value` may be any numeric
/// type; it is converted to the column's element type, exactly for 64-bit
/// integers (unlike a double scalar).
template <class T>
inline Series add_scalar(const Series& a, T value) {
    return Series{dftu_series_add_scalar(a.handle(), to_scalar(value))};
}
template <class T>
inline Series sub_scalar(const Series& a, T value) {
    return Series{dftu_series_sub_scalar(a.handle(), to_scalar(value))};
}
template <class T>
inline Series mul_scalar(const Series& a, T value) {
    return Series{dftu_series_mul_scalar(a.handle(), to_scalar(value))};
}
template <class T>
inline Series div_scalar(const Series& a, T value) {
    return Series{dftu_series_div_scalar(a.handle(), to_scalar(value))};
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_ARITHMETIC_H
