#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_STRING_OPS_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_STRING_OPS_H

#include <dftracer/utils/dataframe/dataframe.h>

#include <string_view>

namespace dftracer::utils::dataframe {

/// String predicates over a String/Binary column, each returning a Bool column.
/// A DICTIONARY input evaluates the predicate once per dictionary entry then
/// maps codes (O(dict + n)). Invalid (empty) for non-string inputs.
Series str_eq(const Series& v, std::string_view rhs);
Series str_contains(const Series& v, std::string_view needle);
Series str_starts_with(const Series& v, std::string_view prefix);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_STRING_OPS_H
