#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_EXPR_HANDLE_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_EXPR_HANDLE_H

#include <dftracer/utils/dataframe/expr.h>

namespace dftracer::utils::dataframe {

// Access the Expr wrapped by a dftu_expr handle. Internal bridge so a C ABI in
// another translation unit (agg_expr.cpp) can compose the expr C ABI without
// re-declaring the opaque handle struct.
const Expr& expr_handle_unwrap(const dftu_expr* h);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_EXPR_HANDLE_H
