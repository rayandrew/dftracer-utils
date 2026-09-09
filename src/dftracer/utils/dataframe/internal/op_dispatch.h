#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_OP_DISPATCH_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_OP_DISPATCH_H

#include <dftracer/utils/dataframe/abi.h>

namespace dftracer::utils::dataframe::internal {

using CS = const dftu_series*;
using CDF = const dftu_dataframe*;
using DF = dftu_dataframe*;

// Maps a packed op signature to the function-pointer type a registered fn of
// that shape must have. Left undefined for a signature dftu_op_run/
// dftu_op_run_aggregate/dftu_op_run_frame has no case for, so a .def row of an
// unhandled shape fails to compile instead of silently returning nullptr
// forever at run time.
template <dftu_op_sig S>
struct op_fn;

#define DFTU_OP_FN(sig, FnType) \
    template <>                 \
    struct op_fn<sig> {         \
        using type = FnType;    \
    };

// dftu_op_run (DFTU_OP_KIND_SERIES).
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, NONE, NONE), dftu_series* (*)(CS))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, SERIES, NONE), dftu_series* (*)(CS, CS))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, SCALAR, NONE),
           dftu_series* (*)(CS, dftu_scalar))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, CMP, SCALAR),
           dftu_series* (*)(CS, dftu_cmp_op, dftu_scalar))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, PRIM, NONE),
           dftu_series* (*)(CS, dftu_prim_op))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, SERIES, LOGICAL),
           dftu_series* (*)(CS, CS, dftu_logical_op))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, DTYPE, NONE),
           dftu_series* (*)(CS, dftu_dtype))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, STR, NONE),
           dftu_series* (*)(CS, const char*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, STR, STR),
           dftu_series* (*)(CS, const char*, int32_t, const char*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, I64, I64),
           dftu_series* (*)(CS, int64_t, int64_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, I64, CHAR),
           dftu_series* (*)(CS, int64_t, char))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, I64, NONE),
           dftu_series* (*)(CS, int64_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, SCALAR, SCALAR),
           dftu_series* (*)(CS, dftu_scalar, dftu_scalar))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, I32, NONE),
           dftu_series* (*)(CS, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, F64, NONE), dftu_series* (*)(CS, double))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, I64, F64),
           dftu_series* (*)(CS, int64_t, double))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, I64, ROLLING),
           dftu_series* (*)(CS, int64_t, dftu_rolling_op))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, STR, NONE, NONE),
           dftu_series* (*)(const char*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, STR, STR, NONE),
           dftu_series* (*)(const char*, int32_t, const char*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(SERIES, SERIES, RANK, I64),
           dftu_series* (*)(CS, dftu_rank_method, int32_t))

// dftu_op_run_aggregate (DFTU_OP_KIND_AGGREGATE).
DFTU_OP_FN(DFTU_OP_SIG(SCALAR, SERIES, NONE, NONE), dftu_scalar (*)(CS))
DFTU_OP_FN(DFTU_OP_SIG(SCALAR, SERIES, REDUCE, NONE),
           dftu_scalar (*)(CS, dftu_reduce_op))
DFTU_OP_FN(DFTU_OP_SIG(SCALAR, SERIES, SERIES, NONE), dftu_scalar (*)(CS, CS))
DFTU_OP_FN(DFTU_OP_SIG(I64, SERIES, NONE, NONE), int64_t (*)(CS))
DFTU_OP_FN(DFTU_OP_SIG(BOOL, SERIES, NONE, NONE), int32_t (*)(CS))
DFTU_OP_FN(DFTU_OP_SIG(BOOL, SERIES, I32, NONE), int32_t (*)(CS, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(F64, SERIES, NONE, NONE), double (*)(CS))
DFTU_OP_FN(DFTU_OP_SIG(F64, SERIES, I32, NONE), double (*)(CS, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(BOOL, STR, STR, NONE),
           int32_t (*)(const char*, int32_t, const char*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(F64, SERIES, F64, NONE), double (*)(CS, double))

// dftu_op_run_frame (DFTU_OP_KIND_FRAME).
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, NONE, NONE), DF (*)(CDF))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, I64, NONE), DF (*)(CDF, int64_t))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, I64, I64), DF (*)(CDF, int64_t, int64_t))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, SCALAR, NONE), DF (*)(CDF, dftu_scalar))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, SERIES, NONE), DF (*)(CDF, CS))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STR, NONE), DF (*)(CDF, const char*))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STR, I32),
           DF (*)(CDF, const char*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STR, SERIES), DF (*)(CDF, const char*, CS))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STRLIST, NONE),
           DF (*)(CDF, const char* const*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STRLIST, I32),
           DF (*)(CDF, const char* const*, int32_t, int32_t))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STRLIST, STRLIST),
           DF (*)(CDF, const char* const*, int32_t, const char* const*,
                  int32_t))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, FRAME, STRLIST, I32LIST),
           DF (*)(CDF, const char* const*, int32_t, const int32_t*, int32_t))
DFTU_OP_FN(DFTU_OP_SIG6(FRAME, FRAME, STR, I64, I32, NONE),
           DF (*)(CDF, const char*, int64_t, int32_t))
DFTU_OP_FN(DFTU_OP_SIG6(FRAME, FRAME, STR, STR, STR, STR),
           DF (*)(CDF, const char*, const char*, const char*, const char*))
DFTU_OP_FN(DFTU_OP_SIG(FRAME, SERIES, NONE, NONE), DF (*)(CS))

#undef DFTU_OP_FN

}  // namespace dftracer::utils::dataframe::internal

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_OP_DISPATCH_H
