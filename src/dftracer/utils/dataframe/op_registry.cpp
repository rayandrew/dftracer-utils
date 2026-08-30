#include <dftracer/utils/dataframe/abi.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace {

// A function-to-void* cast is not constexpr, so this array is dynamically
// initialized before main; fine, since lookups only run at scan time.
const dftu_op_desc BUILTINS[] = {
#define DFTU_SERIES_OP(name, fn, ret, o0, o1, o2) \
    {#name, DFTU_OP_SIG(ret, o0, o1, o2), reinterpret_cast<const void*>(&fn)},
#include <dftracer/utils/dataframe/exported_series_ops.def>
#define DFTU_FRAME_OP(name, fn, ret, o0, o1, o2, o3, o4)    \
    {"frame." #name, DFTU_OP_SIG6(ret, o0, o1, o2, o3, o4), \
     reinterpret_cast<const void*>(&fn)},
#include <dftracer/utils/dataframe/exported_frame_ops.def>
};
constexpr uint32_t BUILTIN_COUNT =
    static_cast<uint32_t>(sizeof(BUILTINS) / sizeof(BUILTINS[0]));

std::mutex& reg_mutex() {
    static std::mutex m;
    return m;
}

// unique_ptr storage keeps each record's address stable as the vector grows, so
// a pointer from dftu_op_find stays valid after a later registration.
std::vector<std::unique_ptr<dftu_op_desc>>& user_ops() {
    static std::vector<std::unique_ptr<dftu_op_desc>> v;
    return v;
}

const dftu_op_desc* find_locked(const char* name) {
    for (uint32_t i = 0; i < BUILTIN_COUNT; ++i)
        if (std::strcmp(BUILTINS[i].name, name) == 0) return &BUILTINS[i];
    for (const auto& op : user_ops())
        if (std::strcmp(op->name, name) == 0) return op.get();
    return nullptr;
}

using CS = const dftu_series*;
template <class F>
F as(const void* fn) {
    return reinterpret_cast<F>(const_cast<void*>(fn));
}

const char* tok_name(dftu_op_tok t) {
    switch (t) {
        case DFTU_TOK_NONE:
            return "";
        case DFTU_TOK_SERIES:
            return "series";
        case DFTU_TOK_SCALAR:
            return "scalar";
        case DFTU_TOK_I64:
            return "i64";
        case DFTU_TOK_BOOL:
            return "bool";
        case DFTU_TOK_CMP:
            return "cmp";
        case DFTU_TOK_PRIM:
            return "prim";
        case DFTU_TOK_LOGICAL:
            return "logical";
        case DFTU_TOK_DTYPE:
            return "dtype";
        case DFTU_TOK_REDUCE:
            return "reduce";
        case DFTU_TOK_STR:
            return "str";
        case DFTU_TOK_CHAR:
            return "char";
        case DFTU_TOK_F64:
            return "f64";
        case DFTU_TOK_I32:
            return "i32";
        case DFTU_TOK_RANK:
            return "rank";
        case DFTU_TOK_ROLLING:
            return "rolling";
        case DFTU_TOK_FRAME:
            return "frame";
        case DFTU_TOK_STRLIST:
            return "strlist";
    }
    return "?";
}

// An operand token that is neither a column nor empty is read from dftu_op_arg.
bool needs_arg(dftu_op_sig sig) {
    for (int i = 0; i < DFTU_OP_MAX_ARGS; ++i) {
        dftu_op_tok t = DFTU_OP_SIG_ARG(sig, i);
        if (t != DFTU_TOK_NONE && t != DFTU_TOK_SERIES && t != DFTU_TOK_FRAME)
            return true;
    }
    return false;
}

}  // namespace

extern "C" {

dftu_op_kind dftu_op_kind_of(dftu_op_sig sig) {
    dftu_op_tok r = DFTU_OP_SIG_RET(sig);
    if (r == DFTU_TOK_SERIES) return DFTU_OP_KIND_SERIES;
    if (r == DFTU_TOK_FRAME) return DFTU_OP_KIND_FRAME;
    return DFTU_OP_KIND_AGGREGATE;
}

uint32_t dftu_op_arity(dftu_op_sig sig) {
    // The primary positional operand: FRAME for a frame op (in frames[]), else
    // SERIES (in in[]). A frame op's series/scalar operands ride dftu_op_arg.
    dftu_op_tok primary = dftu_op_kind_of(sig) == DFTU_OP_KIND_FRAME
                              ? DFTU_TOK_FRAME
                              : DFTU_TOK_SERIES;
    uint32_t n = 0;
    for (int i = 0; i < DFTU_OP_MAX_ARGS; ++i)
        if (DFTU_OP_SIG_ARG(sig, i) == primary) ++n;
    return n;
}

const char* dftu_op_signature(dftu_op_sig sig) {
    static thread_local std::string s;
    s = "(";
    bool first = true;
    for (int i = 0; i < DFTU_OP_MAX_ARGS; ++i) {
        dftu_op_tok t = DFTU_OP_SIG_ARG(sig, i);
        if (t == DFTU_TOK_NONE) break;
        if (!first) s += ", ";
        s += tok_name(t);
        first = false;
    }
    s += ") -> ";
    s += tok_name(DFTU_OP_SIG_RET(sig));
    return s.c_str();
}

const dftu_op_desc* dftu_op_find(const char* name) {
    if (!name) return nullptr;
    std::lock_guard<std::mutex> lock(reg_mutex());
    return find_locked(name);
}

uint32_t dftu_op_count(void) {
    std::lock_guard<std::mutex> lock(reg_mutex());
    return BUILTIN_COUNT + static_cast<uint32_t>(user_ops().size());
}

const dftu_op_desc* dftu_op_at(uint32_t i) {
    std::lock_guard<std::mutex> lock(reg_mutex());
    if (i < BUILTIN_COUNT) return &BUILTINS[i];
    i -= BUILTIN_COUNT;
    if (i < user_ops().size()) return user_ops()[i].get();
    return nullptr;
}

int dftu_op_register(const dftu_op_desc* desc) {
    if (!desc || !desc->name) return 1;
    std::lock_guard<std::mutex> lock(reg_mutex());
    if (find_locked(desc->name)) return 1;  // no silent shadowing
    user_ops().push_back(std::make_unique<dftu_op_desc>(*desc));
    return 0;
}

dftu_series* dftu_op_run(const dftu_op_desc* op, const dftu_series* const* in,
                         uint32_t n, const dftu_op_arg* a) {
    if (!op || !op->fn || !in) return nullptr;
    if (dftu_op_kind_of(op->sig) != DFTU_OP_KIND_SERIES) return nullptr;
    if (n != dftu_op_arity(op->sig)) return nullptr;
    if (needs_arg(op->sig) && !a) return nullptr;
    const dftu_op_val* g = a ? a->args : nullptr;
    switch (op->sig) {
        case DFTU_OP_SIG(SERIES, SERIES, NONE, NONE):
            return as<dftu_series* (*)(CS)>(op->fn)(in[0]);
        case DFTU_OP_SIG(SERIES, SERIES, SERIES, NONE):
            return as<dftu_series* (*)(CS, CS)>(op->fn)(in[0], in[1]);
        case DFTU_OP_SIG(SERIES, SERIES, SCALAR, NONE):
            return as<dftu_series* (*)(CS, dftu_scalar)>(op->fn)(in[0],
                                                                 g[1].scalar);
        case DFTU_OP_SIG(SERIES, SERIES, CMP, SCALAR):
            return as<dftu_series* (*)(CS, dftu_cmp_op, dftu_scalar)>(op->fn)(
                in[0], static_cast<dftu_cmp_op>(g[1].i32), g[2].scalar);
        case DFTU_OP_SIG(SERIES, SERIES, PRIM, NONE):
            return as<dftu_series* (*)(CS, dftu_prim_op)>(op->fn)(
                in[0], static_cast<dftu_prim_op>(g[1].i32));
        case DFTU_OP_SIG(SERIES, SERIES, SERIES, LOGICAL):
            return as<dftu_series* (*)(CS, CS, dftu_logical_op)>(op->fn)(
                in[0], in[1], static_cast<dftu_logical_op>(g[2].i32));
        case DFTU_OP_SIG(SERIES, SERIES, DTYPE, NONE):
            return as<dftu_series* (*)(CS, dftu_dtype)>(op->fn)(
                in[0], static_cast<dftu_dtype>(g[1].i32));
        case DFTU_OP_SIG(SERIES, SERIES, STR, NONE):
            return as<dftu_series* (*)(CS, const char*, int32_t)>(op->fn)(
                in[0], g[1].str.ptr, g[1].str.len);
        case DFTU_OP_SIG(SERIES, SERIES, STR, STR):
            return as<dftu_series* (*)(CS, const char*, int32_t, const char*,
                                       int32_t)>(op->fn)(
                in[0], g[1].str.ptr, g[1].str.len, g[2].str.ptr, g[2].str.len);
        case DFTU_OP_SIG(SERIES, SERIES, I64, I64):
            return as<dftu_series* (*)(CS, int64_t, int64_t)>(op->fn)(
                in[0], g[1].i64, g[2].i64);
        case DFTU_OP_SIG(SERIES, SERIES, I64, CHAR):
            return as<dftu_series* (*)(CS, int64_t, char)>(op->fn)(
                in[0], g[1].i64, g[2].ch);
        case DFTU_OP_SIG(SERIES, SERIES, I64, NONE):
            return as<dftu_series* (*)(CS, int64_t)>(op->fn)(in[0], g[1].i64);
        case DFTU_OP_SIG(SERIES, SERIES, SCALAR, SCALAR):
            return as<dftu_series* (*)(CS, dftu_scalar, dftu_scalar)>(op->fn)(
                in[0], g[1].scalar, g[2].scalar);
        case DFTU_OP_SIG(SERIES, SERIES, I32, NONE):
            return as<dftu_series* (*)(CS, int32_t)>(op->fn)(in[0], g[1].i32);
        case DFTU_OP_SIG(SERIES, SERIES, F64, NONE):
            return as<dftu_series* (*)(CS, double)>(op->fn)(in[0], g[1].f64);
        case DFTU_OP_SIG(SERIES, SERIES, I64, F64):
            return as<dftu_series* (*)(CS, int64_t, double)>(op->fn)(
                in[0], g[1].i64, g[2].f64);
        case DFTU_OP_SIG(SERIES, SERIES, I64, ROLLING):
            return as<dftu_series* (*)(CS, int64_t, dftu_rolling_op)>(op->fn)(
                in[0], g[1].i64, static_cast<dftu_rolling_op>(g[2].i32));
        case DFTU_OP_SIG(SERIES, SERIES, RANK, I64):
            return as<dftu_series* (*)(CS, dftu_rank_method, int32_t)>(op->fn)(
                in[0], static_cast<dftu_rank_method>(g[1].i32),
                static_cast<int32_t>(g[2].i64));
        default:
            return nullptr;
    }
}

dftu_scalar dftu_op_run_aggregate(const dftu_op_desc* op, const dftu_series* v,
                                  const dftu_op_arg* a, int* ok) {
    dftu_scalar z;
    z.kind = DFTU_SCALAR_TAG_I64;
    z.value.i = 0;
    if (!op || !op->fn || dftu_op_kind_of(op->sig) != DFTU_OP_KIND_AGGREGATE) {
        if (ok) *ok = 0;
        return z;
    }
    if (needs_arg(op->sig) && !a) {
        if (ok) *ok = 0;
        return z;
    }
    if (ok) *ok = 1;
    const dftu_op_val* g = a ? a->args : nullptr;
    switch (op->sig) {
        case DFTU_OP_SIG(SCALAR, SERIES, NONE, NONE):
            return as<dftu_scalar (*)(CS)>(op->fn)(v);
        case DFTU_OP_SIG(SCALAR, SERIES, REDUCE, NONE):
            return as<dftu_scalar (*)(CS, dftu_reduce_op)>(op->fn)(
                v, static_cast<dftu_reduce_op>(g[1].i32));
        case DFTU_OP_SIG(I64, SERIES, NONE, NONE):
            z.value.i = as<int64_t (*)(CS)>(op->fn)(v);
            return z;
        case DFTU_OP_SIG(BOOL, SERIES, NONE, NONE):
            z.value.i = as<int32_t (*)(CS)>(op->fn)(v);
            return z;
        case DFTU_OP_SIG(BOOL, SERIES, I32, NONE):
            z.value.i = as<int32_t (*)(CS, int32_t)>(op->fn)(v, g[1].i32);
            return z;
        case DFTU_OP_SIG(F64, SERIES, NONE, NONE):
            z.kind = DFTU_SCALAR_TAG_F64;
            z.value.d = as<double (*)(CS)>(op->fn)(v);
            return z;
        case DFTU_OP_SIG(F64, SERIES, I32, NONE):
            z.kind = DFTU_SCALAR_TAG_F64;
            z.value.d = as<double (*)(CS, int32_t)>(op->fn)(v, g[1].i32);
            return z;
        case DFTU_OP_SIG(F64, SERIES, F64, NONE):
            z.kind = DFTU_SCALAR_TAG_F64;
            z.value.d = as<double (*)(CS, double)>(op->fn)(v, g[1].f64);
            return z;
        default:
            if (ok) *ok = 0;
            return z;
    }
}

dftu_dataframe* dftu_op_run_frame(const dftu_op_desc* op,
                                  const dftu_dataframe* const* frames,
                                  uint32_t n, const dftu_op_arg* a) {
    if (!op || !op->fn) return nullptr;
    if (dftu_op_kind_of(op->sig) != DFTU_OP_KIND_FRAME) return nullptr;
    if (n != dftu_op_arity(op->sig)) return nullptr;
    using CDF = const dftu_dataframe*;
    using DF = dftu_dataframe*;
    const dftu_op_val* g = a ? a->args : nullptr;
    CDF df = (n >= 1 && frames) ? frames[0] : nullptr;
    switch (op->sig) {
        case DFTU_OP_SIG(FRAME, FRAME, NONE, NONE):
            return as<DF (*)(CDF)>(op->fn)(df);
        case DFTU_OP_SIG(FRAME, FRAME, I64, NONE):
            return as<DF (*)(CDF, int64_t)>(op->fn)(df, g[1].i64);
        case DFTU_OP_SIG(FRAME, FRAME, I64, I64):
            return as<DF (*)(CDF, int64_t, int64_t)>(op->fn)(df, g[1].i64,
                                                             g[2].i64);
        case DFTU_OP_SIG(FRAME, FRAME, SCALAR, NONE):
            return as<DF (*)(CDF, dftu_scalar)>(op->fn)(df, g[1].scalar);
        case DFTU_OP_SIG(FRAME, FRAME, SERIES, NONE):
            if (!g[1].series) return nullptr;
            return as<DF (*)(CDF, CS)>(op->fn)(df, g[1].series);
        case DFTU_OP_SIG(FRAME, FRAME, STR, NONE):
            return as<DF (*)(CDF, const char*)>(op->fn)(df, g[1].str.ptr);
        case DFTU_OP_SIG(FRAME, FRAME, STR, I32):
            return as<DF (*)(CDF, const char*, int32_t)>(op->fn)(
                df, g[1].str.ptr, g[2].i32);
        case DFTU_OP_SIG(FRAME, FRAME, STR, SERIES):
            if (!g[2].series) return nullptr;
            return as<DF (*)(CDF, const char*, CS)>(op->fn)(df, g[1].str.ptr,
                                                            g[2].series);
        case DFTU_OP_SIG(FRAME, FRAME, STRLIST, NONE):
            return as<DF (*)(CDF, const char* const*, int32_t)>(op->fn)(
                df, g[1].list.items, g[1].list.n);
        case DFTU_OP_SIG(FRAME, FRAME, STRLIST, I32):
            return as<DF (*)(CDF, const char* const*, int32_t, int32_t)>(
                op->fn)(df, g[1].list.items, g[1].list.n, g[2].i32);
        case DFTU_OP_SIG(FRAME, FRAME, STRLIST, STRLIST):
            return as<DF (*)(CDF, const char* const*, int32_t,
                             const char* const*, int32_t)>(op->fn)(
                df, g[1].list.items, g[1].list.n, g[2].list.items, g[2].list.n);
        case DFTU_OP_SIG6(FRAME, FRAME, STR, I64, I32, NONE):
            return as<DF (*)(CDF, const char*, int64_t, int32_t)>(op->fn)(
                df, g[1].str.ptr, g[2].i64, g[3].i32);
        case DFTU_OP_SIG6(FRAME, FRAME, STR, STR, STR, STR):
            return as<DF (*)(CDF, const char*, const char*, const char*,
                             const char*)>(op->fn)(
                df, g[1].str.ptr, g[2].str.ptr, g[3].str.ptr, g[4].str.ptr);
        case DFTU_OP_SIG(FRAME, SERIES, NONE,
                         NONE):  // value_counts: series -> frame
            if (!g[0].series) return nullptr;
            return as<DF (*)(CS)>(op->fn)(g[0].series);
        default:
            return nullptr;
    }
}

}  // extern "C"
