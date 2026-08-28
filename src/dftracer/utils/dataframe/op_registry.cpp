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
    }
    return "?";
}

// An operand token that is neither a column nor empty is read from dftu_op_arg.
bool needs_arg(dftu_op_sig sig) {
    for (int i = 0; i < 3; ++i) {
        dftu_op_tok t = DFTU_OP_SIG_ARG(sig, i);
        if (t != DFTU_TOK_NONE && t != DFTU_TOK_SERIES) return true;
    }
    return false;
}

}  // namespace

extern "C" {

dftu_op_kind dftu_op_kind_of(dftu_op_sig sig) {
    return DFTU_OP_SIG_RET(sig) == DFTU_TOK_SERIES ? DFTU_OP_KIND_SERIES
                                                   : DFTU_OP_KIND_AGGREGATE;
}

uint32_t dftu_op_arity(dftu_op_sig sig) {
    uint32_t n = 0;
    for (int i = 0; i < 3; ++i)
        if (DFTU_OP_SIG_ARG(sig, i) == DFTU_TOK_SERIES) ++n;
    return n;
}

const char* dftu_op_signature(dftu_op_sig sig) {
    static thread_local std::string s;
    s = "(";
    bool first = true;
    for (int i = 0; i < 3; ++i) {
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
    switch (op->sig) {
        case DFTU_OP_SIG(SERIES, SERIES, NONE, NONE):
            return as<dftu_series* (*)(CS)>(op->fn)(in[0]);
        case DFTU_OP_SIG(SERIES, SERIES, SERIES, NONE):
            return as<dftu_series* (*)(CS, CS)>(op->fn)(in[0], in[1]);
        case DFTU_OP_SIG(SERIES, SERIES, SCALAR, NONE):
            return as<dftu_series* (*)(CS, dftu_scalar)>(op->fn)(in[0],
                                                                 a->scalar);
        case DFTU_OP_SIG(SERIES, SERIES, CMP, SCALAR):
            return as<dftu_series* (*)(CS, dftu_cmp_op, dftu_scalar)>(op->fn)(
                in[0], static_cast<dftu_cmp_op>(a->op_code), a->scalar);
        case DFTU_OP_SIG(SERIES, SERIES, PRIM, NONE):
            return as<dftu_series* (*)(CS, dftu_prim_op)>(op->fn)(
                in[0], static_cast<dftu_prim_op>(a->op_code));
        case DFTU_OP_SIG(SERIES, SERIES, SERIES, LOGICAL):
            return as<dftu_series* (*)(CS, CS, dftu_logical_op)>(op->fn)(
                in[0], in[1], static_cast<dftu_logical_op>(a->op_code));
        case DFTU_OP_SIG(SERIES, SERIES, DTYPE, NONE):
            return as<dftu_series* (*)(CS, dftu_dtype)>(op->fn)(
                in[0], static_cast<dftu_dtype>(a->op_code));
        case DFTU_OP_SIG(SERIES, SERIES, STR, NONE):
            return as<dftu_series* (*)(CS, const char*, int32_t)>(op->fn)(
                in[0], a->s0, a->s0_len);
        case DFTU_OP_SIG(SERIES, SERIES, STR, STR):
            return as<dftu_series* (*)(CS, const char*, int32_t, const char*,
                                       int32_t)>(op->fn)(
                in[0], a->s0, a->s0_len, a->s1, a->s1_len);
        case DFTU_OP_SIG(SERIES, SERIES, I64, I64):
            return as<dftu_series* (*)(CS, int64_t, int64_t)>(op->fn)(
                in[0], a->i0, a->i1);
        case DFTU_OP_SIG(SERIES, SERIES, I64, CHAR):
            return as<dftu_series* (*)(CS, int64_t, char)>(op->fn)(in[0], a->i0,
                                                                   a->ch);
        case DFTU_OP_SIG(SERIES, SERIES, I64, NONE):
            return as<dftu_series* (*)(CS, int64_t)>(op->fn)(in[0], a->i0);
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
    switch (op->sig) {
        case DFTU_OP_SIG(SCALAR, SERIES, NONE, NONE):
            return as<dftu_scalar (*)(CS)>(op->fn)(v);
        case DFTU_OP_SIG(SCALAR, SERIES, REDUCE, NONE):
            return as<dftu_scalar (*)(CS, dftu_reduce_op)>(op->fn)(
                v, static_cast<dftu_reduce_op>(a->op_code));
        case DFTU_OP_SIG(I64, SERIES, NONE, NONE):
            z.value.i = as<int64_t (*)(CS)>(op->fn)(v);
            return z;
        case DFTU_OP_SIG(BOOL, SERIES, NONE, NONE):
            z.value.i = as<int32_t (*)(CS)>(op->fn)(v);
            return z;
        default:
            if (ok) *ok = 0;
            return z;
    }
}

}  // extern "C"
