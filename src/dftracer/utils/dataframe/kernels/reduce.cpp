#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/numeric_dispatch.h>
#include <dftracer/utils/dataframe/internal/reduce_simd.h>
#include <dftracer/utils/dataframe/kernels/reduce.h>

#include <cstdint>
#include <type_traits>
#include <unordered_map>

namespace dftracer::utils::dataframe {

dftu_scalar sum(const Series& v) {
    return dftu_series_reduce(v.handle(), DFTU_REDUCE_SUM);
}
dftu_scalar min(const Series& v) {
    return dftu_series_reduce(v.handle(), DFTU_REDUCE_MIN);
}
dftu_scalar max(const Series& v) {
    return dftu_series_reduce(v.handle(), DFTU_REDUCE_MAX);
}
std::int64_t count(const Series& v) { return dftu_series_count(v.handle()); }

double mean(const Series& v) {
    std::int64_t n = count(v);
    if (n == 0) return 0.0;
    return scalar_value<double>(sum(v)) / static_cast<double>(n);
}

dftu_scalar product(const Series& v) { return dftu_series_product(v.handle()); }
bool all(const Series& v) { return dftu_series_all(v.handle()) != 0; }
bool any(const Series& v) { return dftu_series_any(v.handle()) != 0; }
std::int64_t arg_min(const Series& v) {
    return dftu_series_arg_min(v.handle());
}
std::int64_t arg_max(const Series& v) {
    return dftu_series_arg_max(v.handle());
}
dftu_scalar mode(const Series& v) { return dftu_series_mode(v.handle()); }

}  // namespace dftracer::utils::dataframe

namespace {

bool is_valid(const dftu_series& v, std::int64_t i) {
    if (!v.validity) return true;
    return (v.validity->data()[i >> 3] & (1u << (i & 7))) != 0;
}

template <class T>
void store_domain(dftu_scalar& out, T value) {
    if constexpr (std::is_floating_point_v<T>) {
        out.kind = DFTU_SCALAR_TAG_F64;
        out.value.d = static_cast<double>(value);
    } else if constexpr (std::is_unsigned_v<T>) {
        out.kind = DFTU_SCALAR_TAG_U64;
        out.value.u = static_cast<std::uint64_t>(value);
    } else {
        out.kind = DFTU_SCALAR_TAG_I64;
        out.value.i = static_cast<std::int64_t>(value);
    }
}

// Integer sums accumulate in the widest same-signedness type (no overflow of a
// narrow input); float/f32 sums accumulate in double to hold precision.
template <class T>
using sum_acc_t = std::conditional_t<
    std::is_floating_point_v<T>, double,
    std::conditional_t<std::is_unsigned_v<T>, std::uint64_t, std::int64_t>>;

template <class T>
void reduce_one(const dftu_series& v, std::int32_t op, dftu_scalar& out) {
    const T* p = reinterpret_cast<const T*>(v.data->data());
    const std::int64_t n = v.length;
    // A validity bitmap forces a per-element branch that blocks the compiler
    // from vectorizing the reduction; the far more common no-null column runs a
    // branch-free loop that auto-vectorizes (reduce_simd handles f64 sum before
    // we get here, but integer sum and all min/max land on this path).
    const bool has_null = v.validity != nullptr;

    if (op == DFTU_REDUCE_SUM) {
        sum_acc_t<T> acc = 0;
        if (has_null) {
            for (std::int64_t i = 0; i < n; ++i)
                if (is_valid(v, i)) acc += static_cast<sum_acc_t<T>>(p[i]);
        } else {
            for (std::int64_t i = 0; i < n; ++i)
                acc += static_cast<sum_acc_t<T>>(p[i]);
        }
        if constexpr (std::is_floating_point_v<T>) {
            out.kind = DFTU_SCALAR_TAG_F64;
            out.value.d = acc;
        } else if constexpr (std::is_unsigned_v<T>) {
            out.kind = DFTU_SCALAR_TAG_U64;
            out.value.u = acc;
        } else {
            out.kind = DFTU_SCALAR_TAG_I64;
            out.value.i = acc;
        }
        return;
    }

    const bool is_min = (op == DFTU_REDUCE_MIN);
    T best{};
    if (has_null) {
        bool found = false;
        for (std::int64_t i = 0; i < n; ++i) {
            if (!is_valid(v, i)) continue;
            if (!found) {
                best = p[i];
                found = true;
            } else if (is_min) {
                if (p[i] < best) best = p[i];
            } else if (p[i] > best) {
                best = p[i];
            }
        }
    } else if (n > 0) {
        // op hoisted out of the loop so each stays a branch-free min/max
        // reduction the compiler can vectorize.
        best = p[0];
        if (is_min) {
            for (std::int64_t i = 1; i < n; ++i)
                if (p[i] < best) best = p[i];
        } else {
            for (std::int64_t i = 1; i < n; ++i)
                if (p[i] > best) best = p[i];
        }
    }
    store_domain<T>(out, best);
}

// Product accumulates in the column's wide domain, skipping nulls (identity 1).
template <class T>
void product_one(const dftu_series& v, dftu_scalar& out) {
    const T* p = reinterpret_cast<const T*>(v.data->data());
    const std::int64_t n = v.length;
    const bool has_null = v.validity != nullptr;
    sum_acc_t<T> acc = 1;
    for (std::int64_t i = 0; i < n; ++i)
        if (!has_null || is_valid(v, i)) acc *= static_cast<sum_acc_t<T>>(p[i]);
    if constexpr (std::is_floating_point_v<T>) {
        out.kind = DFTU_SCALAR_TAG_F64;
        out.value.d = static_cast<double>(acc);
    } else if constexpr (std::is_unsigned_v<T>) {
        out.kind = DFTU_SCALAR_TAG_U64;
        out.value.u = static_cast<std::uint64_t>(acc);
    } else {
        out.kind = DFTU_SCALAR_TAG_I64;
        out.value.i = static_cast<std::int64_t>(acc);
    }
}

template <class T>
void argextreme_one(const dftu_series& v, bool is_min, std::int64_t& out_idx) {
    const T* p = reinterpret_cast<const T*>(v.data->data());
    const std::int64_t n = v.length;
    const bool has_null = v.validity != nullptr;
    out_idx = -1;
    T best{};
    for (std::int64_t i = 0; i < n; ++i) {
        if (has_null && !is_valid(v, i)) continue;
        if (out_idx < 0) {
            best = p[i];
            out_idx = i;
        } else if (is_min ? (p[i] < best) : (p[i] > best)) {
            best = p[i];
            out_idx = i;
        }
    }
}

template <class T>
void mode_one(const dftu_series& v, dftu_scalar& out) {
    const T* p = reinterpret_cast<const T*>(v.data->data());
    const std::int64_t n = v.length;
    const bool has_null = v.validity != nullptr;
    std::unordered_map<T, std::int64_t> counts;
    T best{};
    std::int64_t best_count = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if (has_null && !is_valid(v, i)) continue;
        const std::int64_t c = ++counts[p[i]];
        // Strictly-greater keeps the value that first reached the top count,
        // so ties resolve to the earlier-appearing value.
        if (c > best_count) {
            best_count = c;
            best = p[i];
        }
    }
    store_domain<T>(out, best);
}

}  // namespace

dftu_scalar dftu_series_reduce(const dftu_series* v, dftu_reduce_op op) {
    using dftracer::utils::dataframe::TypeId;
    dftu_scalar out{};
    out.kind = DFTU_SCALAR_TAG_I64;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat) return out;
    if (v->type == TypeId::Bool || v->type == TypeId::String ||
        v->type == TypeId::Binary)
        return out;

    if (dftracer::utils::dataframe::reduce(*v, op, out)) return out;
    DF_NUMERIC_DISPATCH(v->type, reduce_one, *v, op, out)
    return out;
}

int64_t dftu_series_count(const dftu_series* v) {
    return v->length - v->null_count;
}

dftu_scalar dftu_series_product(const dftu_series* v) {
    using dftracer::utils::dataframe::TypeId;
    dftu_scalar out{};
    out.kind = DFTU_SCALAR_TAG_I64;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat) return out;
    if (v->type == TypeId::Bool || v->type == TypeId::String ||
        v->type == TypeId::Binary)
        return out;
    if (dftracer::utils::dataframe::product_simd(*v, out)) return out;
    DF_NUMERIC_DISPATCH(v->type, product_one, *v, out)
    return out;
}

int32_t dftu_series_all(const dftu_series* v) {
    using dftracer::utils::dataframe::TypeId;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat ||
        v->type != TypeId::Bool)
        return 1;
    return dftracer::utils::dataframe::all_bits(*v) ? 1 : 0;
}

int32_t dftu_series_any(const dftu_series* v) {
    using dftracer::utils::dataframe::TypeId;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat ||
        v->type != TypeId::Bool)
        return 0;
    return dftracer::utils::dataframe::any_bits(*v) ? 1 : 0;
}

int64_t dftu_series_arg_min(const dftu_series* v) {
    using dftracer::utils::dataframe::TypeId;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat ||
        v->type == TypeId::Bool || v->type == TypeId::String ||
        v->type == TypeId::Binary)
        return -1;
    std::int64_t idx = -1;
    DF_NUMERIC_DISPATCH(v->type, argextreme_one, *v, true, idx)
    return idx;
}

int64_t dftu_series_arg_max(const dftu_series* v) {
    using dftracer::utils::dataframe::TypeId;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat ||
        v->type == TypeId::Bool || v->type == TypeId::String ||
        v->type == TypeId::Binary)
        return -1;
    std::int64_t idx = -1;
    DF_NUMERIC_DISPATCH(v->type, argextreme_one, *v, false, idx)
    return idx;
}

dftu_scalar dftu_series_mode(const dftu_series* v) {
    using dftracer::utils::dataframe::TypeId;
    dftu_scalar out{};
    out.kind = DFTU_SCALAR_TAG_I64;
    if (v->encoding != dftracer::utils::dataframe::Encoding::Flat) return out;
    if (v->type == TypeId::Bool || v->type == TypeId::String ||
        v->type == TypeId::Binary)
        return out;
    DF_NUMERIC_DISPATCH(v->type, mode_one, *v, out)
    return out;
}
