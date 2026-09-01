#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/numeric_dispatch.h>
#include <dftracer/utils/dataframe/internal/scalar.h>
#include <dftracer/utils/dataframe/kernels/arithmetic.h>
#include <dftracer/utils/dataframe/parallel.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>

// Highway runtime dispatch: foreach_target.h recompiles this TU once per ISA,
// then HWY_DYNAMIC_DISPATCH selects the best at load time.
#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dftracer/utils/dataframe/kernels/arithmetic.cpp"
#include <hwy/foreach_target.h>  // must precede highway.h
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace dftracer::utils::dataframe {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

#define DF_DEFINE_BINOP(NAME, HWYOP, SCALAROP)                             \
    template <class T>                                                     \
    void NAME(const void* av, const void* bv, void* ov, std::size_t n) {   \
        const T* a = static_cast<const T*>(av);                            \
        const T* b = static_cast<const T*>(bv);                            \
        T* out = static_cast<T*>(ov);                                      \
        const hn::ScalableTag<T> d;                                        \
        const std::size_t lanes = hn::Lanes(d);                            \
        std::size_t i = 0;                                                 \
        for (; i + lanes <= n; i += lanes)                                 \
            hn::StoreU(HWYOP(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d, \
                       out + i);                                           \
        for (; i < n; ++i) out[i] = SCALAROP;                              \
    }

DF_DEFINE_BINOP(AddImpl, hn::Add, a[i] + b[i])
DF_DEFINE_BINOP(SubImpl, hn::Sub, a[i] - b[i])
DF_DEFINE_BINOP(MulImpl, hn::Mul, a[i] * b[i])

#undef DF_DEFINE_BINOP

// Division: floats use the SIMD hn::Div; integers have no SIMD divide, so a
// scalar loop with a divide-by-zero guard (result 0). The DSL promotes to
// double before dividing, so the integer path is a defensive fallback.
template <class T>
void DivImpl(const void* av, const void* bv, void* ov, std::size_t n) {
    const T* a = static_cast<const T*>(av);
    const T* b = static_cast<const T*>(bv);
    T* out = static_cast<T*>(ov);
    if constexpr (std::is_floating_point_v<T>) {
        const hn::ScalableTag<T> d;
        const std::size_t lanes = hn::Lanes(d);
        std::size_t i = 0;
        for (; i + lanes <= n; i += lanes)
            hn::StoreU(hn::Div(hn::LoadU(d, a + i), hn::LoadU(d, b + i)), d,
                       out + i);
        for (; i < n; ++i) out[i] = a[i] / b[i];
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = b[i] != 0 ? static_cast<T>(a[i] / b[i]) : T{0};
    }
}

// Broadcast a scalar across the column. The scalar is converted to the column
// type T, so every numeric type is represented exactly (no double round-trip).
#define DF_DEFINE_SCALAR(NAME, HWYOP, SCALAROP)                          \
    template <class T>                                                   \
    void NAME(const void* av, dftu_scalar sc, void* ov, std::size_t n) { \
        const T* a = static_cast<const T*>(av);                          \
        T* out = static_cast<T*>(ov);                                    \
        const T s = scalar_as<T>(sc);                                    \
        const hn::ScalableTag<T> d;                                      \
        const auto vs = hn::Set(d, s);                                   \
        const std::size_t lanes = hn::Lanes(d);                          \
        std::size_t i = 0;                                               \
        for (; i + lanes <= n; i += lanes)                               \
            hn::StoreU(HWYOP(hn::LoadU(d, a + i), vs), d, out + i);      \
        for (; i < n; ++i) out[i] = SCALAROP;                            \
    }

DF_DEFINE_SCALAR(AddSImpl, hn::Add, a[i] + s)
DF_DEFINE_SCALAR(SubSImpl, hn::Sub, a[i] - s)
DF_DEFINE_SCALAR(MulSImpl, hn::Mul, a[i] * s)

#undef DF_DEFINE_SCALAR

// Broadcast division: floats SIMD, integers scalar with a zero guard.
template <class T>
void DivSImpl(const void* av, dftu_scalar sc, void* ov, std::size_t n) {
    const T* a = static_cast<const T*>(av);
    T* out = static_cast<T*>(ov);
    const T s = scalar_as<T>(sc);
    if constexpr (std::is_floating_point_v<T>) {
        const hn::ScalableTag<T> d;
        const auto vs = hn::Set(d, s);
        const std::size_t lanes = hn::Lanes(d);
        std::size_t i = 0;
        for (; i + lanes <= n; i += lanes)
            hn::StoreU(hn::Div(hn::LoadU(d, a + i), vs), d, out + i);
        for (; i < n; ++i) out[i] = a[i] / s;
    } else {
        for (std::size_t i = 0; i < n; ++i)
            out[i] = s != 0 ? static_cast<T>(a[i] / s) : T{0};
    }
}

// EXPERIMENT (op 2 measurement, see benchmarks/dataframe_parallel_bench.cpp):
// fan the add out across row ranges via parallel_for to measure whether a
// bandwidth-bound elementwise op is worth parallelizing on this machine.
constexpr std::size_t ADD_PARALLEL_GRAIN = 1 << 20;
template <class T>
void AddParallelImpl(const void* av, const void* bv, void* ov, std::size_t n) {
    if (!parallel_backend_installed() || n < ADD_PARALLEL_GRAIN) {
        AddImpl<T>(av, bv, ov, n);
        return;
    }
    const T* a = static_cast<const T*>(av);
    const T* b = static_cast<const T*>(bv);
    T* out = static_cast<T*>(ov);
    parallel_for(static_cast<std::int64_t>(n),
                 static_cast<std::int64_t>(ADD_PARALLEL_GRAIN),
                 [&](std::int64_t beg, std::int64_t end) {
                     AddImpl<T>(a + beg, b + beg, out + beg,
                                static_cast<std::size_t>(end - beg));
                 });
}

void AddKernel(std::int32_t type, const void* a, const void* b, void* out,
               std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), AddParallelImpl, a, b, out,
                        n)
}
void SubKernel(std::int32_t type, const void* a, const void* b, void* out,
               std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), SubImpl, a, b, out, n)
}
void MulKernel(std::int32_t type, const void* a, const void* b, void* out,
               std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), MulImpl, a, b, out, n)
}
void AddScalarKernel(std::int32_t type, const void* a, dftu_scalar s, void* out,
                     std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), AddSImpl, a, s, out, n)
}
void SubScalarKernel(std::int32_t type, const void* a, dftu_scalar s, void* out,
                     std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), SubSImpl, a, s, out, n)
}
void MulScalarKernel(std::int32_t type, const void* a, dftu_scalar s, void* out,
                     std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), MulSImpl, a, s, out, n)
}
void DivKernel(std::int32_t type, const void* a, const void* b, void* out,
               std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), DivImpl, a, b, out, n)
}
void DivScalarKernel(std::int32_t type, const void* a, dftu_scalar s, void* out,
                     std::size_t n) {
    DF_NUMERIC_DISPATCH(static_cast<TypeId>(type), DivSImpl, a, s, out, n)
}

}  // namespace HWY_NAMESPACE
}  // namespace dftracer::utils::dataframe
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace dftracer::utils::dataframe {

HWY_EXPORT(AddKernel);
HWY_EXPORT(SubKernel);
HWY_EXPORT(MulKernel);
HWY_EXPORT(AddScalarKernel);
HWY_EXPORT(SubScalarKernel);
HWY_EXPORT(MulScalarKernel);
HWY_EXPORT(DivKernel);
HWY_EXPORT(DivScalarKernel);

namespace {

enum class BinOp { Add, Sub, Mul, Div };

bool is_numeric(TypeId t) {
    return t != TypeId::Bool && t != TypeId::String && t != TypeId::Binary;
}

bool is_float_type(TypeId t) {
    return t == TypeId::Float32 || t == TypeId::Float64;
}
bool is_signed_int_type(TypeId t) {
    return t == TypeId::Int8 || t == TypeId::Int16 || t == TypeId::Int32 ||
           t == TypeId::Int64;
}
bool is_unsigned_int_type(TypeId t) {
    return t == TypeId::Uint8 || t == TypeId::Uint16 || t == TypeId::Uint32 ||
           t == TypeId::Uint64;
}

// numpy/pandas array-array result_type for a numeric binary op: widen within
// the same signedness, promote a float/int mix to Float64, and promote a
// signed/unsigned mix to Float64 unless the signed side is strictly wider (so
// it already covers the unsigned side's range).
TypeId promote_common(TypeId a, TypeId b) {
    if (a == b) return a;
    if (is_float_type(a) || is_float_type(b)) return TypeId::Float64;
    const std::size_t wa = byte_width(a);
    const std::size_t wb = byte_width(b);
    if (is_signed_int_type(a) && is_signed_int_type(b)) return wa >= wb ? a : b;
    if (is_unsigned_int_type(a) && is_unsigned_int_type(b))
        return wa >= wb ? a : b;
    TypeId signed_t = is_signed_int_type(a) ? a : b;
    TypeId unsigned_t = is_unsigned_int_type(a) ? a : b;
    return byte_width(signed_t) > byte_width(unsigned_t) ? signed_t
                                                         : TypeId::Float64;
}

// Weak-scalar promotion: a Python int scalar never forces a wider column
// dtype (numpy semantics), but a float scalar against an integer column does.
dftu_series* scalar_op(const dftu_series* a, dftu_scalar s, BinOp op) {
    if (a->encoding != Encoding::Flat || !is_numeric(a->type)) return nullptr;

    dftu_series* casted = nullptr;
    const dftu_series* src = a;
    if (s.kind == DFTU_SCALAR_TAG_F64 && !is_float_type(a->type)) {
        casted = dftu_series_cast(a, static_cast<dftu_dtype>(TypeId::Float64));
        if (!casted) return nullptr;
        src = casted;
    }

    auto* out = new dftu_series();
    out->type = src->type;
    out->encoding = Encoding::Flat;
    out->length = src->length;
    out->null_count = src->null_count;
    out->validity = src->validity;
    out->data = Buffer::allocate(static_cast<std::size_t>(src->length) *
                                 byte_width(src->type));

    std::int32_t t = static_cast<std::int32_t>(src->type);
    const void* pa = src->data->data();
    void* po = out->data->data();
    std::size_t n = static_cast<std::size_t>(src->length);
    switch (op) {
        case BinOp::Add:
            HWY_DYNAMIC_DISPATCH(AddScalarKernel)(t, pa, s, po, n);
            break;
        case BinOp::Sub:
            HWY_DYNAMIC_DISPATCH(SubScalarKernel)(t, pa, s, po, n);
            break;
        case BinOp::Mul:
            HWY_DYNAMIC_DISPATCH(MulScalarKernel)(t, pa, s, po, n);
            break;
        case BinOp::Div:
            HWY_DYNAMIC_DISPATCH(DivScalarKernel)(t, pa, s, po, n);
            break;
    }
    if (casted) dftu_series_free(casted);
    return out;
}

dftu_series* binop(const dftu_series* a, const dftu_series* b, BinOp op) {
    if (a->encoding != Encoding::Flat || b->encoding != Encoding::Flat)
        return nullptr;
    if (a->length != b->length) return nullptr;
    if (!is_numeric(a->type) || !is_numeric(b->type)) return nullptr;

    dftu_series* casted_a = nullptr;
    dftu_series* casted_b = nullptr;
    const dftu_series* pa_src = a;
    const dftu_series* pb_src = b;
    if (a->type != b->type) {
        TypeId common = promote_common(a->type, b->type);
        if (a->type != common) {
            casted_a = dftu_series_cast(a, static_cast<dftu_dtype>(common));
            if (!casted_a) return nullptr;
            pa_src = casted_a;
        }
        if (b->type != common) {
            casted_b = dftu_series_cast(b, static_cast<dftu_dtype>(common));
            if (!casted_b) {
                if (casted_a) dftu_series_free(casted_a);
                return nullptr;
            }
            pb_src = casted_b;
        }
    }

    auto* out = new dftu_series();
    out->type = pa_src->type;
    out->encoding = Encoding::Flat;
    out->length = pa_src->length;
    out->data = Buffer::allocate(static_cast<std::size_t>(pa_src->length) *
                                 byte_width(pa_src->type));

    std::int32_t t = static_cast<std::int32_t>(pa_src->type);
    const void* pa = pa_src->data->data();
    const void* pb = pb_src->data->data();
    void* po = out->data->data();
    std::size_t n = static_cast<std::size_t>(pa_src->length);
    switch (op) {
        case BinOp::Add:
            HWY_DYNAMIC_DISPATCH(AddKernel)(t, pa, pb, po, n);
            break;
        case BinOp::Sub:
            HWY_DYNAMIC_DISPATCH(SubKernel)(t, pa, pb, po, n);
            break;
        case BinOp::Mul:
            HWY_DYNAMIC_DISPATCH(MulKernel)(t, pa, pb, po, n);
            break;
        case BinOp::Div:
            HWY_DYNAMIC_DISPATCH(DivKernel)(t, pa, pb, po, n);
            break;
    }
    if (casted_a) dftu_series_free(casted_a);
    if (casted_b) dftu_series_free(casted_b);
    return out;
}

}  // namespace

Series add(const Series& a, const Series& b) {
    return Series{dftu_series_add(a.handle(), b.handle())};
}
Series sub(const Series& a, const Series& b) {
    return Series{dftu_series_sub(a.handle(), b.handle())};
}
Series mul(const Series& a, const Series& b) {
    return Series{dftu_series_mul(a.handle(), b.handle())};
}
Series div(const Series& a, const Series& b) {
    return Series{dftu_series_div(a.handle(), b.handle())};
}
dftu_series* add_columns(const dftu_series* a, const dftu_series* b) {
    return binop(a, b, BinOp::Add);
}
dftu_series* sub_columns(const dftu_series* a, const dftu_series* b) {
    return binop(a, b, BinOp::Sub);
}
dftu_series* mul_columns(const dftu_series* a, const dftu_series* b) {
    return binop(a, b, BinOp::Mul);
}
dftu_series* div_columns(const dftu_series* a, const dftu_series* b) {
    return binop(a, b, BinOp::Div);
}
dftu_series* add_scalar_col(const dftu_series* a, dftu_scalar s) {
    return scalar_op(a, s, BinOp::Add);
}
dftu_series* sub_scalar_col(const dftu_series* a, dftu_scalar s) {
    return scalar_op(a, s, BinOp::Sub);
}
dftu_series* mul_scalar_col(const dftu_series* a, dftu_scalar s) {
    return scalar_op(a, s, BinOp::Mul);
}
dftu_series* div_scalar_col(const dftu_series* a, dftu_scalar s) {
    return scalar_op(a, s, BinOp::Div);
}

}  // namespace dftracer::utils::dataframe

dftu_series* dftu_series_add(const dftu_series* a, const dftu_series* b) {
    return dftracer::utils::dataframe::add_columns(a, b);
}
dftu_series* dftu_series_sub(const dftu_series* a, const dftu_series* b) {
    return dftracer::utils::dataframe::sub_columns(a, b);
}
dftu_series* dftu_series_mul(const dftu_series* a, const dftu_series* b) {
    return dftracer::utils::dataframe::mul_columns(a, b);
}
dftu_series* dftu_series_div(const dftu_series* a, const dftu_series* b) {
    return dftracer::utils::dataframe::div_columns(a, b);
}
dftu_series* dftu_series_add_scalar(const dftu_series* a, dftu_scalar s) {
    return dftracer::utils::dataframe::add_scalar_col(a, s);
}
dftu_series* dftu_series_sub_scalar(const dftu_series* a, dftu_scalar s) {
    return dftracer::utils::dataframe::sub_scalar_col(a, s);
}
dftu_series* dftu_series_mul_scalar(const dftu_series* a, dftu_scalar s) {
    return dftracer::utils::dataframe::mul_scalar_col(a, s);
}
dftu_series* dftu_series_div_scalar(const dftu_series* a, dftu_scalar s) {
    return dftracer::utils::dataframe::div_scalar_col(a, s);
}

#endif  // HWY_ONCE
