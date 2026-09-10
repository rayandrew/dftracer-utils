#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/decimal.h>
#include <dftracer/utils/dataframe/internal/float16.h>
#include <dftracer/utils/dataframe/internal/numeric_dispatch.h>
#include <dftracer/utils/dataframe/kernels/cast.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dftracer::utils::dataframe {
namespace {

template <class S>
void cast_from(const void* sv, std::int32_t dtype, void* dv, std::size_t n) {
    const S* s = static_cast<const S*>(sv);
#define DF_CAST_TO(D)                                                    \
    do {                                                                 \
        D* d = static_cast<D*>(dv);                                      \
        for (std::size_t i = 0; i < n; ++i) d[i] = static_cast<D>(s[i]); \
    } while (0)
    switch (static_cast<TypeId>(dtype)) {
        case TypeId::Int8:
            DF_CAST_TO(std::int8_t);
            break;
        case TypeId::Int16:
            DF_CAST_TO(std::int16_t);
            break;
        case TypeId::Int32:
            DF_CAST_TO(std::int32_t);
            break;
        case TypeId::Int64:
            DF_CAST_TO(std::int64_t);
            break;
        case TypeId::Uint8:
            DF_CAST_TO(std::uint8_t);
            break;
        case TypeId::Uint16:
            DF_CAST_TO(std::uint16_t);
            break;
        case TypeId::Uint32:
            DF_CAST_TO(std::uint32_t);
            break;
        case TypeId::Uint64:
            DF_CAST_TO(std::uint64_t);
            break;
        case TypeId::Float32:
            DF_CAST_TO(float);
            break;
        case TypeId::Float64:
            DF_CAST_TO(double);
            break;
        default:
            break;
    }
#undef DF_CAST_TO
}

// Float16 is physically uint16 bits, not an integer: it needs to be decoded
// (never reinterpreted) before any numeric op sees it. Decoding is exact, so
// this is a lossless stand-in source for the generic cast_from<float> path.
bool cast_from_float16(const void* sv, std::int32_t dtype, void* dv,
                       std::size_t n) {
    const auto* h = static_cast<const std::uint16_t*>(sv);
    std::vector<float> decoded(n);
    for (std::size_t i = 0; i < n; ++i) decoded[i] = half_to_float(h[i]);
    if (static_cast<TypeId>(dtype) == TypeId::Float16) {
        auto* d = static_cast<std::uint16_t*>(dv);
        for (std::size_t i = 0; i < n; ++i) d[i] = h[i];
        return true;
    }
    cast_from<float>(decoded.data(), dtype, dv, n);
    return true;
}

// Decimal128/256 are opaque fixed-width bytes to every numeric kernel; the
// only cast this engine defines for them is decoding to Float64 (lossy - see
// is_arithmetic_type's documentation). Casting INTO a decimal type would need
// a rounding policy this engine does not define yet, so that direction is
// refused (dftu_series_cast returns null).
bool cast_from_decimal(TypeId src, const void* sv, std::int32_t width,
                       std::int32_t scale, std::int32_t dtype, void* dv,
                       std::size_t n) {
    if (static_cast<TypeId>(dtype) != TypeId::Float64) return false;
    const auto* bytes = static_cast<const std::uint8_t*>(sv);
    auto* d = static_cast<double*>(dv);
    for (std::size_t i = 0; i < n; ++i) {
        const void* row = bytes + i * static_cast<std::size_t>(width);
        d[i] = src == TypeId::Decimal128 ? decimal128_to_double(row, scale)
                                         : decimal256_to_double(row, scale);
    }
    return true;
}

// Encodes an already-computed Float32 buffer down to Float16 (round-to-
// nearest-even); used when the cast TARGET is Float16.
void encode_to_float16(const float* src, std::uint16_t* dst, std::size_t n) {
    for (std::size_t i = 0; i < n; ++i) dst[i] = float_to_half(src[i]);
}

bool is_numeric(TypeId t) { return is_arithmetic_type(t); }

}  // namespace

Series cast(const Series& v, TypeId target) {
    return Series{
        dftu_series_cast(v.handle(), static_cast<dftu_dtype>(target))};
}

}  // namespace dftracer::utils::dataframe

dftu_series* dftu_series_cast(const dftu_series* v, dftu_dtype target) {
    using dftracer::utils::dataframe::Buffer;
    using dftracer::utils::dataframe::buffer_bytes;
    using dftracer::utils::dataframe::byte_width;
    using dftracer::utils::dataframe::Encoding;
    using dftracer::utils::dataframe::TypeId;
    TypeId src = v->type;
    TypeId dst = static_cast<TypeId>(target);
    if (v->encoding != Encoding::Flat) return nullptr;
    if (!dftracer::utils::dataframe::is_numeric(src) ||
        !dftracer::utils::dataframe::is_numeric(dst))
        return nullptr;
    if ((src == TypeId::Decimal128 || src == TypeId::Decimal256) &&
        dst != TypeId::Float64)
        return nullptr;
    if (dst == TypeId::Decimal128 || dst == TypeId::Decimal256) return nullptr;

    auto* out = new dftu_series();
    out->type = dst;
    out->encoding = Encoding::Flat;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;
    out->data = Buffer::allocate(buffer_bytes(dst, v->length));

    const void* sv = v->data->data();
    void* dv = out->data->data();
    std::size_t n = static_cast<std::size_t>(v->length);

    if (src == TypeId::Float16) {
        dftracer::utils::dataframe::cast_from_float16(
            sv, static_cast<std::int32_t>(dst), dv, n);
    } else if (src == TypeId::Decimal128 || src == TypeId::Decimal256) {
        dftracer::utils::dataframe::cast_from_decimal(
            src, sv, static_cast<std::int32_t>(byte_width(src)),
            v->decimal_scale, static_cast<std::int32_t>(dst), dv, n);
    } else if (dst == TypeId::Float16) {
        auto tmp = std::vector<float>(n);
        DF_NUMERIC_DISPATCH(src, dftracer::utils::dataframe::cast_from, sv,
                            static_cast<std::int32_t>(TypeId::Float32),
                            tmp.data(), n)
        dftracer::utils::dataframe::encode_to_float16(
            tmp.data(), static_cast<std::uint16_t*>(dv), n);
    } else if (!dftracer::utils::dataframe::cast_simd(
                   static_cast<std::int32_t>(src),
                   static_cast<std::int32_t>(dst), sv, dv, n)) {
        DF_NUMERIC_DISPATCH(src, dftracer::utils::dataframe::cast_from, sv,
                            target, dv, n)
    }
    return out;
}
