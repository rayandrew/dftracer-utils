#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/numeric_dispatch.h>
#include <dftracer/utils/dataframe/kernels/cast.h>

#include <cstddef>
#include <cstdint>

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

bool is_numeric(TypeId t) {
    return t != TypeId::Bool && t != TypeId::String && t != TypeId::Binary;
}

}  // namespace

Series cast(const Series& v, TypeId target) {
    return Series{
        dftu_series_cast(v.handle(), static_cast<dftu_dtype>(target))};
}

}  // namespace dftracer::utils::dataframe

dftu_series* dftu_series_cast(const dftu_series* v, dftu_dtype target) {
    using dftracer::utils::dataframe::Buffer;
    using dftracer::utils::dataframe::buffer_bytes;
    using dftracer::utils::dataframe::Encoding;
    using dftracer::utils::dataframe::TypeId;
    TypeId src = v->type;
    TypeId dst = static_cast<TypeId>(target);
    if (v->encoding != Encoding::Flat) return nullptr;
    if (!dftracer::utils::dataframe::is_numeric(src) ||
        !dftracer::utils::dataframe::is_numeric(dst))
        return nullptr;

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
    if (!dftracer::utils::dataframe::cast_simd(static_cast<std::int32_t>(src),
                                               static_cast<std::int32_t>(dst),
                                               sv, dv, n))
        DF_NUMERIC_DISPATCH(src, dftracer::utils::dataframe::cast_from, sv,
                            target, dv, n)
    return out;
}
