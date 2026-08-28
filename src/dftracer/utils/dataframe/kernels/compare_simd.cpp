#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/compare_simd.h>
#include <dftracer/utils/dataframe/internal/scalar.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dftracer/utils/dataframe/kernels/compare_simd.cpp"
#include <hwy/foreach_target.h>  // must precede highway.h
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace dftracer::utils::dataframe {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// `op` is validated at the ABI boundary (dftu_series_compare) before dispatch;
// the trailing return is unreachable and satisfies -Wreturn-type.
template <class D, class V>
auto compare_mask(D, V v, V vr, std::int32_t op) {
    switch (static_cast<CmpOp>(op)) {
        case CmpOp::Gt:
            return hn::Gt(v, vr);
        case CmpOp::Ge:
            return hn::Ge(v, vr);
        case CmpOp::Lt:
            return hn::Lt(v, vr);
        case CmpOp::Le:
            return hn::Le(v, vr);
        case CmpOp::Eq:
            return hn::Eq(v, vr);
        case CmpOp::Ne:
            return hn::Ne(v, vr);
    }
    return hn::Ne(v, vr);
}

template <class T>
bool compare_scalar(T a, T b, std::int32_t op) {
    switch (static_cast<CmpOp>(op)) {
        case CmpOp::Gt:
            return a > b;
        case CmpOp::Ge:
            return a >= b;
        case CmpOp::Lt:
            return a < b;
        case CmpOp::Le:
            return a <= b;
        case CmpOp::Eq:
            return a == b;
        case CmpOp::Ne:
            return a != b;
    }
    return a != b;
}

// Vectorized compare that packs the boolean result into `out`. Bits are built
// 64 at a time so every block store lands on a byte boundary regardless of the
// lane count; the shorter tail is packed scalar. Lane j of block i sets bit
// (i + j), matching the little-endian byte layout of the accumulator.
template <class T>
void compare_bits(const T* p, std::int64_t n, T r, std::int32_t op,
                  std::uint8_t* out) {
    const hn::ScalableTag<T> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto vr = hn::Set(d, r);
    const std::uint64_t lane_mask =
        lanes >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << lanes) - 1);
    std::int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        std::uint64_t bits = 0;
        for (std::size_t c = 0; c < 64; c += lanes) {
            const auto v = hn::LoadU(d, p + i + static_cast<std::int64_t>(c));
            const auto m = compare_mask(d, v, vr, op);
            std::uint64_t cb = 0;
            hn::StoreMaskBits(d, m, reinterpret_cast<std::uint8_t*>(&cb));
            bits |= (cb & lane_mask) << c;
        }
        std::memcpy(out + (i >> 3), &bits, 8);
    }
    for (; i < n; ++i)
        if (compare_scalar(p[i], r, op))
            out[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
}

void CompareI64(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<std::int64_t>(static_cast<const std::int64_t*>(p), n,
                               scalar_as<std::int64_t>(r), op, out);
}
void CompareU64(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<std::uint64_t>(static_cast<const std::uint64_t*>(p), n,
                                scalar_as<std::uint64_t>(r), op, out);
}
void CompareF64(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<double>(static_cast<const double*>(p), n, scalar_as<double>(r),
                         op, out);
}
void CompareI32(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<std::int32_t>(static_cast<const std::int32_t*>(p), n,
                               scalar_as<std::int32_t>(r), op, out);
}
void CompareU32(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<std::uint32_t>(static_cast<const std::uint32_t*>(p), n,
                                scalar_as<std::uint32_t>(r), op, out);
}
void CompareF32(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<float>(static_cast<const float*>(p), n, scalar_as<float>(r),
                        op, out);
}
void CompareI16(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<std::int16_t>(static_cast<const std::int16_t*>(p), n,
                               scalar_as<std::int16_t>(r), op, out);
}
void CompareU16(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
                std::uint8_t* out) {
    compare_bits<std::uint16_t>(static_cast<const std::uint16_t*>(p), n,
                                scalar_as<std::uint16_t>(r), op, out);
}
void CompareI8(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
               std::uint8_t* out) {
    compare_bits<std::int8_t>(static_cast<const std::int8_t*>(p), n,
                              scalar_as<std::int8_t>(r), op, out);
}

// Pack per-row byte flags (nonzero = set) into a bit-packed bitmap, 64 bits at
// a time so each block store is byte-aligned; the tail is packed scalar.
void PackFlags(const char* flags, std::int64_t n, std::uint8_t* out) {
    const std::uint8_t* f = reinterpret_cast<const std::uint8_t*>(flags);
    const hn::ScalableTag<std::uint8_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto zero = hn::Zero(d);
    const std::uint64_t lane_mask =
        lanes >= 64 ? ~std::uint64_t{0} : ((std::uint64_t{1} << lanes) - 1);
    std::int64_t i = 0;
    for (; i + 64 <= n; i += 64) {
        std::uint64_t bits = 0;
        for (std::size_t c = 0; c < 64; c += lanes) {
            const auto m = hn::Ne(
                hn::LoadU(d, f + i + static_cast<std::int64_t>(c)), zero);
            std::uint64_t cb = 0;
            hn::StoreMaskBits(d, m, reinterpret_cast<std::uint8_t*>(&cb));
            bits |= (cb & lane_mask) << c;
        }
        std::memcpy(out + (i >> 3), &bits, 8);
    }
    for (; i < n; ++i)
        if (f[i]) out[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
}
void CompareU8(const void* p, std::int64_t n, dftu_scalar r, std::int32_t op,
               std::uint8_t* out) {
    compare_bits<std::uint8_t>(static_cast<const std::uint8_t*>(p), n,
                               scalar_as<std::uint8_t>(r), op, out);
}

}  // namespace HWY_NAMESPACE
}  // namespace dftracer::utils::dataframe
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace dftracer::utils::dataframe {

HWY_EXPORT(CompareI64);
HWY_EXPORT(CompareU64);
HWY_EXPORT(CompareF64);
HWY_EXPORT(CompareI32);
HWY_EXPORT(CompareU32);
HWY_EXPORT(CompareF32);
HWY_EXPORT(CompareI16);
HWY_EXPORT(CompareU16);
HWY_EXPORT(CompareI8);
HWY_EXPORT(CompareU8);
HWY_EXPORT(PackFlags);

void pack_flags(const char* flags, std::int64_t n, std::uint8_t* out) {
    if (n > 0) HWY_DYNAMIC_DISPATCH(PackFlags)(flags, n, out);
}

bool compare(const dftu_series& v, std::int32_t op, dftu_scalar rhs,
             std::uint8_t* out) {
    const void* p = v.data->data();
    std::int64_t n = v.length;
    switch (v.type) {
        case TypeId::Int64:
            HWY_DYNAMIC_DISPATCH(CompareI64)(p, n, rhs, op, out);
            return true;
        case TypeId::Uint64:
            HWY_DYNAMIC_DISPATCH(CompareU64)(p, n, rhs, op, out);
            return true;
        case TypeId::Float64:
            HWY_DYNAMIC_DISPATCH(CompareF64)(p, n, rhs, op, out);
            return true;
        case TypeId::Int32:
            HWY_DYNAMIC_DISPATCH(CompareI32)(p, n, rhs, op, out);
            return true;
        case TypeId::Uint32:
            HWY_DYNAMIC_DISPATCH(CompareU32)(p, n, rhs, op, out);
            return true;
        case TypeId::Float32:
            HWY_DYNAMIC_DISPATCH(CompareF32)(p, n, rhs, op, out);
            return true;
        case TypeId::Int16:
            HWY_DYNAMIC_DISPATCH(CompareI16)(p, n, rhs, op, out);
            return true;
        case TypeId::Uint16:
            HWY_DYNAMIC_DISPATCH(CompareU16)(p, n, rhs, op, out);
            return true;
        case TypeId::Int8:
            HWY_DYNAMIC_DISPATCH(CompareI8)(p, n, rhs, op, out);
            return true;
        case TypeId::Uint8:
            HWY_DYNAMIC_DISPATCH(CompareU8)(p, n, rhs, op, out);
            return true;
        default:
            return false;  // non-numeric: scalar path
    }
}

}  // namespace dftracer::utils::dataframe
#endif  // HWY_ONCE
