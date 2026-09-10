#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/compare_simd.h>
#include <dftracer/utils/dataframe/internal/numeric_dispatch.h>
#include <dftracer/utils/dataframe/internal/scalar.h>
#include <dftracer/utils/dataframe/kernels/comparison.h>

#include <cstring>

namespace dftracer::utils::dataframe {
namespace {

// Compare in the column's own type domain (no double round-trip), so 64-bit
// integers compare exactly. The scalar is converted to the column type T.
// `op` is validated at the ABI boundary (dftu_series_compare) before dispatch.
template <class T>
void compare_impl(const void* data, std::int64_t length, std::int32_t op,
                  dftu_scalar rhs, std::uint8_t* out) {
    const T* p = static_cast<const T*>(data);
    const T r = scalar_as<T>(rhs);
    for (std::int64_t i = 0; i < length; ++i) {
        bool res = false;
        switch (static_cast<CmpOp>(op)) {
            case CmpOp::Gt:
                res = p[i] > r;
                break;
            case CmpOp::Ge:
                res = p[i] >= r;
                break;
            case CmpOp::Lt:
                res = p[i] < r;
                break;
            case CmpOp::Le:
                res = p[i] <= r;
                break;
            case CmpOp::Eq:
                res = p[i] == r;
                break;
            case CmpOp::Ne:
                res = p[i] != r;
                break;
        }
        if (res) out[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
    }
}

}  // namespace
}  // namespace dftracer::utils::dataframe

dftu_series* dftu_series_compare(const dftu_series* v, dftu_cmp_op op,
                                 dftu_scalar rhs) {
    using dftracer::utils::dataframe::Buffer;
    using dftracer::utils::dataframe::buffer_bytes;
    using dftracer::utils::dataframe::CmpOp;
    using dftracer::utils::dataframe::compare_impl;
    using dftracer::utils::dataframe::Encoding;
    using dftracer::utils::dataframe::TypeId;
    // A STR rhs is a string comparison, which is a different kernel: equality
    // only, and dictionary-aware (dftu_series_str_eq tests each dictionary
    // entry once and then compares codes, rather than resolving per row).
    if (rhs.kind == DFTU_SCALAR_TAG_STR) {
        if (op != static_cast<int32_t>(CmpOp::Eq) &&
            op != static_cast<int32_t>(CmpOp::Ne))
            return nullptr;
        dftu_series* eq =
            dftu_series_str_eq(v, rhs.value.s, static_cast<int32_t>(rhs.len));
        if (eq == nullptr || op == static_cast<int32_t>(CmpOp::Eq)) return eq;
        dftu_series* ne = dftu_series_logical_not(eq);
        dftu_series_free(eq);
        return ne;
    }

    if (v->encoding != Encoding::Flat) return nullptr;
    const TypeId phys = physical_type(v->type);
    if (!is_numeric_dispatchable(phys)) return nullptr;
    if (op < static_cast<int32_t>(CmpOp::Gt) ||
        op > static_cast<int32_t>(CmpOp::Ne))
        return nullptr;

    auto* out = new dftu_series();
    out->type = TypeId::Bool;
    out->encoding = Encoding::Flat;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;  // shared: nulls propagate

    std::size_t bytes = buffer_bytes(TypeId::Bool, v->length);
    out->data = Buffer::allocate(bytes);
    std::memset(out->data->data(), 0, bytes);
    std::uint8_t* bits = out->data->data();
    const void* data = v->data->data();
    std::int64_t n = v->length;

    if (!dftracer::utils::dataframe::compare(*v, op, rhs, bits)) {
        DF_NUMERIC_DISPATCH(phys, compare_impl, data, n, op, rhs, bits)
    }
    return out;
}
