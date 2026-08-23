#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/filter_simd.h>
#include <dftracer/utils/dataframe/internal/numeric_dispatch.h>
#include <dftracer/utils/dataframe/internal/scalar.h>
#include <dftracer/utils/dataframe/kernels/filter.h>

#include <cstring>
#include <string>
#include <vector>

using dftracer::utils::dataframe::Buffer;
using dftracer::utils::dataframe::Encoding;
using dftracer::utils::dataframe::scalar_as;
using dftracer::utils::dataframe::Series;
using dftracer::utils::dataframe::TypeId;

namespace {

bool is_valid(const dftu_series& v, std::int64_t i) {
    if (!v.validity) return true;
    const std::uint8_t* bm = v.validity->data();
    return (bm[i >> 3] & (1u << (i & 7))) != 0;
}

// Compare in the column's own type domain; the scalar is converted to the
// column type T, so 64-bit integers are exact.
template <class T>
void select_gt(const dftu_series& v, dftu_scalar threshold,
               std::vector<std::int64_t>& sel) {
    const T* p = reinterpret_cast<const T*>(v.data->data());
    const T t = scalar_as<T>(threshold);
    for (std::int64_t i = 0; i < v.length; ++i) {
        if (is_valid(v, i) && p[i] > t) sel.push_back(i);
    }
}

// Build a SELECTION column over `base` from row indices `sel`.
dftu_series* make_selection(const dftu_series& base,
                            const std::vector<std::int64_t>& sel) {
    auto* out = new dftu_series();
    out->type = base.type;
    out->encoding = Encoding::Selection;
    out->length = static_cast<std::int64_t>(sel.size());
    out->data = Buffer::allocate(sel.size() * sizeof(std::int64_t));
    if (!sel.empty())
        std::memcpy(out->data->data(), sel.data(),
                    sel.size() * sizeof(std::int64_t));
    out->child = std::make_shared<dftu_series>(base);
    return out;
}

}  // namespace

dftu_series* dftu_series_filter_gt(const dftu_series* v,
                                   dftu_scalar threshold) {
    if (v->encoding != Encoding::Flat) return nullptr;
    if (v->type == TypeId::Bool || v->type == TypeId::String ||
        v->type == TypeId::Binary)
        return nullptr;

    std::vector<std::int64_t> sel;
    if (!v->validity) {
        sel.resize(static_cast<std::size_t>(v->length));
        std::int64_t k =
            dftracer::utils::dataframe::compact_gt(*v, threshold, sel.data());
        if (k >= 0) {
            sel.resize(static_cast<std::size_t>(k));
            return make_selection(*v, sel);
        }
        sel.clear();
    }
    DF_NUMERIC_DISPATCH(v->type, select_gt, *v, threshold, sel)
    return make_selection(*v, sel);
}

dftu_series* dftu_series_filter(const dftu_series* v, const dftu_series* mask) {
    if (mask->type != TypeId::Bool || mask->encoding != Encoding::Flat)
        return nullptr;
    if (mask->length != v->length) return nullptr;

    const std::uint8_t* bits = mask->data->data();
    std::vector<std::int64_t> sel;
    for (std::int64_t i = 0; i < v->length; ++i)
        if ((bits[i >> 3] >> (i & 7)) & 1) sel.push_back(i);
    return make_selection(*v, sel);
}

using dftracer::utils::dataframe::buffer_bytes;

// Gather `base`'s validity bitmap by `idx`; sets `null_count`. A negative index
// gathers a null (the join OUTER-fill sentinel), so validity is materialized
// when the base has nulls OR any index is negative. Null otherwise.
static std::shared_ptr<Buffer> gather_validity(const dftu_series& base,
                                               const std::int64_t* idx,
                                               std::int64_t n,
                                               std::int64_t& null_count) {
    null_count = 0;
    bool has_null_idx = false;
    for (std::int64_t i = 0; i < n; ++i)
        if (idx[i] < 0) {
            has_null_idx = true;
            break;
        }
    if (!base.validity && !has_null_idx) return nullptr;
    const std::uint8_t* bm = base.validity ? base.validity->data() : nullptr;
    auto out = Buffer::allocate(buffer_bytes(TypeId::Bool, n));
    std::memset(out->data(), 0, out->size());
    std::uint8_t* ob = out->data();
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t k = idx[i];
        const bool valid = k >= 0 && (!bm || ((bm[k >> 3] >> (k & 7)) & 1));
        if (valid)
            ob[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
        else
            ++null_count;
    }
    return out;
}

static dftu_series* gather_column(const dftu_series& base,
                                  const std::int64_t* idx, std::int64_t n);

// String/Binary: concatenate the selected slices, rebuild offsets.
static dftu_series* gather_varwidth(const dftu_series& base,
                                    const std::int64_t* idx, std::int64_t n) {
    const std::int32_t* boff =
        reinterpret_cast<const std::int32_t*>(base.offsets->data());
    const char* bdata = reinterpret_cast<const char*>(base.data->data());
    std::vector<std::int32_t> offsets{0};
    offsets.reserve(static_cast<std::size_t>(n) + 1);
    std::string out;
    for (std::int64_t i = 0; i < n; ++i) {
        std::int64_t k = idx[i];
        if (k >= 0)  // negative = null: an empty slice
            out.append(bdata + boff[k],
                       static_cast<std::size_t>(boff[k + 1] - boff[k]));
        offsets.push_back(static_cast<std::int32_t>(out.size()));
    }
    std::int64_t nulls = 0;
    auto validity = gather_validity(base, idx, n, nulls);
    return dftu_series_new_string(static_cast<dftu_dtype>(base.type),
                                  offsets.data(), out.data(), n,
                                  validity ? validity->data() : nullptr);
}

// Struct: gather every field column by the same row indices.
static dftu_series* gather_struct(const dftu_series& base,
                                  const std::int64_t* idx, std::int64_t n) {
    auto* out = new dftu_series();
    out->type = TypeId::Struct;
    out->encoding = Encoding::Flat;
    out->length = n;
    out->field_names = base.field_names;
    out->children.reserve(base.children.size());
    for (const auto& ch : base.children)
        out->children.push_back(
            std::shared_ptr<dftu_series>(gather_column(*ch, idx, n)));
    out->validity = gather_validity(base, idx, n, out->null_count);
    return out;
}

// List: expand the selected rows to child-row indices (each row's sublist),
// gather the child by them, and rebuild per-row offsets.
static dftu_series* gather_list(const dftu_series& base,
                                const std::int64_t* idx, std::int64_t n) {
    const std::int32_t* boff =
        reinterpret_cast<const std::int32_t*>(base.offsets->data());
    std::vector<std::int64_t> child_idx;
    std::vector<std::int32_t> offsets{0};
    offsets.reserve(static_cast<std::size_t>(n) + 1);
    for (std::int64_t i = 0; i < n; ++i) {
        const std::int64_t r = idx[i];
        if (r >= 0)  // negative = null: an empty sublist
            for (std::int32_t j = boff[r]; j < boff[r + 1]; ++j)
                child_idx.push_back(j);
        offsets.push_back(static_cast<std::int32_t>(child_idx.size()));
    }
    auto* out = new dftu_series();
    out->type = TypeId::List;
    out->encoding = Encoding::Flat;
    out->length = n;
    out->offsets = Buffer::allocate((static_cast<std::size_t>(n) + 1) *
                                    sizeof(std::int32_t));
    std::memcpy(out->offsets->data(), offsets.data(),
                offsets.size() * sizeof(std::int32_t));
    out->child = std::shared_ptr<dftu_series>(
        gather_column(*base.child, child_idx.data(),
                      static_cast<std::int64_t>(child_idx.size())));
    out->validity = gather_validity(base, idx, n, out->null_count);
    return out;
}

// Gather `n` rows of a FLAT `base` at row indices `idx`, recursively for nested
// types. Propagates validity. Returns null for a non-gatherable base (Bool).
static dftu_series* gather_column(const dftu_series& base,
                                  const std::int64_t* idx, std::int64_t n) {
    if (base.type == TypeId::String || base.type == TypeId::Binary)
        return gather_varwidth(base, idx, n);
    if (base.type == TypeId::Struct) return gather_struct(base, idx, n);
    if (base.type == TypeId::List) return gather_list(base, idx, n);

    const std::size_t width = byte_width(base.type);
    if (width == 0) return nullptr;  // Bool (bit-packed) not gatherable here

    auto* out = new dftu_series();
    out->type = base.type;
    out->encoding = Encoding::Flat;
    out->length = n;
    out->data = Buffer::allocate(static_cast<std::size_t>(n) * width);
    const std::uint8_t* src = base.data->data();
    std::uint8_t* dst = out->data->data();
    for (std::int64_t i = 0; i < n; ++i) {
        if (idx[i] < 0)  // negative = null: zero the (masked) cell
            std::memset(dst + static_cast<std::size_t>(i) * width, 0, width);
        else
            std::memcpy(dst + static_cast<std::size_t>(i) * width,
                        src + static_cast<std::size_t>(idx[i]) * width, width);
    }
    out->validity = gather_validity(base, idx, n, out->null_count);
    return out;
}

dftu_series* dftu_series_materialize(const dftu_series* v) {
    if (v->encoding == Encoding::Flat) return new dftu_series(*v);
    if ((v->encoding != Encoding::Selection &&
         v->encoding != Encoding::Dictionary) ||
        !v->child)
        return nullptr;

    const dftu_series& base = *v->child;
    if (base.encoding != Encoding::Flat) return nullptr;

    if (v->encoding == Encoding::Selection) {
        const std::int64_t* idx =
            reinterpret_cast<const std::int64_t*>(v->data->data());
        return gather_column(base, idx, v->length);
    }
    // Dictionary codes are int32 (they index a small distinct-value set); widen
    // to int64 for the shared gather path.
    const std::int32_t* codes =
        reinterpret_cast<const std::int32_t*>(v->data->data());
    std::vector<std::int64_t> idx(static_cast<std::size_t>(v->length));
    for (std::int64_t i = 0; i < v->length; ++i) idx[i] = codes[i];
    return gather_column(base, idx.data(), v->length);
}

dftu_series* dftu_series_take(const dftu_series* v, const std::int64_t* idx,
                              std::int64_t n) {
    if (v->encoding == Encoding::Flat) return gather_column(*v, idx, n);
    // Resolve indirection to FLAT first, then gather.
    dftu_series* flat = dftu_series_materialize(v);
    if (!flat) return nullptr;
    dftu_series* out = gather_column(*flat, idx, n);
    dftu_series_free(flat);
    return out;
}

namespace dftracer::utils::dataframe {

Series filter(const Series& v, const Series& mask) {
    return Series{dftu_series_filter(v.handle(), mask.handle())};
}

Series materialize(const Series& v) {
    return Series{dftu_series_materialize(v.handle())};
}

Series take(const Series& v, const std::vector<std::int64_t>& indices) {
    return Series{dftu_series_take(v.handle(), indices.data(),
                                   static_cast<std::int64_t>(indices.size()))};
}

}  // namespace dftracer::utils::dataframe
