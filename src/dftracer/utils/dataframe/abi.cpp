#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>

#include <cstring>

using dftracer::utils::dataframe::Buffer;
using dftracer::utils::dataframe::buffer_bytes;
using dftracer::utils::dataframe::byte_width;
using dftracer::utils::dataframe::Encoding;
using dftracer::utils::dataframe::TypeId;

namespace {

std::int64_t validity_bytes(std::int64_t n) { return (n + 7) / 8; }

std::int64_t count_nulls(const std::uint8_t* validity, std::int64_t n) {
    if (validity == nullptr) return 0;
    std::int64_t nulls = 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if ((validity[i >> 3] & (1u << (i & 7))) == 0) ++nulls;
    }
    return nulls;
}

}  // namespace

dftu_series* dftu_series_new_flat(dftu_dtype type, const void* data, int64_t n,
                                  const uint8_t* validity) {
    TypeId t = static_cast<TypeId>(type);
    // Fixed-width only; String/Binary are offset+data and need their own
    // builder.
    if (byte_width(t) == 0) return nullptr;

    auto* col = new dftu_series();
    col->type = t;
    col->encoding = Encoding::Flat;
    col->length = n;

    std::size_t bytes = buffer_bytes(t, n);
    col->data = Buffer::allocate(bytes);
    if (bytes != 0 && data != nullptr) {
        std::memcpy(col->data->data(), data, bytes);
    }

    if (validity != nullptr) {
        std::size_t vbytes = static_cast<std::size_t>(validity_bytes(n));
        col->validity = Buffer::allocate(vbytes);
        std::memcpy(col->validity->data(), validity, vbytes);
        col->null_count = count_nulls(validity, n);
    }
    return col;
}

dftu_series* dftu_series_new_flat_borrowed(dftu_dtype type, const void* data,
                                           int64_t n, const uint8_t* validity,
                                           void (*release)(void* ctx),
                                           void* release_ctx) {
    TypeId t = static_cast<TypeId>(type);
    if (byte_width(t) == 0) {
        if (release != nullptr) release(release_ctx);
        return nullptr;
    }

    auto* col = new dftu_series();
    col->type = t;
    col->encoding = Encoding::Flat;
    col->length = n;

    std::size_t bytes = buffer_bytes(t, n);
    auto* base = static_cast<std::uint8_t*>(const_cast<void*>(data));
    col->data = Buffer::wrap(base, bytes, [release, release_ctx](void*) {
        if (release != nullptr) release(release_ctx);
    });

    if (validity != nullptr) {
        std::size_t vbytes = static_cast<std::size_t>(validity_bytes(n));
        col->validity = Buffer::allocate(vbytes);
        std::memcpy(col->validity->data(), validity, vbytes);
        col->null_count = count_nulls(validity, n);
    }
    return col;
}

dftu_series* dftu_series_new_string(dftu_dtype type, const int32_t* offsets,
                                    const void* data, int64_t n,
                                    const uint8_t* validity) {
    TypeId t = static_cast<TypeId>(type);
    if (t != TypeId::String && t != TypeId::Binary) return nullptr;

    auto* col = new dftu_series();
    col->type = t;
    col->encoding = Encoding::Flat;
    col->length = n;

    std::size_t off_bytes =
        static_cast<std::size_t>(n + 1) * sizeof(std::int32_t);
    col->offsets = Buffer::allocate(off_bytes);
    if (offsets != nullptr)
        std::memcpy(col->offsets->data(), offsets, off_bytes);
    std::int32_t data_len = offsets != nullptr ? offsets[n] : 0;

    col->data = Buffer::allocate(static_cast<std::size_t>(data_len));
    if (data != nullptr && data_len > 0)
        std::memcpy(col->data->data(), data,
                    static_cast<std::size_t>(data_len));

    if (validity != nullptr) {
        std::size_t vbytes = static_cast<std::size_t>(validity_bytes(n));
        col->validity = Buffer::allocate(vbytes);
        std::memcpy(col->validity->data(), validity, vbytes);
        col->null_count = count_nulls(validity, n);
    }
    return col;
}

dftu_series* dftu_series_new_struct(dftu_series** fields, const char** names,
                                    int32_t n_fields) {
    if (n_fields < 0 || (n_fields > 0 && fields == nullptr)) return nullptr;
    auto* col = new dftu_series();
    col->type = TypeId::Struct;
    col->encoding = Encoding::Flat;
    // An empty batch is a valid 0-field, 0-row struct, not an error; returning
    // null here made an empty result crash the Arrow export (null handle
    // deref).
    col->length = n_fields > 0 ? fields[0]->length : 0;
    for (int32_t i = 0; i < n_fields; ++i) {
        col->children.push_back(std::shared_ptr<dftu_series>(fields[i]));
        col->field_names.push_back(names && names[i] ? names[i] : "");
    }
    return col;
}

dftu_series* dftu_series_new_list(const int32_t* offsets, int64_t n,
                                  dftu_series* values) {
    auto* col = new dftu_series();
    col->type = TypeId::List;
    col->encoding = Encoding::Flat;
    col->length = n;
    std::size_t off_bytes =
        static_cast<std::size_t>(n + 1) * sizeof(std::int32_t);
    col->offsets = Buffer::allocate(off_bytes);
    if (offsets != nullptr)
        std::memcpy(col->offsets->data(), offsets, off_bytes);
    col->child = std::shared_ptr<dftu_series>(values);
    return col;
}

void dftu_series_free(dftu_series* col) { delete col; }

int32_t dftu_series_type(const dftu_series* col) {
    return static_cast<int32_t>(col->type);
}

int32_t dftu_series_encoding(const dftu_series* col) {
    return static_cast<int32_t>(col->encoding);
}

int64_t dftu_series_length(const dftu_series* col) { return col->length; }

int64_t dftu_series_null_count(const dftu_series* col) {
    return col->null_count;
}

const void* dftu_series_data(const dftu_series* col) {
    if (col->encoding != Encoding::Flat || !col->data) return nullptr;
    return col->data->data();
}

const int32_t* dftu_series_offsets(const dftu_series* col) {
    if (!col->offsets) return nullptr;
    return reinterpret_cast<const int32_t*>(col->offsets->data());
}

int32_t dftu_series_is_null(const dftu_series* col, int64_t i) {
    if (!col->validity) return 0;
    const std::uint8_t* bm = col->validity->data();
    return ((bm[i >> 3] >> (i & 7)) & 1) ? 0 : 1;
}

int32_t dftu_series_num_children(const dftu_series* col) {
    if (col->type == TypeId::List) return col->child ? 1 : 0;
    return static_cast<int32_t>(col->children.size());
}

dftu_series* dftu_series_child(const dftu_series* col, int32_t i) {
    const std::shared_ptr<dftu_series>* ch = nullptr;
    if (col->type == TypeId::List) {
        if (i == 0 && col->child) ch = &col->child;
    } else if (i >= 0 && static_cast<std::size_t>(i) < col->children.size()) {
        ch = &col->children[static_cast<std::size_t>(i)];
    }
    if (!ch || !*ch) return nullptr;
    // Owned copy sharing the child's buffers (shared_ptr members).
    return new dftu_series(**ch);
}

dftu_series* dftu_series_share(const dftu_series* col) {
    if (!col) return nullptr;
    // Copy the handle struct; its buffer/child shared_ptr members bump their
    // refcounts, so the new column shares the same data with no copy.
    return new dftu_series(*col);
}

dftu_series* dftu_series_slice(const dftu_series* col, int64_t offset,
                               int64_t len) {
    if (!col || col->encoding != Encoding::Flat) return nullptr;
    const std::size_t w = byte_width(col->type);
    if (w == 0 || !col->data) return nullptr;  // variable-width unsupported
    if (offset < 0) offset = 0;
    if (offset > col->length) offset = col->length;
    if (len < 0 || len > col->length - offset) len = col->length - offset;

    auto* out = new dftu_series();
    out->type = col->type;
    out->encoding = Encoding::Flat;
    out->length = len;
    auto parent = col->data;  // shared_ptr copy keeps the buffer alive
    if (col->type == TypeId::Bool) {
        // Bool is bit-packed ((n+7)/8 bytes)
        const std::size_t nbytes = static_cast<std::size_t>((len + 7) / 8);
        auto dbuf = Buffer::allocate(nbytes);
        std::memset(dbuf->data(), 0, nbytes);
        const std::uint8_t* src = parent->data();
        std::uint8_t* dst = dbuf->data();
        for (int64_t i = 0; i < len; ++i) {
            const int64_t p = offset + i;
            if ((src[p >> 3] >> (p & 7)) & 1u)
                dst[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
        }
        out->data = std::move(dbuf);
    } else {
        std::uint8_t* base =
            parent->data() + static_cast<std::size_t>(offset) * w;
        out->data =
            Buffer::wrap(base, static_cast<std::size_t>(len) * w,
                         [parent](void*) { /* view: parent owns it */ });
    }

    // Carry the validity bitmap so nulls survive into the expression engine.
    // The bitmap is indexed from bit 0, so a byte-aligned offset can share the
    // parent buffer at a shifted base; otherwise re-pack the [offset, offset+
    // len) bits down to bit 0.
    if (col->validity && len > 0) {
        const std::uint8_t* src = col->validity->data();
        std::shared_ptr<Buffer> vbuf;
        if ((offset & 7) == 0) {
            auto vparent = col->validity;
            std::uint8_t* vbase =
                vparent->data() + static_cast<std::size_t>(offset >> 3);
            vbuf = Buffer::wrap(vbase, static_cast<std::size_t>((len + 7) / 8),
                                [vparent](void*) { /* view */ });
        } else {
            const std::size_t nbytes = static_cast<std::size_t>((len + 7) / 8);
            vbuf = Buffer::allocate(nbytes);
            std::memset(vbuf->data(), 0, nbytes);
            std::uint8_t* dst = vbuf->data();
            for (int64_t i = 0; i < len; ++i) {
                const int64_t p = offset + i;
                if ((src[p >> 3] >> (p & 7)) & 1u)
                    dst[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
            }
        }
        // Recount nulls in the slice; the parent's count spans the whole
        // column.
        out->null_count = count_nulls(vbuf->data(), len);
        out->validity = std::move(vbuf);
    }
    return out;
}
