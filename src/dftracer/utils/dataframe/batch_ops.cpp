#include <dftracer/utils/core/common/hash/fnv1a.h>       // string cell hashing
#include <dftracer/utils/core/common/hash/splitmix64.h>  // row/cell hashing
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/dataframe/containment.h>
#include <dftracer/utils/dataframe/internal/column_read.h>  // read_u64
#include <dftracer/utils/dataframe/kernels/filter.h>
#include <dftracer/utils/dataframe/kernels/group_by.h>
#include <dftracer/utils/dataframe/kernels/sort.h>
#include <dftracer/utils/dataframe/parallel.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>

namespace dftracer::utils::dataframe {

namespace {

std::int64_t index_of(const DataFrame& b, const std::string& name) {
    for (std::size_t i = 0; i < b.names.size(); ++i)
        if (b.names[i] == name) return static_cast<std::int64_t>(i);
    return -1;
}

AggOp agg_op_of(const std::string& op) {
    if (op == "sum") return AggOp::Sum;
    if (op == "min") return AggOp::Min;
    if (op == "max") return AggOp::Max;
    if (op == "mean") return AggOp::Mean;
    if (op == "count") return AggOp::Count;
    if (op == "var") return AggOp::Var;
    if (op == "std") return AggOp::Std;
    if (op == "skew") return AggOp::Skew;
    if (op == "kurt") return AggOp::Kurt;
    if (op == "first") return AggOp::First;
    if (op == "last") return AggOp::Last;
    if (op == "pct") return AggOp::Pct;
    if (op == "hist") return AggOp::Hist;
    throw std::out_of_range("group_by: unknown aggregate op " + op);
}

}  // namespace

const char* to_string(Agg agg) noexcept {
    switch (agg) {
        case Agg::Sum:
            return "sum";
        case Agg::Min:
            return "min";
        case Agg::Max:
            return "max";
        case Agg::Count:
            return "count";
        case Agg::Mean:
            return "mean";
        case Agg::Var:
            return "var";
        case Agg::Std:
            return "std";
        case Agg::Skew:
            return "skew";
        case Agg::Kurt:
            return "kurt";
        case Agg::First:
            return "first";
        case Agg::Last:
            return "last";
        case Agg::Pct:
            return "pct";
        case Agg::Hist:
            return "hist";
    }
    return "count";
}

Agg agg_from_string(std::string_view name) {
    if (name == "sum") return Agg::Sum;
    if (name == "min") return Agg::Min;
    if (name == "max") return Agg::Max;
    if (name == "count") return Agg::Count;
    if (name == "mean") return Agg::Mean;
    if (name == "var") return Agg::Var;
    if (name == "std") return Agg::Std;
    if (name == "skew") return Agg::Skew;
    if (name == "kurt") return Agg::Kurt;
    if (name == "first") return Agg::First;
    if (name == "last") return Agg::Last;
    if (name == "pct") return Agg::Pct;
    if (name == "hist") return Agg::Hist;
    throw std::out_of_range("agg_from_string: unknown aggregate op " +
                            std::string(name));
}

DataFrame take(const DataFrame& b, const std::vector<std::int64_t>& indices) {
    DataFrame out;
    out.names = b.names;
    out.columns.reserve(b.columns.size());
    for (const Series& c : b.columns) out.columns.push_back(take(c, indices));
    return out;
}

DataFrame filter(const DataFrame& b, const Series& mask) {
    return take(b, mask_to_indices(mask.data<std::uint8_t>(), mask.length()));
}

DataFrame slice(const DataFrame& b, std::int64_t offset, std::int64_t len) {
    const std::int64_t nrows = b.num_rows();
    offset = std::clamp<std::int64_t>(offset, 0, nrows);
    len = std::clamp<std::int64_t>(len, 0, nrows - offset);
    std::vector<std::int64_t> indices;
    indices.reserve(static_cast<std::size_t>(len));
    for (std::int64_t i = 0; i < len; ++i) indices.push_back(offset + i);
    return take(b, indices);
}

DataFrame head(const DataFrame& b, std::int64_t n) { return slice(b, 0, n); }

DataFrame select(const DataFrame& b, const std::vector<std::string>& names) {
    DataFrame out;
    out.names = names;
    out.columns.reserve(names.size());
    for (const std::string& name : names) {
        std::int64_t i = index_of(b, name);
        if (i < 0) throw std::out_of_range("select: no column named " + name);
        out.columns.push_back(b.columns[static_cast<std::size_t>(i)].share());
    }
    return out;
}

DataFrame rename(const DataFrame& b,
                 const std::vector<std::string>& new_names) {
    if (new_names.size() != b.columns.size())
        throw std::invalid_argument("rename: name count must match columns");
    DataFrame out;
    out.names = new_names;
    out.columns.reserve(b.columns.size());
    for (const Series& c : b.columns) out.columns.push_back(c.share());
    return out;
}

DataFrame with_column(const DataFrame& b, const std::string& name,
                      const Series& col) {
    DataFrame out;
    out.names = b.names;
    out.columns.reserve(b.columns.size() + 1);
    for (const Series& c : b.columns) out.columns.push_back(c.share());
    std::int64_t i = index_of(b, name);
    if (i >= 0) {
        out.columns[static_cast<std::size_t>(i)] = col.share();
    } else {
        out.names.push_back(name);
        out.columns.push_back(col.share());
    }
    return out;
}

DataFrame sort_by(const DataFrame& b, const std::string& name,
                  bool descending) {
    std::int64_t k = index_of(b, name);
    if (k < 0) throw std::out_of_range("sort_by: no column named " + name);
    Series order = argsort(b.columns[static_cast<std::size_t>(k)], descending);
    const std::int64_t* p = order.data<std::int64_t>();
    std::vector<std::int64_t> indices(p, p + order.length());
    return take(b, indices);
}

DataFrame topk(const DataFrame& b, const std::string& name, std::int64_t k,
               bool largest) {
    std::int64_t j = index_of(b, name);
    if (j < 0) throw std::out_of_range("topk: no column named " + name);
    Series ind =
        topk_indices(b.columns[static_cast<std::size_t>(j)], k, largest);
    const std::int64_t* p = ind.data<std::int64_t>();
    std::vector<std::int64_t> indices(p, p + ind.length());
    return take(b, indices);
}

Series concat_columns(const std::vector<const Series*>& parts) {
    if (parts.empty()) return Series{};
    // Materialize each part to FLAT so the value buffers are contiguous; concat
    // is a merge, so it copies (a single-buffer column can't alias many
    // inputs).
    std::vector<Series> mats;
    mats.reserve(parts.size());
    for (const Series* p : parts)
        mats.emplace_back(dftu_series_materialize(p->handle()));

    const TypeId t = mats.front().type();
    std::int64_t total = 0;
    bool any_null = false;
    for (const Series& m : mats) {
        if (m.type() != t)
            throw std::invalid_argument("concat: columns must share a type");
        if (t == TypeId::List || t == TypeId::Struct)
            throw std::invalid_argument("concat: nested columns unsupported");
        total += m.length();
        any_null |= m.null_count() > 0;
    }

    std::vector<std::uint8_t> validity;
    if (any_null) {
        validity.assign(static_cast<std::size_t>((total + 7) / 8), 0xFF);
        std::int64_t off = 0;
        for (const Series& m : mats) {
            for (std::int64_t i = 0; i < m.length(); ++i)
                if (m.is_null(i)) {
                    std::int64_t r = off + i;
                    validity[static_cast<std::size_t>(r >> 3)] &=
                        ~(1u << (r & 7));
                }
            off += m.length();
        }
    }
    const std::uint8_t* vptr = any_null ? validity.data() : nullptr;

    if (t == TypeId::String) {
        std::string data;
        std::vector<std::int32_t> offs(static_cast<std::size_t>(total) + 1, 0);
        std::int64_t r = 0;
        for (const Series& m : mats)
            for (std::int64_t i = 0; i < m.length(); ++i) {
                if (!m.is_null(i)) data += m.string_at(i);
                offs[static_cast<std::size_t>(r) + 1] =
                    static_cast<std::int32_t>(data.size());
                ++r;
            }
        return Series{
            dftu_series_new_string(static_cast<dftu_dtype>(TypeId::String),
                                   offs.data(), data.data(), total, vptr)};
    }

    if (t == TypeId::Bool) {
        std::vector<std::uint8_t> bits(
            static_cast<std::size_t>((total + 7) / 8), 0);
        std::int64_t r = 0;
        for (const Series& m : mats) {
            const std::uint8_t* mb = m.data<std::uint8_t>();
            for (std::int64_t i = 0; i < m.length(); ++i) {
                if ((mb[i >> 3] >> (i & 7)) & 1)
                    bits[static_cast<std::size_t>(r >> 3)] |= (1u << (r & 7));
                ++r;
            }
        }
        return Series{dftu_series_new_flat(
            static_cast<dftu_dtype>(TypeId::Bool), bits.data(), total, vptr)};
    }

    const std::size_t w = byte_width(t);
    std::vector<std::uint8_t> buf(static_cast<std::size_t>(total) * w);
    std::int64_t off = 0;
    for (const Series& m : mats) {
        std::memcpy(buf.data() + static_cast<std::size_t>(off) * w,
                    m.data<std::uint8_t>(),
                    static_cast<std::size_t>(m.length()) * w);
        off += m.length();
    }
    return Series{dftu_series_new_flat(static_cast<dftu_dtype>(t), buf.data(),
                                       total, vptr)};
}

DataFrame group_by(const DataFrame& b, const std::string& key,
                   const std::vector<GroupAgg>& aggs) {
    std::int64_t ki = index_of(b, key);
    if (ki < 0) throw std::out_of_range("group_by: no column named " + key);
    const Series& key_col = b.columns[static_cast<std::size_t>(ki)];

    // Lower to the fused aggregation engine: value columns referenced by index
    // (deduped), all aggregates computed in one parallel pass over the batch.
    std::vector<const Series*> values;
    std::map<std::string, std::int32_t> col_idx;
    std::vector<AggSpec> specs;
    specs.reserve(aggs.size());
    for (const GroupAgg& a : aggs) {
        AggSpec sp;
        sp.op = agg_op_of(to_string(a.op));
        sp.out = a.out;
        sp.param = a.param;
        if (sp.op == AggOp::Count) {
            sp.value_col = -1;
        } else {
            auto it = col_idx.find(a.column);
            if (it != col_idx.end()) {
                sp.value_col = it->second;
            } else {
                std::int64_t vi = index_of(b, a.column);
                if (vi < 0)
                    throw std::out_of_range("group_by: no column named " +
                                            a.column);
                sp.value_col = static_cast<std::int32_t>(values.size());
                values.push_back(&b.columns[static_cast<std::size_t>(vi)]);
                col_idx.emplace(a.column, sp.value_col);
            }
        }
        specs.push_back(std::move(sp));
    }
    return group_agg(key_col, values, std::move(specs), key);
}

namespace {

using dftracer::utils::hash::splitmix64;

std::uint64_t read_f64_bits(const Series& c, std::int64_t i) {
    if (c.type() == TypeId::Float32)
        return std::bit_cast<std::uint32_t>(c.data<float>()[i]);
    return std::bit_cast<std::uint64_t>(c.data<double>()[i]);
}

// A stable hash of one cell, so the same value hashes identically on every
// node.
std::uint64_t hash_cell(const Series& c, std::int64_t i) {
    switch (c.type()) {
        case TypeId::String:
            return hash::fnv1a_hash(c.string_at(i));
        case TypeId::Bool: {
            const std::uint8_t* b = c.data<std::uint8_t>();
            return splitmix64((b[i >> 3] >> (i & 7)) & 1);
        }
        case TypeId::Float32:
        case TypeId::Float64:
            return splitmix64(static_cast<std::uint64_t>(read_f64_bits(c, i)));
        default:
            return splitmix64(read_u64(c, i));
    }
}

}  // namespace

std::vector<DataFrame> hash_partition(const DataFrame& b,
                                      const std::vector<std::string>& keys,
                                      std::int64_t n_parts) {
    if (keys.empty()) throw std::invalid_argument("hash_partition: no keys");
    if (n_parts < 1) throw std::invalid_argument("hash_partition: n_parts < 1");
    const std::int64_t n = b.num_rows();

    std::vector<const Series*> key_cols;
    for (const std::string& k : keys) {
        std::int64_t ki = index_of(b, k);
        if (ki < 0)
            throw std::out_of_range("hash_partition: no column named " + k);
        key_cols.push_back(&b.columns[static_cast<std::size_t>(ki)]);
    }

    std::vector<std::vector<std::int64_t>> buckets(
        static_cast<std::size_t>(n_parts));
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint64_t h = 1469598103934665603ULL;
        for (const Series* c : key_cols) h = splitmix64(h ^ hash_cell(*c, i));
        buckets[static_cast<std::size_t>(h %
                                         static_cast<std::uint64_t>(n_parts))]
            .push_back(i);
    }

    std::vector<DataFrame> parts;
    parts.reserve(static_cast<std::size_t>(n_parts));
    for (auto& idx : buckets) parts.push_back(take(b, idx));
    return parts;
}

namespace {

// The type the same-named column takes across parts. Same type stays; a mix of
// numeric types widens to Float64; a numeric/String or Bool/other clash has no
// common type and throws.
TypeId promote_type(TypeId a, TypeId b, const std::string& name) {
    if (a == b) return a;
    auto numeric = [](TypeId t) {
        return t != TypeId::String && t != TypeId::Binary &&
               t != TypeId::Bool && t != TypeId::List && t != TypeId::Struct;
    };
    if (numeric(a) && numeric(b)) return TypeId::Float64;
    throw std::invalid_argument("concat(diagonal): column '" + name +
                                "' has incompatible types across parts");
}

DataFrame concat_diagonal(const std::vector<const DataFrame*>& parts) {
    // Column order = first appearance; track each column's promoted type.
    std::vector<std::string> names;
    std::vector<TypeId> types;
    std::unordered_map<std::string, std::size_t> idx;
    for (const DataFrame* p : parts)
        for (std::size_t c = 0; c < p->names.size(); ++c) {
            const std::string& nm = p->names[c];
            const TypeId t = p->columns[c].type();
            auto it = idx.find(nm);
            if (it == idx.end()) {
                idx.emplace(nm, names.size());
                names.push_back(nm);
                types.push_back(t);
            } else {
                types[it->second] = promote_type(types[it->second], t, nm);
            }
        }

    DataFrame out;
    out.names = names;
    out.columns.reserve(names.size());
    for (std::size_t c = 0; c < names.size(); ++c) {
        const TypeId target = types[c];
        // Hold cast/null-filled parts alive while concat_columns reads them.
        std::vector<Series> owned;
        owned.reserve(parts.size());
        std::vector<const Series*> cols;
        cols.reserve(parts.size());
        for (const DataFrame* p : parts) {
            std::int64_t at = -1;
            for (std::size_t j = 0; j < p->names.size(); ++j)
                if (p->names[j] == names[c]) {
                    at = static_cast<std::int64_t>(j);
                    break;
                }
            if (at < 0) {
                owned.push_back(Series::nulls(target, p->num_rows()));
            } else {
                const Series& src = p->columns[static_cast<std::size_t>(at)];
                owned.push_back(src.type() == target ? src.share()
                                                     : src.cast(target));
            }
            cols.push_back(&owned.back());
        }
        out.columns.push_back(concat_columns(cols));
    }
    return out;
}

}  // namespace

DataFrame concat(const std::vector<const DataFrame*>& parts, ConcatHow how) {
    DataFrame out;
    if (parts.empty()) return out;
    // A single part is the identity: share its columns zero-copy, no merge.
    if (parts.size() == 1) {
        out.names = parts.front()->names;
        out.columns.reserve(parts.front()->columns.size());
        for (const Series& c : parts.front()->columns)
            out.columns.push_back(c.share());
        return out;
    }
    if (how == ConcatHow::Diagonal) return concat_diagonal(parts);
    const DataFrame& first = *parts.front();
    out.names = first.names;
    const std::size_t ncols = first.names.size();
    for (const DataFrame* p : parts)
        if (p->names != first.names)
            throw std::invalid_argument("concat: batches must share a schema");
    out.columns.reserve(ncols);
    for (std::size_t c = 0; c < ncols; ++c) {
        std::vector<const Series*> cols;
        cols.reserve(parts.size());
        for (const DataFrame* p : parts) cols.push_back(&p->columns[c]);
        out.columns.push_back(concat_columns(cols));
    }
    return out;
}

namespace {

// A FLAT copy of every column (sharing already-FLAT ones), so cell accessors
// below can read the value buffers directly.
std::vector<Series> materialized_columns(const DataFrame& b) {
    std::vector<Series> cols;
    cols.reserve(b.columns.size());
    for (const Series& c : b.columns) {
        if (c.encoding() == Encoding::Flat)
            cols.push_back(c.share());
        else
            cols.emplace_back(dftu_series_materialize(c.handle()));
    }
    return cols;
}

// Append cell (col, i) to `key` as raw bytes, prefixed by a present/null flag,
// so the concatenation of a row's cells is a stable composite dedupe key.
void append_cell(std::string& key, const Series& c, std::int64_t i) {
    if (c.is_null(i)) {
        key.push_back('\0');
        return;
    }
    key.push_back('\1');
    const TypeId t = c.type();
    if (t == TypeId::String || t == TypeId::Binary) {
        std::string_view s = c.string_at(i);
        std::int32_t len = static_cast<std::int32_t>(s.size());
        key.append(reinterpret_cast<const char*>(&len), sizeof(len));
        key.append(s.data(), s.size());
        return;
    }
    const std::uint8_t* d = c.data<std::uint8_t>();
    if (t == TypeId::Bool) {
        key.push_back((d[i >> 3] >> (i & 7)) & 1 ? '\1' : '\0');
        return;
    }
    const std::size_t w = byte_width(t);
    key.append(
        reinterpret_cast<const char*>(d + static_cast<std::size_t>(i) * w), w);
}

// Composite dedupe key per row over `cols` (already FLAT).
std::vector<std::string> row_keys(const std::vector<Series>& cols,
                                  std::int64_t n) {
    std::vector<std::string> keys(static_cast<std::size_t>(n));
    for (const Series& c : cols)
        for (std::int64_t i = 0; i < n; ++i)
            append_cell(keys[static_cast<std::size_t>(i)], c, i);
    return keys;
}

template <class T>
int cmp_num(const Series& c, std::int64_t a, std::int64_t b) {
    const T va = c.data<T>()[a];
    const T vb = c.data<T>()[b];
    return va < vb ? -1 : (va > vb ? 1 : 0);
}

// Ordering of two non-null cells of the same column (ascending value order).
int raw_cmp(const Series& c, std::int64_t a, std::int64_t b) {
    switch (c.type()) {
        case TypeId::Bool: {
            const std::uint8_t* p = c.data<std::uint8_t>();
            int va = (p[a >> 3] >> (a & 7)) & 1;
            int vb = (p[b >> 3] >> (b & 7)) & 1;
            return va < vb ? -1 : (va > vb ? 1 : 0);
        }
        case TypeId::Int8:
            return cmp_num<std::int8_t>(c, a, b);
        case TypeId::Int16:
            return cmp_num<std::int16_t>(c, a, b);
        case TypeId::Int32:
            return cmp_num<std::int32_t>(c, a, b);
        case TypeId::Int64:
            return cmp_num<std::int64_t>(c, a, b);
        case TypeId::Uint8:
            return cmp_num<std::uint8_t>(c, a, b);
        case TypeId::Uint16:
            return cmp_num<std::uint16_t>(c, a, b);
        case TypeId::Uint32:
            return cmp_num<std::uint32_t>(c, a, b);
        case TypeId::Uint64:
            return cmp_num<std::uint64_t>(c, a, b);
        case TypeId::Float32:
            return cmp_num<float>(c, a, b);
        case TypeId::Float64:
            return cmp_num<double>(c, a, b);
        case TypeId::String:
        case TypeId::Binary: {
            int r = c.string_at(a).compare(c.string_at(b));
            return r < 0 ? -1 : (r > 0 ? 1 : 0);
        }
        default:
            return 0;
    }
}

Series row_mask(const DataFrame& b, bool want_unique) {
    const std::int64_t n = b.num_rows();
    std::vector<Series> cols = materialized_columns(b);
    std::vector<std::string> keys = row_keys(cols, n);
    std::unordered_map<std::string, std::int64_t> counts;
    counts.reserve(static_cast<std::size_t>(n));
    for (const std::string& k : keys) ++counts[k];
    std::vector<std::uint8_t> bits(static_cast<std::size_t>((n + 7) / 8), 0);
    for (std::int64_t i = 0; i < n; ++i) {
        bool unique = counts[keys[static_cast<std::size_t>(i)]] == 1;
        if (want_unique == unique)
            bits[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    }
    return Series{dftu_series_new_flat(static_cast<dftu_dtype>(TypeId::Bool),
                                       bits.data(), n, nullptr)};
}

double scalar_to_double(dftu_scalar s) {
    switch (s.kind) {
        case DFTU_SCALAR_TAG_F64:
            return s.value.d;
        case DFTU_SCALAR_TAG_U64:
            return static_cast<double>(s.value.u);
        default:
            return static_cast<double>(s.value.i);
    }
}

bool is_numeric_type(TypeId t) {
    switch (t) {
        case TypeId::Int8:
        case TypeId::Int16:
        case TypeId::Int32:
        case TypeId::Int64:
        case TypeId::Uint8:
        case TypeId::Uint16:
        case TypeId::Uint32:
        case TypeId::Uint64:
        case TypeId::Float32:
        case TypeId::Float64:
            return true;
        default:
            return false;
    }
}

}  // namespace

DataFrame drop_nulls(const DataFrame& b) {
    const std::int64_t n = b.num_rows();
    std::vector<std::int64_t> idx;
    for (std::int64_t i = 0; i < n; ++i) {
        bool keep = true;
        for (const Series& c : b.columns)
            if (c.is_null(i)) {
                keep = false;
                break;
            }
        if (keep) idx.push_back(i);
    }
    return take(b, idx);
}

DataFrame fill_null(const DataFrame& b, dftu_scalar value) {
    DataFrame out;
    out.names = b.names;
    out.columns.reserve(b.columns.size());
    for (const Series& c : b.columns) {
        Series f = c.fillna(value);
        out.columns.push_back(f.valid() ? std::move(f) : c.share());
    }
    return out;
}

DataFrame unique(const DataFrame& b) {
    const std::int64_t n = b.num_rows();
    std::vector<Series> cols = materialized_columns(b);
    std::vector<std::string> keys = row_keys(cols, n);
    std::unordered_set<std::string> seen;
    seen.reserve(static_cast<std::size_t>(n));
    std::vector<std::int64_t> idx;
    for (std::int64_t i = 0; i < n; ++i)
        if (seen.insert(keys[static_cast<std::size_t>(i)]).second)
            idx.push_back(i);
    return take(b, idx);
}

DataFrame sort_by_multi(const DataFrame& b,
                        const std::vector<std::string>& names,
                        bool descending) {
    std::vector<const Series*> keys;
    keys.reserve(names.size());
    for (const std::string& name : names) {
        std::int64_t k = index_of(b, name);
        if (k < 0)
            throw std::out_of_range("sort_by_multi: no column named " + name);
        keys.push_back(&b.columns[static_cast<std::size_t>(k)]);
    }
    const std::int64_t n = b.num_rows();
    std::vector<std::int64_t> order(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) order[static_cast<std::size_t>(i)] = i;
    std::stable_sort(order.begin(), order.end(),
                     [&](std::int64_t a, std::int64_t bb) {
                         for (const Series* c : keys) {
                             bool na = c->is_null(a);
                             bool nb = c->is_null(bb);
                             if (na || nb) {
                                 if (na && nb) continue;
                                 return !na;  // nulls last in both directions
                             }
                             int r = raw_cmp(*c, a, bb);
                             if (descending) r = -r;
                             if (r != 0) return r < 0;
                         }
                         return false;
                     });
    return take(b, order);
}

DataFrame tail(const DataFrame& b, std::int64_t n) {
    const std::int64_t len = b.num_rows();
    n = std::clamp<std::int64_t>(n, 0, len);
    return slice(b, len - n, n);
}

DataFrame reverse(const DataFrame& b) {
    const std::int64_t n = b.num_rows();
    std::vector<std::int64_t> idx(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        idx[static_cast<std::size_t>(i)] = n - 1 - i;
    return take(b, idx);
}

DataFrame sample(const DataFrame& b, std::int64_t n, std::uint64_t seed) {
    const std::int64_t len = b.num_rows();
    n = std::clamp<std::int64_t>(n, 0, len);
    std::vector<std::int64_t> order(static_cast<std::size_t>(len));
    for (std::int64_t i = 0; i < len; ++i)
        order[static_cast<std::size_t>(i)] = i;
    auto key = [seed](std::int64_t i) {
        return splitmix64(static_cast<std::uint64_t>(i) + seed);
    };
    if (n < len)
        std::nth_element(
            order.begin(), order.begin() + n, order.end(),
            [&](std::int64_t a, std::int64_t c) { return key(a) < key(c); });
    order.resize(static_cast<std::size_t>(n));
    std::sort(order.begin(), order.end());
    return take(b, order);
}

DataFrame with_row_index(const DataFrame& b, const std::string& name) {
    const std::int64_t n = b.num_rows();
    std::vector<std::int64_t> ids(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) ids[static_cast<std::size_t>(i)] = i;
    DataFrame out;
    out.names.reserve(b.names.size() + 1);
    out.columns.reserve(b.columns.size() + 1);
    out.names.push_back(name);
    out.columns.push_back(Series::flat_i64(ids.data(), n));
    for (std::size_t i = 0; i < b.columns.size(); ++i) {
        out.names.push_back(b.names[i]);
        out.columns.push_back(b.columns[i].share());
    }
    return out;
}

DataFrame describe(const DataFrame& b) {
    const std::vector<std::string> stat_names = {"count", "null_count", "mean",
                                                 "std",   "min",        "max"};
    DataFrame out;
    out.names.push_back("statistic");
    out.columns.push_back(Series::strings(stat_names));
    for (std::size_t i = 0; i < b.columns.size(); ++i) {
        const Series& c = b.columns[i];
        if (!is_numeric_type(c.type())) continue;
        double vals[6];
        vals[0] = static_cast<double>(c.count());
        vals[1] = static_cast<double>(c.null_count());
        vals[2] = c.mean();
        vals[3] = c.stddev();
        vals[4] = scalar_to_double(c.min());
        vals[5] = scalar_to_double(c.max());
        out.names.push_back(b.names[i]);
        out.columns.push_back(Series::flat_f64(vals, 6));
    }
    return out;
}

DataFrame null_count(const DataFrame& b) {
    DataFrame out;
    out.names = b.names;
    out.columns.reserve(b.columns.size());
    for (const Series& c : b.columns) {
        std::int64_t nc = c.null_count();
        out.columns.push_back(Series::flat_i64(&nc, 1));
    }
    return out;
}

Series is_duplicated(const DataFrame& b) { return row_mask(b, false); }
Series is_unique(const DataFrame& b) { return row_mask(b, true); }

DataFrame value_counts(const Series& v) {
    Series mat = v.encoding() == Encoding::Flat
                     ? v.share()
                     : Series{dftu_series_materialize(v.handle())};
    const std::int64_t n = mat.length();
    std::unordered_map<std::string, std::int64_t> idx_of;
    std::vector<std::int64_t> first_index;
    std::vector<std::int64_t> counts;
    for (std::int64_t i = 0; i < n; ++i) {
        if (mat.is_null(i)) continue;
        std::string key;
        append_cell(key, mat, i);
        auto [it, ins] = idx_of.try_emplace(
            std::move(key), static_cast<std::int64_t>(first_index.size()));
        if (ins) {
            first_index.push_back(i);
            counts.push_back(1);
        } else {
            ++counts[static_cast<std::size_t>(it->second)];
        }
    }
    DataFrame df;
    df.names = {"value", "count"};
    df.columns.push_back(mat.take(first_index));
    df.columns.push_back(Series::flat_i64(
        counts.data(), static_cast<std::int64_t>(counts.size())));
    return sort_by(df, "count", true);
}

namespace {

// A FLAT copy of `c` (sharing an already-FLAT column), so offset/value buffers
// can be read directly.
Series flat_copy(const Series& c) {
    return c.encoding() == Encoding::Flat
               ? c.share()
               : Series{dftu_series_materialize(c.handle())};
}

// Human-readable rendering of cell (c, i) for a dummy column label. `c` is
// FLAT.
std::string cell_to_string(const Series& c, std::int64_t i) {
    switch (c.type()) {
        case TypeId::String:
        case TypeId::Binary:
            return std::string(c.string_at(i));
        case TypeId::Bool: {
            const std::uint8_t* b = c.data<std::uint8_t>();
            return ((b[i >> 3] >> (i & 7)) & 1) ? "true" : "false";
        }
        case TypeId::Int8:
            return std::to_string(c.data<std::int8_t>()[i]);
        case TypeId::Int16:
            return std::to_string(c.data<std::int16_t>()[i]);
        case TypeId::Int32:
            return std::to_string(c.data<std::int32_t>()[i]);
        case TypeId::Int64:
            return std::to_string(c.data<std::int64_t>()[i]);
        case TypeId::Uint8:
            return std::to_string(c.data<std::uint8_t>()[i]);
        case TypeId::Uint16:
            return std::to_string(c.data<std::uint16_t>()[i]);
        case TypeId::Uint32:
            return std::to_string(c.data<std::uint32_t>()[i]);
        case TypeId::Uint64:
            return std::to_string(c.data<std::uint64_t>()[i]);
        case TypeId::Float32:
            return std::to_string(c.data<float>()[i]);
        case TypeId::Float64:
            return std::to_string(c.data<double>()[i]);
        default:
            return std::string();
    }
}

bool is_float_type(TypeId t) {
    return t == TypeId::Float32 || t == TypeId::Float64;
}

}  // namespace

DataFrame unpivot(const DataFrame& b, const std::vector<std::string>& id_vars,
                  const std::vector<std::string>& value_vars) {
    if (value_vars.empty())
        throw std::invalid_argument("unpivot: value_vars must be non-empty");

    for (const std::string& name : id_vars)
        if (index_of(b, name) < 0)
            throw std::out_of_range("unpivot: no column named " + name);
    std::vector<std::int64_t> val_idx;
    val_idx.reserve(value_vars.size());
    for (const std::string& name : value_vars) {
        std::int64_t i = index_of(b, name);
        if (i < 0) throw std::out_of_range("unpivot: no column named " + name);
        val_idx.push_back(i);
    }

    const TypeId first = b.columns[static_cast<std::size_t>(val_idx[0])].type();
    bool all_same = true, any_float = false, all_numeric = true;
    for (std::int64_t vi : val_idx) {
        TypeId t = b.columns[static_cast<std::size_t>(vi)].type();
        if (t != first) all_same = false;
        if (is_float_type(t)) any_float = true;
        if (!is_numeric_type(t)) all_numeric = false;
    }
    TypeId common;
    if (all_same) {
        common = first;
    } else if (all_numeric) {
        common = any_float ? TypeId::Float64 : TypeId::Int64;
    } else {
        throw std::invalid_argument(
            "unpivot: value_vars must share a type or all be numeric");
    }

    const std::int64_t n = b.num_rows();
    std::vector<std::int64_t> identity(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        identity[static_cast<std::size_t>(i)] = i;

    DataFrame id_df = select(b, id_vars);
    std::vector<DataFrame> blocks;
    blocks.reserve(value_vars.size());
    for (std::size_t j = 0; j < value_vars.size(); ++j) {
        DataFrame block = take(id_df, identity);
        std::vector<std::string> varnames(static_cast<std::size_t>(n),
                                          value_vars[j]);
        block = with_column(block, "variable", Series::strings(varnames));

        const Series& src = b.columns[static_cast<std::size_t>(val_idx[j])];
        Series matv = flat_copy(src);
        Series valcol =
            matv.type() == common ? std::move(matv) : matv.cast(common);
        block = with_column(block, "value", valcol);
        blocks.push_back(std::move(block));
    }

    std::vector<const DataFrame*> parts;
    parts.reserve(blocks.size());
    for (const DataFrame& blk : blocks) parts.push_back(&blk);
    return concat(parts);
}

DataFrame explode(const DataFrame& b, const std::string& column) {
    std::int64_t ci = index_of(b, column);
    if (ci < 0) throw std::out_of_range("explode: no column named " + column);
    Series lst = flat_copy(b.columns[static_cast<std::size_t>(ci)]);
    if (lst.type() != TypeId::List)
        throw std::invalid_argument("explode: " + column +
                                    " is not a List column");

    const std::int64_t n = lst.length();
    const std::int32_t* offs = lst.offsets();
    Series child = lst.child(0);

    std::vector<std::int64_t> rep_idx;
    std::vector<std::int64_t> child_idx;
    for (std::int64_t i = 0; i < n; ++i) {
        std::int32_t s = offs[i];
        std::int32_t e = offs[i + 1];
        if (e == s) {  // empty or null list -> one null row
            rep_idx.push_back(i);
            child_idx.push_back(-1);
        } else {
            for (std::int32_t j = s; j < e; ++j) {
                rep_idx.push_back(i);
                child_idx.push_back(j);
            }
        }
    }

    DataFrame out;
    out.names = b.names;
    out.columns.reserve(b.columns.size());
    for (std::size_t k = 0; k < b.columns.size(); ++k) {
        if (static_cast<std::int64_t>(k) == ci)
            out.columns.push_back(child.take(child_idx));
        else
            out.columns.push_back(b.columns[k].take(rep_idx));
    }
    return out;
}

DataFrame to_dummies(const DataFrame& b, const std::string& column) {
    std::int64_t ci = index_of(b, column);
    if (ci < 0)
        throw std::out_of_range("to_dummies: no column named " + column);
    Series mat = flat_copy(b.columns[static_cast<std::size_t>(ci)]);
    const std::int64_t n = mat.length();

    // Distinct non-null values, ascending, for a deterministic column order.
    Series uniq = flat_copy(mat.unique());
    const std::int64_t d = uniq.length();

    std::vector<std::string> row_key(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        append_cell(row_key[static_cast<std::size_t>(i)], mat, i);

    DataFrame out;
    out.names.reserve(b.names.size() + static_cast<std::size_t>(d) - 1);
    out.columns.reserve(out.names.capacity());
    for (std::size_t k = 0; k < b.columns.size(); ++k) {
        if (static_cast<std::int64_t>(k) != ci) {
            out.names.push_back(b.names[k]);
            out.columns.push_back(b.columns[k].share());
            continue;
        }
        for (std::int64_t u = 0; u < d; ++u) {
            std::string ukey;
            append_cell(ukey, uniq, u);
            std::vector<std::int8_t> bits(static_cast<std::size_t>(n), 0);
            for (std::int64_t i = 0; i < n; ++i)
                bits[static_cast<std::size_t>(i)] =
                    row_key[static_cast<std::size_t>(i)] == ukey ? 1 : 0;
            out.names.push_back(column + "_" + cell_to_string(uniq, u));
            out.columns.push_back(Series::flat(TypeId::Int8, bits.data(), n));
        }
    }
    return out;
}

namespace {

enum class PivotMode { First, Last, Reduce };

PivotMode pivot_mode_of(const std::string& agg, AggOp& op) {
    if (agg == "first") return PivotMode::First;
    if (agg == "last") return PivotMode::Last;
    op = agg_op_of(agg);  // sum|min|max|mean (and the other AggOps)
    return PivotMode::Reduce;
}

// Floor `x` to a multiple of `m` (m > 0), toward negative infinity.
std::int64_t floor_to_multiple(std::int64_t x, std::int64_t m) {
    std::int64_t q = x / m;
    if ((x % m) != 0 && x < 0) --q;
    return q * m;
}

}  // namespace

DataFrame pivot(const DataFrame& b, const std::string& index_name,
                const std::string& columns_name, const std::string& values_name,
                const std::string& agg) {
    std::int64_t ii = index_of(b, index_name);
    std::int64_t ci = index_of(b, columns_name);
    std::int64_t vi = index_of(b, values_name);
    if (ii < 0) throw std::out_of_range("pivot: no column named " + index_name);
    if (ci < 0)
        throw std::out_of_range("pivot: no column named " + columns_name);
    if (vi < 0)
        throw std::out_of_range("pivot: no column named " + values_name);

    AggOp reduce_op = AggOp::Sum;
    PivotMode mode = pivot_mode_of(agg, reduce_op);

    Series idx_col = flat_copy(b.columns[static_cast<std::size_t>(ii)]);
    Series col_col = flat_copy(b.columns[static_cast<std::size_t>(ci)]);
    Series val_col = flat_copy(b.columns[static_cast<std::size_t>(vi)]);
    const std::int64_t n = idx_col.length();

    // Output rows = distinct index values (sorted asc, non-null); output value
    // columns = distinct column values (sorted asc, non-null).
    Series uniq_idx = flat_copy(idx_col.unique());
    Series uniq_col = flat_copy(col_col.unique());
    const std::int64_t R = uniq_idx.length();
    const std::int64_t C = uniq_col.length();

    std::unordered_map<std::string, std::int64_t> row_of, col_of;
    row_of.reserve(static_cast<std::size_t>(R));
    col_of.reserve(static_cast<std::size_t>(C));
    for (std::int64_t r = 0; r < R; ++r) {
        std::string k;
        append_cell(k, uniq_idx, r);
        row_of.emplace(std::move(k), r);
    }
    for (std::int64_t c = 0; c < C; ++c) {
        std::string k;
        append_cell(k, uniq_col, c);
        col_of.emplace(std::move(k), c);
    }

    // For each output cell (r, c), the source row into `source` (or -1 = absent
    // -> null). first/last take a row of `val_col` directly; the numeric modes
    // aggregate via group_agg, then take a row of the aggregate result. Both
    // reduce to a type-generic gather.
    Series source;
    std::vector<std::int64_t> cell_src(static_cast<std::size_t>(R * C), -1);

    if (mode == PivotMode::Reduce) {
        // Key each valid row by its flattened cell id; invalid (null-key) rows
        // go to a sentinel bucket that is dropped after aggregation.
        const std::int64_t sentinel = R * C;
        std::vector<std::int64_t> keyv(static_cast<std::size_t>(n), sentinel);
        for (std::int64_t i = 0; i < n; ++i) {
            if (idx_col.is_null(i) || col_col.is_null(i)) continue;
            std::string rk, ck;
            append_cell(rk, idx_col, i);
            append_cell(ck, col_col, i);
            keyv[static_cast<std::size_t>(i)] =
                row_of.at(rk) * C + col_of.at(ck);
        }
        Series keyc = Series::flat_i64(keyv.data(), n);
        std::vector<const Series*> values{&val_col};
        std::vector<AggSpec> specs;
        specs.push_back(AggSpec{reduce_op, 0, "v"});
        DataFrame agg_res = group_agg(keyc, values, std::move(specs), "cell");
        source = agg_res.column("v");
        Series cells_col = agg_res.column("cell");
        const std::int64_t m = cells_col.length();
        const std::int64_t* cells = cells_col.data<std::int64_t>();
        for (std::int64_t p = 0; p < m; ++p) {
            std::int64_t cell = cells[p];
            if (cell == sentinel) continue;
            cell_src[static_cast<std::size_t>(cell)] = p;
        }
    } else {
        source = val_col.share();
        for (std::int64_t i = 0; i < n; ++i) {
            if (idx_col.is_null(i) || col_col.is_null(i)) continue;
            std::string rk, ck;
            append_cell(rk, idx_col, i);
            append_cell(ck, col_col, i);
            std::int64_t cell = row_of.at(rk) * C + col_of.at(ck);
            std::int64_t& slot = cell_src[static_cast<std::size_t>(cell)];
            if (mode == PivotMode::First) {
                if (slot < 0) slot = i;
            } else {
                slot = i;  // last wins
            }
        }
    }

    DataFrame out;
    out.names.reserve(static_cast<std::size_t>(C) + 1);
    out.columns.reserve(static_cast<std::size_t>(C) + 1);
    out.names.push_back(index_name);
    out.columns.push_back(uniq_idx.share());
    for (std::int64_t c = 0; c < C; ++c) {
        std::vector<std::int64_t> take_idx(static_cast<std::size_t>(R));
        for (std::int64_t r = 0; r < R; ++r)
            take_idx[static_cast<std::size_t>(r)] =
                cell_src[static_cast<std::size_t>(r * C + c)];
        out.names.push_back(cell_to_string(uniq_col, c));
        out.columns.push_back(source.take(take_idx));
    }
    return out;
}

DataFrame group_by_dynamic(const DataFrame& b, const std::string& time_col,
                           std::int64_t every, std::int64_t period,
                           const std::vector<GroupAgg>& aggs,
                           std::int64_t origin, bool origin_min) {
    if (every <= 0)
        throw std::invalid_argument("group_by_dynamic: every must be > 0");
    if (period <= 0) period = every;

    std::int64_t ti = index_of(b, time_col);
    if (ti < 0)
        throw std::out_of_range("group_by_dynamic: no column named " +
                                time_col);
    Series tcol = flat_copy(b.columns[static_cast<std::size_t>(ti)]);
    if (tcol.type() != TypeId::Int64)
        throw std::invalid_argument("group_by_dynamic: " + time_col +
                                    " must be an Int64 column");
    const std::int64_t n = tcol.length();
    const std::int64_t* t = tcol.data<std::int64_t>();

    // `origin_min` aligns the grid to the minimum time value (buckets begin
    // exactly at min ts), the frame-native analogue of time_bucket("min").
    if (origin_min) {
        bool seen = false;
        for (std::int64_t i = 0; i < n; ++i)
            if (!tcol.is_null(i)) {
                if (!seen || t[i] < origin) origin = t[i];
                seen = true;
            }
        if (!seen) origin = 0;
    }

    // The window grid is `origin + k*every`; the first non-null time picks the
    // first window (origin defaults to 0 = the classic ts-floored grid).
    std::int64_t start0 = 0;
    bool have_anchor = false;
    for (std::int64_t i = 0; i < n; ++i)
        if (!tcol.is_null(i)) {
            start0 = origin + floor_to_multiple(t[i] - origin, every);
            have_anchor = true;
            break;
        }

    // (window start, source row) pairs: a row belongs to every window whose
    // start S = start0 + k*every satisfies S <= t < S + period.
    std::vector<std::int64_t> keyv;
    std::vector<std::int64_t> rowsv;
    if (have_anchor) {
        for (std::int64_t i = 0; i < n; ++i) {
            if (tcol.is_null(i)) continue;
            const std::int64_t ts = t[i];
            std::int64_t k = (ts - start0) / every;  // largest S <= ts
            for (; k >= 0; --k) {
                std::int64_t s = start0 + k * every;
                if (s <= ts - period) break;  // smaller k only makes s smaller
                keyv.push_back(s);
                rowsv.push_back(i);
            }
        }
    }

    const std::int64_t pairs = static_cast<std::int64_t>(keyv.size());
    Series keyc = Series::flat_i64(keyv.data(), pairs);

    // Lower aggs to specs + deduped value columns, gathered at the emitted rows
    // (mirrors dfops::group_by's lowering).
    std::vector<Series> gathered;
    std::vector<AggSpec> specs;
    std::map<std::string, std::int32_t> col_idx;
    specs.reserve(aggs.size());
    for (const GroupAgg& a : aggs) {
        AggSpec sp;
        sp.op = agg_op_of(to_string(a.op));
        sp.out = a.out;
        sp.param = a.param;
        if (sp.op == AggOp::Count) {
            sp.value_col = -1;
        } else {
            auto it = col_idx.find(a.column);
            if (it != col_idx.end()) {
                sp.value_col = it->second;
            } else {
                std::int64_t vidx = index_of(b, a.column);
                if (vidx < 0)
                    throw std::out_of_range(
                        "group_by_dynamic: no column named " + a.column);
                sp.value_col = static_cast<std::int32_t>(gathered.size());
                gathered.push_back(
                    b.columns[static_cast<std::size_t>(vidx)].take(rowsv));
                col_idx.emplace(a.column, sp.value_col);
            }
        }
        specs.push_back(std::move(sp));
    }
    std::vector<const Series*> values;
    values.reserve(gathered.size());
    for (const Series& g : gathered) values.push_back(&g);

    DataFrame res = group_agg(keyc, values, std::move(specs), time_col);
    return sort_by(res, time_col, false);
}

}  // namespace dftracer::utils::dataframe
