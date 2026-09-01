#include <dftracer/utils/core/common/hash/hash_combine.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/field_stat.h>
#include <dftracer/utils/dataframe/internal/column_read.h>  // read_i64/u64/f64
#include <dftracer/utils/dataframe/parallel.h>              // parallel_for
#include <dftracer/utils/dataframe/sketch.h>  // DDSketch, sketch_bucket_keys

#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <set>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dftracer::utils::dataframe {

namespace {

// The value domain a column accumulates in - fixes both which FieldStat::add
// overload runs (keeping integers exact) and the finalized Sum/Min/Max column
// type. Derived once from the column type, never per row, so a group with no
// present rows still lands in the same output column as its populated peers.
FieldStatDomain col_domain(TypeId t) {
    switch (t) {
        case TypeId::Uint8:
        case TypeId::Uint16:
        case TypeId::Uint32:
        case TypeId::Uint64:
            return FieldStatDomain::U64;
        case TypeId::Float32:
        case TypeId::Float64:
            return FieldStatDomain::F64;
        default:
            return FieldStatDomain::I64;
    }
}

// Raw value bits for a fixed-width cell, reinterpreted on finalize by the
// column's domain. Keeps first/last exact for every numeric type.
std::uint64_t read_bits(const Series& c, std::int64_t i, FieldStatDomain d) {
    switch (d) {
        case FieldStatDomain::F64:
            return std::bit_cast<std::uint64_t>(read_f64(c, i));
        case FieldStatDomain::U64:
            return read_u64(c, i);
        default:
            return std::bit_cast<std::uint64_t>(read_i64(c, i));
    }
}

double read_as_double(const Series& c, std::int64_t i, FieldStatDomain d) {
    switch (d) {
        case FieldStatDomain::F64:
            return read_f64(c, i);
        case FieldStatDomain::U64:
            return static_cast<double>(read_u64(c, i));
        default:
            return static_cast<double>(read_i64(c, i));
    }
}

void fs_add(FieldStat& fs, const Series& c, std::int64_t i, FieldStatDomain d) {
    switch (d) {
        case FieldStatDomain::F64:
            fs.add(read_f64(c, i));
            break;
        case FieldStatDomain::U64:
            fs.add(read_u64(c, i));
            break;
        default:
            fs.add(read_i64(c, i));
    }
}

// Generic string form of a cell, for ArgMax's repr and SetUnion's values:
// matches the View's to_str (std::to_string for numbers, the raw text for a
// String column) so the two engines agree on ArgMax/SetUnion output.
std::string cell_repr(const Series& c, std::int64_t i) {
    if (c.type() == TypeId::String) return std::string(c.string_at(i));
    switch (col_domain(c.type())) {
        case FieldStatDomain::F64:
            return std::to_string(read_f64(c, i));
        case FieldStatDomain::U64:
            return std::to_string(read_u64(c, i));
        default:
            return std::to_string(read_i64(c, i));
    }
}

// Separator joining a SetUnion group's distinct values into one text cell;
// matches views/view_aggregate.h SET_SEP so the two engines agree.
constexpr char AGG_SET_SEP = '\x1e';

}  // namespace

// Groups fold by distinct value column, not by spec: sum(dur), mean(dur) and
// var(dur) share one FieldStat per group (accumulated once). Count is
// field-independent (group rows).
class AggState {
   public:
    std::vector<AggSpec> specs;
    std::vector<int> spec_field;         // spec -> field index, or -1 (Count)
    std::vector<std::int32_t> field_vc;  // field -> value_col
    std::vector<FieldStatDomain> field_domain;  // field -> accumulation domain
    std::size_t nf = 0;
    bool inited = false;

    // Composite key over N columns: each column keeps its own type (Int64
    // stays Int64, String stays String) - `key_is_str[k]` selects which of
    // `ikey_cols[k]` / `skey_cols[k]` holds column k's per-group values. Groups
    // are located by a combined hash of the N cells (hash_combine, no string
    // concatenation) with a column-by-column equality check on collision.
    std::size_t nkeys = 0;
    std::vector<char> key_is_str;                      // nkeys
    std::vector<std::vector<std::int64_t>> ikey_cols;  // nkeys * ngroups
    std::vector<std::vector<std::string>> skey_cols;   // nkeys * ngroups
    std::unordered_map<std::uint64_t, std::vector<std::int64_t>> key_buckets;
    std::vector<std::uint64_t> counts;                 // per group: rows seen
    std::vector<FieldStat> fstats;                     // groups * nf

    // First/Last state, allocated (groups * nf) only when `has_fl`. First/Last
    // are order-independent: each keeps the value at the smallest / largest
    // global row index seen for the group, so parallel-chunk merge order does
    // not matter. `fl_*_idx == -1` means no non-null value yet.
    bool has_fl = false;
    std::vector<char> field_is_str;       // field -> value column is String
    std::vector<std::uint64_t> fl_first;  // fixed-width raw bits
    std::vector<std::uint64_t> fl_last;
    std::vector<std::string> fl_first_s;  // string values
    std::vector<std::string> fl_last_s;
    std::vector<std::int64_t> fl_first_idx;
    std::vector<std::int64_t> fl_last_idx;

    // Per-group DDSketch state for Pct, allocated (groups * n_sketch) only when
    // `has_sketch`. All Pct specs on one field share a single sketch slot (the
    // quantiles read the same sketch); `field_sketch[field] == -1` marks a
    // field with no sketch.
    bool has_sketch = false;
    std::size_t n_sketch = 0;
    std::vector<int> field_sketch;
    std::vector<DDSketch> sketches;

    // ArgMax: one slot per ArgMax spec (no field-level dedup, mirroring the
    // View's AggSchema - a slot tracks the max `by_col` seen and the repr of
    // `value_col` at that row). `argmax_by_col`/`argmax_val_col` are raw
    // indices into the `values` array passed to accumulate.
    bool has_argmax = false;
    std::size_t n_argmax = 0;
    std::vector<int> spec_argmax;   // spec -> argmax slot, or -1
    std::vector<std::int32_t> argmax_by_col;
    std::vector<std::int32_t> argmax_val_col;
    std::vector<double> argmax_by;  // groups * n_argmax
    std::vector<char> argmax_has;   // groups * n_argmax
    std::vector<std::string> argmax_repr;

    // SetUnion: one slot per SetUnion spec (no field-level dedup, mirroring the
    // View's AggSchema). `set_val_col` is a raw index into `values`.
    bool has_set = false;
    std::size_t n_set = 0;
    std::vector<int> spec_set;                // spec -> set slot, or -1
    std::vector<std::int32_t> set_val_col;
    std::vector<std::set<std::string>> sets;  // groups * n_set

    std::size_t nspecs() const { return specs.size(); }
    std::int64_t ngroups() const {
        return static_cast<std::int64_t>(counts.size());
    }

    // The distinct-field layout is a pure function of the specs.
    void init_layout() {
        spec_field.assign(specs.size(), -1);
        field_vc.clear();
        std::unordered_map<std::int32_t, int> seen;
        for (std::size_t s = 0; s < specs.size(); ++s) {
            const AggSpec& sp = specs[s];
            if (sp.op == AggOp::Count || sp.op == AggOp::ArgMax ||
                sp.op == AggOp::SetUnion || sp.value_col < 0)
                continue;
            auto it = seen.find(sp.value_col);
            if (it == seen.end()) {
                int fi = static_cast<int>(field_vc.size());
                seen.emplace(sp.value_col, fi);
                field_vc.push_back(sp.value_col);
                spec_field[s] = fi;
            } else {
                spec_field[s] = it->second;
            }
        }
        nf = field_vc.size();
        has_fl = false;
        for (const AggSpec& sp : specs)
            if (sp.op == AggOp::First || sp.op == AggOp::Last) has_fl = true;

        // One shared sketch slot per field that any Pct or Hist spec
        // references (all quantiles and the histogram read the same sketch).
        field_sketch.assign(nf, -1);
        n_sketch = 0;
        for (std::size_t s = 0; s < specs.size(); ++s) {
            if (specs[s].op != AggOp::Pct && specs[s].op != AggOp::Hist)
                continue;
            const int fi = spec_field[s];
            if (fi >= 0 && field_sketch[static_cast<std::size_t>(fi)] < 0)
                field_sketch[static_cast<std::size_t>(fi)] =
                    static_cast<int>(n_sketch++);
        }
        has_sketch = n_sketch > 0;

        spec_argmax.assign(specs.size(), -1);
        argmax_by_col.clear();
        argmax_val_col.clear();
        for (std::size_t s = 0; s < specs.size(); ++s) {
            if (specs[s].op != AggOp::ArgMax) continue;
            spec_argmax[s] = static_cast<int>(n_argmax++);
            argmax_by_col.push_back(specs[s].by_col);
            argmax_val_col.push_back(specs[s].value_col);
        }
        has_argmax = n_argmax > 0;

        spec_set.assign(specs.size(), -1);
        set_val_col.clear();
        for (std::size_t s = 0; s < specs.size(); ++s) {
            if (specs[s].op != AggOp::SetUnion) continue;
            spec_set[s] = static_cast<int>(n_set++);
            set_val_col.push_back(specs[s].value_col);
        }
        has_set = n_set > 0;
    }

    void grow_group() {
        counts.push_back(0);
        fstats.resize(fstats.size() + nf);
        if (has_fl) {
            fl_first.resize(fl_first.size() + nf, 0);
            fl_last.resize(fl_last.size() + nf, 0);
            fl_first_s.resize(fl_first_s.size() + nf);
            fl_last_s.resize(fl_last_s.size() + nf);
            fl_first_idx.resize(fl_first_idx.size() + nf, -1);
            fl_last_idx.resize(fl_last_idx.size() + nf, -1);
        }
        if (has_sketch) sketches.resize(sketches.size() + n_sketch);
        if (has_argmax) {
            argmax_by.resize(argmax_by.size() + n_argmax, 0.0);
            argmax_has.resize(argmax_has.size() + n_argmax, 0);
            argmax_repr.resize(argmax_repr.size() + n_argmax);
        }
        if (has_set) sets.resize(sets.size() + n_set);
    }
    // Shared lookup for both live rows (get_int/get_str read a Series cell) and
    // merge (they read another AggState's already-materialized key columns):
    // hash the N cells, probe the bucket for an exact column-by-column match,
    // else append a new group.
    template <class GetInt, class GetStr>
    std::int64_t find_or_add_group(GetInt&& get_int, GetStr&& get_str) {
        std::size_t h = 0;
        for (std::size_t k = 0; k < nkeys; ++k) {
            const std::size_t cv =
                key_is_str[k] ? std::hash<std::string_view>{}(get_str(k))
                              : static_cast<std::size_t>(get_int(k));
            dftracer::utils::hash_combine(h, cv);
        }
        auto it = key_buckets.find(static_cast<std::uint64_t>(h));
        if (it != key_buckets.end()) {
            for (std::int64_t g : it->second) {
                bool match = true;
                for (std::size_t k = 0; k < nkeys && match; ++k)
                    match = key_is_str[k]
                                ? (skey_cols[k][static_cast<std::size_t>(g)] ==
                                   get_str(k))
                                : (ikey_cols[k][static_cast<std::size_t>(g)] ==
                                   get_int(k));
                if (match) return g;
            }
        }
        const std::int64_t g = ngroups();
        for (std::size_t k = 0; k < nkeys; ++k) {
            if (key_is_str[k])
                skey_cols[k].emplace_back(get_str(k));
            else
                ikey_cols[k].push_back(get_int(k));
        }
        key_buckets[static_cast<std::uint64_t>(h)].push_back(g);
        grow_group();
        return g;
    }
    std::int64_t group_of(const std::vector<const Series*>& keys,
                          std::int64_t i) {
        return find_or_add_group(
            [&](std::size_t k) { return read_i64(*keys[k], i); },
            [&](std::size_t k) -> std::string_view {
                return keys[k]->string_at(i);
            });
    }
    std::int64_t group_of_other(const AggState& other, std::int64_t j) {
        return find_or_add_group(
            [&](std::size_t k) {
                return other.ikey_cols[k][static_cast<std::size_t>(j)];
            },
            [&](std::size_t k) -> std::string_view {
                return other.skey_cols[k][static_cast<std::size_t>(j)];
            });
    }

    // Recompute key_buckets from already-populated ikey_cols/skey_cols (after
    // deserialize), without re-appending group storage.
    void rebuild_key_buckets(std::int64_t ng) {
        key_buckets.clear();
        for (std::int64_t g = 0; g < ng; ++g) {
            std::size_t h = 0;
            for (std::size_t k = 0; k < nkeys; ++k) {
                const std::size_t cv =
                    key_is_str[k]
                        ? std::hash<std::string_view>{}(
                              skey_cols[k][static_cast<std::size_t>(g)])
                        : static_cast<std::size_t>(
                              ikey_cols[k][static_cast<std::size_t>(g)]);
                dftracer::utils::hash_combine(h, cv);
            }
            key_buckets[static_cast<std::uint64_t>(h)].push_back(g);
        }
    }
};

void AggStateDeleter::operator()(AggState* p) const noexcept { delete p; }

AggStatePtr agg_new(std::vector<AggSpec> specs) {
    AggStatePtr s(new AggState());
    s->specs = std::move(specs);
    s->init_layout();
    return s;
}

void agg_accumulate(AggState& st, const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    std::int64_t begin, std::int64_t end) {
    if (!st.inited) {
        st.nkeys = keys.size();
        st.key_is_str.resize(st.nkeys);
        st.ikey_cols.resize(st.nkeys);
        st.skey_cols.resize(st.nkeys);
        for (std::size_t k = 0; k < st.nkeys; ++k)
            st.key_is_str[k] = keys[k]->type() == TypeId::String ? 1 : 0;
        st.field_domain.resize(st.nf);
        st.field_is_str.resize(st.nf);
        for (std::size_t fj = 0; fj < st.nf; ++fj) {
            const Series* vc =
                values[static_cast<std::size_t>(st.field_vc[fj])];
            st.field_domain[fj] = col_domain(vc->type());
            st.field_is_str[fj] = vc->type() == TypeId::String ? 1 : 0;
        }
        st.inited = true;
    }
    if (end < 0) end = keys.empty() ? 0 : keys[0]->length();
    const std::int64_t clen = end - begin;

    // Precompute DDSketch bucket keys for each sketch field in one SIMD pass
    // over the chunk (the logarithm is the costly step); the per-group scatter
    // below just increments a bin. Null rows get a garbage key that the scatter
    // skips.
    std::vector<std::vector<std::int32_t>> chunk_keys;
    if (st.has_sketch && clen > 0) {
        const double lg = DDSketch{}.log_gamma();
        std::vector<double> tmp(static_cast<std::size_t>(clen));
        chunk_keys.assign(st.n_sketch, {});
        for (std::size_t fj = 0; fj < st.nf; ++fj) {
            const int sk = st.field_sketch[fj];
            if (sk < 0) continue;
            const Series* vc =
                values[static_cast<std::size_t>(st.field_vc[fj])];
            for (std::int64_t r = 0; r < clen; ++r)
                tmp[static_cast<std::size_t>(r)] =
                    read_as_double(*vc, begin + r, st.field_domain[fj]);
            chunk_keys[static_cast<std::size_t>(sk)].resize(
                static_cast<std::size_t>(clen));
            sketch_bucket_keys(tmp.data(), clen, lg,
                               chunk_keys[static_cast<std::size_t>(sk)].data());
        }
    }

    for (std::int64_t i = begin; i < end; ++i) {
        std::int64_t g = st.group_of(keys, i);
        st.counts[static_cast<std::size_t>(g)]++;
        const std::size_t base = static_cast<std::size_t>(g) * st.nf;
        for (std::size_t fj = 0; fj < st.nf; ++fj) {
            const Series* vc =
                values[static_cast<std::size_t>(st.field_vc[fj])];
            if (vc->is_null(i)) continue;
            if (!st.field_is_str[fj])
                fs_add(st.fstats[base + fj], *vc, i, st.field_domain[fj]);
            if (st.has_sketch && st.field_sketch[fj] >= 0) {
                const std::size_t sk =
                    static_cast<std::size_t>(st.field_sketch[fj]);
                st.sketches[static_cast<std::size_t>(g) * st.n_sketch + sk]
                    .add_key(
                        chunk_keys[sk][static_cast<std::size_t>(i - begin)]);
            }
            if (st.has_fl) {
                const std::size_t sl = base + fj;
                if (st.fl_first_idx[sl] < 0 || i < st.fl_first_idx[sl]) {
                    st.fl_first_idx[sl] = i;
                    if (st.field_is_str[fj])
                        st.fl_first_s[sl] = std::string(vc->string_at(i));
                    else
                        st.fl_first[sl] =
                            read_bits(*vc, i, st.field_domain[fj]);
                }
                if (i > st.fl_last_idx[sl]) {
                    st.fl_last_idx[sl] = i;
                    if (st.field_is_str[fj])
                        st.fl_last_s[sl] = std::string(vc->string_at(i));
                    else
                        st.fl_last[sl] = read_bits(*vc, i, st.field_domain[fj]);
                }
            }
        }
        if (st.has_argmax) {
            for (std::size_t slot = 0; slot < st.n_argmax; ++slot) {
                const Series* byc =
                    values[static_cast<std::size_t>(st.argmax_by_col[slot])];
                if (byc->is_null(i)) continue;
                const double byv =
                    read_as_double(*byc, i, col_domain(byc->type()));
                const std::size_t as =
                    static_cast<std::size_t>(g) * st.n_argmax + slot;
                if (!st.argmax_has[as] || byv > st.argmax_by[as]) {
                    st.argmax_by[as] = byv;
                    st.argmax_has[as] = 1;
                    const Series* vc = values[static_cast<std::size_t>(
                        st.argmax_val_col[slot])];
                    st.argmax_repr[as] = cell_repr(*vc, i);
                }
            }
        }
        if (st.has_set) {
            for (std::size_t slot = 0; slot < st.n_set; ++slot) {
                const Series* vc =
                    values[static_cast<std::size_t>(st.set_val_col[slot])];
                if (vc->is_null(i)) continue;
                std::string v = cell_repr(*vc, i);
                if (!v.empty())
                    st.sets[static_cast<std::size_t>(g) * st.n_set + slot]
                        .insert(std::move(v));
            }
        }
    }
}

void agg_accumulate(AggState& st, const Series& key,
                    const std::vector<const Series*>& values,
                    std::int64_t begin, std::int64_t end) {
    const std::vector<const Series*> keys{&key};
    agg_accumulate(st, keys, values, begin, end);
}

void agg_merge(AggState& into, const AggState& other) {
    if (!other.inited) return;
    if (!into.inited) {
        into.specs = other.specs;
        into.spec_field = other.spec_field;
        into.field_vc = other.field_vc;
        into.field_domain = other.field_domain;
        into.field_is_str = other.field_is_str;
        into.nf = other.nf;
        into.has_fl = other.has_fl;
        into.has_sketch = other.has_sketch;
        into.n_sketch = other.n_sketch;
        into.field_sketch = other.field_sketch;
        into.has_argmax = other.has_argmax;
        into.n_argmax = other.n_argmax;
        into.spec_argmax = other.spec_argmax;
        into.argmax_by_col = other.argmax_by_col;
        into.argmax_val_col = other.argmax_val_col;
        into.has_set = other.has_set;
        into.n_set = other.n_set;
        into.spec_set = other.spec_set;
        into.set_val_col = other.set_val_col;
        into.nkeys = other.nkeys;
        into.key_is_str = other.key_is_str;
        into.ikey_cols.assign(into.nkeys, {});
        into.skey_cols.assign(into.nkeys, {});
        into.inited = true;
    }
    const std::int64_t og = other.ngroups();
    for (std::int64_t j = 0; j < og; ++j) {
        std::int64_t g = into.group_of_other(other, j);
        into.counts[static_cast<std::size_t>(g)] +=
            other.counts[static_cast<std::size_t>(j)];
        const std::size_t db = static_cast<std::size_t>(g) * into.nf;
        const std::size_t sb = static_cast<std::size_t>(j) * into.nf;
        for (std::size_t fj = 0; fj < into.nf; ++fj)
            into.fstats[db + fj].merge(other.fstats[sb + fj]);
        if (into.has_fl) {
            for (std::size_t fj = 0; fj < into.nf; ++fj) {
                const std::size_t d = db + fj, s = sb + fj;
                const std::int64_t ofi = other.fl_first_idx[s];
                if (ofi >= 0 &&
                    (into.fl_first_idx[d] < 0 || ofi < into.fl_first_idx[d])) {
                    into.fl_first_idx[d] = ofi;
                    into.fl_first[d] = other.fl_first[s];
                    into.fl_first_s[d] = other.fl_first_s[s];
                }
                const std::int64_t oli = other.fl_last_idx[s];
                if (oli > into.fl_last_idx[d]) {
                    into.fl_last_idx[d] = oli;
                    into.fl_last[d] = other.fl_last[s];
                    into.fl_last_s[d] = other.fl_last_s[s];
                }
            }
        }
        if (into.has_sketch) {
            const std::size_t ds = static_cast<std::size_t>(g) * into.n_sketch;
            const std::size_t ss = static_cast<std::size_t>(j) * into.n_sketch;
            for (std::size_t sk = 0; sk < into.n_sketch; ++sk)
                into.sketches[ds + sk].merge(other.sketches[ss + sk]);
        }
        if (into.has_argmax) {
            const std::size_t da = static_cast<std::size_t>(g) * into.n_argmax;
            const std::size_t sa = static_cast<std::size_t>(j) * into.n_argmax;
            for (std::size_t slot = 0; slot < into.n_argmax; ++slot) {
                if (!other.argmax_has[sa + slot]) continue;
                if (!into.argmax_has[da + slot] ||
                    other.argmax_by[sa + slot] > into.argmax_by[da + slot]) {
                    into.argmax_by[da + slot] = other.argmax_by[sa + slot];
                    into.argmax_has[da + slot] = 1;
                    into.argmax_repr[da + slot] = other.argmax_repr[sa + slot];
                }
            }
        }
        if (into.has_set) {
            const std::size_t ds = static_cast<std::size_t>(g) * into.n_set;
            const std::size_t ss = static_cast<std::size_t>(j) * into.n_set;
            for (std::size_t slot = 0; slot < into.n_set; ++slot)
                into.sets[ds + slot].insert(other.sets[ss + slot].begin(),
                                            other.sets[ss + slot].end());
        }
    }
}

DataFrame agg_finalize(const AggState& st,
                       const std::vector<std::string>& key_names) {
    const std::int64_t ng = st.ngroups();
    const std::size_t ns = st.nspecs();
    DataFrame out;
    for (std::size_t k = 0; k < st.nkeys; ++k) {
        out.names.push_back(k < key_names.size() ? key_names[k]
                                                 : "key" + std::to_string(k));
        if (st.key_is_str[k])
            out.columns.push_back(Series::strings(st.skey_cols[k]));
        else
            out.columns.push_back(Series::flat_i64(st.ikey_cols[k].data(), ng));
    }

    for (std::size_t s = 0; s < ns; ++s) {
        const AggSpec& sp = st.specs[s];
        const int fi = st.spec_field[s];
        out.names.push_back(sp.out);
        auto fs_at = [&](std::int64_t g) -> const FieldStat& {
            return st.fstats[static_cast<std::size_t>(g) * st.nf +
                             static_cast<std::size_t>(fi)];
        };
        const FieldStatDomain dom =
            fi >= 0 ? st.field_domain[static_cast<std::size_t>(fi)]
                    : FieldStatDomain::I64;

        if (sp.op == AggOp::Pct) {
            const int sk =
                fi >= 0 ? st.field_sketch[static_cast<std::size_t>(fi)] : -1;
            std::vector<double> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] =
                    sk >= 0 ? st.sketches[static_cast<std::size_t>(g) *
                                              st.n_sketch +
                                          static_cast<std::size_t>(sk)]
                                  .quantile(sp.param)
                            : 0.0;
            out.columns.push_back(Series::flat_f64(v.data(), ng));
        } else if (sp.op == AggOp::Hist) {
            // One list<struct{lo,hi,count}> row per group, from the shared
            // sketch's occupied bins (same shape the View emits).
            const int sk =
                fi >= 0 ? st.field_sketch[static_cast<std::size_t>(fi)] : -1;
            std::vector<std::int32_t> off{0};
            std::vector<double> lo, hi;
            std::vector<std::uint64_t> cnt;
            for (std::int64_t g = 0; g < ng; ++g) {
                if (sk >= 0) {
                    for (const auto& bn :
                         st.sketches[static_cast<std::size_t>(g) * st.n_sketch +
                                     static_cast<std::size_t>(sk)]
                             .bins()) {
                        lo.push_back(bn.lower);
                        hi.push_back(bn.upper);
                        cnt.push_back(bn.count);
                    }
                }
                off.push_back(static_cast<std::int32_t>(lo.size()));
            }
            const std::int64_t nb = static_cast<std::int64_t>(lo.size());
            std::vector<Series> fields;
            fields.push_back(Series::flat(TypeId::Float64, lo.data(), nb));
            fields.push_back(Series::flat(TypeId::Float64, hi.data(), nb));
            fields.push_back(Series::flat(TypeId::Uint64, cnt.data(), nb));
            out.columns.push_back(Series::list(
                off,
                Series::structs({"lo", "hi", "count"}, std::move(fields))));
        } else if (sp.op == AggOp::First || sp.op == AggOp::Last) {
            const bool first = sp.op == AggOp::First;
            const std::vector<std::uint64_t>& bits =
                first ? st.fl_first : st.fl_last;
            const std::vector<std::string>& strs =
                first ? st.fl_first_s : st.fl_last_s;
            auto slot = [&](std::int64_t g) {
                return static_cast<std::size_t>(g) * st.nf +
                       static_cast<std::size_t>(fi);
            };
            if (fi >= 0 && st.field_is_str[static_cast<std::size_t>(fi)]) {
                std::vector<std::string> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] = strs[slot(g)];
                out.columns.push_back(Series::strings(v));
            } else if (dom == FieldStatDomain::F64) {
                std::vector<double> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] =
                        std::bit_cast<double>(bits[slot(g)]);
                out.columns.push_back(Series::flat_f64(v.data(), ng));
            } else if (dom == FieldStatDomain::U64) {
                std::vector<std::uint64_t> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] = bits[slot(g)];
                out.columns.push_back(
                    Series::flat(TypeId::Uint64, v.data(), ng));
            } else {
                std::vector<std::int64_t> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] =
                        std::bit_cast<std::int64_t>(bits[slot(g)]);
                out.columns.push_back(Series::flat_i64(v.data(), ng));
            }
        } else if (sp.op == AggOp::SumSq) {
            std::vector<double> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] = fs_at(g).sumsq;
            out.columns.push_back(Series::flat_f64(v.data(), ng));
        } else if (sp.op == AggOp::ArgMax) {
            const int slot = st.spec_argmax[s];
            std::vector<std::string> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g) {
                if (slot < 0) continue;
                const std::size_t as =
                    static_cast<std::size_t>(g) * st.n_argmax +
                    static_cast<std::size_t>(slot);
                if (st.argmax_has[as])
                    v[static_cast<std::size_t>(g)] = st.argmax_repr[as];
            }
            out.columns.push_back(Series::strings(v));
        } else if (sp.op == AggOp::SetUnion) {
            const int slot = st.spec_set[s];
            std::vector<std::string> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g) {
                std::string joined;
                if (slot >= 0) {
                    const std::set<std::string>& gset =
                        st.sets[static_cast<std::size_t>(g) * st.n_set +
                                static_cast<std::size_t>(slot)];
                    for (const std::string& x : gset) {
                        if (!joined.empty()) joined.push_back(AGG_SET_SEP);
                        joined += x;
                    }
                }
                v[static_cast<std::size_t>(g)] = std::move(joined);
            }
            out.columns.push_back(Series::strings(v));
        } else if (sp.op == AggOp::Count) {
            std::vector<std::int64_t> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] = static_cast<std::int64_t>(
                    st.counts[static_cast<std::size_t>(g)]);
            out.columns.push_back(Series::flat_i64(v.data(), ng));
        } else if (sp.op == AggOp::Sum || sp.op == AggOp::Min ||
                   sp.op == AggOp::Max) {
            auto exact = [&](const FieldStat& f) -> std::int64_t {
                return sp.op == AggOp::Sum   ? f.esum
                       : sp.op == AggOp::Min ? f.emin
                                             : f.emax;
            };
            auto approx = [&](const FieldStat& f) -> double {
                return sp.op == AggOp::Sum   ? f.sum
                       : sp.op == AggOp::Min ? f.min
                                             : f.max;
            };
            if (dom == FieldStatDomain::U64) {
                std::vector<std::uint64_t> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] =
                        std::bit_cast<std::uint64_t>(exact(fs_at(g)));
                out.columns.push_back(
                    Series::flat(TypeId::Uint64, v.data(), ng));
            } else if (dom == FieldStatDomain::I64) {
                std::vector<std::int64_t> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] = exact(fs_at(g));
                out.columns.push_back(Series::flat_i64(v.data(), ng));
            } else {
                std::vector<double> v(static_cast<std::size_t>(ng));
                for (std::int64_t g = 0; g < ng; ++g)
                    v[static_cast<std::size_t>(g)] = approx(fs_at(g));
                out.columns.push_back(Series::flat_f64(v.data(), ng));
            }
        } else {  // Mean/Var/Std/Skew/Kurt -> Float64
            std::vector<double> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g) {
                const FieldStat& f = fs_at(g);
                double r = 0.0;
                switch (sp.op) {
                    case AggOp::Mean:
                        r = f.mean();
                        break;
                    case AggOp::Var:
                        r = f.variance(true);
                        break;
                    case AggOp::Std:
                        r = f.stddev(true);
                        break;
                    case AggOp::Skew:
                        r = f.skewness();
                        break;
                    default:
                        r = f.kurtosis();
                }
                v[static_cast<std::size_t>(g)] = r;
            }
            out.columns.push_back(Series::flat_f64(v.data(), ng));
        }
    }
    return out;
}

DataFrame agg_finalize(const AggState& st, const std::string& key_name) {
    return agg_finalize(st, std::vector<std::string>{key_name});
}

std::int64_t agg_num_groups(const AggState& st) { return st.ngroups(); }

std::size_t agg_approx_bytes(const AggState& st) {
    std::size_t total = 0;
    for (std::size_t k = 0; k < st.nkeys; ++k) {
        if (st.key_is_str[k])
            for (const std::string& v : st.skey_cols[k])
                total += v.size() + sizeof(std::string);
        else
            total += st.ikey_cols[k].size() * sizeof(std::int64_t);
    }
    total += st.counts.size() * sizeof(std::uint64_t);
    total += st.fstats.size() * sizeof(FieldStat);
    if (st.has_fl) {
        total +=
            (st.fl_first.size() + st.fl_last.size()) * sizeof(std::uint64_t);
        total += (st.fl_first_idx.size() + st.fl_last_idx.size()) *
                 sizeof(std::int64_t);
        for (const std::string& v : st.fl_first_s) total += v.size();
        for (const std::string& v : st.fl_last_s) total += v.size();
    }
    if (st.has_sketch)
        for (const DDSketch& sk : st.sketches)
            total += sk.bins().size() * 24 + 64;
    if (st.has_argmax) {
        total += st.argmax_by.size() * sizeof(double) + st.argmax_has.size();
        for (const std::string& v : st.argmax_repr) total += v.size();
    }
    if (st.has_set)
        for (const std::set<std::string>& gset : st.sets)
            for (const std::string& v : gset) total += v.size() + 32;
    return total;
}

int agg_key_cmp(const AggState& a, std::int64_t ga, const AggState& b,
                std::int64_t gb) {
    for (std::size_t k = 0; k < a.nkeys; ++k) {
        if (a.key_is_str[k]) {
            const std::string& x = a.skey_cols[k][static_cast<std::size_t>(ga)];
            const std::string& y = b.skey_cols[k][static_cast<std::size_t>(gb)];
            if (x != y) return x < y ? -1 : 1;
        } else {
            const std::int64_t x = a.ikey_cols[k][static_cast<std::size_t>(ga)];
            const std::int64_t y = b.ikey_cols[k][static_cast<std::size_t>(gb)];
            if (x != y) return x < y ? -1 : 1;
        }
    }
    return 0;
}

namespace {
// Reorders `v` (blocks of `stride` elements per group) into `perm` order.
template <class T>
void permute_blocks(std::vector<T>& v, const std::vector<std::int64_t>& perm,
                    std::size_t stride) {
    if (stride == 0 || v.empty()) return;
    std::vector<T> tmp(v.size());
    for (std::size_t g = 0; g < perm.size(); ++g) {
        const std::size_t src = static_cast<std::size_t>(perm[g]) * stride;
        for (std::size_t j = 0; j < stride; ++j)
            tmp[g * stride + j] = std::move(v[src + j]);
    }
    v.swap(tmp);
}
}  // namespace

void agg_sort_groups(AggState& st) {
    const std::int64_t ng = st.ngroups();
    if (ng <= 1) return;
    std::vector<std::int64_t> perm(static_cast<std::size_t>(ng));
    std::iota(perm.begin(), perm.end(), std::int64_t{0});
    std::sort(perm.begin(), perm.end(), [&](std::int64_t x, std::int64_t y) {
        return agg_key_cmp(st, x, st, y) < 0;
    });

    for (std::size_t k = 0; k < st.nkeys; ++k) {
        if (st.key_is_str[k])
            permute_blocks(st.skey_cols[k], perm, 1);
        else
            permute_blocks(st.ikey_cols[k], perm, 1);
    }
    permute_blocks(st.counts, perm, 1);
    permute_blocks(st.fstats, perm, st.nf);
    if (st.has_fl) {
        permute_blocks(st.fl_first, perm, st.nf);
        permute_blocks(st.fl_last, perm, st.nf);
        permute_blocks(st.fl_first_s, perm, st.nf);
        permute_blocks(st.fl_last_s, perm, st.nf);
        permute_blocks(st.fl_first_idx, perm, st.nf);
        permute_blocks(st.fl_last_idx, perm, st.nf);
    }
    if (st.has_sketch) permute_blocks(st.sketches, perm, st.n_sketch);
    if (st.has_argmax) {
        permute_blocks(st.argmax_by, perm, st.n_argmax);
        permute_blocks(st.argmax_has, perm, st.n_argmax);
        permute_blocks(st.argmax_repr, perm, st.n_argmax);
    }
    if (st.has_set) permute_blocks(st.sets, perm, st.n_set);

    st.rebuild_key_buckets(ng);
}

AggStatePtr agg_extract_group(const AggState& st, std::int64_t g) {
    AggStatePtr out(new AggState());
    out->specs = st.specs;
    out->init_layout();
    out->nkeys = st.nkeys;
    out->key_is_str = st.key_is_str;
    out->ikey_cols.assign(out->nkeys, {});
    out->skey_cols.assign(out->nkeys, {});
    out->field_domain = st.field_domain;
    out->field_is_str = st.field_is_str;
    for (std::size_t k = 0; k < st.nkeys; ++k) {
        if (st.key_is_str[k])
            out->skey_cols[k].push_back(
                st.skey_cols[k][static_cast<std::size_t>(g)]);
        else
            out->ikey_cols[k].push_back(
                st.ikey_cols[k][static_cast<std::size_t>(g)]);
    }
    out->grow_group();
    out->inited = true;

    out->counts[0] = st.counts[static_cast<std::size_t>(g)];
    const std::size_t base = static_cast<std::size_t>(g) * st.nf;
    for (std::size_t fj = 0; fj < st.nf; ++fj)
        out->fstats[fj] = st.fstats[base + fj];
    if (st.has_fl) {
        for (std::size_t fj = 0; fj < st.nf; ++fj) {
            out->fl_first[fj] = st.fl_first[base + fj];
            out->fl_last[fj] = st.fl_last[base + fj];
            out->fl_first_s[fj] = st.fl_first_s[base + fj];
            out->fl_last_s[fj] = st.fl_last_s[base + fj];
            out->fl_first_idx[fj] = st.fl_first_idx[base + fj];
            out->fl_last_idx[fj] = st.fl_last_idx[base + fj];
        }
    }
    if (st.has_sketch) {
        const std::size_t sbase = static_cast<std::size_t>(g) * st.n_sketch;
        for (std::size_t sk = 0; sk < st.n_sketch; ++sk)
            out->sketches[sk] = st.sketches[sbase + sk];
    }
    if (st.has_argmax) {
        const std::size_t abase = static_cast<std::size_t>(g) * st.n_argmax;
        for (std::size_t slot = 0; slot < st.n_argmax; ++slot) {
            out->argmax_by[slot] = st.argmax_by[abase + slot];
            out->argmax_has[slot] = st.argmax_has[abase + slot];
            out->argmax_repr[slot] = st.argmax_repr[abase + slot];
        }
    }
    if (st.has_set) {
        const std::size_t setbase = static_cast<std::size_t>(g) * st.n_set;
        for (std::size_t slot = 0; slot < st.n_set; ++slot)
            out->sets[slot] = st.sets[setbase + slot];
    }
    return out;
}

namespace {

template <class T>
void put(std::string& s, T v) {
    s.append(reinterpret_cast<const char*>(&v), sizeof(T));
}
void put_bytes(std::string& s, const std::string& b) {
    put(s, static_cast<std::uint64_t>(b.size()));
    s.append(b);
}
struct Reader {
    const char* p;
    template <class T>
    T get() {
        T v;
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string get_bytes() {
        std::uint64_t n = get<std::uint64_t>();
        std::string r(p, n);
        p += n;
        return r;
    }
};

constexpr std::int64_t AGG_GRAIN = 1 << 16;

}  // namespace

std::string agg_serialize(const AggState& st) {
    std::string s;
    put(s, static_cast<std::uint32_t>(st.specs.size()));
    put(s, static_cast<std::uint32_t>(st.nkeys));
    for (char b : st.key_is_str) put(s, static_cast<std::uint8_t>(b));
    for (const AggSpec& sp : st.specs) {
        put(s, static_cast<std::int32_t>(sp.op));
        put(s, sp.value_col);
        put_bytes(s, sp.out);
        put(s, sp.param);
        put(s, sp.by_col);
    }
    put(s, static_cast<std::uint32_t>(st.nf));
    for (FieldStatDomain d : st.field_domain)
        put(s, static_cast<std::uint8_t>(d));
    put(s, static_cast<std::uint8_t>(st.has_fl ? 1 : 0));
    if (st.has_fl)
        for (std::size_t fj = 0; fj < st.nf; ++fj)
            put(s, static_cast<std::uint8_t>(
                       fj < st.field_is_str.size() ? st.field_is_str[fj] : 0));
    const std::int64_t ng = st.ngroups();
    put(s, ng);
    for (std::size_t k = 0; k < st.nkeys; ++k) {
        if (st.key_is_str[k])
            for (const std::string& v : st.skey_cols[k]) put_bytes(s, v);
        else
            for (std::int64_t v : st.ikey_cols[k]) put(s, v);
    }
    for (std::uint64_t c : st.counts) put(s, c);
    for (const FieldStat& f : st.fstats) put(s, f);
    if (st.has_fl) {
        for (std::uint64_t b : st.fl_first) put(s, b);
        for (std::uint64_t b : st.fl_last) put(s, b);
        for (std::int64_t x : st.fl_first_idx) put(s, x);
        for (std::int64_t x : st.fl_last_idx) put(s, x);
        for (const std::string& v : st.fl_first_s) put_bytes(s, v);
        for (const std::string& v : st.fl_last_s) put_bytes(s, v);
    }
    if (st.has_sketch) {
        std::vector<std::uint8_t> blob;
        for (const DDSketch& sk : st.sketches) {
            sk.serialize_into(blob);
            put(s, static_cast<std::uint32_t>(blob.size()));
            s.append(reinterpret_cast<const char*>(blob.data()), blob.size());
        }
    }
    put(s, static_cast<std::uint8_t>(st.has_argmax ? 1 : 0));
    if (st.has_argmax) {
        for (double b : st.argmax_by) put(s, b);
        for (char h : st.argmax_has) put(s, static_cast<std::uint8_t>(h));
        for (const std::string& r : st.argmax_repr) put_bytes(s, r);
    }
    put(s, static_cast<std::uint8_t>(st.has_set ? 1 : 0));
    if (st.has_set) {
        for (const std::set<std::string>& gset : st.sets) {
            put(s, static_cast<std::uint32_t>(gset.size()));
            for (const std::string& v : gset) put_bytes(s, v);
        }
    }
    return s;
}

AggStatePtr agg_deserialize(const std::string& blob) {
    Reader r{blob.data()};
    AggStatePtr st(new AggState());
    const std::uint32_t ns = r.get<std::uint32_t>();
    const std::uint32_t nkeys = r.get<std::uint32_t>();
    st->nkeys = nkeys;
    st->key_is_str.resize(nkeys);
    for (std::uint32_t k = 0; k < nkeys; ++k)
        st->key_is_str[k] = static_cast<char>(r.get<std::uint8_t>());
    st->specs.resize(ns);
    for (std::uint32_t i = 0; i < ns; ++i) {
        st->specs[i].op = static_cast<AggOp>(r.get<std::int32_t>());
        st->specs[i].value_col = r.get<std::int32_t>();
        st->specs[i].out = r.get_bytes();
        st->specs[i].param = r.get<double>();
        st->specs[i].by_col = r.get<std::int32_t>();
    }
    st->init_layout();
    const std::uint32_t nf = r.get<std::uint32_t>();
    st->field_domain.resize(nf);
    for (std::uint32_t i = 0; i < nf; ++i)
        st->field_domain[i] =
            static_cast<FieldStatDomain>(r.get<std::uint8_t>());
    st->has_fl = r.get<std::uint8_t>() != 0;
    if (st->has_fl) {
        st->field_is_str.resize(nf);
        for (std::uint32_t i = 0; i < nf; ++i)
            st->field_is_str[i] = static_cast<char>(r.get<std::uint8_t>());
    }
    const std::int64_t ng = r.get<std::int64_t>();
    st->ikey_cols.resize(nkeys);
    st->skey_cols.resize(nkeys);
    for (std::uint32_t k = 0; k < nkeys; ++k) {
        if (st->key_is_str[k]) {
            st->skey_cols[k].resize(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                st->skey_cols[k][static_cast<std::size_t>(g)] = r.get_bytes();
        } else {
            st->ikey_cols[k].resize(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                st->ikey_cols[k][static_cast<std::size_t>(g)] =
                    r.get<std::int64_t>();
        }
    }
    st->rebuild_key_buckets(ng);
    st->counts.resize(static_cast<std::size_t>(ng));
    for (std::int64_t g = 0; g < ng; ++g)
        st->counts[static_cast<std::size_t>(g)] = r.get<std::uint64_t>();
    st->fstats.resize(static_cast<std::size_t>(ng) * nf);
    for (std::size_t i = 0; i < st->fstats.size(); ++i)
        st->fstats[i] = r.get<FieldStat>();
    if (st->has_fl) {
        const std::size_t sz = static_cast<std::size_t>(ng) * nf;
        st->fl_first.resize(sz);
        st->fl_last.resize(sz);
        st->fl_first_idx.resize(sz);
        st->fl_last_idx.resize(sz);
        st->fl_first_s.resize(sz);
        st->fl_last_s.resize(sz);
        for (std::size_t i = 0; i < sz; ++i)
            st->fl_first[i] = r.get<std::uint64_t>();
        for (std::size_t i = 0; i < sz; ++i)
            st->fl_last[i] = r.get<std::uint64_t>();
        for (std::size_t i = 0; i < sz; ++i)
            st->fl_first_idx[i] = r.get<std::int64_t>();
        for (std::size_t i = 0; i < sz; ++i)
            st->fl_last_idx[i] = r.get<std::int64_t>();
        for (std::size_t i = 0; i < sz; ++i) st->fl_first_s[i] = r.get_bytes();
        for (std::size_t i = 0; i < sz; ++i) st->fl_last_s[i] = r.get_bytes();
    }
    if (st->has_sketch) {
        const std::size_t sz = static_cast<std::size_t>(ng) * st->n_sketch;
        st->sketches.reserve(sz);
        for (std::size_t i = 0; i < sz; ++i) {
            const std::uint32_t blen = r.get<std::uint32_t>();
            st->sketches.push_back(DDSketch::deserialize(
                reinterpret_cast<const std::uint8_t*>(r.p), blen));
            r.p += blen;
        }
    }
    st->has_argmax = r.get<std::uint8_t>() != 0;
    if (st->has_argmax) {
        const std::size_t sz = static_cast<std::size_t>(ng) * st->n_argmax;
        st->argmax_by.resize(sz);
        st->argmax_has.resize(sz);
        st->argmax_repr.resize(sz);
        for (std::size_t i = 0; i < sz; ++i) st->argmax_by[i] = r.get<double>();
        for (std::size_t i = 0; i < sz; ++i)
            st->argmax_has[i] = static_cast<char>(r.get<std::uint8_t>());
        for (std::size_t i = 0; i < sz; ++i) st->argmax_repr[i] = r.get_bytes();
    }
    st->has_set = r.get<std::uint8_t>() != 0;
    if (st->has_set) {
        const std::size_t sz = static_cast<std::size_t>(ng) * st->n_set;
        st->sets.resize(sz);
        for (std::size_t i = 0; i < sz; ++i) {
            const std::uint32_t cnt = r.get<std::uint32_t>();
            for (std::uint32_t j = 0; j < cnt; ++j)
                st->sets[i].insert(r.get_bytes());
        }
    }
    st->inited = true;
    return st;
}

DataFrame group_agg(const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs,
                    const std::vector<std::string>& key_names) {
    const std::int64_t n = keys.empty() ? 0 : keys[0]->length();
    if (n <= AGG_GRAIN) {
        auto st = agg_new(std::move(specs));
        agg_accumulate(*st, keys, values);
        return agg_finalize(*st, key_names);
    }
    // Parallel driver: each chunk folds into its own partial (no locks), then
    // the partials merge. parallel_for runs serial when no backend is
    // installed.
    const std::int64_t chunks = (n + AGG_GRAIN - 1) / AGG_GRAIN;
    std::vector<AggStatePtr> partials(static_cast<std::size_t>(chunks));
    parallel_for(n, AGG_GRAIN, [&](std::int64_t b, std::int64_t e) {
        auto st = agg_new(specs);
        agg_accumulate(*st, keys, values, b, e);
        partials[static_cast<std::size_t>(b / AGG_GRAIN)] = std::move(st);
    });
    auto acc = agg_new(std::move(specs));
    for (auto& p : partials)
        if (p) agg_merge(*acc, *p);
    return agg_finalize(*acc, key_names);
}

DataFrame group_agg(const Series& key, const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs, const std::string& key_name) {
    const std::vector<const Series*> keys{&key};
    return group_agg(keys, values, std::move(specs),
                     std::vector<std::string>{key_name});
}

}  // namespace dftracer::utils::dataframe
