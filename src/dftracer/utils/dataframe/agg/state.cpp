#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/dataframe/agg/detail.h>
#include <dftracer/utils/dataframe/internal/column_read.h>  // read_i64/u64/f64
#include <dftracer/utils/dataframe/sketch.h>  // DDSketch, sketch_bucket_keys

#include <bit>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

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

// Types group_of can key on: String via the text path, the rest read exactly
// by read_bits. Anything else would key every row on the same zero and
// collapse the batch into one group. No default, so a new TypeId lands here.
bool is_group_key_type(TypeId t) {
    switch (t) {
        case TypeId::Bool:
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
        case TypeId::Date32:
        case TypeId::Date64:
        case TypeId::Time32:
        case TypeId::Time64:
        case TypeId::Timestamp:
        case TypeId::Duration:
        case TypeId::String:
            return true;
        case TypeId::Unknown:
        case TypeId::Binary:
        case TypeId::List:
        case TypeId::Struct:
        case TypeId::Float16:
        case TypeId::Decimal128:
        case TypeId::Decimal256:
        case TypeId::FixedSizeBinary:
        case TypeId::LargeString:
        case TypeId::LargeBinary:
        case TypeId::LargeList:
        case TypeId::FixedSizeList:
        case TypeId::Map:
            return false;
    }
    return false;
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

namespace {

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

// Exact occupancy-endpoint read: an integer column keeps its raw u64/i64 so a
// timestamp above 2^53 stays exact; a Float64 column goes through double.
std::uint64_t read_u64_exact(const Series& c, std::int64_t i,
                             FieldStatDomain d) {
    switch (d) {
        case FieldStatDomain::F64:
            return static_cast<std::uint64_t>(read_f64(c, i));
        case FieldStatDomain::U64:
            return read_u64(c, i);
        default:
            return static_cast<std::uint64_t>(read_i64(c, i));
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

// KMV element hash. FNV-1a avalanches poorly, so the mix is what makes the
// bottom-k order (and the [0,1) normalization the estimator divides by)
// uniform.
std::uint64_t kmv_hash(std::string_view v) {
    return dftracer::utils::hash::fnv1a_mix(
        dftracer::utils::hash::fnv1a_hash(v));
}

}  // namespace

void AggStateDeleter::operator()(AggState* p) const noexcept { delete p; }

AggStatePtr agg_new(std::vector<AggSpec> specs, std::vector<AggDynSpec> dyn) {
    AggStatePtr s(new AggState());
    s->specs = std::move(specs);
    s->dyn_specs = std::move(dyn);
    s->init_layout();
    return s;
}

void agg_accumulate(AggState& st, const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    std::int64_t begin, std::int64_t end) {
    agg_accumulate(st, keys, values, std::vector<AggDynInput>{}, begin, end);
}

void agg_accumulate(AggState& st, const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    const std::vector<AggDynInput>& dyn_in, std::int64_t begin,
                    std::int64_t end) {
    if (st.has_dyn)
        for (const AggDynInput& di : dyn_in)
            st.dyn_domain.emplace(di.name, col_domain(di.col->type()));
    if (!st.inited) {
        st.nkeys = keys.size();
        st.key_is_str.resize(st.nkeys);
        st.key_domain.resize(st.nkeys);
        st.key_type.resize(st.nkeys);
        st.ikey_cols.resize(st.nkeys);
        st.skey_cols.resize(st.nkeys);
        for (std::size_t k = 0; k < st.nkeys; ++k) {
            const TypeId kt = keys[k]->type();
            if (!is_group_key_type(kt)) {
                throw std::invalid_argument(
                    std::string("group_by: key column type '") + type_name(kt) +
                    "' is not supported as a group key (no int64/double "
                    "domain to hash it in)");
            }
            st.key_is_str[k] = kt == TypeId::String ? 1 : 0;
            st.key_domain[k] = col_domain(kt);
            st.key_type[k] = kt;
        }
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
    // A zero-key state (the whole batch is one group) has no key column to read
    // the row count from, so fall back to a value column; a caller with neither
    // must pass `end` explicitly.
    if (end < 0)
        end = !keys.empty()     ? keys[0]->length()
              : !values.empty() ? values[0]->length()
                                : 0;
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
        if (st.has_arg) {
            const std::size_t abase = static_cast<std::size_t>(g) * st.n_arg;
            const std::size_t rbase =
                static_cast<std::size_t>(g) * st.n_arg_repr;
            for (std::size_t slot = 0; slot < st.n_arg; ++slot) {
                const Series* byc =
                    values[static_cast<std::size_t>(st.arg_by_col[slot])];
                if (byc->is_null(i)) continue;
                // arg_by is double-keyed (storage, merge compare, serialize),
                // so an integer by-value above 2^53 loses exactness here.
                const double byv =
                    read_as_double(*byc, i, col_domain(byc->type()));
                const std::size_t as = abase + slot;
                const std::vector<int>& reprs = st.arg_slot_reprs[slot];
                // A missing value at the winning row reprs as "" (matches the
                // View fold's src.value(field) for an absent field).
                auto repr_of = [&](int ri) -> std::string {
                    const Series* vc = values[static_cast<std::size_t>(
                        st.arg_val_col[static_cast<std::size_t>(ri)])];
                    return vc->is_null(i) ? std::string() : cell_repr(*vc, i);
                };
                // Strict compare, so a tie keeps the row seen first; the View
                // fold does the same and the two engines are compared row for
                // row.
                const bool adopt =
                    !st.arg_has[as] ||
                    (st.arg_dir[slot] == ArgDir::Max ? byv > st.arg_by[as]
                                                     : byv < st.arg_by[as]);
                if (!adopt) continue;
                st.arg_by[as] = byv;
                st.arg_has[as] = 1;
                for (int ri : reprs)
                    st.arg_repr[rbase + static_cast<std::size_t>(ri)] =
                        repr_of(ri);
            }
        }
        if (st.has_bitor) {
            const std::size_t bbase = static_cast<std::size_t>(g) * st.n_bitor;
            for (std::size_t slot = 0; slot < st.n_bitor; ++slot) {
                const Series* vc =
                    values[static_cast<std::size_t>(st.bitor_val_col[slot])];
                if (vc->is_null(i)) continue;
                st.bitor_acc[bbase + slot] |=
                    read_u64_exact(*vc, i, col_domain(vc->type()));
            }
        }
        if (st.has_kmv) {
            const std::size_t kbase = static_cast<std::size_t>(g) * st.n_kmv;
            for (std::size_t slot = 0; slot < st.n_kmv; ++slot) {
                const Series* vc =
                    values[static_cast<std::size_t>(st.kmv_val_col[slot])];
                if (vc->is_null(i)) continue;
                std::string v = cell_repr(*vc, i);
                KmvMap& m = st.kmv[kbase + slot];
                const std::uint64_t h = kmv_hash(v);
                // Past the k-th smallest hash the element cannot enter, so the
                // common case costs one compare and no allocation.
                if (m.size() >= st.kmv_k[slot] && h >= m.rbegin()->first)
                    continue;
                m.insert_or_assign(h, std::move(v));
                kmv_trim(m, st.kmv_k[slot]);
            }
        }
        if (st.has_lst) {
            const std::size_t lbase = static_cast<std::size_t>(g) * st.n_lst;
            for (std::size_t slot = 0; slot < st.n_lst; ++slot) {
                const Series* vc =
                    values[static_cast<std::size_t>(st.lst_val_col[slot])];
                const Series* byc =
                    values[static_cast<std::size_t>(st.lst_by_col[slot])];
                if (byc->is_null(i) || vc->is_null(i)) continue;
                ListItems& items = st.lst[lbase + slot];
                items.emplace_back(
                    read_as_double(*byc, i, col_domain(byc->type())),
                    cell_repr(*vc, i));
                const std::size_t k = st.lst_k[slot];
                if (k > 0 && items.size() > k)
                    list_bound(items, st.lst_kind[slot], k);
            }
        }
        if (st.has_ss) {
            const std::size_t sbase = static_cast<std::size_t>(g) * st.n_ss;
            for (std::size_t slot = 0; slot < st.n_ss; ++slot) {
                const Series* vc =
                    values[static_cast<std::size_t>(st.ss_val_col[slot])];
                if (vc->is_null(i)) continue;
                space_saving_offer(st.ss_counters[sbase + slot], st.ss_k[slot],
                                   cell_repr(*vc, i), 1);
            }
        }
        if (st.has_co) {
            const std::size_t cbase = static_cast<std::size_t>(g) * st.n_co;
            for (std::size_t slot = 0; slot < st.n_co; ++slot) {
                const Series* vc =
                    values[static_cast<std::size_t>(st.co_val_col[slot])];
                const Series* byc =
                    values[static_cast<std::size_t>(st.co_by_col[slot])];
                if (vc->is_null(i) || byc->is_null(i)) continue;
                const double x =
                    read_as_double(*byc, i, col_domain(byc->type()));
                const double y = read_as_double(*vc, i, col_domain(vc->type()));
                const std::size_t cs = cbase + slot;
                st.co_n[cs] += 1.0;
                st.co_sx[cs] += x;
                st.co_sy[cs] += y;
                st.co_sxx[cs] += x * x;
                st.co_syy[cs] += y * y;
                st.co_sxy[cs] += x * y;
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
        if (st.has_occ) {
            for (std::size_t slot = 0; slot < st.n_occ; ++slot) {
                const Series* tsc =
                    values[static_cast<std::size_t>(st.occ_val_col[slot])];
                const Series* durc =
                    values[static_cast<std::size_t>(st.occ_by_col[slot])];
                if (tsc->is_null(i) || durc->is_null(i)) continue;
                const FieldStatDomain durd = col_domain(durc->type());
                std::uint64_t dur;
                if (durd == FieldStatDomain::I64) {
                    const std::int64_t d = read_i64(*durc, i);
                    if (d <= 0) continue;
                    dur = static_cast<std::uint64_t>(d);
                } else if (durd == FieldStatDomain::U64) {
                    dur = read_u64(*durc, i);
                    if (dur == 0) continue;
                } else {
                    const double d = read_f64(*durc, i);
                    if (!(d > 0)) continue;
                    dur = static_cast<std::uint64_t>(d);
                }
                std::uint64_t s0 =
                    read_u64_exact(*tsc, i, col_domain(tsc->type()));
                std::uint64_t e0 = s0 + dur;
                const std::uint64_t cell = st.occ_cell[slot];
                if (cell) {  // optional tolerance: snap start down, end up
                    s0 = s0 / cell * cell;
                    e0 = (e0 + cell - 1) / cell * cell;
                }
                const std::size_t base =
                    static_cast<std::size_t>(g) * st.n_occ + slot;
                st.occ_total[base] += dur;
                if (s0 < st.occ_ts[base]) st.occ_ts[base] = s0;
                if (e0 > st.occ_te[base]) st.occ_te[base] = e0;
                st.occ_deltas[base][s0] += 1;
                st.occ_deltas[base][e0] -= 1;
            }
        }
        if (st.has_dyn) {
            for (const AggDynInput& di : dyn_in) {
                if (di.col->is_null(i)) continue;
                const FieldStatDomain d = col_domain(di.col->type());
                std::map<std::string, FieldStat>& gmap =
                    st.dyn_fs[static_cast<std::size_t>(g)];
                fs_add(gmap[di.name], *di.col, i, d);
                if (st.dyn_has_sketch)
                    st.dyn_sketch[static_cast<std::size_t>(g)][di.name].add(
                        read_as_double(*di.col, i, d));
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
        into.adopt_layout(other);
        into.nkeys = other.nkeys;
        into.key_is_str = other.key_is_str;
        into.key_domain = other.key_domain;
        into.key_type = other.key_type;
        into.ikey_cols.assign(into.nkeys, {});
        into.skey_cols.assign(into.nkeys, {});
        into.inited = true;
    }
    for (const auto& [name, d] : other.dyn_domain)
        into.dyn_domain.emplace(name, d);
    const std::int64_t og = other.ngroups();
    for (std::int64_t j = 0; j < og; ++j)
        into.merge_group(into.group_of_other(other, j), other, j);
}

AggStatePtr agg_regroup(const AggState& src,
                        const std::vector<std::int32_t>& keep,
                        std::int64_t bucket_recut) {
    AggStatePtr dst(new AggState());
    dst->specs = src.specs;
    dst->init_layout();
    dst->adopt_layout(src);  // field_domain/field_is_str are value-derived
    dst->nkeys = keep.size();
    dst->key_is_str.resize(dst->nkeys);
    dst->key_domain.resize(dst->nkeys);
    dst->key_type.resize(dst->nkeys);
    for (std::size_t k = 0; k < dst->nkeys; ++k) {
        dst->key_is_str[k] = src.key_is_str[static_cast<std::size_t>(keep[k])];
        dst->key_domain[k] = src.key_domain[static_cast<std::size_t>(keep[k])];
        dst->key_type[k] = src.key_type[static_cast<std::size_t>(keep[k])];
    }
    dst->ikey_cols.assign(dst->nkeys, {});
    dst->skey_cols.assign(dst->nkeys, {});
    dst->inited = true;

    const std::int64_t ng = src.ngroups();
    for (std::int64_t j = 0; j < ng; ++j) {
        const std::int64_t g = dst->find_or_add_group(
            [&](std::size_t k) -> std::int64_t {
                std::int64_t v =
                    src.ikey_cols[static_cast<std::size_t>(keep[k])]
                                 [static_cast<std::size_t>(j)];
                if (k == 0 && bucket_recut > 0)
                    v = (v / bucket_recut) * bucket_recut;
                return v;
            },
            [&](std::size_t k) -> std::string_view {
                return src.skey_cols[static_cast<std::size_t>(keep[k])]
                                    [static_cast<std::size_t>(j)];
            });
        dst->merge_group(g, src, j);
    }
    return dst;
}

}  // namespace dftracer::utils::dataframe
