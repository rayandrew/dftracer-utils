#include <dftracer/utils/dataframe/agg/detail.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

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
    if (st.has_occ) {
        permute_blocks(st.occ_deltas, perm, st.n_occ);
        permute_blocks(st.occ_total, perm, st.n_occ);
        permute_blocks(st.occ_ts, perm, st.n_occ);
        permute_blocks(st.occ_te, perm, st.n_occ);
    }
    if (st.has_dyn) {
        permute_blocks(st.dyn_fs, perm, 1);
        if (st.dyn_has_sketch) permute_blocks(st.dyn_sketch, perm, 1);
    }

    st.rebuild_key_buckets(ng);
}

AggStatePtr agg_extract_group(const AggState& st, std::int64_t g) {
    AggStatePtr out(new AggState());
    out->specs = st.specs;
    out->dyn_specs = st.dyn_specs;
    out->dyn_domain = st.dyn_domain;
    out->init_layout();
    out->nkeys = st.nkeys;
    out->key_is_str = st.key_is_str;
    out->key_domain = st.key_domain;
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
    if (st.has_occ) {
        const std::size_t obase = static_cast<std::size_t>(g) * st.n_occ;
        for (std::size_t slot = 0; slot < st.n_occ; ++slot) {
            out->occ_deltas[slot] = st.occ_deltas[obase + slot];
            out->occ_total[slot] = st.occ_total[obase + slot];
            out->occ_ts[slot] = st.occ_ts[obase + slot];
            out->occ_te[slot] = st.occ_te[obase + slot];
        }
    }
    if (st.has_dyn) {
        out->dyn_fs[0] = st.dyn_fs[static_cast<std::size_t>(g)];
        if (st.dyn_has_sketch)
            out->dyn_sketch[0] = st.dyn_sketch[static_cast<std::size_t>(g)];
    }
    return out;
}

// The seed path rebuilds a group only from a FieldStat + optional DDSketch;
// First/Last, ArgMax, SetUnion and occupancy keep no such state. No default:
// -Wswitch keeps this complete.
static bool agg_op_seedable(AggOp op) {
    switch (op) {
        case AggOp::Count:
        case AggOp::Sum:
        case AggOp::Min:
        case AggOp::Max:
        case AggOp::Mean:
        case AggOp::Var:
        case AggOp::Std:
        case AggOp::Skew:
        case AggOp::Kurt:
        case AggOp::Pct:
        case AggOp::Hist:
        case AggOp::SumSq:
        case AggOp::CountValid:
            return true;
        case AggOp::First:
        case AggOp::Last:
        case AggOp::ArgMax:
        case AggOp::SetUnion:
        case AggOp::Busy:
        case AggOp::Concurrency:
        case AggOp::Utilization:
        case AggOp::Active:
            return false;
    }
    return false;
}

void agg_seed_begin(AggState& st, std::size_t nkeys) {
    st.nkeys = nkeys;
    st.key_is_str.assign(nkeys, 1);
    st.key_domain.assign(nkeys, FieldStatDomain::I64);
    st.ikey_cols.assign(nkeys, {});
    st.skey_cols.assign(nkeys, {});
    st.field_domain.assign(st.nf, FieldStatDomain::F64);
    st.field_is_str.assign(st.nf, 0);
    st.inited = true;
}

void agg_seed_group(AggState& st, const std::vector<std::string>& str_keys,
                    std::uint64_t count,
                    const std::vector<AggSeedValue>& values) {
    if (!st.inited) agg_seed_begin(st, str_keys.size());
    for (const AggSpec& sp : st.specs)
        if (!agg_op_seedable(sp.op))
            throw std::logic_error(
                "agg_seed_group: an aggregate op has no FieldStat/sketch seed "
                "representation (First/Last/ArgMax/SetUnion/occupancy)");
    const std::int64_t g = st.find_or_add_group(
        [](std::size_t) -> std::int64_t { return 0; },
        [&](std::size_t k) -> std::string_view { return str_keys[k]; });
    st.counts[static_cast<std::size_t>(g)] += count;
    for (const AggSeedValue& v : values) {
        int fj = -1;
        for (std::size_t f = 0; f < st.nf; ++f)
            if (st.field_vc[f] == v.value_col) {
                fj = static_cast<int>(f);
                break;
            }
        if (fj < 0) continue;
        st.fstats[static_cast<std::size_t>(g) * st.nf +
                  static_cast<std::size_t>(fj)]
            .merge(v.stat);
        if (v.sketch && st.has_sketch) {
            const int sk = st.field_sketch[static_cast<std::size_t>(fj)];
            if (sk >= 0)
                st.sketches[static_cast<std::size_t>(g) * st.n_sketch +
                            static_cast<std::size_t>(sk)]
                    .merge(*v.sketch);
        }
    }
}

void agg_seed_finalize(AggState& st) {
    st.inited = true;
    const std::int64_t ng = st.ngroups();
    for (std::size_t fj = 0; fj < st.nf; ++fj) {
        // The columnar materializer picks a column's domain from the first
        // group then downgrades to F64 on any mismatch, treating an absent
        // (n == 0) group as F64. Reproduce that so a field present in every
        // group stays exact-integer and one absent from some groups is Float64.
        bool first = true;
        FieldStatDomain dom = FieldStatDomain::F64;
        for (std::int64_t g = 0; g < ng; ++g) {
            const FieldStat& f =
                st.fstats[static_cast<std::size_t>(g) * st.nf + fj];
            const FieldStatDomain e = f.n > 0 ? f.domain : FieldStatDomain::F64;
            if (first) {
                dom = e;
                first = false;
            } else if (e != dom) {
                dom = FieldStatDomain::F64;
            }
        }
        st.field_domain[fj] = dom;
    }
}

}  // namespace dftracer::utils::dataframe
