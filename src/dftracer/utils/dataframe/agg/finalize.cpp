#include <ankerl/unordered_dense.h>
#include <dftracer/utils/dataframe/agg/detail.h>
#include <dftracer/utils/dataframe/dataframe.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

namespace {

// Separator joining a SetUnion group's distinct values into one text cell;
// matches views/view_aggregate.h SET_SEP so the two engines agree.
constexpr char AGG_SET_SEP = '\x1e';

// One group's occupancy scalars from its endpoint delta-map. Reproduces
// view_aggregate.cpp occupancy_summary byte-for-byte: busy is the exact
// interval union (depth-sweep over sorted deltas), active the peak depth, span
// the makespan, total the raw sum(dur).
struct OccResult {
    std::uint64_t busy = 0;
    std::uint64_t active = 0;
    std::uint64_t total = 0;
    std::uint64_t span = 0;
};

OccResult occ_summarize(
    const ankerl::unordered_dense::map<std::uint64_t, std::int64_t>& deltas,
    std::uint64_t total, std::uint64_t ts, std::uint64_t te) {
    OccResult o;
    o.total = total;
    o.span = te > ts ? te - ts : 0;
    if (deltas.empty()) return o;
    std::vector<std::pair<std::uint64_t, std::int64_t>> pts(deltas.begin(),
                                                            deltas.end());
    std::sort(pts.begin(), pts.end(),
              [](const auto& x, const auto& y) { return x.first < y.first; });
    std::int64_t depth = 0;
    std::uint64_t last = 0, busy = 0, peak = 0;
    bool started = false;
    for (const auto& [t, dlt] : pts) {
        if (started && depth > 0) busy += t - last;
        depth += dlt;
        if (depth > 0 && static_cast<std::uint64_t>(depth) > peak)
            peak = static_cast<std::uint64_t>(depth);
        last = t;
        started = true;
    }
    o.busy = busy;
    o.active = peak;
    return o;
}

}  // namespace

DataFrame agg_finalize(const AggState& st,
                       const std::vector<std::string>& key_names) {
    const std::int64_t ng = st.ngroups();
    const std::size_t ns = st.nspecs();
    DataFrame out;
    // A state finalized without accumulating (empty stream) has no key layout;
    // still emit one empty column per requested key so the schema is complete.
    const std::size_t nk =
        st.nkeys > key_names.size() ? st.nkeys : key_names.size();
    for (std::size_t k = 0; k < nk; ++k) {
        out.names.push_back(k < key_names.size() ? key_names[k]
                                                 : "key" + std::to_string(k));
        if (k >= st.nkeys) {
            out.columns.push_back(Series::strings(
                std::vector<std::string>(static_cast<std::size_t>(ng))));
            continue;
        }
        if (st.key_is_str[k]) {
            out.columns.push_back(Series::strings(st.skey_cols[k]));
            continue;
        }
        const FieldStatDomain kd =
            k < st.key_domain.size() ? st.key_domain[k] : FieldStatDomain::I64;
        const std::vector<std::int64_t>& bits = st.ikey_cols[k];
        if (kd == FieldStatDomain::U64) {
            std::vector<std::uint64_t> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] = std::bit_cast<std::uint64_t>(
                    bits[static_cast<std::size_t>(g)]);
            out.columns.push_back(Series::flat(TypeId::Uint64, v.data(), ng));
        } else if (kd == FieldStatDomain::F64) {
            std::vector<double> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] =
                    std::bit_cast<double>(bits[static_cast<std::size_t>(g)]);
            out.columns.push_back(Series::flat_f64(v.data(), ng));
        } else {
            out.columns.push_back(Series::flat_i64(bits.data(), ng));
        }
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
        } else if (sp.op == AggOp::Busy || sp.op == AggOp::Concurrency ||
                   sp.op == AggOp::Utilization || sp.op == AggOp::Active) {
            const int slot = st.spec_occ[s];
            std::vector<double> v(static_cast<std::size_t>(ng), 0.0);
            for (std::int64_t g = 0; g < ng; ++g) {
                if (slot < 0) continue;
                const std::size_t b = static_cast<std::size_t>(g) * st.n_occ +
                                      static_cast<std::size_t>(slot);
                const OccResult o =
                    occ_summarize(st.occ_deltas[b], st.occ_total[b],
                                  st.occ_ts[b], st.occ_te[b]);
                double r = 0.0;
                switch (sp.op) {
                    case AggOp::Busy:
                        r = static_cast<double>(o.busy);
                        break;
                    case AggOp::Concurrency:
                        r = o.busy ? static_cast<double>(o.total) /
                                         static_cast<double>(o.busy)
                                   : 0.0;
                        break;
                    case AggOp::Utilization:
                        r = (o.busy && o.span) ? static_cast<double>(o.busy) /
                                                     static_cast<double>(o.span)
                                               : 0.0;
                        break;
                    default:  // Active
                        r = static_cast<double>(o.active);
                }
                v[static_cast<std::size_t>(g)] = r;
            }
            out.columns.push_back(Series::flat_f64(v.data(), ng));
        } else if (sp.op == AggOp::Count) {
            std::vector<std::int64_t> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] = static_cast<std::int64_t>(
                    st.counts[static_cast<std::size_t>(g)]);
            out.columns.push_back(Series::flat_i64(v.data(), ng));
        } else if (sp.op == AggOp::CountValid) {
            std::vector<std::int64_t> v(static_cast<std::size_t>(ng));
            for (std::int64_t g = 0; g < ng; ++g)
                v[static_cast<std::size_t>(g)] =
                    static_cast<std::int64_t>(fs_at(g).n);
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

    if (st.has_dyn) {
        // Union of discovered argument names, sorted (std::set) so the dyn
        // column order is deterministic and matches the GroupMap path. Columns
        // are laid out name-outer, reduction-inner.
        std::set<std::string> names;
        for (const std::map<std::string, FieldStat>& gmap : st.dyn_fs)
            for (const auto& [name, fs] : gmap) names.insert(name);
        for (const std::string& name : names) {
            const auto dit = st.dyn_domain.find(name);
            const FieldStatDomain dom =
                dit != st.dyn_domain.end() ? dit->second : FieldStatDomain::I64;
            for (const AggDynSpec& d : st.dyn_specs) {
                out.names.push_back(d.out_prefix + name);
                auto fs_of = [&](std::int64_t g) -> const FieldStat* {
                    const std::map<std::string, FieldStat>& gmap =
                        st.dyn_fs[static_cast<std::size_t>(g)];
                    const auto it = gmap.find(name);
                    return it != gmap.end() ? &it->second : nullptr;
                };
                if (d.op == AggOp::Pct) {
                    std::vector<double> v(static_cast<std::size_t>(ng), 0.0);
                    for (std::int64_t g = 0; g < ng; ++g) {
                        const std::map<std::string, DDSketch>& gsk =
                            st.dyn_sketch[static_cast<std::size_t>(g)];
                        const auto it = gsk.find(name);
                        if (it != gsk.end())
                            v[static_cast<std::size_t>(g)] =
                                it->second.quantile(d.param);
                    }
                    out.columns.push_back(Series::flat_f64(v.data(), ng));
                } else if (d.op == AggOp::Count) {
                    std::vector<std::int64_t> v(static_cast<std::size_t>(ng),
                                                0);
                    for (std::int64_t g = 0; g < ng; ++g)
                        if (const FieldStat* f = fs_of(g))
                            v[static_cast<std::size_t>(g)] =
                                static_cast<std::int64_t>(f->n);
                    out.columns.push_back(Series::flat_i64(v.data(), ng));
                } else if ((d.op == AggOp::Sum || d.op == AggOp::Min ||
                            d.op == AggOp::Max) &&
                           dom != FieldStatDomain::F64) {
                    auto exact = [&](const FieldStat& f) -> std::int64_t {
                        return d.op == AggOp::Sum   ? f.esum
                               : d.op == AggOp::Min ? f.emin
                                                    : f.emax;
                    };
                    if (dom == FieldStatDomain::U64) {
                        std::vector<std::uint64_t> v(
                            static_cast<std::size_t>(ng), 0);
                        for (std::int64_t g = 0; g < ng; ++g)
                            if (const FieldStat* f = fs_of(g))
                                v[static_cast<std::size_t>(g)] =
                                    std::bit_cast<std::uint64_t>(exact(*f));
                        out.columns.push_back(
                            Series::flat(TypeId::Uint64, v.data(), ng));
                    } else {
                        std::vector<std::int64_t> v(
                            static_cast<std::size_t>(ng), 0);
                        for (std::int64_t g = 0; g < ng; ++g)
                            if (const FieldStat* f = fs_of(g))
                                v[static_cast<std::size_t>(g)] = exact(*f);
                        out.columns.push_back(Series::flat_i64(v.data(), ng));
                    }
                } else {
                    std::vector<double> v(static_cast<std::size_t>(ng), 0.0);
                    for (std::int64_t g = 0; g < ng; ++g) {
                        const FieldStat* f = fs_of(g);
                        if (!f) continue;
                        double r = 0.0;
                        switch (d.op) {
                            case AggOp::Sum:
                                r = f->sum;
                                break;
                            case AggOp::Min:
                                r = f->n ? f->min : 0.0;
                                break;
                            case AggOp::Max:
                                r = f->n ? f->max : 0.0;
                                break;
                            case AggOp::SumSq:
                                r = f->sumsq;
                                break;
                            case AggOp::Mean:
                                r = f->mean();
                                break;
                            case AggOp::Var:
                                r = f->variance(true);
                                break;
                            case AggOp::Std:
                                r = f->stddev(true);
                                break;
                            case AggOp::Skew:
                                r = f->skewness();
                                break;
                            case AggOp::Kurt:
                                r = f->kurtosis();
                                break;
                            default:
                                r = 0.0;
                        }
                        v[static_cast<std::size_t>(g)] = r;
                    }
                    out.columns.push_back(Series::flat_f64(v.data(), ng));
                }
            }
        }
    }
    return out;
}

DataFrame agg_finalize(const AggState& st, const std::string& key_name) {
    return agg_finalize(st, std::vector<std::string>{key_name});
}

}  // namespace dftracer::utils::dataframe
