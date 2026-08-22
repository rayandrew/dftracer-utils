#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/field_stat.h>
#include <dftracer/utils/dataframe/internal/column_read.h>  // read_i64/u64/f64
#include <dftracer/utils/dataframe/parallel.h>              // parallel_for

#include <bit>
#include <cstdint>
#include <cstring>
#include <string>
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
    bool key_string = false;
    std::unordered_map<std::int64_t, std::int64_t> imap;
    std::unordered_map<std::string, std::int64_t> smap;
    std::vector<std::int64_t> ikeys;
    std::vector<std::string> skeys;
    std::vector<std::uint64_t> counts;  // per group: rows seen
    std::vector<FieldStat> fstats;      // groups * nf

    std::size_t nspecs() const { return specs.size(); }
    std::int64_t ngroups() const {
        return static_cast<std::int64_t>(key_string ? skeys.size()
                                                    : ikeys.size());
    }

    // The distinct-field layout is a pure function of the specs.
    void init_layout() {
        spec_field.assign(specs.size(), -1);
        field_vc.clear();
        std::unordered_map<std::int32_t, int> seen;
        for (std::size_t s = 0; s < specs.size(); ++s) {
            const AggSpec& sp = specs[s];
            if (sp.op == AggOp::Count || sp.value_col < 0) continue;
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
    }

    void grow_group() {
        counts.push_back(0);
        fstats.resize(fstats.size() + nf);
    }
    std::int64_t group_of_i64(std::int64_t k) {
        auto it = imap.find(k);
        if (it != imap.end()) return it->second;
        std::int64_t g = ngroups();
        ikeys.push_back(k);
        imap.emplace(k, g);
        grow_group();
        return g;
    }
    std::int64_t group_of_str(const std::string& k) {
        auto it = smap.find(k);
        if (it != smap.end()) return it->second;
        std::int64_t g = ngroups();
        skeys.push_back(k);
        smap.emplace(k, g);
        grow_group();
        return g;
    }
};

void AggStateDeleter::operator()(AggState* p) const noexcept { delete p; }

AggStatePtr agg_new(std::vector<AggSpec> specs) {
    AggStatePtr s(new AggState());
    s->specs = std::move(specs);
    s->init_layout();
    return s;
}

void agg_accumulate(AggState& st, const Series& key,
                    const std::vector<const Series*>& values,
                    std::int64_t begin, std::int64_t end) {
    if (!st.inited) {
        st.key_string = key.type() == TypeId::String;
        st.field_domain.resize(st.nf);
        for (std::size_t fj = 0; fj < st.nf; ++fj)
            st.field_domain[fj] = col_domain(
                values[static_cast<std::size_t>(st.field_vc[fj])]->type());
        st.inited = true;
    }
    if (end < 0) end = key.length();
    for (std::int64_t i = begin; i < end; ++i) {
        std::int64_t g = st.key_string
                             ? st.group_of_str(std::string(key.string_at(i)))
                             : st.group_of_i64(read_i64(key, i));
        st.counts[static_cast<std::size_t>(g)]++;
        const std::size_t base = static_cast<std::size_t>(g) * st.nf;
        for (std::size_t fj = 0; fj < st.nf; ++fj) {
            const Series* vc =
                values[static_cast<std::size_t>(st.field_vc[fj])];
            if (vc->is_null(i)) continue;
            fs_add(st.fstats[base + fj], *vc, i, st.field_domain[fj]);
        }
    }
}

void agg_merge(AggState& into, const AggState& other) {
    if (!other.inited) return;
    if (!into.inited) {
        into.specs = other.specs;
        into.spec_field = other.spec_field;
        into.field_vc = other.field_vc;
        into.field_domain = other.field_domain;
        into.nf = other.nf;
        into.key_string = other.key_string;
        into.inited = true;
    }
    const std::int64_t og = other.ngroups();
    for (std::int64_t j = 0; j < og; ++j) {
        std::int64_t g =
            into.key_string
                ? into.group_of_str(other.skeys[static_cast<std::size_t>(j)])
                : into.group_of_i64(other.ikeys[static_cast<std::size_t>(j)]);
        into.counts[static_cast<std::size_t>(g)] +=
            other.counts[static_cast<std::size_t>(j)];
        const std::size_t db = static_cast<std::size_t>(g) * into.nf;
        const std::size_t sb = static_cast<std::size_t>(j) * into.nf;
        for (std::size_t fj = 0; fj < into.nf; ++fj)
            into.fstats[db + fj].merge(other.fstats[sb + fj]);
    }
}

DataFrame agg_finalize(const AggState& st, const std::string& key_name) {
    const std::int64_t ng = st.ngroups();
    const std::size_t ns = st.nspecs();
    DataFrame out;
    out.names.push_back(key_name);
    if (st.key_string)
        out.columns.push_back(Series::strings(st.skeys));
    else
        out.columns.push_back(Series::flat_i64(st.ikeys.data(), ng));

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

        if (sp.op == AggOp::Count) {
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
    put(s, static_cast<std::uint8_t>(st.key_string ? 1 : 0));
    for (const AggSpec& sp : st.specs) {
        put(s, static_cast<std::int32_t>(sp.op));
        put(s, sp.value_col);
        put_bytes(s, sp.out);
    }
    put(s, static_cast<std::uint32_t>(st.nf));
    for (FieldStatDomain d : st.field_domain)
        put(s, static_cast<std::uint8_t>(d));
    const std::int64_t ng = st.ngroups();
    put(s, ng);
    if (st.key_string)
        for (const std::string& k : st.skeys) put_bytes(s, k);
    else
        for (std::int64_t k : st.ikeys) put(s, k);
    for (std::uint64_t c : st.counts) put(s, c);
    for (const FieldStat& f : st.fstats) put(s, f);
    return s;
}

AggStatePtr agg_deserialize(const std::string& blob) {
    Reader r{blob.data()};
    AggStatePtr st(new AggState());
    const std::uint32_t ns = r.get<std::uint32_t>();
    st->key_string = r.get<std::uint8_t>() != 0;
    st->specs.resize(ns);
    for (std::uint32_t i = 0; i < ns; ++i) {
        st->specs[i].op = static_cast<AggOp>(r.get<std::int32_t>());
        st->specs[i].value_col = r.get<std::int32_t>();
        st->specs[i].out = r.get_bytes();
    }
    st->init_layout();
    const std::uint32_t nf = r.get<std::uint32_t>();
    st->field_domain.resize(nf);
    for (std::uint32_t i = 0; i < nf; ++i)
        st->field_domain[i] =
            static_cast<FieldStatDomain>(r.get<std::uint8_t>());
    const std::int64_t ng = r.get<std::int64_t>();
    if (st->key_string) {
        st->skeys.resize(static_cast<std::size_t>(ng));
        for (std::int64_t g = 0; g < ng; ++g) {
            st->skeys[static_cast<std::size_t>(g)] = r.get_bytes();
            st->smap.emplace(st->skeys[static_cast<std::size_t>(g)], g);
        }
    } else {
        st->ikeys.resize(static_cast<std::size_t>(ng));
        for (std::int64_t g = 0; g < ng; ++g) {
            st->ikeys[static_cast<std::size_t>(g)] = r.get<std::int64_t>();
            st->imap.emplace(st->ikeys[static_cast<std::size_t>(g)], g);
        }
    }
    st->counts.resize(static_cast<std::size_t>(ng));
    for (std::int64_t g = 0; g < ng; ++g)
        st->counts[static_cast<std::size_t>(g)] = r.get<std::uint64_t>();
    st->fstats.resize(static_cast<std::size_t>(ng) * nf);
    for (std::size_t i = 0; i < st->fstats.size(); ++i)
        st->fstats[i] = r.get<FieldStat>();
    st->inited = true;
    return st;
}

DataFrame group_agg(const Series& key, const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs, const std::string& key_name) {
    const std::int64_t n = key.length();
    if (n <= AGG_GRAIN) {
        auto st = agg_new(std::move(specs));
        agg_accumulate(*st, key, values);
        return agg_finalize(*st, key_name);
    }
    // Parallel driver: each chunk folds into its own partial (no locks), then
    // the partials merge. parallel_for runs serial when no backend is
    // installed.
    const std::int64_t chunks = (n + AGG_GRAIN - 1) / AGG_GRAIN;
    std::vector<AggStatePtr> partials(static_cast<std::size_t>(chunks));
    parallel_for(n, AGG_GRAIN, [&](std::int64_t b, std::int64_t e) {
        auto st = agg_new(specs);
        agg_accumulate(*st, key, values, b, e);
        partials[static_cast<std::size_t>(b / AGG_GRAIN)] = std::move(st);
    });
    auto acc = agg_new(std::move(specs));
    for (auto& p : partials)
        if (p) agg_merge(*acc, *p);
    return agg_finalize(*acc, key_name);
}

}  // namespace dftracer::utils::dataframe
