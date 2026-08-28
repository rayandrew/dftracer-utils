#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/field_stat.h>
#include <dftracer/utils/dataframe/internal/column_read.h>  // read_i64/u64/f64
#include <dftracer/utils/dataframe/parallel.h>              // parallel_for
#include <dftracer/utils/dataframe/sketch.h>  // DDSketch, sketch_bucket_keys

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
        st.field_is_str.resize(st.nf);
        for (std::size_t fj = 0; fj < st.nf; ++fj) {
            const Series* vc =
                values[static_cast<std::size_t>(st.field_vc[fj])];
            st.field_domain[fj] = col_domain(vc->type());
            st.field_is_str[fj] = vc->type() == TypeId::String ? 1 : 0;
        }
        st.inited = true;
    }
    if (end < 0) end = key.length();
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
        std::int64_t g = st.key_string
                             ? st.group_of_str(std::string(key.string_at(i)))
                             : st.group_of_i64(read_i64(key, i));
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
    }
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
        put(s, sp.param);
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
    if (st.key_string)
        for (const std::string& k : st.skeys) put_bytes(s, k);
    else
        for (std::int64_t k : st.ikeys) put(s, k);
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
        st->specs[i].param = r.get<double>();
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
