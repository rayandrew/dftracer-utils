#include <dftracer/utils/core/common/to_chars.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/trace/aggregators/reserved_args.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/agg_fold.h>
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_resolver.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace dataframe = dftracer::utils::dataframe;

namespace dftracer::utils::trace::views::detail {

std::optional<double> to_number(simdjson::dom::element e) {
    double d;
    if (e.get_double().get(d) == simdjson::SUCCESS) return d;
    std::int64_t i;
    if (e.get_int64().get(i) == simdjson::SUCCESS)
        return static_cast<double>(i);
    std::uint64_t u;
    if (e.get_uint64().get(u) == simdjson::SUCCESS)
        return static_cast<double>(u);
    return std::nullopt;
}

std::string to_str(simdjson::dom::element e) {
    std::string_view s;
    if (e.get_string().get(s) == simdjson::SUCCESS) return std::string(s);
    std::int64_t i;
    if (e.get_int64().get(i) == simdjson::SUCCESS) return std::to_string(i);
    std::uint64_t u;
    if (e.get_uint64().get(u) == simdjson::SUCCESS) return std::to_string(u);
    double d;
    if (e.get_double().get(d) == simdjson::SUCCESS) return std::to_string(d);
    bool b;
    if (e.get_bool().get(b) == simdjson::SUCCESS) return b ? "true" : "false";
    return {};
}

std::string top_or_args_str(simdjson::dom::element root,
                            const std::string& key) {
    auto r = root[key];
    if (!r.error()) return to_str(r.value_unsafe());
    auto args = root["args"];
    if (!args.error() && args.is_object()) {
        auto rr = args[key];
        if (!rr.error()) return to_str(rr.value_unsafe());
    }
    return {};
}

// Byte size of an event, derived with the reader's io-cat rule (ret is the
// byte count for POSIX/STDIO read|write) so "size" aggregates match dfanalyzer.
static std::optional<double> derived_size(simdjson::dom::element root) {
    namespace dfi = trace::internal;
    auto num_of = [&](const char* key) -> std::optional<std::int64_t> {
        auto r = root[key];
        if (!r.error())
            if (auto n = to_number(r.value_unsafe()))
                return static_cast<std::int64_t>(*n);
        auto args = root["args"];
        if (!args.error() && args.is_object()) {
            auto rr = args[key];
            if (!rr.error())
                if (auto n = to_number(rr.value_unsafe()))
                    return static_cast<std::int64_t>(*n);
        }
        return std::nullopt;
    };
    auto s = dfi::derive_io_size(
        top_or_args_str(root, "cat"), top_or_args_str(root, "name"),
        num_of("size_sum"), num_of("ret"), num_of("image_size"));
    return s ? std::optional<double>(static_cast<double>(*s)) : std::nullopt;
}

std::optional<double> agg_field(simdjson::dom::element root,
                                const std::string& field) {
    // "size" is the io-cat-derived byte size, not a raw field lookup.
    if (field == "size") return derived_size(root);
    auto r = root[field];
    if (!r.error()) return to_number(r.value_unsafe());
    auto args = root["args"];
    if (!args.error() && args.is_object()) {
        auto rr = args[field];
        if (!rr.error()) return to_number(rr.value_unsafe());
    }
    // "te" is the event end time; derive it from ts + dur when not stored.
    if (field == "te") {
        auto ts = agg_field(root, "ts");
        auto dur = agg_field(root, "dur");
        if (ts && dur) return *ts + *dur;
    }
    return std::nullopt;
}

AggSchema make_agg_schema(const ViewPlan& plan) {
    AggSchema s;
    s.spec_field.assign(plan.agg.size(), -1);
    s.spec_argmax.assign(plan.agg.size(), -1);
    s.spec_set.assign(plan.agg.size(), -1);
    auto field_index = [&](const std::string& f) -> int {
        for (std::size_t i = 0; i < s.fields.size(); ++i)
            if (s.fields[i] == f) return static_cast<int>(i);
        s.fields.push_back(f);
        return static_cast<int>(s.fields.size() - 1);
    };
    for (std::size_t i = 0; i < plan.agg.size(); ++i) {
        const auto& spec = plan.agg[i];
        if (spec.op == AggOp::ArgMax) {
            s.spec_argmax[i] = static_cast<int>(s.argmax_count++);
            continue;
        }
        // SetUnion collects a field's distinct string values, not a numeric
        // stat, so it gets its own slot and no FieldStat field.
        if (spec.op == AggOp::SetUnion) {
            if (!spec.field.empty())
                s.spec_set[i] = static_cast<int>(s.set_count++);
            continue;
        }
        if (is_occupancy_op(spec.op)) s.want_occupancy = true;
        // Count() reduces over rows (group count), not a field; every other op
        // (including Count(field)) reads its field's stat.
        if (spec.field.empty()) continue;
        s.spec_field[i] = field_index(spec.field);
    }
    s.field_scaled.assign(s.fields.size(), false);
    for (std::size_t i = 0; i < s.fields.size(); ++i)
        s.field_scaled[i] = (s.fields[i] == "ts" || s.fields[i] == "dur" ||
                             s.fields[i] == "te");
    // A field named by a Pct or Hist op gets one shared DDSketch slot (all
    // quantiles and the histogram on that field read the same sketch).
    s.field_sketch.assign(s.fields.size(), -1);
    for (std::size_t i = 0; i < plan.agg.size(); ++i) {
        if (plan.agg[i].op != AggOp::Pct && plan.agg[i].op != AggOp::Hist)
            continue;
        const int fi = s.spec_field[i];
        if (fi >= 0 && s.field_sketch[fi] < 0)
            s.field_sketch[fi] = static_cast<int>(s.sketch_count++);
    }
    if (s.want_occupancy) s.occ_cell_us = plan.occ_cell_us;
    for (const auto& r : plan.numeric_arg_aggs)
        if (r.op == AggOp::Pct) {
            s.dyn_sketch = true;
            break;
        }
    return s;
}

const AggSchema& ensure_schema(const ViewPlan& plan) {
    if (!plan.schema)
        plan.schema = std::make_shared<AggSchema>(make_agg_schema(plan));
    return *plan.schema;
}

void fold_event(GroupMap& map, simdjson::dom::element root,
                const ViewPlan& plan, std::string& keybuf) {
    DomSource src(root);
    fold_event_over(map, src, plan, keybuf);
}

int schema_field_index(const AggSchema& s, const std::string& field) {
    for (std::size_t i = 0; i < s.fields.size(); ++i)
        if (s.fields[i] == field) return static_cast<int>(i);
    return -1;
}

// Occupancy summary from the endpoint delta-map: busy is the exact interval
// union (prefix-sum sweep over sorted deltas), active the peak depth, total
// the raw sum(dur) (for concurrency), span the makespan (for utilization).
// Empty when the group had no timed events.
struct OccSummary {
    std::uint64_t busy = 0;
    std::uint64_t active = 0;
    std::uint64_t total = 0;
    std::uint64_t span = 0;
};

static OccSummary occupancy_summary(const AggAccum& a) {
    OccSummary o;
    o.total = a.occ_total;
    o.span = a.occ_te > a.occ_ts ? a.occ_te - a.occ_ts : 0;
    if (a.occ_deltas.empty()) return o;
    std::vector<std::pair<std::uint64_t, std::int64_t>> pts(
        a.occ_deltas.begin(), a.occ_deltas.end());
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

double finalize_value(const AggAccum& a, const ViewPlan& plan, std::size_t i) {
    const AggSchema& sch = *plan.schema;
    const auto& spec = plan.agg[i];
    const int fi = sch.spec_field[i];
    const FieldStat* fs = fi >= 0 ? &a.fields[fi] : nullptr;
    const double N = static_cast<double>(a.count);
    switch (spec.op) {
        case AggOp::Count:
            // Count() counts rows; Count(field) the field-present events.
            return spec.field.empty() ? N
                                      : (fs ? static_cast<double>(fs->n) : 0.0);
        case AggOp::Sum:
            return fs ? fs->sum : 0.0;
        case AggOp::Min:
            return fs ? fs->min : 0.0;
        case AggOp::Max:
            return fs ? fs->max : 0.0;
        case AggOp::SumSq:
            return fs ? fs->sumsq : 0.0;
        case AggOp::Mean:
            // Mean/Var/Std/Skew/Kurt delegate to the shared FieldStat (over the
            // field-present count, sample variance) so the View matches the
            // dataframe engine and pandas/polars exactly.
            return fs ? fs->mean() : 0.0;
        case AggOp::Var:
            return fs ? fs->variance(true) : 0.0;
        case AggOp::Std:
            return fs ? fs->stddev(true) : 0.0;
        case AggOp::Pct: {
            const int sk = fi >= 0 ? sch.field_sketch[fi] : -1;
            return sk >= 0 ? a.sketches[sk].quantile(spec.q) : 0.0;
        }
        case AggOp::Skew:
            return fs ? fs->skewness() : 0.0;
        case AggOp::Kurt:
            return fs ? fs->kurtosis() : 0.0;
        case AggOp::ArgMax:
            return 0.0;  // emitted as a text column
        case AggOp::Hist:
            return 0.0;  // emitted as a list<struct> column via finalize_hist
        case AggOp::SetUnion:
            return 0.0;  // emitted as a joined text column
        case AggOp::Busy:
            return static_cast<double>(occupancy_summary(a).busy);
        case AggOp::Concurrency: {
            const OccSummary o = occupancy_summary(a);
            return o.busy ? static_cast<double>(o.total) /
                                static_cast<double>(o.busy)
                          : 0.0;
        }
        case AggOp::Utilization: {
            const OccSummary o = occupancy_summary(a);
            return (o.busy && o.span) ? static_cast<double>(o.busy) /
                                            static_cast<double>(o.span)
                                      : 0.0;
        }
        case AggOp::Active:
            return static_cast<double>(occupancy_summary(a).active);
    }
    return 0.0;
}

// Like finalize_value but keeps the exact integer domain for Count/Sum/Min/Max
// so an integer field's aggregate is not rounded through a double. Every other
// op (Mean/Var/Std/quantiles/...) is inherently double.
static dftracer::utils::dataframe::FieldNum finalize_value_typed(
    const AggAccum& a, const ViewPlan& plan, std::size_t i) {
    using dftracer::utils::dataframe::FieldNum;
    using dftracer::utils::dataframe::FieldStatDomain;
    const AggSchema& sch = *plan.schema;
    const auto& spec = plan.agg[i];
    const int fi = sch.spec_field[i];
    const FieldStat* fs = fi >= 0 ? &a.fields[fi] : nullptr;

    auto exact = [&](std::int64_t bits, double dbl) -> FieldNum {
        if (!fs) return FieldNum::of(dbl);
        switch (fs->domain) {
            case FieldStatDomain::I64:
                return FieldNum::of(bits);
            case FieldStatDomain::U64:
                return FieldNum::of(std::bit_cast<std::uint64_t>(bits));
            default:
                return FieldNum::of(dbl);
        }
    };
    switch (spec.op) {
        case AggOp::Count:
            return FieldNum::of(static_cast<std::int64_t>(
                spec.field.empty() ? a.count : (fs ? fs->n : 0)));
        case AggOp::Sum:
            return fs ? exact(fs->esum, fs->sum) : FieldNum::of(0.0);
        case AggOp::Min:
            return fs ? exact(fs->emin, fs->min) : FieldNum::of(0.0);
        case AggOp::Max:
            return fs ? exact(fs->emax, fs->max) : FieldNum::of(0.0);
        default:
            return FieldNum::of(finalize_value(a, plan, i));
    }
}

// The distinct values of a SetUnion spec joined into one text cell (sorted;
// empty when the spec has no set slot).
static std::string finalize_set(const AggAccum& a, const ViewPlan& plan,
                                std::size_t i) {
    const int si = plan.schema->spec_set[i];
    if (si < 0 || static_cast<std::size_t>(si) >= a.sets.size()) return {};
    std::string out;
    for (const auto& v : a.sets[si]) {
        if (!out.empty()) out.push_back(SET_SEP);
        out += v;
    }
    return out;
}

std::vector<utilities::common::statistics::HistogramBin> finalize_hist(
    const AggAccum& a, const ViewPlan& plan, std::size_t i) {
    const AggSchema& sch = *plan.schema;
    const int fi = sch.spec_field[i];
    const int sk = fi >= 0 ? sch.field_sketch[fi] : -1;
    if (sk < 0) return {};
    return a.sketches[sk].bins();
}

std::pair<std::string, bool> source_agg_field(const ViewPlan& plan) {
    std::string field;
    for (const auto& spec : plan.agg) {
        // ArgMax reduces over `by`; the others over `field`. Count() is
        // neutral.
        const std::string& f = spec.op == AggOp::ArgMax ? spec.by : spec.field;
        if (f.empty()) continue;
        if (field.empty())
            field = f;
        else if (field != f)
            return {"", false};
    }
    return {field, true};
}

const AggSpec* find_argmax(const ViewPlan& plan) {
    for (const auto& spec : plan.agg)
        if (spec.op == AggOp::ArgMax) return &spec;
    return nullptr;
}

void merge_accum(AggAccum& da, const AggAccum& sa, const ViewPlan& plan) {
    // plan.schema is prebuilt by the terminal (merge runs concurrently across
    // shards, so it must not lazily build here).
    const AggSchema& sch = *plan.schema;
    if (da.count == 0) {
        da.keys = sa.keys;
        da.fields.resize(sch.fields.size());
        da.argmax.resize(sch.argmax_count);
        da.sketches.resize(sch.sketch_count);
        da.sets.resize(sch.set_count);
    }
    da.count += sa.count;
    for (std::size_t i = 0; i < da.fields.size() && i < sa.fields.size(); ++i)
        da.fields[i].merge(sa.fields[i]);
    for (std::size_t i = 0; i < da.argmax.size() && i < sa.argmax.size(); ++i) {
        const auto& s = sa.argmax[i];
        if (!s.has) continue;
        auto& d = da.argmax[i];
        if (!d.has || s.by > d.by) d = s;
    }
    for (std::size_t i = 0; i < da.sketches.size() && i < sa.sketches.size();
         ++i)
        da.sketches[i].merge(sa.sketches[i]);
    for (std::size_t i = 0; i < da.sets.size() && i < sa.sets.size(); ++i)
        da.sets[i].insert(sa.sets[i].begin(), sa.sets[i].end());
    for (const auto& [name, sm] : sa.dyn) da.dyn[name].merge(sm);
    for (const auto& [name, sk] : sa.dyn_sketches)
        da.dyn_sketches[name].merge(sk);
    if (!sa.occ_deltas.empty() || sa.occ_total) {
        da.occ_total += sa.occ_total;
        if (sa.occ_ts < da.occ_ts) da.occ_ts = sa.occ_ts;
        if (sa.occ_te > da.occ_te) da.occ_te = sa.occ_te;
        for (const auto& [t, dlt] : sa.occ_deltas) da.occ_deltas[t] += dlt;
        if (sa.occ_cell_us) da.occ_cell_us = sa.occ_cell_us;
    }
}

void merge_maps(GroupMap& dst, const GroupMap& src, const ViewPlan& plan) {
    for (const auto& [k, sa] : src) merge_accum(dst[k], sa, plan);
}

const GroupResolver* ensure_resolver(const ViewPlan& plan) {
    bool needs = false;
    for (const auto& gk : plan.group_by)
        if (gk.kind == GroupKey::Kind::FilePath ||
            gk.kind == GroupKey::Kind::FileName ||
            gk.kind == GroupKey::Kind::HostName ||
            gk.kind == GroupKey::Kind::Rank) {
            needs = true;
            break;
        }
    if (!needs) return nullptr;
    if (!plan.resolver) {
        std::vector<std::string> paths;
        for (const auto& f : plan.files) {
            if (f.index_path.empty()) continue;
            bool seen = false;
            for (const auto& p : paths)
                if (p == f.index_path) {
                    seen = true;
                    break;
                }
            if (!seen) paths.push_back(f.index_path);
        }
        plan.resolver = std::make_shared<GroupResolver>(paths);
    }
    return plan.resolver.get();
}

void apply_ranks(const ViewPlan& plan,
                 std::unordered_map<std::uint64_t, std::string>& ranks) {
    if (ranks.empty()) return;
    ensure_resolver(plan);  // Rank counts as a resolved key, so this builds it
    if (!plan.resolver) return;
    for (auto& [pid, rank] : ranks)
        plan.resolver->set_rank(std::to_string(pid), std::move(rank));
    ranks.clear();
}

std::string resolve_group_value(const GroupResolver& r, GroupKey::Kind kind,
                                const std::string& hash) {
    if (kind == GroupKey::Kind::FilePath) return r.file_path(hash);
    if (kind == GroupKey::Kind::FileName) {
        const std::string& p = r.file_path(hash);
        const std::size_t slash = p.find_last_of('/');
        return slash == std::string::npos ? p : p.substr(slash + 1);
    }
    if (kind == GroupKey::Kind::HostName) return r.host_name(hash);
    if (kind == GroupKey::Kind::Rank) return r.rank(hash);
    return hash;
}

std::string apply_group_transform(const GroupKey& gk, std::string v) {
    switch (gk.transform) {
        case GroupKey::Transform::None:
            return v;
        case GroupKey::Transform::Dirname: {
            const auto slash = v.find_last_of('/');
            if (slash == std::string::npos) return std::string();
            return v.substr(0, slash);
        }
        case GroupKey::Transform::Basename: {
            const auto slash = v.find_last_of('/');
            return slash == std::string::npos ? v : v.substr(slash + 1);
        }
        case GroupKey::Transform::Lower: {
            for (auto& ch : v)
                ch = static_cast<char>(
                    ::tolower(static_cast<unsigned char>(ch)));
            return v;
        }
        case GroupKey::Transform::Bucket:
            for (const auto& b : gk.transform_args)
                if (v.find(b) != std::string::npos) return b;
            return std::string();
    }
    return v;
}

void resolve_group_keys(GroupMap& map, const ViewPlan& plan) {
    const GroupResolver* r = ensure_resolver(plan);
    // Transforms coarsen keys that need no resolver at all (lower(cat)), so a
    // missing resolver must not skip them.
    const bool any_transform = std::any_of(
        plan.group_by.begin(), plan.group_by.end(),
        [](const auto& g) { return g.transform != GroupKey::Transform::None; });
    if (!r && !any_transform) return;
    // AggAccum.keys layout is [time_bucket?] + group_by, so the bucket takes
    // index 0 when present.
    const std::size_t off = plan.time_bucket_us > 0 ? 1 : 0;
    GroupMap out;
    std::string newkey;
    for (auto& [k, accum] : map) {
        (void)k;
        for (std::size_t j = 0; j < plan.group_by.size(); ++j) {
            const std::size_t idx = off + j;
            if (idx >= accum.keys.size()) continue;
            std::string v = std::move(accum.keys[idx]);
            if (r) v = resolve_group_value(*r, plan.group_by[j].kind, v);
            accum.keys[idx] =
                apply_group_transform(plan.group_by[j], std::move(v));
        }
        newkey.clear();
        for (const auto& part : accum.keys) {
            newkey += part;
            newkey += GROUP_SEP;
        }
        merge_accum(out[newkey], accum, plan);
    }
    map = std::move(out);
}

std::string group_col_name(const GroupKey& gk) {
    switch (gk.kind) {
        case GroupKey::Kind::Name:
            return "name";
        case GroupKey::Kind::Cat:
            return "cat";
        case GroupKey::Kind::Pid:
            return "pid";
        case GroupKey::Kind::Tid:
            return "tid";
        case GroupKey::Kind::Fhash:
            return "fhash";
        case GroupKey::Kind::Hhash:
            return "hhash";
        case GroupKey::Kind::IoCat:
            return "io_cat";
        case GroupKey::Kind::AccPat:
            return "acc_pat";
        case GroupKey::Kind::FilePath:
            return "file_path";
        case GroupKey::Kind::FileName:
            return "file_name";
        case GroupKey::Kind::HostName:
            return "host_name";
        case GroupKey::Kind::Rank:
            return "rank";
        case GroupKey::Kind::Arg:
        case GroupKey::Kind::Field:
            return gk.arg;
    }
    return {};
}

// A discovered numeric arg reduces over the values that were present (N =
// fs.n), matching the legacy bare-mean (sum/n). Only FieldStat-derivable ops
// are reachable here; a Pct/Hist over a dyn arg needs a per-arg sketch and is
// rejected at the builder.
static double reduce_dyn(const FieldStat& fs, AggOp op, double /*q*/) {
    switch (op) {
        case AggOp::Count:
            return static_cast<double>(fs.n);
        case AggOp::Sum:
            return fs.sum;
        case AggOp::Min:
            return fs.n ? fs.min : 0.0;
        case AggOp::Max:
            return fs.n ? fs.max : 0.0;
        case AggOp::SumSq:
            return fs.sumsq;
        // Moment reductions delegate to the shared FieldStat (field-present n,
        // sample variance) so the dyn path matches the named-field path, the
        // dataframe engine, and pandas/polars.
        case AggOp::Mean:
            return fs.mean();
        case AggOp::Var:
            return fs.variance(true);
        case AggOp::Std:
            return fs.stddev(true);
        case AggOp::Skew:
            return fs.skewness();
        case AggOp::Kurt:
            return fs.kurtosis();
        default:
            return 0.0;
    }
}

// Column name for a per-arg reduction: `<op>_<arg>`, matching the named-field
// convention (sum_field). The legacy mean path keeps the bare arg name. Pct
// uses the spec's out_name prefix (e.g. "p90_" from the pNN shorthand) so the
// quantile is legible; otherwise it falls back to "pct_".
static std::string dyn_col_name(const AggSpec& spec, const std::string& key) {
    switch (spec.op) {
        case AggOp::Count:
            return "count_" + key;
        case AggOp::Sum:
            return "sum_" + key;
        case AggOp::Min:
            return "min_" + key;
        case AggOp::Max:
            return "max_" + key;
        case AggOp::SumSq:
            return "sumsq_" + key;
        case AggOp::Mean:
            return "mean_" + key;
        case AggOp::Var:
            return "var_" + key;
        case AggOp::Std:
            return "std_" + key;
        case AggOp::Skew:
            return "skew_" + key;
        case AggOp::Kurt:
            return "kurt_" + key;
        case AggOp::Pct: {
            std::string pfx = spec.out_name.empty() ? "pct_" : spec.out_name;
            if (pfx.back() != '_') pfx += '_';
            return pfx + key;
        }
        default:
            return key;
    }
}

std::string agg_col_name(const AggSpec& spec) {
    if (!spec.out_name.empty()) return spec.out_name;
    switch (spec.op) {
        case AggOp::Count:
            return "count";
        case AggOp::Sum:
            return "sum_" + spec.field;
        case AggOp::SumSq:
            return "sumsq_" + spec.field;
        case AggOp::Min:
            return "min_" + spec.field;
        case AggOp::Max:
            return "max_" + spec.field;
        case AggOp::Mean:
            return "mean_" + spec.field;
        case AggOp::Var:
            return "var_" + spec.field;
        case AggOp::Std:
            return "std_" + spec.field;
        case AggOp::Pct:
            return "pct_" + spec.field;  // Python sets a precise out_name
        case AggOp::Skew:
            return "skew_" + spec.field;
        case AggOp::Kurt:
            return "kurt_" + spec.field;
        case AggOp::Hist:
            return "hist_" + spec.field;
        case AggOp::ArgMax:
            return "argmax_" + spec.field;
        case AggOp::SetUnion:
            return "set_" + spec.field;
        case AggOp::Busy:
            return "busy";
        case AggOp::Concurrency:
            return "concurrency";
        case AggOp::Utilization:
            return "utilization";
        case AggOp::Active:
            return "active";
    }
    return {};
}

// Materialize the group map straight into a columnar dataframe::DataFrame.
// Column order: group keys (time_bucket first when bucketing), value columns
// (count or per-agg, then dyn), text columns, hist columns. Types: count is
// Int64, integer Sum/Min/Max exact as Int64/Uint64, other value columns
// Float64; text is String; hist is list<struct<lo, hi, count>>.
dataframe::DataFrame to_batch(const GroupMap& map, const ViewPlan& plan) {
    std::vector<std::string> group_cols, value_cols, text_cols, hist_cols;
    if (plan.time_bucket_us > 0) group_cols.push_back("time_bucket");
    for (const auto& gk : plan.group_by)
        group_cols.push_back(group_col_name(gk));
    const bool count_col = plan.agg.empty();
    if (count_col) {
        value_cols.push_back("count");
    } else {
        for (const auto& spec : plan.agg) {
            if (spec.op == AggOp::ArgMax || spec.op == AggOp::SetUnion)
                text_cols.push_back(agg_col_name(spec));
            else if (spec.op == AggOp::Hist)
                hist_cols.push_back(agg_col_name(spec));
            else
                value_cols.push_back(agg_col_name(spec));
        }
    }
    struct DynCol {
        std::string key;
        AggSpec spec;
    };
    std::vector<DynCol> dyn_cols;
    if (plan.auto_numeric_metrics) {
        std::set<std::string> names;
        for (const auto& [k, a] : map) {
            (void)k;
            for (const auto& [name, m] : a.dyn) names.insert(name);
        }
        if (plan.numeric_arg_aggs.empty()) {
            // Legacy: one bare-named per-arg mean column.
            for (const auto& n : names) {
                dyn_cols.push_back({n, AggSpec(AggOp::Mean)});
                value_cols.push_back(n);
            }
        } else {
            for (const auto& n : names)
                for (const auto& spec : plan.numeric_arg_aggs) {
                    dyn_cols.push_back({n, spec});
                    value_cols.push_back(dyn_col_name(spec, n));
                }
        }
    }
    // Effective grid resolution, so a caller can tell a grid-derived busy from
    // a clamped one.
    bool occ_cell_col = false;
    for (const auto& spec : plan.agg)
        if (spec.op == AggOp::Busy || spec.op == AggOp::Concurrency ||
            spec.op == AggOp::Utilization) {
            occ_cell_col = true;
            break;
        }
    if (occ_cell_col) value_cols.push_back("busy_cell_us");

    const std::size_t ng = map.size();
    std::vector<std::vector<std::string>> gk(group_cols.size());
    for (auto& v : gk) v.reserve(ng);
    // One typed cell per (group, value column); reconciled to a single column
    // type at build time, so Sum/Min/Max of an integer field emit an exact
    // Int64/Uint64 column rather than a rounded Float64.
    std::vector<std::vector<dftracer::utils::dataframe::FieldNum>> vcells(
        value_cols.size());
    for (auto& v : vcells) v.reserve(ng);
    std::vector<std::vector<std::string>> tvals(text_cols.size());
    for (auto& v : tvals) v.reserve(ng);
    struct HistBuild {
        std::vector<std::int32_t> off{0};
        std::vector<double> lo, hi;
        std::vector<std::uint64_t> cnt;
    };
    std::vector<HistBuild> hvals(hist_cols.size());

    const AggSchema& sch = ensure_schema(plan);
    for (const auto& [k, a] : map) {
        (void)k;
        for (std::size_t i = 0; i < group_cols.size(); ++i)
            gk[i].push_back(a.keys[i]);
        std::size_t vc = 0, ti = 0, hi = 0;
        if (count_col) {
            vcells[vc++].push_back(dftracer::utils::dataframe::FieldNum::of(
                static_cast<std::int64_t>(a.count)));
        } else {
            for (std::size_t i = 0; i < plan.agg.size(); ++i) {
                const AggOp op = plan.agg[i].op;
                if (op == AggOp::ArgMax) {
                    tvals[ti++].push_back(a.argmax[sch.spec_argmax[i]].repr);
                } else if (op == AggOp::SetUnion) {
                    tvals[ti++].push_back(finalize_set(a, plan, i));
                } else if (op == AggOp::Hist) {
                    auto bins = finalize_hist(a, plan, i);
                    auto& h = hvals[hi++];
                    for (const auto& b : bins) {
                        h.lo.push_back(b.lower);
                        h.hi.push_back(b.upper);
                        h.cnt.push_back(b.count);
                    }
                    h.off.push_back(static_cast<std::int32_t>(h.lo.size()));
                } else {
                    vcells[vc++].push_back(finalize_value_typed(a, plan, i));
                }
            }
        }
        for (const auto& dc : dyn_cols) {
            double v = 0.0;
            if (dc.spec.op == AggOp::Pct) {
                auto it = a.dyn_sketches.find(dc.key);
                if (it != a.dyn_sketches.end())
                    v = it->second.quantile(dc.spec.q);
            } else if (auto it = a.dyn.find(dc.key); it != a.dyn.end()) {
                v = reduce_dyn(it->second, dc.spec.op, dc.spec.q);
            }
            vcells[vc++].push_back(dftracer::utils::dataframe::FieldNum::of(v));
        }
        if (occ_cell_col)
            vcells[vc++].push_back(dftracer::utils::dataframe::FieldNum::of(
                static_cast<std::int64_t>(sch.occ_cell_us)));
    }

    dataframe::DataFrame batch;
    const std::int64_t nrows = static_cast<std::int64_t>(ng);
    for (std::size_t i = 0; i < group_cols.size(); ++i) {
        batch.names.push_back(group_cols[i]);
        batch.columns.push_back(dataframe::Series::strings(gk[i]));
    }
    // A value column is Int64/Uint64 only if every group shares that integer
    // domain; a single float (or mixed) cell makes the whole column Float64.
    auto build_value_column =
        [&](const std::vector<dftracer::utils::dataframe::FieldNum>& cells) {
            using dftracer::utils::dataframe::FieldStatDomain;
            FieldStatDomain dom =
                cells.empty() ? FieldStatDomain::F64 : cells[0].domain;
            for (const auto& c : cells)
                if (c.domain != dom) {
                    dom = FieldStatDomain::F64;
                    break;
                }
            if (dom == FieldStatDomain::I64) {
                std::vector<std::int64_t> v;
                v.reserve(cells.size());
                for (const auto& c : cells) v.push_back(c.i);
                return dataframe::Series::flat(dataframe::TypeId::Int64,
                                               v.data(), nrows);
            }
            if (dom == FieldStatDomain::U64) {
                std::vector<std::uint64_t> v;
                v.reserve(cells.size());
                for (const auto& c : cells) v.push_back(c.u);
                return dataframe::Series::flat(dataframe::TypeId::Uint64,
                                               v.data(), nrows);
            }
            std::vector<double> v;
            v.reserve(cells.size());
            for (const auto& c : cells) v.push_back(c.as_double());
            return dataframe::Series::flat_f64(v.data(), nrows);
        };
    for (std::size_t i = 0; i < value_cols.size(); ++i) {
        batch.names.push_back(value_cols[i]);
        batch.columns.push_back(build_value_column(vcells[i]));
    }
    for (std::size_t i = 0; i < text_cols.size(); ++i) {
        batch.names.push_back(text_cols[i]);
        batch.columns.push_back(dataframe::Series::strings(tvals[i]));
    }
    for (std::size_t i = 0; i < hist_cols.size(); ++i) {
        auto& h = hvals[i];
        const auto n = static_cast<std::int64_t>(h.lo.size());
        std::vector<dataframe::Series> fields;
        fields.push_back(dataframe::Series::flat(dataframe::TypeId::Float64,
                                                 h.lo.data(), n));
        fields.push_back(dataframe::Series::flat(dataframe::TypeId::Float64,
                                                 h.hi.data(), n));
        fields.push_back(dataframe::Series::flat(dataframe::TypeId::Uint64,
                                                 h.cnt.data(), n));
        dataframe::Series st = dataframe::Series::structs({"lo", "hi", "count"},
                                                          std::move(fields));
        batch.names.push_back(hist_cols[i]);
        batch.columns.push_back(dataframe::Series::list(h.off, std::move(st)));
    }
    return batch;
}

void project_columns(dftracer::utils::dataframe::DataFrame& batch,
                     const std::vector<std::string>& select) {
    if (select.empty()) return;
    const std::set<std::string> keep(select.begin(), select.end());
    std::vector<std::string> names;
    std::vector<dftracer::utils::dataframe::Series> columns;
    for (std::size_t i = 0; i < batch.names.size(); ++i) {
        if (keep.count(batch.names[i])) {
            names.push_back(std::move(batch.names[i]));
            columns.push_back(std::move(batch.columns[i]));
        }
    }
    batch.names = std::move(names);
    batch.columns = std::move(columns);
}

dataframe::DataFrame finalize_collect_batch(const GroupMap& map,
                                            const ViewPlan& plan) {
    return to_batch(map, plan);
}

dataframe::DataFrame apply_agg_post_ops(dataframe::DataFrame batch,
                                        const ViewPlan& plan) {
    if (!plan.sort_col.empty())
        batch = batch.sort_by(plan.sort_col, plan.sort_desc);
    if (!plan.topk_col.empty())
        batch = batch.topk(plan.topk_col, plan.topk_k, plan.topk_largest);
    if (plan.offset || plan.limit) {
        const std::int64_t off = static_cast<std::int64_t>(plan.offset);
        const std::int64_t len = plan.limit
                                     ? static_cast<std::int64_t>(plan.limit)
                                     : batch.num_rows();
        batch = batch.slice(off, len);
    }
    project_columns(batch, plan.select);
    return batch;
}

}  // namespace dftracer::utils::trace::views::detail
