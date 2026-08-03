#include <dftracer/utils/core/common/to_chars.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/reserved_args.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/agg_fold.h>
#include <dftracer/utils/utilities/composites/dft/views/event_source.h>
#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>
#include <dftracer/utils/utilities/composites/dft/views/view_resolver.h>

#include <cctype>
#include <cmath>
#include <set>
#include <string>
#include <type_traits>

namespace dftracer::utils::utilities::composites::dft::views::detail {

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
    namespace dfi = composites::dft::internal;
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
            // Mean/Var/Std use N = group rows (matches the pre-FieldStat
            // behavior), which differs from fs->n only for sparse fields.
            return (fs && a.count) ? fs->sum / N : 0.0;
        case AggOp::Var:
        case AggOp::Std: {
            double var = 0.0;
            if (fs && a.count) {
                const double mean = fs->sum / N;
                var = fs->sumsq / N - mean * mean;
                if (var < 0.0) var = 0.0;  // clamp round-off
            }
            return spec.op == AggOp::Std ? std::sqrt(var) : var;
        }
        case AggOp::Pct: {
            const int sk = fi >= 0 ? sch.field_sketch[fi] : -1;
            return sk >= 0 ? a.sketches[sk].quantile(spec.q) : 0.0;
        }
        case AggOp::Skew:
        case AggOp::Kurt: {
            // Population skewness/excess-kurtosis from the raw power sums, over
            // N = group rows (matching Mean/Var).
            if (!fs || a.count == 0) return 0.0;
            const double mu = fs->sum / N;
            const double cm2 = fs->sumsq / N - mu * mu;
            if (cm2 <= 0.0) return 0.0;
            const double cm3 =
                fs->m3 / N - 3.0 * mu * (fs->sumsq / N) + 2.0 * mu * mu * mu;
            if (spec.op == AggOp::Skew) return cm3 / std::pow(cm2, 1.5);
            const double cm4 = fs->m4 / N - 4.0 * mu * (fs->m3 / N) +
                               6.0 * mu * mu * (fs->sumsq / N) -
                               3.0 * mu * mu * mu * mu;
            return cm4 / (cm2 * cm2) - 3.0;
        }
        case AggOp::ArgMax:
            return 0.0;  // emitted as a text column
        case AggOp::Hist:
            return 0.0;  // emitted as a list<struct> column via finalize_hist
        case AggOp::SetUnion:
            return 0.0;  // emitted as a joined text column
    }
    return 0.0;
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

std::vector<common::statistics::HistogramBin> finalize_hist(
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
}

void merge_maps(GroupMap& dst, const GroupMap& src, const ViewPlan& plan) {
    for (const auto& [k, sa] : src) merge_accum(dst[k], sa, plan);
}

const GroupResolver* ensure_resolver(const ViewPlan& plan) {
    bool needs = false;
    for (const auto& gk : plan.group_by)
        if (gk.kind == GroupKey::Kind::FilePath ||
            gk.kind == GroupKey::Kind::FileName ||
            gk.kind == GroupKey::Kind::HostName) {
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

std::string resolve_group_value(const GroupResolver& r, GroupKey::Kind kind,
                                const std::string& hash) {
    if (kind == GroupKey::Kind::FilePath) return r.file_path(hash);
    if (kind == GroupKey::Kind::FileName) {
        const std::string& p = r.file_path(hash);
        const std::size_t slash = p.find_last_of('/');
        return slash == std::string::npos ? p : p.substr(slash + 1);
    }
    if (kind == GroupKey::Kind::HostName) return r.host_name(hash);
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
        case GroupKey::Kind::Arg:
            return gk.arg;
    }
    return {};
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
    }
    return {};
}

ResultTable to_result_table(const GroupMap& map, const ViewPlan& plan) {
    ResultTable t;
    if (plan.time_bucket_us > 0) t.group_columns.push_back("time_bucket");
    for (const auto& gk : plan.group_by)
        t.group_columns.push_back(group_col_name(gk));
    const bool count_col = plan.agg.empty();
    if (count_col) {
        t.value_columns.push_back("count");
    } else {
        for (const auto& spec : plan.agg) {
            if (spec.op == AggOp::ArgMax || spec.op == AggOp::SetUnion)
                t.text_columns.push_back(agg_col_name(spec));
            else if (spec.op == AggOp::Hist)
                t.hist_columns.push_back(agg_col_name(spec));
            else
                t.value_columns.push_back(agg_col_name(spec));
        }
    }

    // Dynamic numeric-arg columns: the stable, sorted union of every arg seen
    // across groups. Each is emitted as the per-group mean (0 where absent).
    std::vector<std::string> dyn_cols;
    if (plan.auto_numeric_metrics) {
        std::set<std::string> names;
        for (const auto& [k, a] : map) {
            (void)k;
            for (const auto& [name, m] : a.dyn) names.insert(name);
        }
        dyn_cols.assign(names.begin(), names.end());
        for (const auto& n : dyn_cols) t.value_columns.push_back(n);
    }

    const AggSchema& sch = ensure_schema(plan);
    for (const auto& [k, a] : map) {
        (void)k;
        ResultRow row;
        row.keys = a.keys;
        if (count_col) {
            row.values.push_back(static_cast<double>(a.count));
        } else {
            for (std::size_t i = 0; i < plan.agg.size(); ++i) {
                if (plan.agg[i].op == AggOp::ArgMax)
                    row.texts.push_back(a.argmax[sch.spec_argmax[i]].repr);
                else if (plan.agg[i].op == AggOp::SetUnion)
                    row.texts.push_back(finalize_set(a, plan, i));
                else if (plan.agg[i].op == AggOp::Hist)
                    row.hists.push_back(finalize_hist(a, plan, i));
                else
                    row.values.push_back(finalize_value(a, plan, i));
            }
        }
        for (const auto& n : dyn_cols) {
            auto it = a.dyn.find(n);
            row.values.push_back(it != a.dyn.end() && it->second.n
                                     ? it->second.sum /
                                           static_cast<double>(it->second.n)
                                     : 0.0);
        }
        t.rows.push_back(std::move(row));
    }
    return t;
}

void project_columns(ResultTable& table,
                     const std::vector<std::string>& select) {
    if (select.empty()) return;
    const std::set<std::string> keep(select.begin(), select.end());

    // For one column category: keep only selected columns (in result order) and
    // drop the aligned cell from every row via a pointer-to-member.
    auto filter = [&](std::vector<std::string>& cols, auto row_member) {
        std::vector<std::size_t> keep_idx;
        for (std::size_t i = 0; i < cols.size(); ++i)
            if (keep.count(cols[i])) keep_idx.push_back(i);
        if (keep_idx.size() == cols.size()) return;

        std::vector<std::string> new_cols;
        new_cols.reserve(keep_idx.size());
        for (auto i : keep_idx) new_cols.push_back(std::move(cols[i]));
        cols = std::move(new_cols);

        for (auto& row : table.rows) {
            auto& cells = row.*row_member;
            std::remove_reference_t<decltype(cells)> kept;
            kept.reserve(keep_idx.size());
            for (auto i : keep_idx)
                if (i < cells.size()) kept.push_back(std::move(cells[i]));
            cells = std::move(kept);
        }
    };

    filter(table.group_columns, &ResultRow::keys);
    filter(table.value_columns, &ResultRow::values);
    filter(table.text_columns, &ResultRow::texts);
    filter(table.hist_columns, &ResultRow::hists);
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
