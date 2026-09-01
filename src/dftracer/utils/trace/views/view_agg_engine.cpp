#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/aggfold.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/native_row_fold.h>
#include <dftracer/utils/trace/views/view_agg_engine.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_executor.h>
#include <dftracer/utils/trace/views/view_scan.h>
#include <dftracer/utils/trace/views/view_source.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <iterator>
#include <optional>
#include <set>
#include <string>
#include <vector>

namespace dftracer::utils::trace::views::detail {

namespace dataframe = dftracer::utils::dataframe;

namespace {

// Value/by fields eligible for an agg spec: the always-present top-level
// event fields. ViewSource now fixes the streamed schema to the raw scan's
// select list (see view_source.cpp), so an arg/fhash/hhash value column is no
// longer unsafe to reference this way, but widening agg targets past the
// stable set is a later phase; this phase only widens group KEYS.
bool is_stable_field(const std::string& f) {
    return f.empty() || f == "name" || f == "cat" || f == "pid" || f == "tid" ||
           f == "ts" || f == "dur";
}

bool agg_op_engine_supported(AggOp op) {
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
        case AggOp::SumSq:
        case AggOp::Pct:
        case AggOp::Hist:
        case AggOp::ArgMax:
        case AggOp::SetUnion:
            return true;
        case AggOp::Busy:
        case AggOp::Concurrency:
        case AggOp::Utilization:
        case AggOp::Active:
            return false;
    }
    return false;
}

dataframe::Agg to_engine_agg(AggOp op) {
    switch (op) {
        case AggOp::Count:
            return dataframe::Agg::Count;
        case AggOp::Sum:
            return dataframe::Agg::Sum;
        case AggOp::Min:
            return dataframe::Agg::Min;
        case AggOp::Max:
            return dataframe::Agg::Max;
        case AggOp::Mean:
            return dataframe::Agg::Mean;
        case AggOp::Var:
            return dataframe::Agg::Var;
        case AggOp::Std:
            return dataframe::Agg::Std;
        case AggOp::Skew:
            return dataframe::Agg::Skew;
        case AggOp::Kurt:
            return dataframe::Agg::Kurt;
        case AggOp::SumSq:
            return dataframe::Agg::SumSq;
        case AggOp::Pct:
            return dataframe::Agg::Pct;
        case AggOp::Hist:
            return dataframe::Agg::Hist;
        case AggOp::ArgMax:
            return dataframe::Agg::ArgMax;
        case AggOp::SetUnion:
            return dataframe::Agg::SetUnion;
        case AggOp::Busy:
        case AggOp::Concurrency:
        case AggOp::Utilization:
        case AggOp::Active:
            break;
    }
    throw DFTUtilsException::cat(
        ErrorCode::INTERNAL,
        "agg engine: occupancy ops are not engine-eligible");
}

dataframe::GroupAgg to_group_agg(const AggSpec& spec) {
    dataframe::GroupAgg g;
    g.op = to_engine_agg(spec.op);
    g.out = agg_col_name(spec);
    g.param = spec.q;
    if (spec.op == AggOp::ArgMax) {
        g.column = spec.field;
        g.by = spec.by;
    } else if (spec.op != AggOp::Count) {
        g.column = spec.field;
    }
    return g;
}

// to_batch's group key column is always a String (the fold's key is text);
// agg_finalize instead keeps a non-string key's native domain collapsed to
// Int64. Render that back to the same decimal text the GroupMap path would
// have produced (plain integer text - the group key values reachable here are
// pid/tid, always small enough that the int64/uint64 bit pattern round-trips
// through decimal identically).
dataframe::Series key_column_to_string(const dataframe::Series& col) {
    using dataframe::TypeId;
    if (col.type() == TypeId::String) return col.share();
    if (col.type() == TypeId::Int64) {
        const std::int64_t n = col.length();
        std::vector<std::string> vals(static_cast<std::size_t>(n));
        const std::int64_t* d = col.data<std::int64_t>();
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] = std::to_string(d[i]);
        return dataframe::Series::strings(vals);
    }
    throw DFTUtilsException::cat(
        ErrorCode::INTERNAL, "agg engine: unexpected group-key column type");
}

// Keys whose group column is an opaque identifier the post-aggregation re-key
// pass relabels to a human-readable name: fhash/hhash for resolved-name keys,
// pid for Rank (the PR-metadata rank map keys on pid).
bool key_is_resolved(GroupKey::Kind kind) {
    return kind == GroupKey::Kind::FilePath ||
           kind == GroupKey::Kind::FileName ||
           kind == GroupKey::Kind::HostName || kind == GroupKey::Kind::Rank;
}

// The raw scan/group-by field a key groups on: fhash/hhash for a resolved
// name key (the fold groups on the hash, a bijection, and relabels to the
// resolved name only after aggregation), pid for Rank (the rank map keys on
// pid, matching agg_fold.h's append_group_dim), a group-key-string sentinel for
// an Arg/Field key (rendered like the GroupMap fold, then relabeled to
// group_col_name after aggregation), else the key's own column.
std::string key_group_field(const GroupKey& gk) {
    switch (gk.kind) {
        case GroupKey::Kind::FilePath:
        case GroupKey::Kind::FileName:
            return "fhash";
        case GroupKey::Kind::HostName:
            return "hhash";
        case GroupKey::Kind::Rank:
            return "pid";
        case GroupKey::Kind::Arg:
            return std::string(AGG_KEY_ARG_PREFIX) + gk.arg;
        case GroupKey::Kind::Field:
            return std::string(AGG_KEY_FIELD_PREFIX) + gk.arg;
        default:
            return group_col_name(gk);
    }
}

// Resolve one already-stringified hash group-key column to its resolved name,
// matching resolve_group_keys/resolve_group_value exactly (same GroupResolver,
// same FileName-from-FilePath basename derivation).
dataframe::Series resolve_key_column(const dataframe::Series& hashes,
                                     const GroupResolver& resolver,
                                     GroupKey::Kind kind) {
    const std::int64_t n = hashes.length();
    std::vector<std::string> vals(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        vals[static_cast<std::size_t>(i)] = resolve_group_value(
            resolver, kind, std::string(hashes.string_at(i)));
    return dataframe::Series::strings(vals);
}

// Rank is a query-time side channel: the pid -> rank map lives in PR metadata
// records, not the index or the event columns the engine streams. Harvest it
// with the same AggFold logic the GroupMap path uses (identical records over
// the same trace, so the map is byte-identical) and feed it to the shared
// resolver the post-aggregation re-key reads.
coro::CoroTask<void> harvest_ranks(const ViewPlan& plan) {
    ViewPlan hp = plan;
    hp.group_by.assign(1, GroupKey::rank());
    hp.agg.clear();
    hp.numeric_arg_aggs.clear();
    hp.auto_numeric_metrics = false;
    hp.time_bucket_us = 0;
    hp.bucket_origin_us = 0;
    hp.bucket_origin_min = false;
    hp.materialize = false;
    hp.sort_col.clear();
    hp.topk_col.clear();
    hp.offset = 0;
    hp.limit = 0;
    hp.select.clear();
    hp.schema.reset();
    ensure_schema(hp);
    ViewDefinition vdef = make_vdef(hp, /*for_aggregation=*/true);
    dftracer::utils::StringIntern intern;
    AggFold agg(hp, intern);
    std::array<Fold*, 1> folds{&agg};
    co_await fuse(hp, vdef, folds, intern);
    apply_ranks(plan, agg.ranks());
}

// The numeric args auto_numeric_metrics discovers are data-dependent (the arg
// set is only known after a scan; see docs/plans
// lazyframe_async_unification 3.3 / 5.4). Run the SAME GroupMap fold that the
// legacy path folds with (a single group, no per-arg sketch) and read the
// sorted union of the arg names it saw - byte-identical to the set to_batch
// would emit (fold_numeric_args_t discovers, reserved/pre-agg-filtered, plus
// the io-cat-derived "size"). Its finish_map is discarded; only the names are
// kept, then fed back as engine value columns.
coro::CoroTask<std::vector<std::string>> harvest_numeric_arg_names(
    const ViewPlan& plan) {
    ViewPlan hp = plan;
    hp.group_by.clear();
    hp.agg.clear();
    hp.auto_numeric_metrics = true;
    hp.numeric_arg_aggs.clear();
    hp.time_bucket_us = 0;
    hp.bucket_origin_us = 0;
    hp.bucket_origin_min = false;
    hp.materialize = false;
    hp.sort_col.clear();
    hp.topk_col.clear();
    hp.offset = 0;
    hp.limit = 0;
    hp.select.clear();
    hp.schema.reset();
    ensure_schema(hp);
    ViewDefinition vdef = make_vdef(hp, /*for_aggregation=*/true);
    dftracer::utils::StringIntern intern;
    AggFold agg(hp, intern);
    std::array<Fold*, 1> folds{&agg};
    co_await fuse(hp, vdef, folds, intern);
    std::set<std::string> names;  // to_batch unions the dyn keys via std::set
    GroupMap m = agg.finish_map();
    for (const auto& [k, a] : m) {
        (void)k;
        for (const auto& [name, stat] : a.dyn) {
            (void)stat;
            names.insert(name);
        }
    }
    co_return std::vector<std::string>(names.begin(), names.end());
}

}  // namespace

bool agg_engine_eligible(const ViewPlan& plan) {
    if (plan.group_by.empty() && plan.time_bucket_us == 0) return false;
    for (const GroupKey& gk : plan.group_by) {
        switch (gk.kind) {
            case GroupKey::Kind::Name:
            case GroupKey::Kind::Pid:
            case GroupKey::Kind::Tid:
            case GroupKey::Kind::Fhash:
            case GroupKey::Kind::Hhash:
            case GroupKey::Kind::Cat:
            case GroupKey::Kind::FilePath:
            case GroupKey::Kind::FileName:
            case GroupKey::Kind::HostName:
            case GroupKey::Kind::IoCat:
            case GroupKey::Kind::Rank:
            case GroupKey::Kind::Arg:
            case GroupKey::Kind::Field:
                break;
            case GroupKey::Kind::AccPat:
                return false;
        }
        if (gk.transform != GroupKey::Transform::None) return false;
    }
    // The dyn (auto_numeric_metrics) path feeds each discovered numeric arg as
    // a Float64 value column, so every reduction maps to an engine Agg EXCEPT
    // Count: the engine's Count is the group's row count, while the dyn Count
    // is the arg-present count (FieldStat::n), which the engine has no agg for.
    // A dyn Count therefore stays on GroupMap (same reason as Count(field)
    // below).
    for (const AggSpec& r : plan.numeric_arg_aggs)
        if (r.op == AggOp::Count) return false;
    // Not yet converged: materialize writes a rollup CF; Hist emits a nested
    // column the engine's spill cannot concat. Both stay on GroupMap for now.
    if (plan.materialize) return false;

    for (const auto& spec : plan.agg) {
        if (!agg_op_engine_supported(spec.op)) return false;
        if (spec.op == AggOp::Hist) return false;
        // The engine's Count is always the group's row count; the View's
        // Count(field) counts only the field-present rows, a different value.
        if (spec.op == AggOp::Count && !spec.field.empty()) return false;
        if (!is_stable_field(spec.field) || !is_stable_field(spec.by))
            return false;
    }
    return true;
}

coro::CoroTask<dataframe::DataFrame> run_collect_via_engine(
    const ViewPlan& plan_in) {
    const ViewPlan plan = resolve_bucket_origin(plan_in);
    ensure_schema(plan);

    {
        GroupMap served;
        if (co_await try_serve_aggregate_no_scan(plan, served))
            co_return finalize_collect_batch(served, plan);
    }

    // A Rank key groups on pid and relabels to the PR-metadata rank; harvest
    // that map into the resolver before the scan, mirroring run_scan_aggregate.
    if (std::any_of(
            plan.group_by.begin(), plan.group_by.end(),
            [](const GroupKey& gk) { return gk.kind == GroupKey::Kind::Rank; }))
        co_await harvest_ranks(plan);

    const bool has_bucket = plan.time_bucket_us > 0;

    std::vector<std::string> key_names;
    std::vector<std::string> key_fields;
    key_names.reserve(plan.group_by.size());
    key_fields.reserve(plan.group_by.size());
    for (const GroupKey& gk : plan.group_by) {
        key_names.push_back(group_col_name(gk));
        key_fields.push_back(key_group_field(gk));
    }

    // Scaled-field renaming (bucket only, see below): ts/dur route through a
    // hidden pre-scaled column instead of the raw field name.
    static constexpr const char* SCALED_TS_COL = "__view_agg_engine_scaled_ts";
    static constexpr const char* SCALED_DUR_COL =
        "__view_agg_engine_scaled_dur";
    const bool needs_value_scale = has_bucket && plan.time_scale != 1.0;
    auto scaled_name = [&](const std::string& f) -> std::string {
        if (!needs_value_scale) return f;
        if (f == "ts") return SCALED_TS_COL;
        if (f == "dur") return SCALED_DUR_COL;
        return f;
    };

    // Auto-discovered numeric-arg reductions (the auto_numeric_metrics / dyn
    // path). The arg set is only known after a scan, so discover it first, then
    // feed each (arg, reduction) to the engine as a Float64 value column.
    // Output column names and order match to_batch's dyn columns exactly (arg
    // outer, reduction inner; legacy empty-reductions = one bare-named per-arg
    // mean).
    struct DynAgg {
        std::string column;  // AGG_NUM_ARG_PREFIX + arg (the frame column name)
        std::string out;  // dyn_col_name(spec, arg), or the bare arg (legacy)
        AggOp op;
        double q;
    };
    std::vector<DynAgg> dyn_aggs;
    if (plan.auto_numeric_metrics) {
        std::vector<std::string> names =
            co_await harvest_numeric_arg_names(plan);
        for (const std::string& name : names) {
            const std::string col = std::string(AGG_NUM_ARG_PREFIX) + name;
            if (plan.numeric_arg_aggs.empty()) {
                dyn_aggs.push_back({col, name, AggOp::Mean, 0.0});
            } else {
                for (const AggSpec& r : plan.numeric_arg_aggs)
                    dyn_aggs.push_back({col, dyn_col_name(r, name), r.op, r.q});
            }
        }
    }

    // to_batch lays out value columns as [named non-text/hist aggs, dyn cols]
    // then text columns (ArgMax/SetUnion) then hist. The engine emits columns
    // in gaggs order, so partition named specs into value vs text and slot the
    // dyn columns between them to reproduce that order.
    std::vector<dataframe::GroupAgg> gaggs;
    std::vector<dataframe::GroupAgg> text_gaggs;
    if (plan.agg.empty()) {
        dataframe::GroupAgg g;
        g.op = dataframe::Agg::Count;
        g.out = agg_col_name(AggSpec(AggOp::Count));
        gaggs.push_back(std::move(g));
    } else {
        for (const auto& spec : plan.agg) {
            dataframe::GroupAgg g = to_group_agg(spec);
            g.column = scaled_name(g.column);
            g.by = scaled_name(g.by);
            if (spec.op == AggOp::ArgMax || spec.op == AggOp::SetUnion)
                text_gaggs.push_back(std::move(g));
            else
                gaggs.push_back(std::move(g));
        }
    }
    for (const DynAgg& d : dyn_aggs) {
        dataframe::GroupAgg g;
        g.op = to_engine_agg(d.op);
        g.out = d.out;
        g.param = d.q;
        g.column = d.column;
        gaggs.push_back(std::move(g));
    }
    gaggs.insert(gaggs.end(), std::make_move_iterator(text_gaggs.begin()),
                 std::make_move_iterator(text_gaggs.end()));

    // A fixed select list on the raw scan (instead of the default, per-batch-
    // discovered schema) guarantees every streamed morsel carries the same
    // columns in the same order: the streaming group_by resolves key/value
    // columns once against the Source's schema, so a morsel with a different
    // column layout would silently misalign otherwise. Group keys select the
    // field they fold on (key_fields), which for a resolved name key is the
    // raw hash, not the resolved-name output column.
    std::vector<std::string> select = key_fields;
    auto add_field = [&](const std::string& f) {
        if (f.empty()) return;
        if (std::find(select.begin(), select.end(), f) == select.end())
            select.push_back(f);
    };
    for (const auto& spec : plan.agg) {
        add_field(spec.field);
        add_field(spec.by);
    }
    for (const DynAgg& d : dyn_aggs) add_field(d.column);
    if (has_bucket) add_field("ts");

    auto next = std::make_shared<ViewPlan>(plan);
    next->group_by.clear();
    next->agg.clear();
    next->auto_numeric_metrics = false;
    next->numeric_arg_aggs.clear();
    next->sort_col.clear();
    next->topk_col.clear();
    next->offset = 0;
    next->limit = 0;
    next->select = select;
    next->schema.reset();
    next->resolver.reset();
    next->time_bucket_us = 0;
    next->bucket_origin_us = 0;
    next->bucket_origin_min = false;
    // The raw scan (native_row_fold.cpp) pre-scales+rounds ts/dur to the
    // nearest integer via time_scale before this function ever sees them; the
    // GroupMap fold instead applies time_scale as an unrounded double, once,
    // at bucket/aggregate time (agg_fold.h). Bucketing needs the exact
    // fold formula, so pull the raw (unscaled) values here and reapply
    // time_scale ourselves below, byte-for-byte like the fold.
    if (has_bucket) next->time_scale = 1.0;

    View raw(std::move(next));
    dataframe::LazyFrame lf =
        dataframe::LazyFrame::scan(std::make_shared<ViewSource>(raw))
            .memory_budget(plan.memory_budget);

    // cat is a computed key: the GroupMap path lowercases it for grouping only
    // (agg_fold.h's lower_ascii) while a value agg (e.g. SetUnion(cat)) still
    // sees the raw-case text, so the lowered key is materialized into a hidden
    // column rather than overwriting "cat" in place.
    static constexpr const char* CAT_KEY_COL = "__view_agg_engine_cat_key";
    static constexpr const char* BUCKET_KEY_COL =
        "__view_agg_engine_time_bucket";
    std::vector<std::string> group_key_names = key_fields;
    std::optional<std::size_t> cat_pos;
    for (std::size_t i = 0; i < plan.group_by.size(); ++i) {
        if (plan.group_by[i].kind != GroupKey::Kind::Cat) continue;
        cat_pos = i;
        group_key_names[i] = CAT_KEY_COL;
        lf = lf.with_column(CAT_KEY_COL,
                            dataframe::expr_lower(dataframe::expr_col(
                                static_cast<std::int32_t>(i))));
    }

    // Bucket key: match agg_fold.h's fold_event_over exactly. `ts` there is the
    // raw (unscaled) timestamp; the fold applies time_scale itself, floors
    // (ts*scale - origin)/interval toward -inf, then rescales by the interval
    // width and shifts back by origin. All in int64 once floored, so the
    // multiply-back and origin add cannot introduce float error the fold does
    // not also have.
    if (has_bucket) {
        const auto ts_it = std::find(select.begin(), select.end(), "ts");
        const auto ts_idx = static_cast<std::int32_t>(ts_it - select.begin());
        const double interval = static_cast<double>(plan.time_bucket_us);
        const auto w = static_cast<std::int64_t>(plan.time_bucket_us);
        const auto origin = static_cast<std::int64_t>(plan.bucket_origin_us);
        dataframe::Expr ts_d = dataframe::expr_cast(
            dataframe::TypeId::Float64, dataframe::expr_col(ts_idx));
        dataframe::Expr rel = ts_d * dataframe::expr_lit(plan.time_scale) -
                              dataframe::expr_lit(static_cast<double>(origin));
        dataframe::Expr floored = dataframe::expr_unary(
            static_cast<std::int32_t>(dataframe::UnaryOp::Floor),
            rel / dataframe::expr_lit(interval));
        dataframe::Expr bucket =
            dataframe::expr_cast(dataframe::TypeId::Int64, floored) *
                dataframe::expr_lit(w) +
            dataframe::expr_lit(origin);
        lf = lf.with_column(BUCKET_KEY_COL, bucket);
        group_key_names.insert(group_key_names.begin(), BUCKET_KEY_COL);

        // Value fields that need the same unrounded time_scale (agg_fold.h's
        // field_scaled: ts/dur/te - te is never engine-eligible, see
        // is_stable_field). Only materialize the ones an agg spec actually
        // references.
        if (needs_value_scale) {
            bool need_ts = false, need_dur = false;
            for (const auto& spec : plan.agg) {
                need_ts = need_ts || spec.field == "ts" || spec.by == "ts";
                need_dur = need_dur || spec.field == "dur" || spec.by == "dur";
            }
            auto scale_col = [&](const std::string& field, const char* out) {
                const auto it = std::find(select.begin(), select.end(), field);
                const auto idx = static_cast<std::int32_t>(it - select.begin());
                lf = lf.with_column(
                    out, dataframe::expr_cast(dataframe::TypeId::Float64,
                                              dataframe::expr_col(idx)) *
                             dataframe::expr_lit(plan.time_scale));
            };
            if (need_ts) scale_col("ts", SCALED_TS_COL);
            if (need_dur) scale_col("dur", SCALED_DUR_COL);
        }
    }

    dataframe::DataFrame r =
        co_await lf.group_by(group_key_names, gaggs).collect();
    const std::size_t off = has_bucket ? 1 : 0;
    if (has_bucket) r.names[0] = "time_bucket";
    if (cat_pos) r.names[off + *cat_pos] = "cat";
    // An Arg/Field key groups on a sentinel-named string column; relabel it to
    // its user-facing group_col_name (== gk.arg == key_names[j]).
    for (std::size_t j = 0; j < plan.group_by.size(); ++j) {
        const GroupKey::Kind k = plan.group_by[j].kind;
        if (k == GroupKey::Kind::Arg || k == GroupKey::Kind::Field)
            r.names[off + j] = key_names[j];
    }
    for (std::size_t i = 0; i < off + key_names.size(); ++i)
        r.columns[i] = key_column_to_string(r.columns[i]);

    // Post-aggregation re-key: relabel each resolved-name key column (grouped
    // on its raw hash, a bijection) to the resolved name, same as
    // resolve_group_keys/resolve_group_value in the GroupMap path. Runs on the
    // small distinct-group result, never per event.
    bool needs_resolver = false;
    for (const GroupKey& gk : plan.group_by)
        needs_resolver = needs_resolver || key_is_resolved(gk.kind);
    if (needs_resolver) {
        const GroupResolver* resolver = ensure_resolver(plan);
        for (std::size_t j = 0; j < plan.group_by.size(); ++j) {
            if (!key_is_resolved(plan.group_by[j].kind)) continue;
            const std::size_t idx = off + j;
            r.columns[idx] = resolve_key_column(r.columns[idx], *resolver,
                                                plan.group_by[j].kind);
            r.names[idx] = key_names[j];
        }
    }

    // A dyn Pct on a group where the arg never appeared: the engine reads an
    // empty per-group DDSketch, whose quantile is NaN, while the GroupMap dyn
    // path emits 0.0 (dyn_sketches has no entry for the arg). Rewrite that NaN
    // to 0.0 so the two paths match. Only an empty sketch yields NaN, so this
    // touches exactly the absent-arg groups.
    for (const DynAgg& d : dyn_aggs) {
        if (d.op != AggOp::Pct) continue;
        const std::int64_t ci = r.column_index(d.out);
        if (ci < 0) continue;
        dataframe::Series& col = r.columns[static_cast<std::size_t>(ci)];
        if (col.type() != dataframe::TypeId::Float64) continue;
        const std::int64_t n = col.length();
        const double* src = col.data<double>();
        std::vector<double> vals(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] =
                std::isnan(src[i]) ? 0.0 : src[i];
        col = dataframe::Series::flat_f64(vals.data(), n);
    }
    co_return r;
}

}  // namespace dftracer::utils::trace::views::detail
