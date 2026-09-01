#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/env.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/view_agg_engine.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_executor.h>
#include <dftracer/utils/trace/views/view_source.h>

#include <algorithm>
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
                break;
            case GroupKey::Kind::IoCat:
            case GroupKey::Kind::AccPat:
            case GroupKey::Kind::FilePath:
            case GroupKey::Kind::FileName:
            case GroupKey::Kind::HostName:
            case GroupKey::Kind::Rank:
            case GroupKey::Kind::Arg:
            case GroupKey::Kind::Field:
                return false;
        }
        if (gk.transform != GroupKey::Transform::None) return false;
    }
    if (plan.auto_numeric_metrics) return false;
    if (!plan.numeric_arg_aggs.empty()) return false;
    if (!plan.sort_col.empty() || !plan.topk_col.empty()) return false;
    if (plan.offset != 0 || plan.limit != 0) return false;
    if (!plan.select.empty()) return false;

    for (const auto& spec : plan.agg) {
        if (!agg_op_engine_supported(spec.op)) return false;
        // The engine's Count is always the group's row count; the View's
        // Count(field) counts only the field-present rows, a different value.
        if (spec.op == AggOp::Count && !spec.field.empty()) return false;
        if (!is_stable_field(spec.field) || !is_stable_field(spec.by))
            return false;
    }
    return true;
}

bool agg_engine_enabled() {
    return dftracer::utils::Env::get<std::string_view>(
               "DFTRACER_UTILS_AGG_ENGINE")
        .has_value();
}

coro::CoroTask<dataframe::DataFrame> run_collect_via_engine(
    const ViewPlan& plan_in) {
    const ViewPlan plan = resolve_bucket_origin(plan_in);
    const bool has_bucket = plan.time_bucket_us > 0;

    std::vector<std::string> key_names;
    key_names.reserve(plan.group_by.size());
    for (const GroupKey& gk : plan.group_by)
        key_names.push_back(group_col_name(gk));

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

    std::vector<dataframe::GroupAgg> gaggs;
    if (plan.agg.empty()) {
        dataframe::GroupAgg g;
        g.op = dataframe::Agg::Count;
        g.out = agg_col_name(AggSpec(AggOp::Count));
        gaggs.push_back(std::move(g));
    } else {
        gaggs.reserve(plan.agg.size());
        for (const auto& spec : plan.agg) {
            dataframe::GroupAgg g = to_group_agg(spec);
            g.column = scaled_name(g.column);
            g.by = scaled_name(g.by);
            gaggs.push_back(std::move(g));
        }
    }

    // A fixed select list on the raw scan (instead of the default, per-batch-
    // discovered schema) guarantees every streamed morsel carries the same
    // columns in the same order: the streaming group_by resolves key/value
    // columns once against the Source's schema, so a morsel with a different
    // column layout would silently misalign otherwise.
    std::vector<std::string> select = key_names;
    auto add_field = [&](const std::string& f) {
        if (f.empty()) return;
        if (std::find(select.begin(), select.end(), f) == select.end())
            select.push_back(f);
    };
    for (const auto& spec : plan.agg) {
        add_field(spec.field);
        add_field(spec.by);
    }
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
    std::vector<std::string> group_key_names = key_names;
    auto cat_it = std::find(key_names.begin(), key_names.end(), "cat");
    if (cat_it != key_names.end()) {
        const auto cat_idx =
            static_cast<std::int32_t>(cat_it - key_names.begin());
        group_key_names[static_cast<std::size_t>(cat_idx)] = CAT_KEY_COL;
        lf = lf.with_column(
            CAT_KEY_COL, dataframe::expr_lower(dataframe::expr_col(cat_idx)));
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
    if (cat_it != key_names.end())
        r.names[off + static_cast<std::size_t>(cat_it - key_names.begin())] =
            "cat";
    for (std::size_t i = 0; i < off + key_names.size(); ++i)
        r.columns[i] = key_column_to_string(r.columns[i]);
    co_return r;
}

}  // namespace dftracer::utils::trace::views::detail
