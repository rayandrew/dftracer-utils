#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/env.h>
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/view_agg_engine.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_source.h>

#include <algorithm>
#include <string>
#include <vector>

namespace dftracer::utils::trace::views::detail {

namespace dataframe = dftracer::utils::dataframe;

namespace {

// Value/by fields the raw row stream emits at a fixed position in every
// morsel: the always-present top-level event fields. An arg field's column is
// discovered per scan batch (present only when that batch has it), so the
// engine's streaming group_by - which resolves key/value columns once against
// the Source's advertised schema, not per morsel - cannot safely reference
// one without a broader fix to make that schema stable across morsels; the
// same goes for fhash/hhash, whose column is emitted only when the batch has
// at least one event carrying it. Keeping Phase 1 to the stable fields avoids
// that trap entirely.
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
    if (plan.group_by.size() != 1) return false;
    const GroupKey& gk = plan.group_by[0];
    switch (gk.kind) {
        case GroupKey::Kind::Name:
        case GroupKey::Kind::Pid:
        case GroupKey::Kind::Tid:
            break;
        case GroupKey::Kind::Cat:
        case GroupKey::Kind::Fhash:
        case GroupKey::Kind::Hhash:
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
    if (plan.time_bucket_us != 0) return false;
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
    const ViewPlan& plan) {
    const std::string key_name = group_col_name(plan.group_by[0]);

    std::vector<dataframe::GroupAgg> gaggs;
    if (plan.agg.empty()) {
        dataframe::GroupAgg g;
        g.op = dataframe::Agg::Count;
        g.out = agg_col_name(AggSpec(AggOp::Count));
        gaggs.push_back(std::move(g));
    } else {
        gaggs.reserve(plan.agg.size());
        for (const auto& spec : plan.agg) gaggs.push_back(to_group_agg(spec));
    }

    // A fixed select list on the raw scan (instead of the default, per-batch-
    // discovered schema) guarantees every streamed morsel carries the same
    // columns in the same order: the streaming group_by resolves key/value
    // columns once against the Source's schema, so a morsel with a different
    // column layout would silently misalign otherwise.
    std::vector<std::string> select{key_name};
    auto add_field = [&](const std::string& f) {
        if (f.empty()) return;
        if (std::find(select.begin(), select.end(), f) == select.end())
            select.push_back(f);
    };
    for (const auto& spec : plan.agg) {
        add_field(spec.field);
        add_field(spec.by);
    }

    auto next = std::make_shared<ViewPlan>(plan);
    next->group_by.clear();
    next->agg.clear();
    next->auto_numeric_metrics = false;
    next->numeric_arg_aggs.clear();
    next->sort_col.clear();
    next->topk_col.clear();
    next->offset = 0;
    next->limit = 0;
    next->select = std::move(select);
    next->schema.reset();
    next->resolver.reset();

    View raw(std::move(next));
    dataframe::LazyFrame lf =
        dataframe::LazyFrame::scan(std::make_shared<ViewSource>(raw))
            .memory_budget(plan.memory_budget);

    dataframe::DataFrame r = co_await lf.group_by(key_name, gaggs).collect();
    r.columns[0] = key_column_to_string(r.columns[0]);
    co_return r;
}

}  // namespace dftracer::utils::trace::views::detail
