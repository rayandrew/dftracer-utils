#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H

#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/trace/views/view_plan.h>

#include <optional>
#include <string>
#include <vector>

// View -> dataframe engine aggregation convergence: every group_by/agg (and
// global) View aggregation runs through the dataframe engine's streaming
// group_by. run_collect_via_engine itself is declared (as a friend of View) in
// view.h; this header adds the finalize/prepare helpers.
namespace dftracer::utils::trace::views::detail {

/// Finalize a raw engine AggState (key layout [time_bucket?, group_by...] with
/// value/text/dyn specs in engine order) to the result DataFrame, applying the
/// same key rendering, resolver relabel, busy_cell_us column and dyn column
/// fixes as the streaming scan path - so a rollup-served result matches a fresh
/// scan byte-for-byte. `plan` supplies the grouping/resolver/occupancy context.
dftracer::utils::dataframe::DataFrame finalize_engine_result(
    const dftracer::utils::dataframe::AggState& state, const ViewPlan& plan);

/// Scan `plan` through the engine group-by and return the mergeable partial
/// (fine-grain, key layout [time_bucket?, group_by...]) before finalizing. The
/// unit persisted as a rollup and merged across distributed ranks; finalize
/// with finalize_engine_result or coarsen with agg_regroup.
coro::CoroTask<dftracer::utils::dataframe::AggStatePtr> build_engine_agg_state(
    const ViewPlan& plan);

/// One dyn (auto_numeric_metrics) output column needing a post-finalize fix: a
/// Pct on an absent arg reads an empty sketch (NaN -> 0), a Count is emitted
/// Int64 by CountValid but the dyn convention is Float64.
struct DynFix {
    std::string out;
    bool pct = false;
    bool count = false;
};

/// The engine group-by inputs for a bucket-resolved plan: the raw scan
/// LazyFrame with its hidden key/scale columns, the group key column names, the
/// gaggs in [value, dyn, text] order, and the dyn column fixes.
struct EnginePrep {
    std::optional<dftracer::utils::dataframe::LazyFrame> lf;
    std::vector<std::string> group_key_names;
    std::vector<dftracer::utils::dataframe::GroupAgg> gaggs;
    std::vector<DynFix> dynfix;
};

coro::CoroTask<EnginePrep> prepare_engine_group(const ViewPlan& plan);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H
