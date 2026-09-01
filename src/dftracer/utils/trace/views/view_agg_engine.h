#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H

#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/trace/views/view_plan.h>

// View -> dataframe engine aggregation convergence: an alternate group_by/agg
// path that routes an eligible group_by/agg View (one or more direct-column
// keys, no transform) through the dataframe engine's streaming (possibly
// composite-key) group_by instead of the View's own GroupMap fold. This is
// the default for every eligible query; the GroupMap path (view_aggregate.h/
// .cpp) is the fallback for plans agg_engine_eligible rejects.
// run_collect_via_engine itself is declared (as a friend of View) in view.h;
// this header only adds the eligibility check.
namespace dftracer::utils::trace::views::detail {

/// True when `plan` is a shape the engine path can answer byte-for-byte
/// identically to the GroupMap path (see view_agg_engine.cpp for the exact
/// qualifier). False routes to the existing GroupMap fold.
bool agg_engine_eligible(const ViewPlan& plan);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H
