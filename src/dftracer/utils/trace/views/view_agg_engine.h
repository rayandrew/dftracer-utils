#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H

#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/trace/views/view_plan.h>

// Phase 1 of the View -> dataframe engine aggregation convergence: an
// alternate, flag-gated group_by/agg path that routes a single-key
// group_by/agg View through the dataframe engine's streaming group_by instead
// of the View's own GroupMap fold. Behind DFTRACER_UTILS_AGG_ENGINE and
// default-off; the GroupMap path (view_aggregate.h/.cpp) is unchanged and
// stays the default. run_collect_via_engine itself is declared (as a friend
// of View) in view.h; this header only adds the eligibility check.
namespace dftracer::utils::trace::views::detail {

/// True when `plan` is a shape the engine path can answer byte-for-byte
/// identically to the GroupMap path (see view_agg_engine.cpp for the exact
/// qualifier). False routes to the existing GroupMap fold.
bool agg_engine_eligible(const ViewPlan& plan);

/// True when DFTRACER_UTILS_AGG_ENGINE is set (any non-empty value); the
/// engine path is opt-in and off by default.
bool agg_engine_enabled();

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGG_ENGINE_H
