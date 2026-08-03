#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_COUNTER_FORMAT_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_COUNTER_FORMAT_H

#include <dftracer/utils/utilities/composites/dft/views/view.h>
#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>
#include <dftracer/utils/utilities/composites/dft/views/view_plan.h>

#include <string>

// Emitting aggregate rows/groups as dftracer ph="C" counter events: group
// columns map to name/cat/pid/tid/ts, everything else becomes args.
namespace dftracer::utils::utilities::composites::dft::views::detail {

// One aggregate row -> one ph="C" counter event JSON line.
std::string counter_line(const ResultTable& t, const ResultRow& row);

// One merged group -> one ph="C" counter event written to `sink`. Each event
// carries its own args (its own dynamic columns), so no global schema is
// needed.
void emit_group_counter(const std::string& key, const AggAccum& a,
                        const ViewPlan& plan, ExportSink& sink);

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_VIEW_COUNTER_FORMAT_H
