#ifndef DFTRACER_UTILS_TRACE_VIEWS_PIPELINE_H
#define DFTRACER_UTILS_TRACE_VIEWS_PIPELINE_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/view_definition.h>
#include <dftracer/utils/trace/views/view_plan.h>

#include <cstdint>
#include <vector>

namespace dftracer::utils::trace::views::detail {

enum class NodeKind { Source, StreamOp, Reduce, Sink };

/// One node of a lowered pipeline. A node carries only what its kind needs: a
/// Source borrows the scan plan and definition, a Reduce or Sink borrows the
/// fold it drives. Every pointer is borrowed and must outlive `execute`.
struct PipelineNode {
    NodeKind kind = NodeKind::Source;
    const ViewPlan* plan = nullptr;
    const ViewDefinition* vdef = nullptr;
    Fold* fold = nullptr;
};

/// A linear pipeline: one scan source, zero or more streaming ops, one sink.
/// Breakers split pipelines; a Pipeline is one fuse pass or one driver run.
struct Pipeline {
    PipelineNode source;
    std::vector<PipelineNode> ops;
    PipelineNode sink;
};

/// Identity lowering: the scan as the Source, `sink` as the only fold, no
/// streaming ops. Executing it is exactly `fuse(plan, vdef, {&sink}, ...)`.
Pipeline lower_single_fold(const ViewPlan& plan, const ViewDefinition& vdef,
                           Fold& sink);

/// Run a lowered pipeline. A Source feeding fold Sinks with no streaming ops is
/// lowering (a): one direct `fuse` call. Throws std::logic_error on a shape no
/// lowering handles yet, rather than silently dropping nodes.
coro::CoroTask<ExportStats> execute(const Pipeline& pipeline,
                                    dftracer::utils::StringIntern& intern,
                                    const CoverageSet* covered = nullptr,
                                    std::uint64_t limit = 0);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_PIPELINE_H
