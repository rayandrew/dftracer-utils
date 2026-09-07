#include <dftracer/utils/trace/views/pipeline.h>

#include <array>
#include <stdexcept>

namespace dftracer::utils::trace::views::detail {

Pipeline lower_single_fold(const ViewPlan& plan, const ViewDefinition& vdef,
                           Fold& sink) {
    Pipeline p;
    p.source.kind = NodeKind::Source;
    p.source.plan = &plan;
    p.source.vdef = &vdef;
    p.sink.kind = NodeKind::Sink;
    p.sink.fold = &sink;
    return p;
}

coro::CoroTask<ExportStats> execute(const Pipeline& pipeline,
                                    dftracer::utils::StringIntern& intern,
                                    const CoverageSet* covered,
                                    std::uint64_t limit) {
    if (!pipeline.ops.empty())
        throw std::logic_error("pipeline: streaming ops need the pull driver");
    if (pipeline.source.plan == nullptr || pipeline.source.vdef == nullptr ||
        pipeline.sink.fold == nullptr)
        throw std::logic_error("pipeline: incomplete source or sink node");

    std::array<Fold*, 1> folds{pipeline.sink.fold};
    co_return co_await fuse(*pipeline.source.plan, *pipeline.source.vdef, folds,
                            intern, covered, limit);
}

}  // namespace dftracer::utils::trace::views::detail
