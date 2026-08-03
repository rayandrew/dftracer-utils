#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/mv_store.h>
#include <dftracer/utils/utilities/composites/dft/views/view.h>
#include <dftracer/utils/utilities/composites/dft/views/view_executor.h>
#include <dftracer/utils/utilities/composites/dft/views/view_plan.h>
#include <dftracer/utils/utilities/filesystem/pattern_directory_scanner_utility.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace dftracer::utils::utilities::composites::dft::views {

namespace {
// A mutable copy of a plan (or a fresh one), ready for one op to mutate.
std::shared_ptr<detail::ViewPlan> clone(
    const std::shared_ptr<const detail::ViewPlan>& base) {
    return base ? std::make_shared<detail::ViewPlan>(*base)
                : std::make_shared<detail::ViewPlan>();
}
}  // namespace

View::View() : plan_(std::make_shared<detail::ViewPlan>()) {}

View::View(std::shared_ptr<const detail::ViewPlan> plan)
    : plan_(std::move(plan)) {}

View View::from_files(std::vector<ViewFile> files,
                      indexing::BloomFilterCache* bloom_cache) {
    auto plan = std::make_shared<detail::ViewPlan>();
    plan->files = std::move(files);
    plan->bloom_cache = bloom_cache;
    return View(std::move(plan));
}

View View::from_file(std::string file_path, std::string index_path) {
    std::vector<ViewFile> files;
    files.push_back(ViewFile{std::move(file_path), std::move(index_path)});
    return from_files(std::move(files));
}

coro::CoroTask<View> View::from_directory(std::string dir,
                                          std::string index_path) {
    filesystem::PatternDirectoryScannerUtility scanner;
    filesystem::PatternDirectoryScannerUtilityInput input(
        std::move(dir), {".pfw.gz"}, /*recursive=*/true,
        /*populate_size=*/false);

    std::vector<filesystem::FileEntry> entries;
    co_await run_coro_scope([&](CoroScope& scope) -> coro::CoroTask<void> {
        entries = co_await scope.spawn(scanner, input);
    });

    std::vector<ViewFile> files;
    files.reserve(entries.size());
    for (auto& e : entries) {
        std::string fp = e.path.string();
        files.push_back(
            ViewFile{fp, internal::determine_index_path(fp, index_path)});
    }
    std::sort(files.begin(), files.end(),
              [](const ViewFile& a, const ViewFile& b) {
                  return a.file_path < b.file_path;
              });
    co_return from_files(std::move(files));
}

std::vector<TraceConfig> View::config() const {
    std::vector<TraceConfig> out;
    for (const auto& f : plan_->files) {
        auto tail = read_trace_config(f.file_path);
        if (!tail.empty()) {
            for (auto& c : tail) out.push_back(std::move(c));
            continue;
        }
        if (f.index_path.empty()) continue;

        struct Sink : ExportSink {
            std::string buf;
            void write(std::string_view d) override { buf.append(d); }
        } sink;
        View::from_file(f.file_path, f.index_path)
            .query(R"(name == "end")")
            .metadata(false)
            .export_json(sink)
            .get();
        std::size_t pos = 0;
        while (pos < sink.buf.size()) {
            std::size_t nl = sink.buf.find('\n', pos);
            if (nl == std::string::npos) nl = sink.buf.size();
            if (auto c = parse_end_event(
                    std::string_view(sink.buf).substr(pos, nl - pos)))
                out.push_back(std::move(*c));
            pos = nl + 1;
        }
    }
    return out;
}

View View::filter(Query q) const {
    auto next = clone(plan_);
    if (next->query) {
        // AND the two predicates by combining their source strings.
        std::string combined =
            "(" + next->query->source() + ") and (" + q.source() + ")";
        next->query = common::query::parse_or_throw(combined);
    } else {
        next->query = std::move(q);
    }
    return View(std::move(next));
}

View View::query(const std::string& dsl) const {
    return filter(common::query::parse_or_throw(dsl));
}

View View::phase(Phase p) const {
    auto next = clone(plan_);
    next->phase = p;
    return View(std::move(next));
}

View View::time_range(double begin, double end) const {
    auto next = clone(plan_);
    next->time_range = std::make_pair(begin, end);
    return View(std::move(next));
}

View View::time_bucket(std::uint64_t interval_us) const {
    auto next = clone(plan_);
    next->time_bucket_us = interval_us;
    return View(std::move(next));
}

View View::time_scale(double ns_ratio) const {
    auto next = clone(plan_);
    next->time_scale = ns_ratio;
    return View(std::move(next));
}

AggregatedView View::group_by(std::vector<GroupKey> keys) const {
    auto next = clone(plan_);
    next->group_by = std::move(keys);
    return AggregatedView(View(std::move(next)));
}

AggregatedView View::agg(std::vector<AggSpec> specs) const {
    auto next = clone(plan_);
    next->agg = std::move(specs);
    return AggregatedView(View(std::move(next)));
}

AggregatedView View::agg_numeric_args() const {
    auto next = clone(plan_);
    next->auto_numeric_metrics = true;
    return AggregatedView(View(std::move(next)));
}

View View::memory_budget(std::uint64_t bytes) const {
    auto next = clone(plan_);
    next->memory_budget = bytes;
    return View(std::move(next));
}

View View::auto_spill() const {
    // ~1/3 of available memory, floored so tiny machines still get a workable
    // in-core window before spilling.
    std::uint64_t budget =
        static_cast<std::uint64_t>(detect_available_memory() / 3);
    if (budget < MIN_MEMORY_BUDGET_BYTES) budget = MIN_MEMORY_BUDGET_BYTES;
    return memory_budget(budget);
}

View View::select(std::vector<std::string> cols) const {
    auto next = clone(plan_);
    next->select = std::move(cols);
    return View(std::move(next));
}

View View::limit(std::uint64_t n) const {
    auto next = clone(plan_);
    next->limit = n;
    return View(std::move(next));
}

View View::offset(std::uint64_t n) const {
    auto next = clone(plan_);
    next->offset = n;
    return View(std::move(next));
}

View View::materialize(std::uint64_t checkpoint_size,
                       std::uint64_t part_size) const {
    auto next = clone(plan_);
    next->materialize = true;
    next->mv_checkpoint_size = checkpoint_size;
    next->mv_part_size = part_size;
    return View(std::move(next));
}

View View::metadata(bool include) const {
    auto next = clone(plan_);
    next->include_metadata = include;
    return View(std::move(next));
}

View View::with_partial_source(const detail::PartialSource* source) const {
    auto next = clone(plan_);
    next->agg_source = source;
    return View(std::move(next));
}

View View::cancel_when(std::function<bool()> pred) const {
    auto next = clone(plan_);
    next->cancelled = std::move(pred);
    return View(std::move(next));
}

View View::rollup_root(std::string dir) const {
    auto next = clone(plan_);
    next->rollup_root = std::move(dir);
    return View(std::move(next));
}

View View::views_root(std::string dir) const {
    auto next = clone(plan_);
    next->views_root = std::move(dir);
    return View(std::move(next));
}

View View::emit_all_metadata(bool v) const {
    auto next = clone(plan_);
    next->emit_all_metadata = v;
    return View(std::move(next));
}

coro::CoroTask<ExportStats> View::for_each_batch(
    std::function<void(std::size_t, const std::vector<std::string_view>&)>
        on_batch,
    std::size_t num_slots, std::uint64_t limit) const {
    co_return co_await detail::run_scan_batches(plan_, num_slots, limit,
                                                on_batch);
}

bool View::export_would_bootstrap() const {
    return detail::export_bootstrap_eligible(*plan_);
}

bool View::collect_would_bootstrap() const {
    return detail::collect_bootstrap_eligible(*plan_);
}

coro::CoroTask<ExportStats> View::export_json(ExportSink& sink) const {
    co_return co_await detail::run_export(*plan_, sink);
}

coro::CoroTask<ExportStats> View::export_trace(TraceWriteOptions opts) const {
    co_return co_await detail::run_export_trace(*plan_, opts);
}

coro::CoroTask<ResultTable> View::collect() const {
    co_return co_await detail::run_collect(*plan_);
}

coro::CoroTask<ExportStats> View::run(const ProgressFn* progress) const {
    co_return co_await detail::run_materialize(*plan_, progress);
}

std::string View::materialize_dir() const {
    return detail::materialize_view_dir(*plan_);
}

void View::register_materialized(const std::string& dir) const {
    detail::register_view(dir, *plan_);
}

std::vector<std::string> View::mv_source() const {
    std::vector<std::string> out;
    if (auto mv = detail::find_subsuming_view(*plan_))
        for (const auto& f : *mv) out.push_back(f.file_path);
    return out;
}

coro::CoroTask<TypedResult> View::collect_typed(
    int shard_begin, int shard_end, const ProgressFn* progress) const {
    co_return co_await detail::run_collect_typed(*plan_, shard_begin, shard_end,
                                                 progress);
}

coro::CoroTask<ExportStats> View::export_counters(ExportSink& sink) const {
    co_return co_await detail::run_export_counters(*plan_, sink);
}

coro::CoroTask<void> View::materialize_partials(
    const std::vector<std::string_view>& partials) const {
    co_return co_await detail::run_materialize_partials(*plan_, partials);
}

std::optional<ResultTable> View::reconstruct_if_cached() const {
    return detail::run_reconstruct_if_cached(*plan_);
}

coro::CoroTask<std::string> View::aggregate_partial() const {
    co_return co_await detail::run_aggregate_partial(*plan_);
}

ResultTable View::merge_partials_to_table(
    const std::vector<std::string_view>& partials) const {
    return detail::merge_partials_to_table(*plan_, partials);
}

ExportStats View::merge_counter_partials(
    const std::vector<std::string_view>& partials, ExportSink& sink) const {
    return detail::merge_counters_partials(*plan_, partials, sink);
}

PartitionRun View::partition(std::size_t num_slots, std::uint64_t limit) const {
    return PartitionRun(plan_, num_slots, limit);
}

std::shared_ptr<ResultTable> PartitionRun::collect(
    Query predicate, std::vector<GroupKey> group_by, std::vector<AggSpec> agg) {
    auto out = std::make_shared<ResultTable>();
    detail::BranchHooks h = detail::make_collect_branch(
        std::move(group_by), std::move(agg), out, num_slots_);
    h.predicate = std::move(predicate);
    detail::add_branch(*state_, std::move(h));
    return out;
}

std::shared_ptr<ExportStats> PartitionRun::export_json(Query predicate,
                                                       ExportSink& sink) {
    auto out = std::make_shared<ExportStats>();
    detail::BranchHooks h = detail::make_export_branch(sink, out);
    h.predicate = std::move(predicate);
    detail::add_branch(*state_, std::move(h));
    return out;
}

coro::CoroTask<ExportStats> PartitionRun::execute() {
    co_return co_await detail::run_partition(state_);
}

}  // namespace dftracer::utils::utilities::composites::dft::views
