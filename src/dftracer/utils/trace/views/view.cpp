#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/common/platform_compat.h>
#include <dftracer/utils/core/common/string_arena.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/json/parser.h>
#include <dftracer/utils/trace/comparator/compare_view.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/time_metric.h>
#include <dftracer/utils/trace/views/mv_store.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/trace/views/view_executor.h>
#include <dftracer/utils/trace/views/view_plan.h>
#include <dftracer/utils/utilities/common/arrow/column_builder.h>
#include <dftracer/utils/utilities/filesystem/pattern_directory_scanner_utility.h>
#include <dftracer/utils/utilities/reader/internal/arrow_row_builder.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <string>
#include <utility>

namespace dftracer::utils::trace::views {

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
    utilities::filesystem::PatternDirectoryScannerUtility scanner;
    utilities::filesystem::PatternDirectoryScannerUtilityInput input(
        std::move(dir), {".pfw.gz"}, /*recursive=*/true,
        /*populate_size=*/false);

    std::vector<utilities::filesystem::FileEntry> entries;
    co_await run_coro_scope([&](CoroScope& scope) -> coro::CoroTask<void> {
        entries = co_await scanner(scope, input);
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
        next->query = query::parse_or_throw(combined);
    } else {
        next->query = std::move(q);
    }
    return View(std::move(next));
}

View View::query(const std::string& dsl) const {
    return filter(query::parse_or_throw(dsl));
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

View View::occ_cell(std::uint64_t cell_us) const {
    auto next = clone(plan_);
    next->occ_cell_us = cell_us;
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

namespace {

// Lower a named (non-wildcard) field aggregate to an AggSpec. The field is
// carried through verbatim, so dotted args.* names resolve exactly like the
// string / AggSpec forms.
AggSpec lower_field_agg(const FieldAggExpr& e) {
    using dftracer::utils::dataframe::field::AggFn;
    switch (e.fn()) {
        case AggFn::Count:
            return AggSpec(AggOp::Count);
        case AggFn::Sum:
            return AggSpec(AggOp::Sum, e.field());
        case AggFn::Min:
            return AggSpec(AggOp::Min, e.field());
        case AggFn::Max:
            return AggSpec(AggOp::Max, e.field());
        case AggFn::Mean:
            return AggSpec(AggOp::Mean, e.field());
        case AggFn::Var:
            return AggSpec(AggOp::Var, e.field());
        case AggFn::Std:
            return AggSpec(AggOp::Std, e.field());
        case AggFn::Skew:
            return AggSpec(AggOp::Skew, e.field());
        case AggFn::Kurt:
            return AggSpec(AggOp::Kurt, e.field());
        case AggFn::ArgMax:
            return AggSpec(AggOp::ArgMax, e.field(), "", e.by());
    }
    throw DFTUtilsException::cat(ErrorCode::INVALID_ARGUMENT,
                                 "unhandled aggregate function");
}

}  // namespace

AggregatedView View::agg(std::vector<FieldAggExpr> exprs) const {
    using dftracer::utils::dataframe::field::AggFn;
    std::vector<AggSpec> specs;
    specs.reserve(exprs.size());
    bool wildcard_mean = false;
    for (const auto& e : exprs) {
        if (e.wildcard()) {
            // The numeric-args path (agg_numeric_args) only computes a per-arg
            // mean, so F.any honors .mean() (that path) and .count() (the plain
            // group count); any other reduction over the wildcard is rejected.
            if (e.fn() == AggFn::Mean) {
                wildcard_mean = true;
            } else if (e.fn() == AggFn::Count) {
                specs.push_back(AggSpec(AggOp::Count));
            } else {
                throw DFTUtilsException::cat(
                    ErrorCode::INVALID_ARGUMENT,
                    "F.any supports only .mean() (per numeric arg) and "
                    ".count(); name a field for other reductions, e.g. "
                    "F(\"args.level\").sum()");
            }
            continue;
        }
        specs.push_back(lower_field_agg(e));
    }
    View v = *this;
    if (!specs.empty()) v = v.agg(std::move(specs));
    if (wildcard_mean) v = v.agg_numeric_args();
    return AggregatedView(std::move(v));
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

View View::sort_by(std::string column, bool descending) const {
    auto next = clone(plan_);
    next->sort_col = std::move(column);
    next->sort_desc = descending;
    return View(std::move(next));
}

View View::topk(std::string column, std::int64_t k, bool largest) const {
    auto next = clone(plan_);
    next->topk_col = std::move(column);
    next->topk_k = k;
    next->topk_largest = largest;
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

coro::CoroTask<dataframe::DataFrame> View::collect() const {
    detail::GroupMap m = co_await detail::run_collect(*plan_);
    co_return detail::finalize_collect_batch(m, *plan_);
}

coro::CoroTask<ExportStats> View::run_folds(
    std::span<detail::Fold* const> folds,
    dftracer::utils::StringIntern& intern) const {
    co_return co_await detail::run_folds(*plan_, folds, intern);
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

std::optional<dataframe::DataFrame> View::reconstruct_if_cached() const {
    return detail::run_reconstruct_if_cached(*plan_);
}

coro::CoroTask<std::string> View::aggregate_partial() const {
    co_return co_await detail::run_aggregate_partial(*plan_);
}

dataframe::DataFrame View::merge_partials_to_table(
    const std::vector<std::string_view>& partials) const {
    return detail::merge_partials_to_table(*plan_, partials);
}

ExportStats View::merge_counter_partials(
    const std::vector<std::string_view>& partials, ExportSink& sink) const {
    return detail::merge_counters_partials(*plan_, partials, sink);
}

ViewSession View::session() const { return ViewSession(plan_); }

ViewSession::ViewSession(std::shared_ptr<const detail::ViewPlan> plan)
    : num_slots_(available_parallelism()),
      state_(detail::make_view_session_state(std::move(plan), num_slots_)) {}

void ViewSession::attach_fold(
    Query predicate,
    std::function<void(std::size_t, const json::JsonValue&, std::string_view)>
        consume,
    std::function<void()> finalize) {
    detail::add_fold_branch(*state_, std::move(predicate), std::move(consume),
                            std::move(finalize));
}

void ViewSession::attach_fold_factory(
    std::function<std::unique_ptr<detail::Fold>(dftracer::utils::StringIntern&)>
        make,
    std::function<void()> finalize) {
    detail::add_fold_factory(*state_, std::move(make), std::move(finalize));
}

Deferred<dataframe::DataFrame> ViewSession::collect(
    Query predicate, std::vector<GroupKey> group_by, std::vector<AggSpec> agg) {
    auto out = std::make_shared<dataframe::DataFrame>();
    key_counts_.emplace_back(out.get(),
                             static_cast<std::int64_t>(group_by.size()));
    detail::BranchHooks h = detail::make_collect_branch(
        std::move(group_by), std::move(agg), out, num_slots_);
    h.predicate = std::move(predicate);
    detail::add_branch(*state_, std::move(h));
    return {out, executed_};
}

Deferred<dataframe::DataFrame> ViewSession::collect(
    std::vector<GroupKey> group_by, std::vector<AggSpec> agg) {
    auto out = std::make_shared<dataframe::DataFrame>();
    key_counts_.emplace_back(out.get(),
                             static_cast<std::int64_t>(group_by.size()));
    detail::BranchHooks h =
        detail::make_collect_branch(group_by, agg, out, num_slots_);
    // predicate left unset: match all scanned events. The descriptor lets
    // execute() serve this branch from a rollup instead of scanning.
    h.agg = detail::AggBranch{
        std::move(group_by), std::move(agg), out, nullptr, false, nullptr};
    detail::add_branch(*state_, std::move(h));
    return {out, executed_};
}

Deferred<dataframe::DataFrame> ViewSession::collect(const View& branch) {
    auto out = std::make_shared<dataframe::DataFrame>();
    const auto& bp = *branch.plan_;
    // Output key columns are [time_bucket?, group_by...], so a bucketed branch
    // has one more leading key column than its group_by (matches join layout).
    key_counts_.emplace_back(out.get(),
                             static_cast<std::int64_t>(bp.group_by.size()) +
                                 (bp.time_bucket_us > 0 ? 1 : 0));
    detail::BranchHooks h =
        detail::make_collect_branch(bp.group_by, bp.agg, out, num_slots_);
    h.agg = detail::AggBranch{bp.group_by,          bp.agg, out, branch.plan_,
                              bp.query.has_value(), nullptr};
    detail::add_branch(*state_, std::move(h));
    return {out, executed_};
}

std::int64_t ViewSession::key_count_of(const void* out) const {
    for (const auto& [ptr, n] : key_counts_)
        if (ptr == out) return n;
    return -1;
}

Deferred<dataframe::DataFrame> ViewSession::join(
    Deferred<dataframe::DataFrame> left, Deferred<dataframe::DataFrame> right,
    JoinType how, std::int64_t n_key) {
    if (!left.value_ || !right.value_ || left.executed_ != executed_ ||
        right.executed_ != executed_)
        throw DFTUtilsException::cat(
            ErrorCode::INVALID_ARGUMENT,
            "ViewSession::join: both handles must be collect() results of "
            "this session");
    if (n_key < 0) n_key = key_count_of(left.value_.get());
    if (n_key < 0)
        throw DFTUtilsException::cat(
            ErrorCode::INVALID_ARGUMENT,
            "ViewSession::join: cannot infer n_key; pass it explicitly");
    auto out = std::make_shared<dataframe::DataFrame>();
    combines_.emplace_back(
        [out, l = left.value_, r = right.value_, n_key, how]() {
            *out = join_batches(*l, *r, n_key, how);
        });
    return {out, executed_};
}

Deferred<dataframe::DataFrame> ViewSession::compare(
    Deferred<dataframe::DataFrame> baseline,
    Deferred<dataframe::DataFrame> variant, std::int64_t n_key) {
    if (!baseline.value_ || !variant.value_ ||
        baseline.executed_ != executed_ || variant.executed_ != executed_)
        throw DFTUtilsException::cat(
            ErrorCode::INVALID_ARGUMENT,
            "ViewSession::compare: both handles must be collect() results of "
            "this session");
    if (n_key < 0) n_key = key_count_of(baseline.value_.get());
    if (n_key < 0)
        throw DFTUtilsException::cat(
            ErrorCode::INVALID_ARGUMENT,
            "ViewSession::compare: cannot infer n_key; pass it explicitly");
    auto out = std::make_shared<dataframe::DataFrame>();
    combines_.emplace_back(
        [out, b = baseline.value_, v = variant.value_, n_key]() {
            *out = comparator::CompareView::compare_batches(*b, *v, n_key);
        });
    return {out, executed_};
}

Deferred<std::string> ViewSession::aggregate_partial(const View& branch) {
    auto out =
        std::make_shared<dataframe::DataFrame>();  // unused; partial path
    auto partial = std::make_shared<std::string>();
    const auto& bp = *branch.plan_;
    detail::BranchHooks h =
        detail::make_collect_branch(bp.group_by, bp.agg, out, num_slots_);
    h.agg = detail::AggBranch{bp.group_by,          bp.agg, out, branch.plan_,
                              bp.query.has_value(), partial};
    detail::add_branch(*state_, std::move(h));
    return {partial, executed_};
}

void ViewSession::materialize(std::vector<GroupKey> group_by,
                              std::vector<AggSpec> agg) {
    detail::add_materialize_branch(*state_, std::move(group_by),
                                   std::move(agg));
}

Deferred<ExportStats> ViewSession::export_json(Query predicate,
                                               ExportSink& sink) {
    auto out = std::make_shared<ExportStats>();
    detail::BranchHooks h = detail::make_export_branch(sink, out);
    h.predicate = std::move(predicate);
    detail::add_branch(*state_, std::move(h));
    return {out, executed_};
}

Deferred<ExportStats> ViewSession::export_json(ExportSink& sink) {
    auto out = std::make_shared<ExportStats>();
    detail::BranchHooks h = detail::make_export_branch(sink, out);
    detail::add_branch(*state_, std::move(h));
    return {out, executed_};
}

Deferred<dataframe::DataFrame> ViewSession::collect_events(const View& branch) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    (void)branch;
    throw DFTUtilsException(ErrorCode::INVALID_ARGUMENT,
                            "collect_events requires the arrow-enabled build");
#else
    namespace rd = utilities::reader::internal;
    namespace arw = utilities::common::arrow;
    const auto& bp = *branch.plan_;
    auto out = std::make_shared<dataframe::DataFrame>();
    const std::size_t slots = num_slots_ ? num_slots_ : 1;
    // One Arrow row builder per worker slot (indexed, never moved). The JSON
    // row builder is the only JSON -> columns path, so events cross Arrow once.
    auto builders =
        std::make_shared<std::vector<arw::RecordBatchBuilder> >(slots);
    auto parsers = std::make_shared<std::vector<json::JsonParser> >(slots);
    auto arenas = std::make_shared<std::vector<StringArena> >(slots);
    auto tscales = std::make_shared<std::vector<trace::TimeScaleState> >(slots);
    auto keep = std::make_shared<std::vector<std::string> >(bp.select);
    rd::RowBuildOptions ropts;
    if (!keep->empty()) ropts.keep = keep.get();
    ropts.time_scale = bp.time_scale;

    auto consume = [builders, parsers, arenas, tscales, keep, ropts, slots](
                       std::size_t slot, const json::JsonValue&,
                       std::string_view raw) {
        if (slot >= slots) return;
        rd::process_json_line((*builders)[slot], (*parsers)[slot],
                              (*arenas)[slot], raw, /*normalize=*/false,
                              (*tscales)[slot], ropts);
    };
    auto finalize = [builders, slots, out]() {
        std::vector<dataframe::DataFrame> frames;
        frames.reserve(slots);
        for (std::size_t s = 0; s < slots; ++s) {
            if ((*builders)[s].num_rows() == 0) continue;
            auto res = (*builders)[s].finish();
            frames.push_back(dataframe::DataFrame::from_arrow(res.get_schema(),
                                                              res.get_array()));
        }
        if (frames.empty()) return;  // out stays an empty frame
        std::vector<const dataframe::DataFrame*> parts;
        parts.reserve(frames.size());
        for (const auto& f : frames) parts.push_back(&f);
        *out = dataframe::concat(parts);
    };

    if (bp.query)
        detail::add_fold_branch(*state_, *bp.query, std::move(consume),
                                std::move(finalize));
    else
        detail::add_fold_branch(*state_, std::move(consume),
                                std::move(finalize));
    return {out, executed_};
#endif
}

coro::CoroTask<ExportStats> ViewSession::execute() {
    ExportStats stats = co_await detail::run_session(state_);
    // Branch outputs are populated; run the post-scan combines before flipping
    // executed_ so a combine reads the raw branch out, not a resolved handle.
    for (auto& c : combines_) c();
    *executed_ = true;
    co_return stats;
}

}  // namespace dftracer::utils::trace::views
