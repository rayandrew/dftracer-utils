#include <dftracer/utils/core/common/field_ref.h>
#include <dftracer/utils/core/coro/async_semaphore.h>
#include <dftracer/utils/core/coro/channel.h>
#include <dftracer/utils/core/coro/coro.h>
#include <dftracer/utils/core/pipeline/executor.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/scalar.h>
#include <dftracer/utils/query/builder.h>
#include <dftracer/utils/query/query.h>
#include <dftracer/utils/trace/views/native_row_fold.h>
#include <dftracer/utils/trace/views/stream_row_fold.h>
#include <dftracer/utils/trace/views/view_plan.h>
#include <dftracer/utils/trace/views/view_source.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <future>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>

namespace dftracer::utils::trace::views {

bool ViewCursor::has_nested_column() const {
    if (!nested_) {
        nested_ = std::any_of(
            buf_->columns.begin(), buf_->columns.end(),
            [](const dftracer::utils::dataframe::Series& c) {
                return c.type() == dftracer::utils::dataframe::TypeId::List ||
                       c.type() == dftracer::utils::dataframe::TypeId::Struct;
            });
    }
    return *nested_;
}

coro::CoroTask<std::optional<dftracer::utils::dataframe::Morsel>>
ViewCursor::next(std::int64_t max_rows) {
    const std::int64_t nrows = buf_->num_rows();
    if (offset_ >= nrows) co_return std::nullopt;

    if (has_nested_column() || max_rows <= 0) {
        dftracer::utils::dataframe::Morsel m;
        m.rows = nrows;
        m.columns.reserve(buf_->columns.size());
        for (const dftracer::utils::dataframe::Series& c : buf_->columns)
            m.columns.push_back(c.share());
        offset_ = nrows;
        co_return m;
    }

    dftracer::utils::dataframe::DataFrame part = buf_->slice(offset_, max_rows);
    dftracer::utils::dataframe::Morsel m;
    m.rows = part.num_rows();
    m.columns = std::move(part.columns);
    offset_ += m.rows;
    co_return m;
}

namespace {

namespace df = dftracer::utils::dataframe;
namespace q = dftracer::utils::query;

// What a source can push for one predicate: the query, plus whether it is the
// predicate exactly or merely a superset of it (a superset prunes I/O but must
// be re-applied by the engine).
struct Pushable {
    q::Expr expr;
    bool exact;
};

// Translate a `col <cmp> scalar` LazyFrame predicate into an index-pushable
// query on the named event field. nullopt for anything else (col-vs-col,
// arithmetic), which the engine applies itself. ts/dur carry a time_scale in
// the streamed morsel but the query matches the raw field, so those are only
// pushable when the scale is the identity.
std::optional<q::Expr> translate_leaf(const df::Expr& e,
                                      const std::vector<std::string>& fnames,
                                      double time_scale) {
    std::int32_t ci = -1;
    df::CmpOp op{};
    df::Scalar rhs;
    if (!df::expr_as_col_cmp(e, &ci, &op, &rhs)) return std::nullopt;
    if (ci < 0 || static_cast<std::size_t>(ci) >= fnames.size())
        return std::nullopt;
    const std::string& col = fnames[static_cast<std::size_t>(ci)];
    if ((col == "ts" || col == "dur") && time_scale != 1.0) return std::nullopt;
    // The query DSL names a nested arg field by its bare key; a top-level field
    // is unchanged.
    const std::string field(dftracer::utils::strip_args_prefix(col));

    q::CompareOp qop;
    switch (op) {
        case df::CmpOp::Gt:
            qop = q::CompareOp::GT;
            break;
        case df::CmpOp::Ge:
            qop = q::CompareOp::GE;
            break;
        case df::CmpOp::Lt:
            qop = q::CompareOp::LT;
            break;
        case df::CmpOp::Le:
            qop = q::CompareOp::LE;
            break;
        case df::CmpOp::Eq:
            qop = q::CompareOp::EQ;
            break;
        case df::CmpOp::Ne:
            qop = q::CompareOp::NE;
            break;
    }

    q::LiteralNode lit;
    switch (rhs.tag()) {
        case DFTU_SCALAR_TAG_I64:
            lit = q::LiteralNode{rhs.i64()};
            break;
        case DFTU_SCALAR_TAG_U64:
            lit = q::LiteralNode{rhs.u64()};
            break;
        default:
            lit = q::LiteralNode{rhs.f64()};
            break;
    }

    return q::field_cmp(field, qop, std::move(lit));
}

// The whole predicate tree, pushing as much of it as is sound:
//
//   AND - one untranslatable side does not sink the other; pushing just the
//         translatable conjunct selects a SUPERSET, which prunes I/O and is
//         re-applied by the engine (Inexact).
//   OR  - both sides must translate. Pushing one alone would drop rows the
//         other side keeps.
//   NOT - the operand must translate EXACTLY. Negating a superset yields a
//         subset, which drops rows.
std::optional<Pushable> translate_pred(const df::Expr& e,
                                       const std::vector<std::string>& fnames,
                                       double time_scale) {
    df::LogicalOp lop{};
    df::Expr lhs, rhs;
    if (df::expr_as_logical(e, &lop, &lhs, &rhs)) {
        std::optional<Pushable> a = translate_pred(lhs, fnames, time_scale);
        std::optional<Pushable> b = translate_pred(rhs, fnames, time_scale);
        if (lop == df::LogicalOp::And) {
            if (a && b)
                return Pushable{
                    q::all_of(std::move(a->expr), std::move(b->expr)),
                    a->exact && b->exact};
            if (a) return Pushable{std::move(a->expr), false};
            if (b) return Pushable{std::move(b->expr), false};
            return std::nullopt;
        }
        if (!a || !b) return std::nullopt;
        return Pushable{q::any_of(std::move(a->expr), std::move(b->expr)),
                        a->exact && b->exact};
    }

    df::Expr inner;
    if (df::expr_as_not(e, &inner)) {
        std::optional<Pushable> a = translate_pred(inner, fnames, time_scale);
        if (!a || !a->exact) return std::nullopt;
        return Pushable{q::negate(std::move(a->expr)), true};
    }

    std::optional<q::Expr> leaf = translate_leaf(e, fnames, time_scale);
    if (!leaf) return std::nullopt;
    return Pushable{std::move(*leaf), true};
}

coro::Coro run_detached(coro::CoroTask<void> task,
                        std::shared_ptr<std::promise<void>> done) {
    try {
        co_await std::move(task);
        done->set_value();
    } catch (...) {
        done->set_exception(std::current_exception());
    }
}

// Enqueues onto Executor::current() rather than default_runtime(), so the
// producer shares whatever is pulling the cursor - a real pool, or the ad hoc
// RunLoop a bare CoroTask::get() stands up, which cannot wait across pools.
std::shared_future<void> spawn_on_current_executor(coro::CoroTask<void> task) {
    dftracer::utils::Executor* exec = dftracer::utils::Executor::current();
    if (!exec) exec = dftracer::utils::default_runtime().executor();
    if (task.handle()) task.handle().promise().set_executor(exec);
    auto done = std::make_shared<std::promise<void>>();
    std::shared_future<void> fut = done->get_future().share();
    coro::Coro c = run_detached(std::move(task), std::move(done));
    exec->enqueue(c.release());
    return fut;
}

// Rows [offset, offset+n) of `m`, sharing `m`'s schema (name_ids/intern) since
// a row slice does not change it.
dftracer::utils::dataframe::Morsel slice_morsel(
    const dftracer::utils::dataframe::Morsel& m, std::int64_t offset,
    std::int64_t n) {
    dftracer::utils::dataframe::DataFrame tmp;
    tmp.names.assign(m.columns.size() + m.dyn_columns.size(), std::string());
    for (const dftracer::utils::dataframe::Series& c : m.columns)
        tmp.columns.push_back(c.share());
    for (const dftracer::utils::dataframe::Series& c : m.dyn_columns)
        tmp.columns.push_back(c.share());
    dftracer::utils::dataframe::DataFrame s = tmp.slice(offset, n);

    dftracer::utils::dataframe::Morsel out;
    out.rows = n;
    out.name_ids = m.name_ids;
    out.intern = m.intern;
    out.dyn_names = m.dyn_names;
    const std::size_t nc = m.columns.size();
    out.columns.assign(
        std::make_move_iterator(s.columns.begin()),
        std::make_move_iterator(s.columns.begin() +
                                static_cast<std::ptrdiff_t>(nc)));
    out.dyn_columns.assign(
        std::make_move_iterator(s.columns.begin() +
                                static_cast<std::ptrdiff_t>(nc)),
        std::make_move_iterator(s.columns.end()));
    return out;
}

// Pulls morsels off the channel, releasing each one's share of `budget_` back
// to the fold's producers once it is handed off; surfaces the producer's
// exception, if any, once the channel drains. A caller-supplied `max_rows`
// re-chunks a received morsel into <=max_rows-row sub-morsels instead of
// handing the whole (one-per-scan-batch) morsel out at once.
class StreamViewCursor : public dftracer::utils::dataframe::Cursor {
   public:
    StreamViewCursor(
        std::shared_ptr<coro::Channel<dftracer::utils::dataframe::Morsel>>
            channel,
        std::shared_ptr<coro::CoroSemaphore> budget,
        std::shared_future<void> producer,
        std::shared_ptr<std::atomic<bool>> stop)
        : channel_(std::move(channel)),
          budget_(std::move(budget)),
          producer_(std::move(producer)),
          stop_(std::move(stop)) {}

    // Abandoning the cursor must stop the scan behind it: the producer holds
    // its own channel registration, so nothing else ends it, and the byte
    // budget is released only here - a producer that outlives the cursor parks
    // in acquire() forever. Setting the flag alone cannot wake a parked
    // producer (the semaphore has no shutdown), so release enough permits for
    // it to run to its next is_cancelled() check and unwind.
    ~StreamViewCursor() override {
        stop_->store(true, std::memory_order_relaxed);
        budget_->release(std::numeric_limits<std::uint32_t>::max());
    }

    coro::CoroTask<std::optional<dftracer::utils::dataframe::Morsel>> next(
        std::int64_t max_rows) override {
        if (max_rows <= 0) {
            auto item = co_await channel_->receive();
            if (item)
                budget_->release(detail::morsel_bytes(*item));
            else
                producer_.get();
            co_return item;
        }

        if (!pending_ || offset_ >= pending_->rows) {
            auto item = co_await channel_->receive();
            if (!item) {
                producer_.get();
                co_return std::nullopt;
            }
            pending_bytes_ = detail::morsel_bytes(*item);
            pending_ = std::move(item);
            offset_ = 0;
        }

        const std::int64_t n = std::min(max_rows, pending_->rows - offset_);
        dftracer::utils::dataframe::Morsel out =
            slice_morsel(*pending_, offset_, n);
        offset_ += n;
        if (offset_ >= pending_->rows) {
            budget_->release(pending_bytes_);
            pending_.reset();
        }
        co_return out;
    }

   private:
    std::shared_ptr<coro::Channel<dftracer::utils::dataframe::Morsel>> channel_;
    std::shared_ptr<coro::CoroSemaphore> budget_;
    std::shared_future<void> producer_;
    std::shared_ptr<std::atomic<bool>> stop_;
    std::optional<dftracer::utils::dataframe::Morsel> pending_;
    std::int64_t offset_ = 0;
    std::uint64_t pending_bytes_ = 0;
};

}  // namespace

bool ViewSource::can_stream_rows() const {
    if (!view_.is_row_query()) return false;
    const detail::ViewPlan& p = *view_.plan_;
    // View::collect() always strips sort/topk/offset/limit before building a
    // ViewSource, and select unless it needs the resolver (resolved.*/r.*
    // fields, which the raw stream never computes); these checks stay as a
    // defensive guard for any other caller.
    return p.sort_col.empty() && p.topk_col.empty() && p.offset == 0 &&
           p.limit == 0 && !detail::select_needs_resolver(p.select);
}

// Matches build_row_frame's own empty-select order (fixed top-level fields,
// fhash/hhash if present, then sorted args), from index metadata rather than
// a scan - best-effort, may omit an arg key not yet indexed.
std::vector<std::string> ViewSource::row_schema() const {
    static const char* const TOP_LEVEL[] = {"name", "cat", "pid", "tid",
                                            "ts",   "dur", "ph"};
    std::vector<std::string> out(std::begin(TOP_LEVEL), std::end(TOP_LEVEL));

    std::vector<std::string> cols = view_.columns();  // sorted
    const bool has_fhash =
        std::binary_search(cols.begin(), cols.end(), std::string("fhash"));
    const bool has_hhash =
        std::binary_search(cols.begin(), cols.end(), std::string("hhash"));
    if (has_fhash) out.push_back("fhash");
    if (has_hhash) out.push_back("hhash");

    for (std::string& c : cols) {
        if (c == "pid" || c == "tid" || c == "ts" || c == "dur" ||
            c == "name" || c == "cat" || c == "fhash" || c == "hhash")
            continue;
        if (c.find('.') != std::string::npos) continue;  // nested/resolved.*
        // A flattened arg column: build_row_frame's empty-select branch
        // always names these "args.<key>", so the schemas must match.
        out.push_back(std::string(dftracer::utils::ARGS_PREFIX) + c);
    }
    return out;
}

dftracer::utils::dataframe::Schema ViewSource::schema() const {
    dftracer::utils::dataframe::Schema s;
    if (can_stream_rows()) {
        // A non-empty select fixes every streamed morsel's columns to exactly
        // this list (see open_stream()), so the schema must match it, not the
        // broader index-derived row_schema(). Canonicalize each select entry
        // the same way build_row_frame does, so a bare arg name (or one
        // colliding with a top-level field) resolves to the same column name
        // the producer actually emits.
        if (!view_.plan_->select.empty()) {
            s.names.reserve(view_.plan_->select.size());
            s.types.reserve(view_.plan_->select.size());
            for (const std::string& sel : view_.plan_->select) {
                s.names.push_back(detail::canonical_row_column_name(sel));
                s.types.push_back(detail::row_column_type(sel));
            }
        } else {
            s.names = row_schema();
            s.types.reserve(s.names.size());
            for (const std::string& name : s.names)
                s.types.push_back(detail::row_column_type(name));
        }
        return s;
    }
    // Aggregated / post-scan-op view: buffer() already ran the scan, so its
    // columns' real types are known (unlike the row-query branches above).
    const dftracer::utils::dataframe::DataFrame& buf = *buffer();
    s.names = buf.names;
    s.types.reserve(buf.columns.size());
    for (const dftracer::utils::dataframe::Series& col : buf.columns)
        s.types.push_back(col.type());
    return s;
}

const dftracer::utils::dataframe::DataFrame* ViewSource::as_frame() const {
    if (can_stream_rows()) return nullptr;
    return buffer().get();
}

std::unique_ptr<dftracer::utils::dataframe::Cursor> ViewSource::open_stream(
    const View& v, std::uint64_t memory_budget) const {
    // Capacity 0 = an effectively unbounded ring (see Channel's ctor); the
    // shared budget semaphore is the sole backpressure, acquired before send
    // and released once the cursor hands a morsel off.
    auto channel = coro::make_channel<dftracer::utils::dataframe::Morsel>(0);
    auto budget = std::make_shared<coro::CoroSemaphore>(memory_budget);
    auto intern = std::make_shared<dftracer::utils::StringIntern>();
    const double time_scale = v.plan_->time_scale;

    // The cursor's early-out: fuse polls the plan's cancel predicate per unit
    // and per batch, so the flag is composed into it rather than added beside
    // it. Composed, never overwritten, so a caller's own cancel_when survives.
    auto stop = std::make_shared<std::atomic<bool>>(false);
    View scan_view = v.cancel_when([stop, prev = v.plan_->cancelled] {
        return stop->load(std::memory_order_relaxed) || (prev && prev());
    });

    // Empty select: each batch discovers its own columns from the actual
    // scanned events, so morsels can differ batch to batch; name_ids lets
    // drain_to_frame reconcile them. A non-empty select instead fixes every
    // morsel's columns to that exact list (build_row_frame's select branch
    // always emits each one, null-filled where absent), matching schema().
    auto task =
        [](View vv, double ts,
           std::shared_ptr<coro::Channel<dftracer::utils::dataframe::Morsel>>
               ch,
           std::shared_ptr<coro::CoroSemaphore> sem,
           std::shared_ptr<dftracer::utils::StringIntern> iv,
           bool emit_dyn) -> coro::CoroTask<void> {
        detail::StreamRowFold fold(ch, sem, iv, vv.plan_->select, ts, nullptr,
                                   vv.plan_->phase == Phase::Metadata,
                                   emit_dyn);
        std::array<detail::Fold*, 1> folds{&fold};
        co_await vv.run_folds(folds, *iv);
    }(scan_view, time_scale, channel, budget, intern, emit_dyn_);

    std::shared_future<void> producer =
        spawn_on_current_executor(std::move(task));
    return std::make_unique<StreamViewCursor>(
        std::move(channel), std::move(budget), std::move(producer),
        std::move(stop));
}

dftracer::utils::dataframe::ScanResult ViewSource::scan(
    const dftracer::utils::dataframe::ScanRequest& req) const {
    dftracer::utils::dataframe::ScanResult r;
    r.filters.assign(req.filters.size(),
                     dftracer::utils::dataframe::Pushed::No);

    // Aggregated / post-scan-op view: buffer once (the resident fast path uses
    // as_frame()). No filter/query pushdown here; project the buffer to honor
    // the requested column set. The engine applies the residual filters.
    if (!can_stream_rows()) {
        std::shared_ptr<const dftracer::utils::dataframe::DataFrame> buf =
            buffer();
        if (!req.projection.empty())
            buf = std::make_shared<const dftracer::utils::dataframe::DataFrame>(
                buf->select(req.projection));
        r.cursor = std::make_unique<ViewCursor>(std::move(buf));
        return r;
    }

    // Streamable row query: translate each simple predicate into the View's
    // query (Exact) and push projection into View::select so the fold harvests
    // only those columns.
    const std::vector<std::string> fnames =
        req.projection.empty() ? names() : req.projection;
    const double time_scale = view_.plan_->time_scale;
    View v = view_;
    for (std::size_t i = 0; i < req.filters.size(); ++i) {
        std::optional<Pushable> pushed =
            translate_pred(req.filters[i], fnames, time_scale);
        if (!pushed) continue;
        auto built = std::move(pushed->expr).build();
        if (!built.has_value()) continue;
        v = v.filter(std::move(built.value()));
        r.filters[i] = pushed->exact
                           ? dftracer::utils::dataframe::Pushed::Exact
                           : dftracer::utils::dataframe::Pushed::Inexact;
    }
    if (!req.projection.empty()) v = v.select(req.projection);

    r.cursor = open_stream(v, req.memory_budget);
    return r;
}

}  // namespace dftracer::utils::trace::views
