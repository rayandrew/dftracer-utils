#include <dftracer/utils/dataframe/batch_ops.h>  // concat_columns
#include <dftracer/utils/dataframe/lazyframe.h>

#include <algorithm>
#include <deque>
#include <string>
#include <utility>
#include <variant>

namespace dftracer::utils::dataframe {

namespace {

std::vector<const Series*> column_ptrs(const std::vector<Series>& cols) {
    std::vector<const Series*> in;
    in.reserve(cols.size());
    for (const Series& c : cols) in.push_back(&c);
    return in;
}

// ---- cursors ----------------------------------------------------------------

// Reads contiguous chunks off an in-memory frame, materialized FLAT.
class InMemoryCursor : public Cursor {
   public:
    explicit InMemoryCursor(std::shared_ptr<const DataFrame> frame)
        : frame_(std::move(frame)), n_(frame_->num_rows()) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (off_ >= n_) return std::nullopt;
        const std::int64_t len =
            std::min(std::max<std::int64_t>(max_rows, 1), n_ - off_);
        DataFrame chunk = frame_->slice(off_, len);
        off_ += len;
        Morsel m;
        m.rows = len;
        m.columns.reserve(chunk.columns.size());
        for (const Series& c : chunk.columns)
            m.columns.push_back(c.materialize());
        return m;
    }

   private:
    std::shared_ptr<const DataFrame> frame_;
    std::int64_t n_;
    std::int64_t off_ = 0;
};

// Keeps rows where `pred` is true; pulls upstream until it has a non-empty
// morsel or the input ends.
class FilterCursor : public Cursor {
   public:
    FilterCursor(std::unique_ptr<Cursor> in, Expr pred)
        : in_(std::move(in)), pred_(std::move(pred)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        while (auto m = in_->next(max_rows)) {
            Series mask = eval(pred_, column_ptrs(m->columns));
            DataFrame tmp;
            tmp.names.assign(m->columns.size(), std::string());
            for (Series& c : m->columns) tmp.columns.push_back(c.share());
            DataFrame kept = tmp.filter(mask);
            Morsel out;
            out.columns.reserve(kept.columns.size());
            for (const Series& c : kept.columns)
                out.columns.push_back(c.materialize());
            out.rows = out.columns.empty() ? 0 : out.columns.front().length();
            if (out.rows > 0) return out;
        }
        return std::nullopt;
    }

   private:
    std::unique_ptr<Cursor> in_;
    Expr pred_;
};

// Projects columns by index (share, no copy).
class SelectCursor : public Cursor {
   public:
    SelectCursor(std::unique_ptr<Cursor> in, std::vector<int> idx)
        : in_(std::move(in)), idx_(std::move(idx)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        auto m = in_->next(max_rows);
        if (!m) return std::nullopt;
        Morsel out;
        out.rows = m->rows;
        out.columns.reserve(idx_.size());
        for (int i : idx_) out.columns.push_back(m->columns[i].share());
        return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<int> idx_;
};

// Adds or replaces one column from an expr.
class WithColumnCursor : public Cursor {
   public:
    WithColumnCursor(std::unique_ptr<Cursor> in, Expr e, int replace)
        : in_(std::move(in)), expr_(std::move(e)), replace_(replace) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        auto m = in_->next(max_rows);
        if (!m) return std::nullopt;
        Series nc = eval(expr_, column_ptrs(m->columns));
        Morsel out;
        out.rows = m->rows;
        out.columns = std::move(m->columns);
        if (replace_ >= 0)
            out.columns[static_cast<std::size_t>(replace_)] = std::move(nc);
        else
            out.columns.push_back(std::move(nc));
        return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    Expr expr_;
    int replace_;
};

// Slices a morsel's rows [off, off+len) as a fresh FLAT morsel.
Morsel slice_morsel(const Morsel& m, std::int64_t off, std::int64_t len) {
    DataFrame tmp;
    tmp.names.assign(m.columns.size(), std::string());
    for (const Series& c : m.columns) tmp.columns.push_back(c.share());
    DataFrame s = tmp.slice(off, len);
    Morsel out;
    out.rows = len;
    out.columns.reserve(s.columns.size());
    for (const Series& c : s.columns) out.columns.push_back(c.materialize());
    return out;
}

// Emits the global row window [offset, offset+len); stops once len rows are out
// (so head() short-circuits the scan).
class SliceCursor : public Cursor {
   public:
    SliceCursor(std::unique_ptr<Cursor> in, std::int64_t offset,
                std::int64_t len)
        : in_(std::move(in)), offset_(offset), len_(len) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        while (emitted_ < len_) {
            auto m = in_->next(max_rows);
            if (!m) return std::nullopt;
            const std::int64_t start = seen_;
            seen_ += m->rows;
            const std::int64_t w_start = std::max(offset_, start);
            const std::int64_t w_end = std::min(offset_ + len_, seen_);
            if (w_end <= w_start) continue;
            emitted_ += w_end - w_start;
            return slice_morsel(*m, w_start - start, w_end - w_start);
        }
        return std::nullopt;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::int64_t offset_, len_;
    std::int64_t seen_ = 0, emitted_ = 0;
};

// Keeps the last n rows: consumes the input streaming, holding at most n rows
// (plus one morsel) in a ring, then emits the tail once.
class TailCursor : public Cursor {
   public:
    TailCursor(std::unique_ptr<Cursor> in, std::int64_t n)
        : in_(std::move(in)), n_(n) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        if (n_ <= 0) return std::nullopt;
        std::deque<Morsel> buf;
        std::int64_t total = 0;
        while (auto m = in_->next(max_rows)) {
            total += m->rows;
            buf.push_back(std::move(*m));
            while (!buf.empty() && total - buf.front().rows >= n_) {
                total -= buf.front().rows;
                buf.pop_front();
            }
        }
        if (buf.empty()) return std::nullopt;
        const std::size_t ncols = buf.front().columns.size();
        Morsel out;
        out.rows = total;
        out.columns.reserve(ncols);
        for (std::size_t c = 0; c < ncols; ++c) {
            std::vector<const Series*> parts;
            parts.reserve(buf.size());
            for (Morsel& m : buf) parts.push_back(&m.columns[c]);
            out.columns.push_back(concat_columns(parts));
        }
        if (total > n_) return slice_morsel(out, total - n_, n_);
        return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::int64_t n_;
    bool done_ = false;
};

// ---- plan ops (tagged union) ------------------------------------------------

struct FilterOp {
    Expr pred;
};
struct SelectOp {
    std::vector<std::string> names;
};
struct WithColumnOp {
    std::string name;
    Expr expr;
};
struct RenameOp {
    std::vector<std::string> names;
};
struct SliceOp {
    std::int64_t offset;
    std::int64_t len;
};
struct TailOp {
    std::int64_t n;
};

template <class... Ts>
struct overloaded : Ts... {
    using Ts::operator()...;
};
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace

// A plan node: exactly the data its op needs (no fat struct). Held by value in
// the LazyFrame plan.
class LazyOp {
   public:
    std::variant<FilterOp, SelectOp, WithColumnOp, RenameOp, SliceOp, TailOp>
        node;
};

namespace {

std::string join_names(const std::vector<std::string>& v) {
    std::string s;
    for (std::size_t i = 0; i < v.size(); ++i) {
        if (i) s += ", ";
        s += v[i];
    }
    return s;
}

std::vector<std::string> out_schema(const LazyOp& op,
                                    std::vector<std::string> in) {
    return std::visit(
        overloaded{[&](const FilterOp&) { return in; },
                   [&](const SliceOp&) { return in; },
                   [&](const TailOp&) { return in; },
                   [&](const SelectOp& o) { return o.names; },
                   [&](const RenameOp& o) { return o.names; },
                   [&](const WithColumnOp& o) {
                       if (std::find(in.begin(), in.end(), o.name) == in.end())
                           in.push_back(o.name);
                       return in;
                   }},
        op.node);
}

std::string describe(const LazyOp& op) {
    return std::visit(
        overloaded{
            [](const FilterOp&) { return std::string("filter"); },
            [](const SelectOp& o) {
                return "select [" + join_names(o.names) + "]";
            },
            [](const WithColumnOp& o) { return "with_column " + o.name; },
            [](const RenameOp& o) {
                return "rename [" + join_names(o.names) + "]";
            },
            [](const SliceOp&) { return std::string("slice"); },
            [](const TailOp&) { return std::string("tail"); }},
        op.node);
}

int col_index(const std::vector<std::string>& sch, const std::string& name) {
    auto it = std::find(sch.begin(), sch.end(), name);
    return it == sch.end() ? -1 : static_cast<int>(it - sch.begin());
}

// Predicate pushdown: bubble each Filter left past any WithColumn whose output
// column it does not read, so the filter shrinks the with_column's input. Safe
// on column indices because WithColumn appends or replaces in place; filters do
// not cross other ops (Select/Rename reindex, Slice/Tail change row counts).
std::vector<std::shared_ptr<const LazyOp>> pushdown_predicates(
    const std::vector<std::string>& source_names,
    const std::vector<std::shared_ptr<const LazyOp>>& ops) {
    struct Node {
        std::shared_ptr<const LazyOp> op;
        int write_idx;  // WithColumn output column index; -1 otherwise
    };
    std::vector<Node> nodes;
    nodes.reserve(ops.size());
    std::vector<std::string> sch = source_names;
    for (const auto& op : ops) {
        int w = -1;
        if (const auto* wc = std::get_if<WithColumnOp>(&op->node)) {
            w = col_index(sch, wc->name);
            if (w < 0) w = static_cast<int>(sch.size());
        }
        nodes.push_back({op, w});
        sch = out_schema(*op, std::move(sch));
    }

    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            const auto* filt = std::get_if<FilterOp>(&nodes[i].op->node);
            const bool prev_wc =
                std::holds_alternative<WithColumnOp>(nodes[i - 1].op->node);
            if (filt && prev_wc &&
                !expr_references(filt->pred, nodes[i - 1].write_idx)) {
                std::swap(nodes[i - 1], nodes[i]);
                changed = true;
            }
        }
    }

    std::vector<std::shared_ptr<const LazyOp>> out;
    out.reserve(nodes.size());
    for (Node& n : nodes) out.push_back(std::move(n.op));
    return out;
}

// Build the cursor for one op over `in`, resolving names against `sch` (the
// op's input schema).
std::unique_ptr<Cursor> make_cursor(const LazyOp& op,
                                    std::unique_ptr<Cursor> in,
                                    const std::vector<std::string>& sch) {
    return std::visit(
        overloaded{[&](const FilterOp& o) -> std::unique_ptr<Cursor> {
                       return std::make_unique<FilterCursor>(std::move(in),
                                                             o.pred);
                   },
                   [&](const SelectOp& o) -> std::unique_ptr<Cursor> {
                       std::vector<int> idx;
                       idx.reserve(o.names.size());
                       for (const std::string& nm : o.names)
                           idx.push_back(col_index(sch, nm));
                       return std::make_unique<SelectCursor>(std::move(in),
                                                             std::move(idx));
                   },
                   [&](const WithColumnOp& o) -> std::unique_ptr<Cursor> {
                       return std::make_unique<WithColumnCursor>(
                           std::move(in), o.expr, col_index(sch, o.name));
                   },
                   [&](const RenameOp&) -> std::unique_ptr<Cursor> {
                       return std::move(in);  // names-only; data passes through
                   },
                   [&](const SliceOp& o) -> std::unique_ptr<Cursor> {
                       return std::make_unique<SliceCursor>(std::move(in),
                                                            o.offset, o.len);
                   },
                   [&](const TailOp& o) -> std::unique_ptr<Cursor> {
                       return std::make_unique<TailCursor>(std::move(in), o.n);
                   }},
        op.node);
}

}  // namespace

InMemorySource::InMemorySource(DataFrame frame)
    : frame_(std::make_shared<const DataFrame>(std::move(frame))) {}

std::vector<std::string> InMemorySource::names() const { return frame_->names; }

std::unique_ptr<Cursor> InMemorySource::open() const {
    return std::make_unique<InMemoryCursor>(frame_);
}

LazyFrame LazyFrame::scan(std::shared_ptr<const Source> source) {
    return LazyFrame(std::move(source), {});
}

LazyFrame LazyFrame::select(std::vector<std::string> names) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{SelectOp{std::move(names)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::filter(Expr predicate) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{FilterOp{std::move(predicate)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::with_column(std::string name, Expr expr) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{WithColumnOp{std::move(name), std::move(expr)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::rename(std::vector<std::string> names) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{RenameOp{std::move(names)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::slice(std::int64_t offset, std::int64_t len) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{SliceOp{offset, len}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::head(std::int64_t n) const { return slice(0, n); }

LazyFrame LazyFrame::tail(std::int64_t n) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{TailOp{n}}));
    return LazyFrame(source_, std::move(ops));
}

std::vector<std::string> LazyFrame::schema() const {
    std::vector<std::string> s = source_->names();
    for (const auto& op : ops_) s = out_schema(*op, std::move(s));
    return s;
}

std::string LazyFrame::explain() const {
    std::string s = "scan [" + join_names(source_->names()) + "]";
    for (const auto& op : pushdown_predicates(source_->names(), ops_))
        s += "\n" + describe(*op);
    return s;
}

DataFrame LazyFrame::collect(std::int64_t morsel_rows) const {
    auto ops = pushdown_predicates(source_->names(), ops_);
    std::unique_ptr<Cursor> cur = source_->open();
    std::vector<std::string> sch = source_->names();
    for (const auto& op : ops) {
        cur = make_cursor(*op, std::move(cur), sch);
        sch = out_schema(*op, std::move(sch));
    }

    std::vector<std::vector<Series>> chunks;
    while (auto m = cur->next(morsel_rows))
        chunks.push_back(std::move(m->columns));

    DataFrame out;
    out.names = std::move(sch);
    const std::size_t ncols = out.names.size();
    out.columns.reserve(ncols);
    for (std::size_t c = 0; c < ncols; ++c) {
        std::vector<const Series*> parts;
        parts.reserve(chunks.size());
        for (auto& ch : chunks) parts.push_back(&ch[c]);
        out.columns.push_back(concat_columns(parts));
    }
    return out;
}

LazyFrame DataFrame::lazy() const {
    DataFrame shared;
    shared.names = names;
    shared.columns.reserve(columns.size());
    for (const Series& c : columns) shared.columns.push_back(c.share());
    return LazyFrame::scan(std::make_shared<InMemorySource>(std::move(shared)));
}

LazyFrame lazy(DataFrame frame) {
    return LazyFrame::scan(std::make_shared<InMemorySource>(std::move(frame)));
}

}  // namespace dftracer::utils::dataframe
