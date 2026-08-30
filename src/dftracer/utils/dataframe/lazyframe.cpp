#include <dftracer/utils/dataframe/batch_ops.h>  // concat_columns
#include <dftracer/utils/dataframe/lazyframe.h>

#include <algorithm>
#include <utility>

namespace dftracer::utils::dataframe {

namespace {

std::vector<const Series*> column_ptrs(const std::vector<Series>& cols) {
    std::vector<const Series*> in;
    in.reserve(cols.size());
    for (const Series& c : cols) in.push_back(&c);
    return in;
}

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
// morsel (so an all-filtered chunk never leaks downstream) or the input ends.
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

}  // namespace

class LazyOp {
   public:
    enum class Kind { Filter, Select, WithColumn };
    Kind kind;
    Expr expr;                       // Filter / WithColumn
    std::vector<std::string> names;  // Select
    std::string name;                // WithColumn

    std::vector<std::string> out_schema(std::vector<std::string> in) const {
        switch (kind) {
            case Kind::Filter:
                return in;
            case Kind::Select:
                return names;
            case Kind::WithColumn:
                if (std::find(in.begin(), in.end(), name) == in.end())
                    in.push_back(name);
                return in;
        }
        return in;
    }
};

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
    auto op = std::make_shared<LazyOp>();
    op->kind = LazyOp::Kind::Select;
    op->names = std::move(names);
    ops.push_back(std::move(op));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::filter(Expr predicate) const {
    auto ops = ops_;
    auto op = std::make_shared<LazyOp>();
    op->kind = LazyOp::Kind::Filter;
    op->expr = std::move(predicate);
    ops.push_back(std::move(op));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::with_column(std::string name, Expr expr) const {
    auto ops = ops_;
    auto op = std::make_shared<LazyOp>();
    op->kind = LazyOp::Kind::WithColumn;
    op->name = std::move(name);
    op->expr = std::move(expr);
    ops.push_back(std::move(op));
    return LazyFrame(source_, std::move(ops));
}

std::vector<std::string> LazyFrame::schema() const {
    std::vector<std::string> s = source_->names();
    for (const auto& op : ops_) s = op->out_schema(std::move(s));
    return s;
}

DataFrame LazyFrame::collect(std::int64_t morsel_rows) const {
    // Build the cursor chain: source reader wrapped by each op, tracking the
    // schema so a Select resolves names to indices.
    std::unique_ptr<Cursor> cur = source_->open();
    std::vector<std::string> sch = source_->names();
    for (const auto& op : ops_) {
        switch (op->kind) {
            case LazyOp::Kind::Filter:
                cur = std::make_unique<FilterCursor>(std::move(cur), op->expr);
                break;
            case LazyOp::Kind::Select: {
                std::vector<int> idx;
                idx.reserve(op->names.size());
                for (const std::string& nm : op->names) {
                    auto it = std::find(sch.begin(), sch.end(), nm);
                    idx.push_back(it == sch.end()
                                      ? -1
                                      : static_cast<int>(it - sch.begin()));
                }
                cur = std::make_unique<SelectCursor>(std::move(cur),
                                                     std::move(idx));
                break;
            }
            case LazyOp::Kind::WithColumn: {
                auto it = std::find(sch.begin(), sch.end(), op->name);
                int replace =
                    it == sch.end() ? -1 : static_cast<int>(it - sch.begin());
                cur = std::make_unique<WithColumnCursor>(std::move(cur),
                                                         op->expr, replace);
                break;
            }
        }
        sch = op->out_schema(std::move(sch));
    }

    // Drain and concatenate the morsels into the result frame.
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
    // DataFrame is move-only; hand the source a shallow copy that shares column
    // buffers.
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
