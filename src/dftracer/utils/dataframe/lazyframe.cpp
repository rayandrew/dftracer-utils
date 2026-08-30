#include <ankerl/unordered_dense.h>
#include <dftracer/utils/dataframe/agg.h>        // streaming group-by state
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

// Wrap a morsel's columns in a DataFrame (dummy names), apply `fn`, and return
// the result materialized FLAT.
template <class Fn>
Morsel map_frame(Morsel&& m, Fn&& fn) {
    DataFrame tmp;
    tmp.names.assign(m.columns.size(), std::string());
    for (Series& c : m.columns) tmp.columns.push_back(std::move(c));
    DataFrame r = fn(std::move(tmp));
    Morsel out;
    out.columns.reserve(r.columns.size());
    for (const Series& c : r.columns) out.columns.push_back(c.materialize());
    out.rows = out.columns.empty() ? 0 : out.columns.front().length();
    return out;
}

// Drops rows null in any column; skips fully-dropped morsels.
class DropNullsCursor : public Cursor {
   public:
    explicit DropNullsCursor(std::unique_ptr<Cursor> in) : in_(std::move(in)) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        while (auto m = in_->next(max_rows)) {
            Morsel out = map_frame(std::move(*m),
                                   [](DataFrame f) { return f.drop_nulls(); });
            if (out.rows > 0) return out;
        }
        return std::nullopt;
    }

   private:
    std::unique_ptr<Cursor> in_;
};

// Fills nulls per morsel.
class FillNullCursor : public Cursor {
   public:
    FillNullCursor(std::unique_ptr<Cursor> in, dftu_scalar value)
        : in_(std::move(in)), value_(value) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        auto m = in_->next(max_rows);
        if (!m) return std::nullopt;
        return map_frame(std::move(*m),
                         [&](DataFrame f) { return f.fill_null(value_); });
    }

   private:
    std::unique_ptr<Cursor> in_;
    dftu_scalar value_;
};

// Prepends a global Int64 row-index column.
class WithRowIndexCursor : public Cursor {
   public:
    explicit WithRowIndexCursor(std::unique_ptr<Cursor> in)
        : in_(std::move(in)) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        auto m = in_->next(max_rows);
        if (!m) return std::nullopt;
        std::vector<std::int64_t> idx(static_cast<std::size_t>(m->rows));
        for (std::int64_t i = 0; i < m->rows; ++i) idx[i] = pos_ + i;
        pos_ += m->rows;
        Morsel out;
        out.rows = m->rows;
        out.columns.reserve(m->columns.size() + 1);
        out.columns.push_back(Series::flat_i64(idx.data(), m->rows));
        for (Series& c : m->columns) out.columns.push_back(std::move(c));
        return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::int64_t pos_ = 0;
};

// Per-column null counts, accumulated streaming, emitted as one row.
class NullCountCursor : public Cursor {
   public:
    explicit NullCountCursor(std::unique_ptr<Cursor> in) : in_(std::move(in)) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        std::vector<std::int64_t> counts;
        while (auto m = in_->next(max_rows)) {
            if (counts.empty()) counts.assign(m->columns.size(), 0);
            for (std::size_t i = 0; i < m->columns.size(); ++i)
                counts[i] += m->columns[i].null_count();
        }
        if (counts.empty()) return std::nullopt;
        Morsel out;
        out.rows = 1;
        out.columns.reserve(counts.size());
        for (std::int64_t& c : counts)
            out.columns.push_back(Series::flat_i64(&c, 1));
        return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    bool done_ = false;
};

// Wrap a morsel's columns in a DataFrame with real `names`, apply `fn`, return
// the result materialized FLAT.
template <class Fn>
Morsel frame_op(Morsel&& m, const std::vector<std::string>& names, Fn&& fn) {
    DataFrame tmp;
    tmp.names = names;
    for (Series& c : m.columns) tmp.columns.push_back(std::move(c));
    DataFrame r = fn(std::move(tmp));
    Morsel out;
    out.rows = r.num_rows();
    out.columns.reserve(r.columns.size());
    for (const Series& c : r.columns) out.columns.push_back(c.materialize());
    return out;
}

// Explodes a List column per morsel (streaming).
class ExplodeCursor : public Cursor {
   public:
    ExplodeCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                  std::string column)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          column_(std::move(column)) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        auto m = in_->next(max_rows);
        if (!m) return std::nullopt;
        return frame_op(std::move(*m), sch_,
                        [&](DataFrame f) { return f.explode(column_); });
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::string column_;
};

// Reshapes wide->long per morsel (streaming).
class UnpivotCursor : public Cursor {
   public:
    UnpivotCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                  std::vector<std::string> id, std::vector<std::string> val)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          id_(std::move(id)),
          val_(std::move(val)) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        auto m = in_->next(max_rows);
        if (!m) return std::nullopt;
        return frame_op(std::move(*m), sch_,
                        [&](DataFrame f) { return f.unpivot(id_, val_); });
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_, id_, val_;
};

// The k rows with the largest/smallest `name`: keep a running best of <= k
// rows, re-topk after each morsel (bounded state), emit once. Streaming
// ingestion.
class TopkCursor : public Cursor {
   public:
    TopkCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
               std::string name, std::int64_t k, bool largest)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          name_(std::move(name)),
          k_(k),
          largest_(largest) {}
    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        DataFrame best;
        bool has = false;
        while (auto m = in_->next(max_rows)) {
            DataFrame cur;
            cur.names = sch_;
            for (Series& c : m->columns) cur.columns.push_back(std::move(c));
            if (!has) {
                best = cur.topk(name_, k_, largest_);
                has = true;
            } else {
                DataFrame u = concat({&best, &cur});
                best = u.topk(name_, k_, largest_);
            }
        }
        if (!has) return std::nullopt;
        Morsel out;
        out.rows = best.num_rows();
        out.columns.reserve(best.columns.size());
        for (const Series& c : best.columns)
            out.columns.push_back(c.materialize());
        return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::string name_;
    std::int64_t k_;
    bool largest_;
    bool done_ = false;
};

AggOp to_agg_op(Agg a) {
    switch (a) {
        case Agg::Sum:
            return AggOp::Sum;
        case Agg::Min:
            return AggOp::Min;
        case Agg::Max:
            return AggOp::Max;
        case Agg::Count:
            return AggOp::Count;
        case Agg::Mean:
            return AggOp::Mean;
        case Agg::Var:
            return AggOp::Var;
        case Agg::Std:
            return AggOp::Std;
        case Agg::Skew:
            return AggOp::Skew;
        case Agg::Kurt:
            return AggOp::Kurt;
        case Agg::First:
            return AggOp::First;
        case Agg::Last:
            return AggOp::Last;
        case Agg::Pct:
            return AggOp::Pct;
        case Agg::Hist:
            return AggOp::Hist;
    }
    return AggOp::Count;
}

// Streaming group-by: fold every morsel into one mergeable AggState (bounded by
// the group count), finalize once. No materialize-all.
class GroupByCursor : public Cursor {
   public:
    GroupByCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                  std::string key, std::vector<GroupAgg> aggs)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          key_(std::move(key)),
          aggs_(std::move(aggs)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;

        const int key_idx = index_in(sch_, key_);
        std::vector<AggSpec> specs;
        std::vector<int> value_idx;  // sch indices of the deduped value columns
        ankerl::unordered_dense::map<std::string, std::int32_t> dedup;
        specs.reserve(aggs_.size());
        for (const GroupAgg& a : aggs_) {
            AggSpec sp;
            sp.op = to_agg_op(a.op);
            sp.out = a.out;
            sp.param = a.param;
            if (sp.op == AggOp::Count) {
                sp.value_col = -1;
            } else {
                auto it = dedup.find(a.column);
                if (it != dedup.end()) {
                    sp.value_col = it->second;
                } else {
                    sp.value_col = static_cast<std::int32_t>(value_idx.size());
                    value_idx.push_back(index_in(sch_, a.column));
                    dedup.emplace(a.column, sp.value_col);
                }
            }
            specs.push_back(std::move(sp));
        }

        AggStatePtr state = agg_new(specs);
        while (auto m = in_->next(max_rows)) {
            std::vector<const Series*> values;
            values.reserve(value_idx.size());
            for (int vi : value_idx) values.push_back(&m->columns[vi]);
            agg_accumulate(*state, m->columns[key_idx], values);
        }
        DataFrame r = agg_finalize(*state, key_);
        Morsel out;
        out.rows = r.num_rows();
        out.columns.reserve(r.columns.size());
        for (const Series& c : r.columns)
            out.columns.push_back(c.materialize());
        return out;
    }

   private:
    static int index_in(const std::vector<std::string>& s,
                        const std::string& n) {
        auto it = std::find(s.begin(), s.end(), n);
        return it == s.end() ? -1 : static_cast<int>(it - s.begin());
    }
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::string key_;
    std::vector<GroupAgg> aggs_;
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
struct DropNullsOp {};
struct FillNullOp {
    dftu_scalar value;
};
struct WithRowIndexOp {
    std::string name;
};
struct NullCountOp {};
struct ExplodeOp {
    std::string column;
};
struct UnpivotOp {
    std::vector<std::string> id_vars;
    std::vector<std::string> value_vars;
};
struct TopkOp {
    std::string name;
    std::int64_t k;
    bool largest;
};
struct GroupByOp {
    std::string key;
    std::vector<GroupAgg> aggs;
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
    std::variant<FilterOp, SelectOp, WithColumnOp, RenameOp, SliceOp, TailOp,
                 DropNullsOp, FillNullOp, WithRowIndexOp, NullCountOp,
                 ExplodeOp, UnpivotOp, TopkOp, GroupByOp>
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
                   [&](const DropNullsOp&) { return in; },
                   [&](const FillNullOp&) { return in; },
                   [&](const SelectOp& o) { return o.names; },
                   [&](const RenameOp& o) { return o.names; },
                   [&](const WithColumnOp& o) {
                       if (std::find(in.begin(), in.end(), o.name) == in.end())
                           in.push_back(o.name);
                       return in;
                   },
                   [&](const WithRowIndexOp& o) {
                       in.insert(in.begin(), o.name);
                       return in;
                   },
                   [&](const NullCountOp&) { return in; },
                   [&](const ExplodeOp&) { return in; },
                   [&](const TopkOp&) { return in; },
                   [&](const UnpivotOp& o) {
                       std::vector<std::string> s = o.id_vars;
                       s.push_back("variable");
                       s.push_back("value");
                       return s;
                   },
                   [&](const GroupByOp& o) {
                       std::vector<std::string> s{o.key};
                       for (const GroupAgg& a : o.aggs) s.push_back(a.out);
                       return s;
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
            [](const TailOp&) { return std::string("tail"); },
            [](const DropNullsOp&) { return std::string("drop_nulls"); },
            [](const FillNullOp&) { return std::string("fill_null"); },
            [](const WithRowIndexOp& o) { return "with_row_index " + o.name; },
            [](const NullCountOp&) { return std::string("null_count"); },
            [](const ExplodeOp& o) { return "explode " + o.column; },
            [](const UnpivotOp&) { return std::string("unpivot"); },
            [](const TopkOp& o) { return "topk " + o.name; },
            [](const GroupByOp& o) { return "group_by " + o.key; }},
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
        overloaded{
            [&](const FilterOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<FilterCursor>(std::move(in), o.pred);
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
                return std::make_unique<SliceCursor>(std::move(in), o.offset,
                                                     o.len);
            },
            [&](const TailOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<TailCursor>(std::move(in), o.n);
            },
            [&](const DropNullsOp&) -> std::unique_ptr<Cursor> {
                return std::make_unique<DropNullsCursor>(std::move(in));
            },
            [&](const FillNullOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<FillNullCursor>(std::move(in), o.value);
            },
            [&](const WithRowIndexOp&) -> std::unique_ptr<Cursor> {
                return std::make_unique<WithRowIndexCursor>(std::move(in));
            },
            [&](const NullCountOp&) -> std::unique_ptr<Cursor> {
                return std::make_unique<NullCountCursor>(std::move(in));
            },
            [&](const ExplodeOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<ExplodeCursor>(std::move(in), sch,
                                                       o.column);
            },
            [&](const UnpivotOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<UnpivotCursor>(std::move(in), sch,
                                                       o.id_vars, o.value_vars);
            },
            [&](const TopkOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<TopkCursor>(std::move(in), sch, o.name,
                                                    o.k, o.largest);
            },
            [&](const GroupByOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<GroupByCursor>(std::move(in), sch,
                                                       o.key, o.aggs);
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

LazyFrame LazyFrame::drop_nulls() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{DropNullsOp{}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::fill_null(dftu_scalar value) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{FillNullOp{value}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::with_row_index(std::string name) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{WithRowIndexOp{std::move(name)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::null_count() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{NullCountOp{}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::explode(std::string column) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{ExplodeOp{std::move(column)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::unpivot(std::vector<std::string> id_vars,
                             std::vector<std::string> value_vars) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{UnpivotOp{std::move(id_vars), std::move(value_vars)}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::topk(std::string name, std::int64_t k,
                          bool largest) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{TopkOp{std::move(name), k, largest}}));
    return LazyFrame(source_, std::move(ops));
}

LazyFrame LazyFrame::group_by(std::string key,
                              std::vector<GroupAgg> aggs) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{GroupByOp{std::move(key), std::move(aggs)}}));
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
