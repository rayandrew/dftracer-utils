#include <ankerl/unordered_dense.h>
#include <dftracer/utils/core/common/hash/splitmix64.h>  // sample row keys
#include <dftracer/utils/core/common/memory_budget.h>  // compute_memory_budget
#include <dftracer/utils/dataframe/agg.h>        // streaming group-by state
#include <dftracer/utils/dataframe/batch_ops.h>  // concat_columns, take, concat
#include <dftracer/utils/dataframe/field_stat.h>         // FieldStat (describe)
#include <dftracer/utils/dataframe/internal/cell_ops.h>  // row_key, cell_to_string
#include <dftracer/utils/dataframe/internal/spill.h>     // external-merge spill
#include <dftracer/utils/dataframe/kernels/field_stat.h>  // field_stat_reduce (SIMD)
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/dataframe/parallel.h>  // parallel_for (parallel sinks)
#include <dftracer/utils/dataframe/types.h>     // byte_width, buffer_bytes

#include <algorithm>
#include <cstdint>
#include <deque>
#include <memory>
#include <numeric>
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

// Drain a cursor to a single DataFrame, concatenating its morsels under
// `names`.
DataFrame drain_to_frame(Cursor& in, const std::vector<std::string>& names,
                         std::int64_t max_rows) {
    std::vector<std::vector<Series>> chunks;
    while (auto m = in.next(max_rows)) chunks.push_back(std::move(m->columns));
    DataFrame out;
    out.names = names;
    // Trust the produced column count over `names`: a data-dependent sink emits
    // more (or fewer) columns than the static schema, and the caller relabels.
    const std::size_t ncols =
        chunks.empty() ? names.size() : chunks.front().size();
    out.columns.reserve(ncols);
    for (std::size_t c = 0; c < ncols; ++c) {
        std::vector<const Series*> parts;
        parts.reserve(chunks.size());
        for (auto& ch : chunks) parts.push_back(&ch[c]);
        out.columns.push_back(concat_columns(parts));
    }
    return out;
}

Morsel morsel_of(DataFrame&& f) {
    Morsel out;
    out.rows = f.num_rows();
    out.columns.reserve(f.columns.size());
    for (const Series& c : f.columns) out.columns.push_back(c.materialize());
    return out;
}

// Read a numeric cell as a double for key comparison (FLAT columns only).
double read_num(const Series& c, std::int64_t i) {
    switch (c.type()) {
        case TypeId::Bool:
            return (c.data<std::uint8_t>()[i >> 3] >> (i & 7)) & 1;
        case TypeId::Int8:
            return c.data<std::int8_t>()[i];
        case TypeId::Int16:
            return c.data<std::int16_t>()[i];
        case TypeId::Int32:
            return c.data<std::int32_t>()[i];
        case TypeId::Int64:
            return static_cast<double>(c.data<std::int64_t>()[i]);
        case TypeId::Uint8:
            return c.data<std::uint8_t>()[i];
        case TypeId::Uint16:
            return c.data<std::uint16_t>()[i];
        case TypeId::Uint32:
            return c.data<std::uint32_t>()[i];
        case TypeId::Uint64:
            return static_cast<double>(c.data<std::uint64_t>()[i]);
        case TypeId::Float32:
            return static_cast<double>(c.data<float>()[i]);
        case TypeId::Float64:
            return c.data<double>()[i];
        default:
            return 0.0;
    }
}

// Three-way compare of two key cells for a merge, honoring `descending`. Nulls
// always sort last (both directions), matching argsort.
int cmp_cell(const Series& a, std::int64_t ia, const Series& b, std::int64_t ib,
             bool descending) {
    const bool na = a.is_null(ia), nb = b.is_null(ib);
    if (na || nb) return na && nb ? 0 : (na ? 1 : -1);
    int c;
    if (a.type() == TypeId::String) {
        const std::string_view x = a.string_at(ia), y = b.string_at(ib);
        c = x < y ? -1 : (x > y ? 1 : 0);
    } else {
        const double x = read_num(a, ia), y = read_num(b, ib);
        c = x < y ? -1 : (x > y ? 1 : 0);
    }
    return descending ? -c : c;
}

// Approximate in-memory byte size of a set of FLAT columns (spill trigger).
std::size_t morsel_bytes(const std::vector<Series>& cols) {
    std::size_t total = 0;
    for (const Series& c : cols) {
        const std::int64_t n = c.length();
        if (byte_width(c.type()) == 0) {  // String / Binary
            const std::int32_t* offs = dftu_series_offsets(c.handle());
            total += static_cast<std::size_t>(n + 1) * sizeof(std::int32_t) +
                     (n > 0 ? static_cast<std::size_t>(offs[n]) : 0);
        } else {
            total += buffer_bytes(c.type(), n);
        }
    }
    return total;
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
        // Bounded parallel sink: pull a batch of morsels, accumulate each into
        // its own partial AggState in parallel (the mergeable agg IR), then
        // merge the partials into the running state. Memory stays bounded to
        // one batch; a serial-pull single morsel skips the fan-out.
        constexpr std::size_t BATCH = 32;
        std::vector<Morsel> batch;
        batch.reserve(BATCH);
        bool eof = false;
        while (!eof) {
            batch.clear();
            for (std::size_t b = 0; b < BATCH; ++b) {
                auto m = in_->next(max_rows);
                if (!m) {
                    eof = true;
                    break;
                }
                batch.push_back(std::move(*m));
            }
            if (batch.empty()) break;
            auto accumulate = [&](AggState& st, const Morsel& m) {
                std::vector<const Series*> values;
                values.reserve(value_idx.size());
                for (int vi : value_idx) values.push_back(&m.columns[vi]);
                agg_accumulate(st, m.columns[key_idx], values);
            };
            if (batch.size() == 1) {
                accumulate(*state, batch[0]);
                continue;
            }
            std::vector<AggStatePtr> partials(batch.size());
            parallel_for(
                static_cast<std::int64_t>(batch.size()), 1,
                [&](std::int64_t bi, std::int64_t ei) {
                    for (std::int64_t j = bi; j < ei; ++j) {
                        auto st = agg_new(specs);
                        accumulate(*st, batch[static_cast<std::size_t>(j)]);
                        partials[static_cast<std::size_t>(j)] = std::move(st);
                    }
                });
            for (auto& p : partials)
                if (p) agg_merge(*state, *p);
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

// Streaming tumbling/sliding time-window aggregation over an ascending Int64
// time column. Explodes each event into the windows it falls in and feeds one
// mergeable agg state, so state is bounded by the window count (the output) not
// the input. Requires ascending time: the grid is anchored on the first event
// (= the minimum), matching DataFrame::group_by_dynamic.
class GroupByDynamicCursor : public Cursor {
   public:
    GroupByDynamicCursor(std::unique_ptr<Cursor> in,
                         std::vector<std::string> sch, std::string time_col,
                         std::int64_t every, std::int64_t period,
                         std::vector<GroupAgg> aggs, std::int64_t origin,
                         bool origin_min)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          time_col_(std::move(time_col)),
          every_(every),
          period_(period),
          aggs_(std::move(aggs)),
          origin_(origin),
          origin_min_(origin_min) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        if (every_ <= 0)
            throw std::invalid_argument("group_by_dynamic: every must be > 0");
        const std::int64_t period = period_ <= 0 ? every_ : period_;

        const int ti = index_in(sch_, time_col_);
        if (ti < 0)
            throw std::out_of_range("group_by_dynamic: no column named " +
                                    time_col_);

        std::vector<AggSpec> specs;
        std::vector<int> value_idx;
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
        bool anchored = false;
        std::int64_t start0 = 0, origin = origin_;
        while (auto m = in_->next(max_rows)) {
            const Series& tc = m->columns[static_cast<std::size_t>(ti)];
            if (tc.type() != TypeId::Int64)
                throw std::invalid_argument("group_by_dynamic: " + time_col_ +
                                            " must be an Int64 column");
            const std::int64_t n = m->rows;
            const std::int64_t* t = tc.data<std::int64_t>();
            if (!anchored) {
                for (std::int64_t i = 0; i < n; ++i)
                    if (!tc.is_null(i)) {
                        if (origin_min_) origin = t[i];
                        start0 =
                            origin + floor_to_multiple(t[i] - origin, every_);
                        anchored = true;
                        break;
                    }
                if (!anchored) continue;
            }
            std::vector<std::int64_t> keyv, rowsv;
            for (std::int64_t i = 0; i < n; ++i) {
                if (tc.is_null(i)) continue;
                const std::int64_t ts = t[i];
                std::int64_t k = (ts - start0) / every_;
                for (; k >= 0; --k) {
                    const std::int64_t s = start0 + k * every_;
                    if (s <= ts - period) break;
                    keyv.push_back(s);
                    rowsv.push_back(i);
                }
            }
            if (keyv.empty()) continue;
            Series keyc = Series::flat_i64(
                keyv.data(), static_cast<std::int64_t>(keyv.size()));
            std::vector<Series> gathered;
            gathered.reserve(value_idx.size());
            for (int vi : value_idx)
                gathered.push_back(
                    m->columns[static_cast<std::size_t>(vi)].take(rowsv));
            std::vector<const Series*> values;
            values.reserve(gathered.size());
            for (const Series& g : gathered) values.push_back(&g);
            agg_accumulate(*state, keyc, values);
        }
        DataFrame r = agg_finalize(*state, time_col_).sort_by(time_col_, false);
        return morsel_of(std::move(r));
    }

   private:
    static int index_in(const std::vector<std::string>& s,
                        const std::string& n) {
        auto it = std::find(s.begin(), s.end(), n);
        return it == s.end() ? -1 : static_cast<int>(it - s.begin());
    }
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::string time_col_;
    std::int64_t every_, period_;
    std::vector<GroupAgg> aggs_;
    std::int64_t origin_;
    bool origin_min_;
    bool done_ = false;
};

// Streaming min-hash reservoir: keep the n rows with the smallest
// mix64(global_row_index + seed) keys, matching DataFrame::sample. Bounded to n
// rows (plus one morsel) regardless of input size; emits them in original row
// order.
class SampleCursor : public Cursor {
   public:
    SampleCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                 std::int64_t n, std::uint64_t seed)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          n_(std::max<std::int64_t>(n, 0)),
          seed_(seed) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        DataFrame best;                   // <= n_ rows
        std::vector<std::uint64_t> keys;  // parallel to best's rows
        std::vector<std::int64_t> idx;    // original global row indices
        std::int64_t off = 0;
        while (auto m = in_->next(max_rows)) {
            const std::int64_t mrows = m->rows;
            DataFrame mf;
            mf.names = sch_;
            mf.columns = std::move(m->columns);
            DataFrame combined =
                best.num_rows() == 0
                    ? std::move(mf)
                    : concat({&best, &mf}, ConcatHow::Vertical);
            keys.reserve(keys.size() + static_cast<std::size_t>(mrows));
            idx.reserve(idx.size() + static_cast<std::size_t>(mrows));
            for (std::int64_t i = 0; i < mrows; ++i) {
                keys.push_back(hash::splitmix64(
                    static_cast<std::uint64_t>(off + i) + seed_));
                idx.push_back(off + i);
            }
            off += mrows;
            const std::int64_t total = combined.num_rows();
            const std::int64_t keep = std::min(n_, total);
            std::vector<std::int64_t> sel(static_cast<std::size_t>(total));
            std::iota(sel.begin(), sel.end(), std::int64_t{0});
            if (keep < total)
                std::nth_element(sel.begin(), sel.begin() + keep, sel.end(),
                                 [&](std::int64_t a, std::int64_t b) {
                                     return keys[static_cast<std::size_t>(a)] <
                                            keys[static_cast<std::size_t>(b)];
                                 });
            sel.resize(static_cast<std::size_t>(keep));
            best = take(combined, sel);
            std::vector<std::uint64_t> nk(static_cast<std::size_t>(keep));
            std::vector<std::int64_t> ni(static_cast<std::size_t>(keep));
            for (std::int64_t j = 0; j < keep; ++j) {
                nk[static_cast<std::size_t>(j)] = keys[static_cast<std::size_t>(
                    sel[static_cast<std::size_t>(j)])];
                ni[static_cast<std::size_t>(j)] = idx[static_cast<std::size_t>(
                    sel[static_cast<std::size_t>(j)])];
            }
            keys = std::move(nk);
            idx = std::move(ni);
        }
        // DataFrame::sample returns survivors in original row order.
        std::vector<std::int64_t> ord(
            static_cast<std::size_t>(best.num_rows()));
        std::iota(ord.begin(), ord.end(), std::int64_t{0});
        std::sort(ord.begin(), ord.end(), [&](std::int64_t a, std::int64_t b) {
            return idx[static_cast<std::size_t>(a)] <
                   idx[static_cast<std::size_t>(b)];
        });
        return morsel_of(take(best, ord));
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::int64_t n_;
    std::uint64_t seed_;
    bool done_ = false;
};

// External merge sort. Generates sorted runs bounded by `budget` bytes (spilled
// to disk; budget 0 keeps one in-memory run), then k-way range-merges them into
// a sorted stream. Peak memory is O(budget + one output morsel) when spilling.
class SortMergeCursor : public Cursor {
   public:
    SortMergeCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                    std::string key, bool descending, std::uint64_t budget)
        : in_(std::move(in)),
          sch_(std::move(sch)),
          key_(std::move(key)),
          descending_(descending),
          budget_(budget) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (!built_) build(max_rows);

        std::vector<std::vector<Series>> pieces;
        std::int64_t out_rows = 0;
        while (out_rows < max_rows) {
            const int winner = pick(-1, false);
            if (winner < 0) break;
            const int bnd = pick(winner, true);
            Morsel& wm = *cur_[static_cast<std::size_t>(winner)];
            const Series& kw = wm.columns[static_cast<std::size_t>(key_idx_)];
            std::int64_t pos = pos_[static_cast<std::size_t>(winner)];
            const std::int64_t end =
                std::min(wm.rows, pos + (max_rows - out_rows));
            std::int64_t limit;
            if (bnd < 0) {
                limit = end;
            } else {
                const Morsel& bm = *cur_[static_cast<std::size_t>(bnd)];
                const Series& kb =
                    bm.columns[static_cast<std::size_t>(key_idx_)];
                const std::int64_t bpos = pos_[static_cast<std::size_t>(bnd)];
                limit = pos;
                while (limit < end &&
                       cmp_cell(kw, limit, kb, bpos, descending_) <= 0)
                    ++limit;
            }
            Morsel piece = slice_morsel(wm, pos, limit - pos);
            pieces.push_back(std::move(piece.columns));
            out_rows += limit - pos;
            pos_[static_cast<std::size_t>(winner)] = limit;
            if (limit >= wm.rows) advance(winner, max_rows);
        }
        if (pieces.empty()) return std::nullopt;
        Morsel out;
        out.rows = out_rows;
        const std::size_t ncols = pieces.front().size();
        out.columns.reserve(ncols);
        for (std::size_t c = 0; c < ncols; ++c) {
            std::vector<const Series*> parts;
            parts.reserve(pieces.size());
            for (auto& pc : pieces) parts.push_back(&pc[c]);
            out.columns.push_back(concat_columns(parts));
        }
        return out;
    }

   private:
    // Index of the active run whose current key is most "before" the rest;
    // `exclude` skips one run (for the boundary), returns -1 if none active.
    int pick(int exclude, bool /*is_boundary*/) const {
        int best = -1;
        for (std::size_t k = 0; k < cur_.size(); ++k) {
            if (static_cast<int>(k) == exclude) continue;
            if (!cur_[k] || pos_[k] >= cur_[k]->rows) continue;
            if (best < 0) {
                best = static_cast<int>(k);
                continue;
            }
            const Series& kk =
                cur_[k]->columns[static_cast<std::size_t>(key_idx_)];
            const Series& kb =
                cur_[static_cast<std::size_t>(best)]
                    ->columns[static_cast<std::size_t>(key_idx_)];
            if (cmp_cell(kk, pos_[k], kb, pos_[static_cast<std::size_t>(best)],
                         descending_) < 0)
                best = static_cast<int>(k);
        }
        return best;
    }

    void advance(int k, std::int64_t max_rows) {
        cur_[static_cast<std::size_t>(k)] =
            runs_[static_cast<std::size_t>(k)]->next(max_rows);
        pos_[static_cast<std::size_t>(k)] = 0;
    }

    DataFrame concat_pending(std::vector<std::vector<Series>>& pending) const {
        DataFrame buf;
        buf.names = sch_;
        buf.columns.reserve(sch_.size());
        for (std::size_t c = 0; c < sch_.size(); ++c) {
            std::vector<const Series*> parts;
            parts.reserve(pending.size());
            for (auto& ch : pending) parts.push_back(&ch[c]);
            buf.columns.push_back(concat_columns(parts));
        }
        return buf;
    }

    void spill_run(std::vector<std::vector<Series>>& pending, int id,
                   std::int64_t chunk) {
        DataFrame sorted = concat_pending(pending).sort_by(key_, descending_);
        spill::Writer w(dir_.run_path(id));
        const std::int64_t total = sorted.num_rows();
        for (std::int64_t off = 0; off < total; off += chunk) {
            const std::int64_t len = std::min(chunk, total - off);
            DataFrame s = sorted.slice(off, len);
            std::vector<Series> cols;
            cols.reserve(s.columns.size());
            for (const Series& c : s.columns) cols.push_back(c.materialize());
            w.write(cols, len);
        }
        w.close();
    }

    void build(std::int64_t max_rows) {
        key_idx_ = static_cast<int>(std::distance(
            sch_.begin(), std::find(sch_.begin(), sch_.end(), key_)));
        if (key_idx_ >= static_cast<int>(sch_.size()))
            throw std::out_of_range("sort_by: no column named " + key_);

        std::vector<std::vector<Series>> pending;
        std::size_t pend_bytes = 0;
        int run_id = 0;
        while (auto m = in_->next(max_rows)) {
            pend_bytes += morsel_bytes(m->columns);
            pending.push_back(std::move(m->columns));
            if (budget_ > 0 && pend_bytes > budget_) {
                spill_run(pending, run_id++, max_rows);
                pending.clear();
                pend_bytes = 0;
            }
        }

        if (run_id == 0) {  // everything fits in memory: one sorted run
            DataFrame buf =
                pending.empty() ? DataFrame{} : concat_pending(pending);
            if (pending.empty()) buf.names = sch_;
            DataFrame sorted = buf.num_rows() ? buf.sort_by(key_, descending_)
                                              : std::move(buf);
            runs_.push_back(std::make_unique<InMemoryCursor>(
                std::make_shared<const DataFrame>(std::move(sorted))));
        } else {
            if (!pending.empty()) spill_run(pending, run_id++, max_rows);
            for (int id = 0; id < run_id; ++id)
                runs_.push_back(
                    std::make_unique<spill::Reader>(dir_.run_path(id)));
        }
        cur_.resize(runs_.size());
        pos_.assign(runs_.size(), 0);
        for (std::size_t k = 0; k < runs_.size(); ++k)
            cur_[k] = runs_[k]->next(max_rows);
        built_ = true;
    }

    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::string key_;
    bool descending_;
    std::uint64_t budget_;
    bool built_ = false;
    int key_idx_ = 0;
    spill::Dir dir_;
    std::vector<std::unique_ptr<Cursor>> runs_;
    std::vector<std::optional<Morsel>> cur_;
    std::vector<std::int64_t> pos_;
};

// Streaming distinct (keep first occurrence, original order). Holds only the
// set of distinct row keys - which is the result itself, materialized by
// collect anyway - and streams input and output morsel by morsel.
class UniqueCursor : public Cursor {
   public:
    UniqueCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch)
        : in_(std::move(in)), sch_(std::move(sch)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        while (auto m = in_->next(max_rows)) {
            const std::int64_t n = m->rows;
            // Build the exact row keys in parallel (scalar string work, one per
            // row, independent), then dedupe serially against the running set
            // to keep first-occurrence order.
            std::vector<std::string> keys(static_cast<std::size_t>(n));
            parallel_for(n, std::int64_t{1} << 13,
                         [&](std::int64_t b, std::int64_t e) {
                             for (std::int64_t i = b; i < e; ++i)
                                 keys[static_cast<std::size_t>(i)] =
                                     row_key(m->columns, i);
                         });
            std::vector<std::int64_t> keep;
            for (std::int64_t i = 0; i < n; ++i)
                if (seen_.insert(std::move(keys[static_cast<std::size_t>(i)]))
                        .second)
                    keep.push_back(i);
            if (keep.empty()) continue;
            DataFrame mf;
            mf.names = sch_;
            mf.columns = std::move(m->columns);
            return morsel_of(take(mf, keep));
        }
        return std::nullopt;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    ankerl::unordered_dense::set<std::string> seen_;
};

// Streaming per-column summary statistics, matching DataFrame::describe. One
// pass with O(numeric columns) state: count/null_count and running min/max plus
// Welford (mean, M2) for mean/sample-std. Output columns are data-dependent
// (one per numeric input column), reported via out_names().
class DescribeCursor : public Cursor {
   public:
    DescribeCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch)
        : in_(std::move(in)), sch_(std::move(sch)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        // Per-morsel SIMD reduction into one mergeable FieldStat per numeric
        // column (the engine's shared aggregation atom), so the numeric work is
        // vectorized and bounded by the column count.
        std::vector<int> num_idx;
        std::vector<FieldStat> acc;
        std::int64_t total_rows = 0;
        bool first = true;
        while (auto m = in_->next(max_rows)) {
            if (first) {
                first = false;
                for (std::size_t c = 0; c < m->columns.size(); ++c)
                    if (is_numeric(m->columns[c].type()))
                        num_idx.push_back(static_cast<int>(c));
                acc.resize(num_idx.size());
            }
            total_rows += m->rows;
            for (std::size_t j = 0; j < num_idx.size(); ++j)
                acc[j].merge(field_stat_reduce(
                    m->columns[static_cast<std::size_t>(num_idx[j])]));
        }
        DataFrame out;
        out.names.push_back("statistic");
        out.columns.push_back(Series::strings(
            {"count", "null_count", "mean", "std", "min", "max"}));
        for (std::size_t j = 0; j < num_idx.size(); ++j) {
            const FieldStat& fs = acc[j];
            const double vals[6] = {
                static_cast<double>(fs.n),
                static_cast<double>(total_rows -
                                    static_cast<std::int64_t>(fs.n)),
                fs.mean(),
                fs.stddev(),
                fs.n ? fs.min : 0.0,
                fs.n ? fs.max : 0.0};
            out.columns.push_back(Series::flat_f64(vals, 6));
            out.names.push_back(sch_[static_cast<std::size_t>(num_idx[j])]);
        }
        produced_ = out.names;
        return morsel_of(std::move(out));
    }

    std::optional<std::vector<std::string>> out_names() const override {
        return produced_;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    bool done_ = false;
    std::vector<std::string> produced_;
};

// Two-pass per-row mask: pass 1 counts each row key while spooling the input
// (RAM up to the budget, overflow to disk); pass 2 replays the spool in order,
// emitting count>1 (is_duplicated) or count==1 (is_unique) as one Bool column
// per morsel. Reads the upstream once; order-preserving; state is the count map
// plus the spool.
class IsDupCursor : public Cursor {
   public:
    IsDupCursor(std::unique_ptr<Cursor> in, std::uint64_t budget, bool unique)
        : first_(std::move(in)), spool_(budget), unique_(unique) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (!counted_) {
            while (auto m = first_->next(max_rows)) {
                for (std::int64_t i = 0; i < m->rows; ++i)
                    ++counts_[row_key(m->columns, i)];
                spool_.add(std::move(m->columns), m->rows);
            }
            first_.reset();
            pass2_ = spool_.reader();
            counted_ = true;
        }
        while (auto m = pass2_->next(max_rows)) {
            const std::int64_t n = m->rows;
            std::vector<std::uint8_t> bits(
                static_cast<std::size_t>((n + 7) / 8), 0);
            for (std::int64_t i = 0; i < n; ++i) {
                auto it = counts_.find(row_key(m->columns, i));
                const std::int64_t c = it != counts_.end() ? it->second : 0;
                if (unique_ ? c == 1 : c > 1)
                    bits[static_cast<std::size_t>(i >> 3)] |=
                        static_cast<std::uint8_t>(1u << (i & 7));
            }
            Morsel out;
            out.rows = n;
            out.columns.push_back(Series::flat(TypeId::Bool, bits.data(), n));
            return out;
        }
        return std::nullopt;
    }

   private:
    std::unique_ptr<Cursor> first_, pass2_;
    spill::Spool spool_;
    bool unique_;
    bool counted_ = false;
    ankerl::unordered_dense::map<std::string, std::int64_t> counts_;
};

// Two-pass one-hot encode. Pass 1 collects the distinct non-null values of
// `column` (bounded by cardinality); pass 2 re-scans, replacing that column in
// place with one Int8 column per distinct value (ascending, named
// "<column>_<value>"), matching DataFrame::to_dummies. Output columns are
// data-dependent, reported via out_names().
class ToDummiesCursor : public Cursor {
   public:
    ToDummiesCursor(std::unique_ptr<Cursor> in, std::uint64_t budget,
                    std::vector<std::string> sch, std::string column)
        : first_(std::move(in)),
          spool_(budget),
          sch_(std::move(sch)),
          column_(std::move(column)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (!built_) build(max_rows);
        auto m = pass2_->next(max_rows);
        if (!m) return std::nullopt;
        const Series& col = m->columns[static_cast<std::size_t>(ci_)];
        const std::int64_t n = m->rows;
        std::vector<Series> one;
        one.push_back(col.share());
        std::vector<std::vector<std::int8_t>> dummies(
            uniq_idx_.size(),
            std::vector<std::int8_t>(static_cast<std::size_t>(n), 0));
        for (std::int64_t i = 0; i < n; ++i) {
            if (col.is_null(i)) continue;
            auto it = uniq_idx_.find(row_key(one, i));
            if (it != uniq_idx_.end())
                dummies[static_cast<std::size_t>(it->second)]
                       [static_cast<std::size_t>(i)] = 1;
        }
        Morsel out;
        out.rows = n;
        for (std::size_t k = 0; k < m->columns.size(); ++k) {
            if (static_cast<int>(k) != ci_) {
                out.columns.push_back(m->columns[k].share());
                continue;
            }
            for (auto& col_bits : dummies)
                out.columns.push_back(
                    Series::flat(TypeId::Int8, col_bits.data(), n));
        }
        return out;
    }

    std::optional<std::vector<std::string>> out_names() const override {
        return produced_;
    }

   private:
    void build(std::int64_t max_rows) {
        ci_ = static_cast<int>(std::distance(
            sch_.begin(), std::find(sch_.begin(), sch_.end(), column_)));
        if (ci_ >= static_cast<int>(sch_.size()))
            throw std::out_of_range("to_dummies: no column named " + column_);

        ankerl::unordered_dense::set<std::string> seen;
        std::vector<Series> chunks;
        while (auto m = first_->next(max_rows)) {
            const Series& col = m->columns[static_cast<std::size_t>(ci_)];
            std::vector<Series> one;
            one.push_back(col.share());
            std::vector<std::int64_t> keep;
            for (std::int64_t i = 0; i < m->rows; ++i)
                if (!col.is_null(i) && seen.insert(row_key(one, i)).second)
                    keep.push_back(i);
            if (!keep.empty()) chunks.push_back(col.take(keep));
            spool_.add(std::move(m->columns), m->rows);
        }
        first_.reset();
        // Distinct values, ascending (Series::unique sorts), matching eager.
        Series uniq;
        if (!chunks.empty())
            uniq = concat_columns(column_ptrs(chunks)).unique().materialize();
        const std::int64_t d = uniq.length();
        std::vector<Series> one;
        one.push_back(uniq.share());
        for (std::int64_t u = 0; u < d; ++u)
            uniq_idx_.emplace(row_key(one, u), static_cast<int>(u));
        for (std::size_t k = 0; k < sch_.size(); ++k) {
            if (static_cast<int>(k) != ci_) {
                produced_.push_back(sch_[k]);
                continue;
            }
            for (std::int64_t u = 0; u < d; ++u)
                produced_.push_back(column_ + "_" + cell_to_string(uniq, u));
        }
        pass2_ = spool_.reader();
        built_ = true;
    }

    std::unique_ptr<Cursor> first_, pass2_;
    spill::Spool spool_;
    std::vector<std::string> sch_;
    std::string column_;
    bool built_ = false;
    int ci_ = 0;
    ankerl::unordered_dense::map<std::string, int> uniq_idx_;
    std::vector<std::string> produced_;
};

// Two-pass long->wide pivot, matching DataFrame::pivot. Pass 1 discovers the
// distinct index rows and `on` columns (both ascending via Series::unique);
// pass 2 aggregates each (index, on) cell through the mergeable agg IR keyed by
// row*C+col (first/last/sum/min/max/mean all map to an AggOp), so state is
// bounded by the output (R*C) not the input. Output columns are data-dependent,
// reported via out_names().
class PivotCursor : public Cursor {
   public:
    PivotCursor(std::unique_ptr<Cursor> in, std::uint64_t budget,
                std::vector<std::string> sch, std::string index, std::string on,
                std::string values, std::string agg)
        : first_(std::move(in)),
          spool_(budget),
          sch_(std::move(sch)),
          index_(std::move(index)),
          on_(std::move(on)),
          values_(std::move(values)),
          agg_(std::move(agg)) {}

    std::optional<Morsel> next(std::int64_t max_rows) override {
        if (done_) return std::nullopt;
        done_ = true;
        const int ii = idx_of(index_), ci = idx_of(on_), vi = idx_of(values_);
        if (ii < 0) throw std::out_of_range("pivot: no column named " + index_);
        if (ci < 0) throw std::out_of_range("pivot: no column named " + on_);
        if (vi < 0)
            throw std::out_of_range("pivot: no column named " + values_);

        // Pass 1: distinct index and `on` values (ascending, matching eager).
        ankerl::unordered_dense::set<std::string> seen_i, seen_c;
        std::vector<Series> ich, cch;
        while (auto m = first_->next(max_rows)) {
            distinct_into(m->columns[static_cast<std::size_t>(ii)], seen_i,
                          ich);
            distinct_into(m->columns[static_cast<std::size_t>(ci)], seen_c,
                          cch);
            spool_.add(std::move(m->columns), m->rows);
        }
        first_.reset();
        Series uniq_idx =
            ich.empty()
                ? Series{}
                : concat_columns(column_ptrs(ich)).unique().materialize();
        Series uniq_col =
            cch.empty()
                ? Series{}
                : concat_columns(column_ptrs(cch)).unique().materialize();
        const std::int64_t R = uniq_idx.length(), C = uniq_col.length();
        ankerl::unordered_dense::map<std::string, std::int64_t> row_of, col_of;
        key_index(uniq_idx, R, row_of);
        key_index(uniq_col, C, col_of);
        const std::int64_t sentinel = R * C;

        // Pass 2: aggregate each cell through the agg IR keyed by row*C+col.
        std::vector<AggSpec> specs;
        AggSpec sp;
        sp.op = to_agg_op(agg_from_string(agg_));
        sp.value_col = 0;
        sp.out = "v";
        specs.push_back(std::move(sp));
        AggStatePtr state = agg_new(std::move(specs));
        auto p2 = spool_.reader();
        while (auto m = p2->next(max_rows)) {
            const Series& ic = m->columns[static_cast<std::size_t>(ii)];
            const Series& cc = m->columns[static_cast<std::size_t>(ci)];
            const Series& vc = m->columns[static_cast<std::size_t>(vi)];
            const std::int64_t n = m->rows;
            std::vector<Series> oi, oc;
            oi.push_back(ic.share());
            oc.push_back(cc.share());
            std::vector<std::int64_t> keyv(static_cast<std::size_t>(n),
                                           sentinel);
            for (std::int64_t i = 0; i < n; ++i) {
                if (ic.is_null(i) || cc.is_null(i)) continue;
                auto ri = row_of.find(row_key(oi, i));
                auto rc = col_of.find(row_key(oc, i));
                if (ri != row_of.end() && rc != col_of.end())
                    keyv[static_cast<std::size_t>(i)] =
                        ri->second * C + rc->second;
            }
            Series keyc = Series::flat_i64(keyv.data(), n);
            std::vector<const Series*> values{&vc};
            agg_accumulate(*state, keyc, values);
        }
        DataFrame agg_res = agg_finalize(*state, "cell");
        Series source = agg_res.column("v");
        Series cells_col = agg_res.column("cell");
        const std::int64_t* cells = cells_col.data<std::int64_t>();
        std::vector<std::int64_t> cell_src(static_cast<std::size_t>(R * C), -1);
        for (std::int64_t p = 0; p < cells_col.length(); ++p) {
            const std::int64_t cell = cells[p];
            if (cell != sentinel) cell_src[static_cast<std::size_t>(cell)] = p;
        }

        DataFrame out;
        out.names.push_back(index_);
        out.columns.push_back(uniq_idx.share());
        for (std::int64_t c = 0; c < C; ++c) {
            std::vector<std::int64_t> ti(static_cast<std::size_t>(R));
            for (std::int64_t r = 0; r < R; ++r)
                ti[static_cast<std::size_t>(r)] =
                    cell_src[static_cast<std::size_t>(r * C + c)];
            out.names.push_back(cell_to_string(uniq_col, c));
            out.columns.push_back(source.take(ti));
        }
        produced_ = out.names;
        return morsel_of(std::move(out));
    }

    std::optional<std::vector<std::string>> out_names() const override {
        return produced_;
    }

   private:
    int idx_of(const std::string& name) const {
        auto it = std::find(sch_.begin(), sch_.end(), name);
        return it == sch_.end() ? -1 : static_cast<int>(it - sch_.begin());
    }
    // Append the first-occurrence non-null cells of `col` to `chunks`.
    static void distinct_into(const Series& col,
                              ankerl::unordered_dense::set<std::string>& seen,
                              std::vector<Series>& chunks) {
        std::vector<Series> one;
        one.push_back(col.share());
        std::vector<std::int64_t> keep;
        for (std::int64_t i = 0; i < col.length(); ++i)
            if (!col.is_null(i) && seen.insert(row_key(one, i)).second)
                keep.push_back(i);
        if (!keep.empty()) chunks.push_back(col.take(keep));
    }
    static void key_index(
        const Series& uniq, std::int64_t n,
        ankerl::unordered_dense::map<std::string, std::int64_t>& out) {
        std::vector<Series> one;
        one.push_back(uniq.share());
        for (std::int64_t i = 0; i < n; ++i) out.emplace(row_key(one, i), i);
    }

    std::unique_ptr<Cursor> first_;
    spill::Spool spool_;
    std::vector<std::string> sch_;
    std::string index_, on_, values_, agg_;
    bool done_ = false;
    std::vector<std::string> produced_;
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
struct SortByOp {
    std::string name;
    bool descending;
};
struct UniqueOp {};
struct SampleOp {
    std::int64_t n;
    std::uint64_t seed;
};
struct IsDupOp {
    bool unique;  // true = is_unique, false = is_duplicated
};
struct GroupByDynamicOp {
    std::string time_col;
    std::int64_t every;
    std::int64_t period;
    std::vector<GroupAgg> aggs;
    std::int64_t origin;
    bool origin_min;
};
struct PivotOp {
    std::string index, on, values, agg;
};
struct ToDummiesOp {
    std::string column;
};
struct DescribeOp {};

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
                 ExplodeOp, UnpivotOp, TopkOp, GroupByOp, SortByOp, UniqueOp,
                 SampleOp, IsDupOp, GroupByDynamicOp, PivotOp, ToDummiesOp,
                 DescribeOp>
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
        overloaded{
            [&](const FilterOp&) { return in; },
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
            },
            [&](const SortByOp&) { return in; },
            [&](const UniqueOp&) { return in; },
            [&](const SampleOp&) { return in; },
            [&](const IsDupOp& o) {
                return std::vector<std::string>{o.unique ? "is_unique"
                                                         : "is_duplicated"};
            },
            [&](const GroupByDynamicOp& o) {
                std::vector<std::string> s{o.time_col};
                for (const GroupAgg& a : o.aggs) s.push_back(a.out);
                return s;
            },
            // Data-dependent schema: known only after running; collect()
            // relabels from the cursor's out_names().
            [&](const PivotOp&) { return std::vector<std::string>{}; },
            [&](const ToDummiesOp&) { return std::vector<std::string>{}; },
            [&](const DescribeOp&) { return std::vector<std::string>{}; }},
        op.node);
}

std::string describe_op(const LazyOp& op) {
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
            [](const GroupByOp& o) { return "group_by " + o.key; },
            [](const SortByOp& o) { return "sort_by " + o.name; },
            [](const UniqueOp&) { return std::string("unique"); },
            [](const SampleOp&) { return std::string("sample"); },
            [](const IsDupOp& o) {
                return std::string(o.unique ? "is_unique" : "is_duplicated");
            },
            [](const GroupByDynamicOp& o) {
                return "group_by_dynamic " + o.time_col;
            },
            [](const PivotOp& o) { return "pivot on " + o.on; },
            [](const ToDummiesOp& o) { return "to_dummies " + o.column; },
            [](const DescribeOp&) { return std::string("describe"); }},
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
                                    const std::vector<std::string>& sch,
                                    std::uint64_t budget) {
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
            },
            [&](const SortByOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<SortMergeCursor>(
                    std::move(in), sch, o.name, o.descending, budget);
            },
            [&](const UniqueOp&) -> std::unique_ptr<Cursor> {
                return std::make_unique<UniqueCursor>(std::move(in), sch);
            },
            [&](const SampleOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<SampleCursor>(std::move(in), sch, o.n,
                                                      o.seed);
            },
            [&](const IsDupOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<IsDupCursor>(std::move(in), budget,
                                                     o.unique);
            },
            [&](const GroupByDynamicOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<GroupByDynamicCursor>(
                    std::move(in), sch, o.time_col, o.every, o.period, o.aggs,
                    o.origin, o.origin_min);
            },
            [&](const PivotOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<PivotCursor>(
                    std::move(in), budget, sch, o.index, o.on, o.values, o.agg);
            },
            [&](const ToDummiesOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<ToDummiesCursor>(std::move(in), budget,
                                                         sch, o.column);
            },
            [&](const DescribeOp&) -> std::unique_ptr<Cursor> {
                return std::make_unique<DescribeCursor>(std::move(in), sch);
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
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::filter(Expr predicate) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{FilterOp{std::move(predicate)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::with_column(std::string name, Expr expr) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{WithColumnOp{std::move(name), std::move(expr)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::rename(std::vector<std::string> names) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{RenameOp{std::move(names)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::slice(std::int64_t offset, std::int64_t len) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{SliceOp{offset, len}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::head(std::int64_t n) const { return slice(0, n); }

LazyFrame LazyFrame::tail(std::int64_t n) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{TailOp{n}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::drop_nulls() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{DropNullsOp{}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::fill_null(dftu_scalar value) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{FillNullOp{value}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::with_row_index(std::string name) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{WithRowIndexOp{std::move(name)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::null_count() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{NullCountOp{}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::explode(std::string column) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{ExplodeOp{std::move(column)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::unpivot(std::vector<std::string> id_vars,
                             std::vector<std::string> value_vars) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{UnpivotOp{std::move(id_vars), std::move(value_vars)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::topk(std::string name, std::int64_t k,
                          bool largest) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{TopkOp{std::move(name), k, largest}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::group_by(std::string key,
                              std::vector<GroupAgg> aggs) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{GroupByOp{std::move(key), std::move(aggs)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::sort_by(std::string name, bool descending) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{SortByOp{std::move(name), descending}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::unique() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{UniqueOp{}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::drop_duplicates() const { return unique(); }

LazyFrame LazyFrame::sample(std::int64_t n, std::uint64_t seed) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{SampleOp{n, seed}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::is_duplicated() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{IsDupOp{false}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::is_unique() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{IsDupOp{true}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::group_by_dynamic(std::string time_col, std::int64_t every,
                                      std::int64_t period,
                                      std::vector<GroupAgg> aggs,
                                      std::int64_t origin,
                                      bool origin_min) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{GroupByDynamicOp{std::move(time_col), every, period,
                                std::move(aggs), origin, origin_min}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::melt(std::vector<std::string> id_vars,
                          std::vector<std::string> value_vars) const {
    return unpivot(std::move(id_vars), std::move(value_vars));
}

LazyFrame LazyFrame::pivot(std::string index, std::string on,
                           std::string values, std::string agg) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{PivotOp{
        std::move(index), std::move(on), std::move(values), std::move(agg)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::to_dummies(std::string column) const {
    auto ops = ops_;
    ops.push_back(
        std::make_shared<LazyOp>(LazyOp{ToDummiesOp{std::move(column)}}));
    return with_ops(std::move(ops));
}

LazyFrame LazyFrame::describe() const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(LazyOp{DescribeOp{}}));
    return with_ops(std::move(ops));
}

std::vector<std::string> LazyFrame::schema() const {
    std::vector<std::string> s = source_->names();
    for (const auto& op : ops_) s = out_schema(*op, std::move(s));
    return s;
}

std::string LazyFrame::explain() const {
    std::string s = "scan [" + join_names(source_->names()) + "]";
    for (const auto& op : pushdown_predicates(source_->names(), ops_))
        s += "\n" + describe_op(*op);
    return s;
}

LazyFrame LazyFrame::memory_budget(std::uint64_t bytes) const {
    return LazyFrame(source_, ops_, bytes);
}

LazyFrame LazyFrame::auto_spill() const {
    // Same policy as the default (0) and as View: ~1/3 of available memory.
    return LazyFrame(source_, ops_, resolve_spill_budget(0));
}

DataFrame LazyFrame::collect(std::int64_t morsel_rows) const {
    auto ops = pushdown_predicates(source_->names(), ops_);
    // 0 resolves to auto (~1/3 RAM), same policy as View.
    const std::uint64_t budget = resolve_spill_budget(memory_budget_);
    std::unique_ptr<Cursor> cur = source_->open();
    std::vector<std::string> sch = source_->names();
    for (const auto& op : ops) {
        cur = make_cursor(*op, std::move(cur), sch, budget);
        sch = out_schema(*op, std::move(sch));
    }
    DataFrame out = drain_to_frame(*cur, sch, morsel_rows);
    // A data-dependent terminal (pivot/to_dummies/describe) knows its true
    // schema only after running; prefer it over the static plan schema.
    if (auto n = cur->out_names()) out.names = std::move(*n);
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
