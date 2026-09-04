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
#include <fstream>
#include <memory>
#include <numeric>
#include <stdexcept>
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

// Default scan chunk when the caller does not set one (morsel_rows <= 0).
constexpr std::int64_t DEFAULT_MORSEL_ROWS = 65536;

// Build one standalone DataFrame from a morsel. Names come from the morsel's
// own schema (streaming, self-describing), else the cursor's data-dependent
// out_names(), else the static plan schema.
DataFrame frame_from_morsel(
    Morsel&& m, const std::vector<std::string>& static_names,
    const std::optional<std::vector<std::string>>& out_names) {
    DataFrame out;
    out.columns = std::move(m.columns);
    if (!m.name_ids.empty()) {
        out.names.reserve(m.name_ids.size());
        for (std::uint32_t id : m.name_ids)
            out.names.emplace_back(m.intern->resolve(id));
    } else if (out_names) {
        out.names = *out_names;
    } else {
        out.names = static_names;
    }
    // Fold the out-of-band dyn set back in as trailing named columns, so a
    // DataFrame-terminal consumer sees the per-morsel dyn columns by name.
    for (std::size_t i = 0; i < m.dyn_columns.size(); ++i) {
        out.names.push_back(std::move(m.dyn_names[i]));
        out.columns.push_back(std::move(m.dyn_columns[i]));
    }
    return out;
}

// Drain a chunk generator to a single DataFrame.
coro::CoroTask<DataFrame> drain_stream(coro::AsyncGenerator<DataFrame> gen) {
    std::vector<DataFrame> parts;
    while (auto df = co_await gen.next()) parts.push_back(std::move(*df));
    if (parts.empty()) co_return DataFrame{};
    // A single chunk needs no merge - concat cannot rejoin a nested
    // (List/Struct) column, which a single already-complete chunk (e.g. a
    // resident source's one morsel) may carry.
    if (parts.size() == 1) co_return std::move(parts[0]);

    bool uniform = true;
    for (std::size_t i = 1; i < parts.size() && uniform; ++i) {
        if (parts[i].names != parts[0].names ||
            parts[i].columns.size() != parts[0].columns.size()) {
            uniform = false;
            break;
        }
        for (std::size_t c = 0; c < parts[0].columns.size(); ++c) {
            if (parts[i].columns[c].type() != parts[0].columns[c].type()) {
                uniform = false;
                break;
            }
        }
    }
    std::vector<const DataFrame*> ptrs;
    ptrs.reserve(parts.size());
    for (const DataFrame& p : parts) ptrs.push_back(&p);
    co_return concat(ptrs, uniform ? ConcatHow::Vertical : ConcatHow::Diagonal);
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

// Fan-out width for the partitioned first-seen dedup below.
constexpr std::size_t DEDUP_PARTITIONS = 32;

// Radix-partition `keys` by hash into `seen_p.size()` buckets, each with its
// own running hash set, and mark keep[i] the first time each key is seen
// (order-preserving): buckets never overlap, so each is probed by exactly one
// worker with no lock. `seen_p` carries state across calls, bounded by the
// distinct-key count rather than the row count. An empty `seen_p` means no
// parallel backend is installed; the single `seen_serial` set is used instead
// so the partition bookkeeping is never paid for nothing.
std::vector<std::uint8_t> first_seen_mask(
    std::vector<std::string>& keys, std::int64_t n,
    ankerl::unordered_dense::set<std::string>& seen_serial,
    std::vector<ankerl::unordered_dense::set<std::string>>& seen_p) {
    std::vector<std::uint8_t> keep_mask(static_cast<std::size_t>(n), 0);
    if (seen_p.empty()) {
        for (std::int64_t i = 0; i < n; ++i)
            if (seen_serial.insert(std::move(keys[static_cast<std::size_t>(i)]))
                    .second)
                keep_mask[static_cast<std::size_t>(i)] = 1;
        return keep_mask;
    }
    const std::size_t p = seen_p.size();
    std::vector<std::vector<std::int64_t>> buckets(p);
    for (std::int64_t i = 0; i < n; ++i)
        buckets[std::hash<std::string>{}(keys[static_cast<std::size_t>(i)]) % p]
            .push_back(i);
    parallel_for(
        static_cast<std::int64_t>(p), 1, [&](std::int64_t pb, std::int64_t pe) {
            for (std::int64_t g = pb; g < pe; ++g)
                for (std::int64_t i : buckets[static_cast<std::size_t>(g)])
                    if (seen_p[static_cast<std::size_t>(g)]
                            .insert(
                                std::move(keys[static_cast<std::size_t>(i)]))
                            .second)
                        keep_mask[static_cast<std::size_t>(i)] = 1;
        });
    return keep_mask;
}

// Approximate per-entry overhead of one string in an unordered_dense::set
// (node + bucket bookkeeping), added to the key's own byte length when sizing
// the distinct-key state for the unique() spill trigger below.
constexpr std::size_t DEDUP_ENTRY_OVERHEAD = 48;

// Grace-hash-distinct spill fan-out and recursion bound (unique() below): a
// partition that still exceeds budget after fanning out is re-partitioned
// with a depth-salted hash, capped so a single hot key (which always lands in
// the same partition, however deep) cannot recurse forever.
constexpr int UNIQUE_SPILL_FANOUT = 16;
constexpr int UNIQUE_SPILL_MAX_DEPTH = 3;
constexpr std::int64_t UNIQUE_SPILL_MIN_LEAF_ROWS = 64;

// Depth-salted hash of a row key: depth 0 matches the plain hash so the first
// pass agrees with any caller hashing the same key; deeper passes mix in the
// depth so a key that collided at one depth spreads differently at the next.
std::size_t unique_spill_hash(const std::string& key, int depth) {
    const std::size_t h = std::hash<std::string>{}(key);
    if (depth == 0) return h;
    return static_cast<std::size_t>(hash::splitmix64(
        static_cast<std::uint64_t>(h) ^
        (static_cast<std::uint64_t>(depth) * 0x9E3779B97F4A7C15ULL)));
}

// All columns but the first (the row-id helper column prepended by the
// unique() spill path), as cheap shared views.
std::vector<Series> drop_first_column(const std::vector<Series>& cols) {
    std::vector<Series> out;
    out.reserve(cols.size() - 1);
    for (std::size_t i = 1; i < cols.size(); ++i)
        out.push_back(cols[i].share());
    return out;
}

// ---- cursors ----------------------------------------------------------------

// Reads contiguous chunks off an in-memory frame as zero-copy offset views.
class InMemoryCursor : public Cursor {
   public:
    explicit InMemoryCursor(std::shared_ptr<const DataFrame> frame)
        : frame_(std::move(frame)), n_(frame_->num_rows()) {}

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (off_ >= n_) co_return std::nullopt;
        const std::int64_t len =
            std::min(std::max<std::int64_t>(max_rows, 1), n_ - off_);
        DataFrame chunk = frame_->slice(off_, len);
        off_ += len;
        Morsel m;
        m.rows = len;
        m.columns.reserve(chunk.columns.size());
        for (Series& c : chunk.columns) m.columns.push_back(std::move(c));
        co_return m;
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        while (auto m = co_await in_->next(max_rows)) {
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
            if (out.rows > 0) co_return out;
        }
        co_return std::nullopt;
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        auto m = co_await in_->next(max_rows);
        if (!m) co_return std::nullopt;
        Morsel out;
        out.rows = m->rows;
        out.columns.reserve(idx_.size());
        for (int i : idx_) out.columns.push_back(m->columns[i].share());
        co_return out;
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        auto m = co_await in_->next(max_rows);
        if (!m) co_return std::nullopt;
        Series nc = eval(expr_, column_ptrs(m->columns));
        Morsel out;
        out.rows = m->rows;
        out.columns = std::move(m->columns);
        out.dyn_names = std::move(m->dyn_names);
        out.dyn_columns = std::move(m->dyn_columns);
        if (replace_ >= 0)
            out.columns[static_cast<std::size_t>(replace_)] = std::move(nc);
        else
            out.columns.push_back(std::move(nc));
        co_return out;
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        while (emitted_ < len_) {
            auto m = co_await in_->next(max_rows);
            if (!m) co_return std::nullopt;
            const std::int64_t start = seen_;
            seen_ += m->rows;
            const std::int64_t w_start = std::max(offset_, start);
            const std::int64_t w_end = std::min(offset_ + len_, seen_);
            if (w_end <= w_start) continue;
            emitted_ += w_end - w_start;
            co_return slice_morsel(*m, w_start - start, w_end - w_start);
        }
        co_return std::nullopt;
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
        done_ = true;
        if (n_ <= 0) co_return std::nullopt;
        std::deque<Morsel> buf;
        std::int64_t total = 0;
        while (auto m = co_await in_->next(max_rows)) {
            total += m->rows;
            buf.push_back(std::move(*m));
            while (!buf.empty() && total - buf.front().rows >= n_) {
                total -= buf.front().rows;
                buf.pop_front();
            }
        }
        if (buf.empty()) co_return std::nullopt;
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
        if (total > n_) co_return slice_morsel(out, total - n_, n_);
        co_return out;
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
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        while (auto m = co_await in_->next(max_rows)) {
            Morsel out = map_frame(std::move(*m),
                                   [](DataFrame f) { return f.drop_nulls(); });
            if (out.rows > 0) co_return out;
        }
        co_return std::nullopt;
    }

   private:
    std::unique_ptr<Cursor> in_;
};

// Fills nulls per morsel.
class FillNullCursor : public Cursor {
   public:
    FillNullCursor(std::unique_ptr<Cursor> in, dftu_scalar value)
        : in_(std::move(in)), value_(value) {}
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        auto m = co_await in_->next(max_rows);
        if (!m) co_return std::nullopt;
        co_return map_frame(std::move(*m),
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
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        auto m = co_await in_->next(max_rows);
        if (!m) co_return std::nullopt;
        std::vector<std::int64_t> idx(static_cast<std::size_t>(m->rows));
        for (std::int64_t i = 0; i < m->rows; ++i) idx[i] = pos_ + i;
        pos_ += m->rows;
        Morsel out;
        out.rows = m->rows;
        out.columns.reserve(m->columns.size() + 1);
        out.columns.push_back(Series::flat_i64(idx.data(), m->rows));
        for (Series& c : m->columns) out.columns.push_back(std::move(c));
        co_return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::int64_t pos_ = 0;
};

// Per-column null counts, accumulated streaming, emitted as one row.
class NullCountCursor : public Cursor {
   public:
    explicit NullCountCursor(std::unique_ptr<Cursor> in) : in_(std::move(in)) {}
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
        done_ = true;
        std::vector<std::int64_t> counts;
        while (auto m = co_await in_->next(max_rows)) {
            if (counts.empty()) counts.assign(m->columns.size(), 0);
            for (std::size_t i = 0; i < m->columns.size(); ++i)
                counts[i] += m->columns[i].null_count();
        }
        if (counts.empty()) co_return std::nullopt;
        Morsel out;
        out.rows = 1;
        out.columns.reserve(counts.size());
        for (std::int64_t& c : counts)
            out.columns.push_back(Series::flat_i64(&c, 1));
        co_return out;
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
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        auto m = co_await in_->next(max_rows);
        if (!m) co_return std::nullopt;
        co_return frame_op(std::move(*m), sch_,
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
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        auto m = co_await in_->next(max_rows);
        if (!m) co_return std::nullopt;
        co_return frame_op(std::move(*m), sch_,
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
    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
        done_ = true;
        DataFrame best;
        bool has = false;
        while (auto m = co_await in_->next(max_rows)) {
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
        if (!has) co_return std::nullopt;
        Morsel out;
        out.rows = best.num_rows();
        out.columns.reserve(best.columns.size());
        for (const Series& c : best.columns)
            out.columns.push_back(c.materialize());
        co_return out;
    }

   private:
    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::string name_;
    std::int64_t k_;
    bool largest_;
    bool done_ = false;
};

// One run: single-group AggState blobs (agg_extract_group + agg_serialize),
// length-prefixed, in ascending composite-key order (agg_sort_groups). The
// on-disk unit a bounded k-way merge reads back one group at a time.
void agg_write_run(AggState& st, const std::string& path) {
    agg_sort_groups(st);
    std::ofstream os(path, std::ios::binary);
    const std::int64_t ng = agg_num_groups(st);
    for (std::int64_t g = 0; g < ng; ++g) {
        const std::string blob = agg_serialize(*agg_extract_group(st, g));
        const std::uint32_t len = static_cast<std::uint32_t>(blob.size());
        os.write(reinterpret_cast<const char*>(&len), sizeof(len));
        os.write(blob.data(), static_cast<std::streamsize>(blob.size()));
    }
}

// Streams one sorted run's single-group states back, one record at a time.
class AggRunReader {
   public:
    explicit AggRunReader(const std::string& path)
        : is_(path, std::ios::binary) {
        advance();
    }
    bool valid() const { return valid_; }
    const AggState& state() const { return *cur_; }
    void advance() {
        std::uint32_t len = 0;
        if (!is_.read(reinterpret_cast<char*>(&len), sizeof(len))) {
            valid_ = false;
            return;
        }
        std::string blob(len, '\0');
        is_.read(blob.data(), static_cast<std::streamsize>(len));
        cur_ = agg_deserialize(blob);
        valid_ = true;
    }

   private:
    std::ifstream is_;
    AggStatePtr cur_;
    bool valid_ = false;
};

// Streaming group-by: fold every morsel into one mergeable AggState. When the
// accumulated state exceeds `budget`, flush it to a sorted-by-key run on disk
// and start a fresh state (mirrors SortMergeCursor's external merge sort). No
// spill needed: finalize the single in-memory state directly (unchanged
// behavior). Spilled: k-way merge the runs, combining equal composite keys,
// emitting rows bounded by max_rows per call.
class GroupByCursor : public Cursor {
   public:
    GroupByCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                  std::vector<std::string> keys, std::vector<GroupAgg> aggs,
                  std::uint64_t budget, std::vector<AggDynSpec> dyn = {},
                  std::string dyn_prefix = {})
        : in_(std::move(in)),
          sch_(std::move(sch)),
          keys_(std::move(keys)),
          aggs_(std::move(aggs)),
          budget_(budget),
          dyn_specs_(std::move(dyn)),
          dyn_prefix_(std::move(dyn_prefix)) {}

    std::optional<std::vector<std::string>> out_names() const override {
        return out_names_;
    }

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (!built_) co_await build(max_rows);
        if (!spilled_) {
            if (done_) co_return std::nullopt;
            done_ = true;
            co_return std::move(result_);
        }
        co_return merge_next(max_rows);
    }

   private:
    static int index_in(const std::vector<std::string>& s,
                        const std::string& n) {
        auto it = std::find(s.begin(), s.end(), n);
        return it == s.end() ? -1 : static_cast<int>(it - s.begin());
    }

    static Morsel to_morsel(const DataFrame& r) {
        Morsel out;
        out.rows = r.num_rows();
        out.columns.reserve(r.columns.size());
        for (const Series& c : r.columns)
            out.columns.push_back(c.materialize());
        return out;
    }

    coro::CoroTask<void> build(std::int64_t max_rows) {
        std::vector<int> key_idx;
        key_idx.reserve(keys_.size());
        for (const std::string& k : keys_) key_idx.push_back(index_in(sch_, k));
        std::vector<int> value_idx;  // sch indices of the deduped value columns
        ankerl::unordered_dense::map<std::string, std::int32_t> dedup;
        auto resolve = [&](const std::string& name) -> std::int32_t {
            auto it = dedup.find(name);
            if (it != dedup.end()) return it->second;
            const std::int32_t idx =
                static_cast<std::int32_t>(value_idx.size());
            value_idx.push_back(index_in(sch_, name));
            dedup.emplace(name, idx);
            return idx;
        };
        specs_.reserve(aggs_.size());
        for (const GroupAgg& a : aggs_) {
            AggSpec sp;
            sp.op = to_agg_op(a.op);
            sp.out = a.out;
            sp.param = a.param;
            sp.value_col = sp.op == AggOp::Count ? -1 : resolve(a.column);
            if (agg_uses_by_col(sp.op)) sp.by_col = resolve(a.by);
            specs_.push_back(std::move(sp));
        }

        // A resident source carries dyn columns in-band (prefix-tagged in
        // sch_); a streaming source carries them out of band (Morsel::dyn_*).
        std::vector<std::pair<int, std::string>> sch_dyn;
        if (!dyn_specs_.empty() && !dyn_prefix_.empty())
            for (std::size_t i = 0; i < sch_.size(); ++i)
                if (sch_[i].rfind(dyn_prefix_, 0) == 0)
                    sch_dyn.emplace_back(static_cast<int>(i),
                                         sch_[i].substr(dyn_prefix_.size()));

        AggStatePtr state = agg_new(specs_, dyn_specs_);
        // Bounded parallel sink: pull a batch of morsels, accumulate each into
        // its own partial AggState in parallel (the mergeable agg IR), then
        // merge the partials into the running state. Memory stays bounded to
        // one batch; a serial-pull single morsel skips the fan-out.
        constexpr std::size_t BATCH = 32;
        std::vector<Morsel> batch;
        batch.reserve(BATCH);
        bool eof = false;
        int run_id = 0;
        while (!eof) {
            batch.clear();
            for (std::size_t b = 0; b < BATCH; ++b) {
                auto m = co_await in_->next(max_rows);
                if (!m) {
                    eof = true;
                    break;
                }
                batch.push_back(std::move(*m));
            }
            if (batch.empty()) break;
            auto accumulate = [&](AggState& st, const Morsel& m) {
                std::vector<const Series*> keys;
                keys.reserve(key_idx.size());
                for (int ki : key_idx) keys.push_back(&m.columns[ki]);
                std::vector<const Series*> values;
                values.reserve(value_idx.size());
                for (int vi : value_idx) values.push_back(&m.columns[vi]);
                if (dyn_specs_.empty()) {
                    agg_accumulate(st, keys, values);
                    return;
                }
                std::vector<AggDynInput> dyn;
                dyn.reserve(sch_dyn.size() + m.dyn_columns.size());
                for (const auto& [ci, name] : sch_dyn)
                    dyn.push_back(
                        {name, &m.columns[static_cast<std::size_t>(ci)]});
                for (std::size_t i = 0; i < m.dyn_columns.size(); ++i) {
                    const std::string& raw = m.dyn_names[i];
                    std::string name = raw.rfind(dyn_prefix_, 0) == 0
                                           ? raw.substr(dyn_prefix_.size())
                                           : raw;
                    dyn.push_back({std::move(name), &m.dyn_columns[i]});
                }
                agg_accumulate(st, keys, values, dyn);
            };
            if (batch.size() == 1) {
                accumulate(*state, batch[0]);
            } else {
                std::vector<AggStatePtr> partials(batch.size());
                parallel_for(
                    static_cast<std::int64_t>(batch.size()), 1,
                    [&](std::int64_t bi, std::int64_t ei) {
                        for (std::int64_t j = bi; j < ei; ++j) {
                            auto st = agg_new(specs_, dyn_specs_);
                            accumulate(*st, batch[static_cast<std::size_t>(j)]);
                            partials[static_cast<std::size_t>(j)] =
                                std::move(st);
                        }
                    });
                for (auto& p : partials)
                    if (p) agg_merge(*state, *p);
            }
            if (budget_ > 0 && agg_approx_bytes(*state) > budget_) {
                agg_write_run(*state, dir_.run_path(run_id++));
                state = agg_new(specs_, dyn_specs_);
            }
        }

        if (run_id == 0) {
            DataFrame r = agg_finalize(*state, keys_);
            out_names_ = r.names;
            result_ = to_morsel(r);
            spilled_ = false;
        } else {
            if (agg_num_groups(*state) > 0)
                agg_write_run(*state, dir_.run_path(run_id++));
            runs_.reserve(static_cast<std::size_t>(run_id));
            for (int i = 0; i < run_id; ++i)
                runs_.push_back(
                    std::make_unique<AggRunReader>(dir_.run_path(i)));
            spilled_ = true;
        }
        built_ = true;
    }

    // Merge every run sharing the smallest composite key into `merged`;
    // distinct keys arrive ascending, so groups append in order. One
    // agg_finalize at the end (not per group + concat_columns) lets a nested
    // Hist column, which concat_columns cannot rejoin, survive spill.
    std::optional<Morsel> merge_next(std::int64_t max_rows) {
        // The dyn column set is the global name union, so a k-way streamed
        // emission would give per-batch-varying dyn columns that cannot
        // vertically concat. Merge all runs into one state, finalize once.
        if (!dyn_specs_.empty()) {
            if (dyn_drained_) return std::nullopt;
            dyn_drained_ = true;
            AggStatePtr merged = agg_new(specs_, dyn_specs_);
            for (auto& run : runs_)
                while (run->valid()) {
                    agg_merge(*merged, run->state());
                    run->advance();
                }
            if (agg_num_groups(*merged) == 0) return std::nullopt;
            DataFrame r = agg_finalize(*merged, keys_);
            out_names_ = r.names;
            return to_morsel(r);
        }
        AggStatePtr merged = agg_new(specs_, dyn_specs_);
        std::int64_t produced = 0;
        while (produced < max_rows) {
            int best = -1;
            for (std::size_t i = 0; i < runs_.size(); ++i) {
                if (!runs_[i]->valid()) continue;
                if (best < 0 ||
                    agg_key_cmp(runs_[i]->state(), 0,
                                runs_[static_cast<std::size_t>(best)]->state(),
                                0) < 0)
                    best = static_cast<int>(i);
            }
            if (best < 0) break;
            // Snapshot the winning key before merging: advancing `best`'s own
            // reader mid-loop would otherwise mutate the very state later
            // iterations compare against.
            const AggStatePtr win_key = agg_extract_group(
                runs_[static_cast<std::size_t>(best)]->state(), 0);
            for (auto& run : runs_) {
                if (!run->valid()) continue;
                if (agg_key_cmp(run->state(), 0, *win_key, 0) != 0) continue;
                agg_merge(*merged, run->state());
                run->advance();
            }
            ++produced;
        }
        if (produced == 0) return std::nullopt;
        DataFrame r = agg_finalize(*merged, keys_);
        out_names_ = r.names;
        return to_morsel(r);
    }

    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::vector<std::string> keys_;
    std::vector<GroupAgg> aggs_;
    std::uint64_t budget_;
    std::vector<AggDynSpec> dyn_specs_;
    std::string dyn_prefix_;
    std::vector<AggSpec> specs_;
    std::optional<std::vector<std::string>> out_names_;
    bool built_ = false;
    bool done_ = false;
    bool spilled_ = false;
    bool dyn_drained_ = false;
    Morsel result_;
    spill::Dir dir_;
    std::vector<std::unique_ptr<AggRunReader>> runs_;
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
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
        auto resolve = [&](const std::string& name) -> std::int32_t {
            auto it = dedup.find(name);
            if (it != dedup.end()) return it->second;
            const std::int32_t idx =
                static_cast<std::int32_t>(value_idx.size());
            value_idx.push_back(index_in(sch_, name));
            dedup.emplace(name, idx);
            return idx;
        };
        specs.reserve(aggs_.size());
        for (const GroupAgg& a : aggs_) {
            AggSpec sp;
            sp.op = to_agg_op(a.op);
            sp.out = a.out;
            sp.param = a.param;
            sp.value_col = sp.op == AggOp::Count ? -1 : resolve(a.column);
            if (agg_uses_by_col(sp.op)) sp.by_col = resolve(a.by);
            specs.push_back(std::move(sp));
        }

        AggStatePtr state = agg_new(specs);
        bool anchored = false;
        std::int64_t start0 = 0, origin = origin_;
        while (auto m = co_await in_->next(max_rows)) {
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
        co_return morsel_of(std::move(r));
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
        done_ = true;
        DataFrame best;                   // <= n_ rows
        std::vector<std::uint64_t> keys;  // parallel to best's rows
        std::vector<std::int64_t> idx;    // original global row indices
        std::int64_t off = 0;
        while (auto m = co_await in_->next(max_rows)) {
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
        co_return morsel_of(take(best, ord));
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (!built_) co_await build(max_rows);

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
            if (limit >= wm.rows) co_await advance(winner, max_rows);
        }
        if (pieces.empty()) co_return std::nullopt;
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
        co_return out;
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

    coro::CoroTask<void> advance(int k, std::int64_t max_rows) {
        cur_[static_cast<std::size_t>(k)] =
            co_await runs_[static_cast<std::size_t>(k)]->next(max_rows);
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

    coro::CoroTask<void> build(std::int64_t max_rows) {
        key_idx_ = static_cast<int>(std::distance(
            sch_.begin(), std::find(sch_.begin(), sch_.end(), key_)));
        if (key_idx_ >= static_cast<int>(sch_.size()))
            throw std::out_of_range("sort_by: no column named " + key_);

        std::vector<std::vector<Series>> pending;
        std::size_t pend_bytes = 0;
        int run_id = 0;
        while (auto m = co_await in_->next(max_rows)) {
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
            cur_[k] = co_await runs_[k]->next(max_rows);
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

// Streaming distinct (keep first occurrence, original order). Fast path holds
// only the set of distinct row keys - which is the result itself, materialized
// by collect anyway - and streams input and output morsel by morsel. When that
// key state would exceed `budget_`, switches to a grace-hash-distinct spill:
// every row still to come gets a global row-id, is hash-partitioned by key to
// disk (skipping any key already resolved by the fast path), each partition is
// deduped independently keeping the row with the minimum row-id (recursing
// with a depth-salted hash if a partition itself does not fit budget), and the
// survivors are k-way merged back into row-id order so the fast-emitted prefix
// and the spilled remainder together reproduce one globally first-occurrence,
// input-order stream.
class UniqueCursor : public Cursor {
   public:
    UniqueCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch,
                 std::uint64_t budget)
        : in_(std::move(in)), sch_(std::move(sch)), budget_(budget) {
        if (parallel_backend_installed()) seen_p_.resize(DEDUP_PARTITIONS);
    }

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (!spilling_) {
            while (auto m = co_await in_->next(max_rows)) {
                const std::int64_t n = m->rows;
                // Build the exact row keys in parallel (scalar string work,
                // one per row, independent), then dedupe (radix-partitioned
                // when a parallel backend is installed, else one serial set).
                std::vector<std::string> keys(static_cast<std::size_t>(n));
                parallel_for(n, std::int64_t{1} << 13,
                             [&](std::int64_t b, std::int64_t e) {
                                 for (std::int64_t i = b; i < e; ++i)
                                     keys[static_cast<std::size_t>(i)] =
                                         row_key(m->columns, i);
                             });
                std::vector<std::uint32_t> key_len(static_cast<std::size_t>(n));
                for (std::int64_t i = 0; i < n; ++i)
                    key_len[static_cast<std::size_t>(i)] =
                        static_cast<std::uint32_t>(
                            keys[static_cast<std::size_t>(i)].size());
                std::vector<std::uint8_t> keep_mask =
                    first_seen_mask(keys, n, seen_, seen_p_);
                std::vector<std::int64_t> keep;
                keep.reserve(static_cast<std::size_t>(n));
                for (std::int64_t i = 0; i < n; ++i) {
                    if (!keep_mask[static_cast<std::size_t>(i)]) continue;
                    keep.push_back(i);
                    fast_bytes_ += key_len[static_cast<std::size_t>(i)] +
                                   DEDUP_ENTRY_OVERHEAD;
                }
                next_row_id_ += n;
                if (budget_ > 0 && fast_bytes_ > budget_) spilling_ = true;
                if (!keep.empty()) {
                    DataFrame mf;
                    mf.names = sch_;
                    mf.columns = std::move(m->columns);
                    co_return morsel_of(take(mf, keep));
                }
                if (spilling_) break;
            }
            if (!spilling_) co_return std::nullopt;
        }
        if (!drained_) co_await drain_and_finalize(max_rows);
        co_return co_await merge_next(max_rows);
    }

   private:
    bool already_seen(const std::string& key) const {
        if (!seen_p_.empty())
            return seen_p_[std::hash<std::string>{}(key) % seen_p_.size()]
                .contains(key);
        return seen_.contains(key);
    }

    std::vector<std::string> rowid_schema() const {
        std::vector<std::string> out;
        out.reserve(sch_.size() + 1);
        out.emplace_back("__row_id");
        for (const std::string& n : sch_) out.push_back(n);
        return out;
    }

    // Hash-partitions `cols` (row-id column first, then `data_cols` for the
    // key) into `fanout` on-disk runs by `unique_spill_hash(key, depth)`,
    // tracking each partition's approximate bytes and row count for the
    // recurse-or-leaf decision at finalize.
    void partition_rows(const std::vector<Series>& tagged_cols,
                        const std::vector<Series>& data_cols, std::int64_t n,
                        int depth, int fanout,
                        std::vector<spill::Writer>& writers,
                        std::vector<std::size_t>& bytes,
                        std::vector<std::int64_t>& rows) {
        std::vector<std::string> keys(static_cast<std::size_t>(n));
        parallel_for(
            n, std::int64_t{1} << 13, [&](std::int64_t b, std::int64_t e) {
                for (std::int64_t i = b; i < e; ++i)
                    keys[static_cast<std::size_t>(i)] = row_key(data_cols, i);
            });
        std::vector<std::vector<std::int64_t>> buckets(
            static_cast<std::size_t>(fanout));
        for (std::int64_t i = 0; i < n; ++i) {
            if (depth == 0 && already_seen(keys[static_cast<std::size_t>(i)]))
                continue;  // already emitted by the fast phase
            const std::size_t p =
                unique_spill_hash(keys[static_cast<std::size_t>(i)], depth) %
                static_cast<std::size_t>(fanout);
            buckets[p].push_back(i);
        }
        DataFrame mf;
        mf.names = rowid_schema();
        mf.columns.reserve(tagged_cols.size());
        for (const Series& c : tagged_cols) mf.columns.push_back(c.share());
        for (int p = 0; p < fanout; ++p) {
            if (buckets[static_cast<std::size_t>(p)].empty()) continue;
            DataFrame sel = take(mf, buckets[static_cast<std::size_t>(p)]);
            bytes[static_cast<std::size_t>(p)] += morsel_bytes(sel.columns);
            rows[static_cast<std::size_t>(p)] += sel.num_rows();
            writers[static_cast<std::size_t>(p)].write(sel.columns,
                                                       sel.num_rows());
        }
    }

    // Dedups one spilled partition (rows whose key was not already resolved
    // by the fast phase, hash-partitioned to land here), keeping the row with
    // the minimum row-id per key. If the partition itself would still exceed
    // budget and is large enough that splitting helps, re-partitions it with
    // a depth-salted hash instead of loading it whole (bounded recursion: a
    // single hot key always lands in the same sub-partition however deep, so
    // depth alone cannot force it smaller - the row-count floor stops the
    // recursion once further splitting cannot shrink it).
    coro::CoroTask<void> finalize_partition(const std::string& path,
                                            std::size_t bytes,
                                            std::int64_t rows, int depth) {
        if (depth < UNIQUE_SPILL_MAX_DEPTH && budget_ > 0 && bytes > budget_ &&
            rows > UNIQUE_SPILL_MIN_LEAF_ROWS) {
            constexpr int FANOUT = UNIQUE_SPILL_FANOUT;
            std::vector<spill::Writer> writers;
            writers.reserve(static_cast<std::size_t>(FANOUT));
            std::vector<int> ids(static_cast<std::size_t>(FANOUT));
            for (int p = 0; p < FANOUT; ++p) {
                ids[static_cast<std::size_t>(p)] = next_run_id_++;
                writers.emplace_back(
                    dir_.run_path(ids[static_cast<std::size_t>(p)]));
            }
            std::vector<std::size_t> sub_bytes(static_cast<std::size_t>(FANOUT),
                                               0);
            std::vector<std::int64_t> sub_rows(static_cast<std::size_t>(FANOUT),
                                               0);
            spill::Reader reader(path);
            while (auto m = co_await reader.next(DEFAULT_MORSEL_ROWS)) {
                std::vector<Series> data_cols = drop_first_column(m->columns);
                partition_rows(m->columns, data_cols, m->rows, depth + 1,
                               FANOUT, writers, sub_bytes, sub_rows);
            }
            for (spill::Writer& w : writers) w.close();
            for (int p = 0; p < FANOUT; ++p)
                co_await finalize_partition(
                    dir_.run_path(ids[static_cast<std::size_t>(p)]),
                    sub_bytes[static_cast<std::size_t>(p)],
                    sub_rows[static_cast<std::size_t>(p)], depth + 1);
            co_return;
        }

        // Leaf: small enough to fit budget (or recursion bottomed out) - load
        // fully, dedupe by minimum row-id, sort survivors by row-id, and write
        // one run for the final k-way merge.
        std::vector<std::vector<Series>> parts;
        std::int64_t total = 0;
        {
            spill::Reader reader(path);
            while (auto m = co_await reader.next(DEFAULT_MORSEL_ROWS)) {
                total += m->rows;
                parts.push_back(std::move(m->columns));
            }
        }
        if (total == 0) co_return;

        const std::size_t ncols = parts.front().size();
        std::vector<Series> whole;
        whole.reserve(ncols);
        for (std::size_t c = 0; c < ncols; ++c) {
            std::vector<const Series*> pcs;
            pcs.reserve(parts.size());
            for (auto& pc : parts) pcs.push_back(&pc[c]);
            whole.push_back(concat_columns(pcs));
        }
        const std::vector<Series> data_cols = drop_first_column(whole);
        const std::int64_t* rowid = whole[0].data<std::int64_t>();

        ankerl::unordered_dense::map<std::string, std::int64_t> best;
        for (std::int64_t i = 0; i < total; ++i) {
            std::string k = row_key(data_cols, i);
            auto it = best.find(k);
            if (it == best.end())
                best.emplace(std::move(k), i);
            else if (rowid[i] < rowid[it->second])
                it->second = i;
        }
        std::vector<std::int64_t> survivors;
        survivors.reserve(best.size());
        for (const auto& kv : best) survivors.push_back(kv.second);
        std::sort(survivors.begin(), survivors.end(),
                  [&](std::int64_t a, std::int64_t b) {
                      return rowid[a] < rowid[b];
                  });

        DataFrame mf;
        mf.names = rowid_schema();
        mf.columns = std::move(whole);
        DataFrame sorted = take(mf, survivors);

        const std::string run_path = dir_.run_path(next_run_id_++);
        spill::Writer w(run_path);
        w.write(sorted.columns, sorted.num_rows());
        w.close();
        survivor_runs_.push_back(std::make_unique<spill::Reader>(run_path));
    }

    // Fully drains the remaining input into UNIQUE_SPILL_FANOUT partitions
    // (skipping rows whose key the fast phase already resolved), then dedupes
    // and orders every partition's survivors for the k-way merge in
    // merge_next.
    coro::CoroTask<void> drain_and_finalize(std::int64_t max_rows) {
        constexpr int FANOUT = UNIQUE_SPILL_FANOUT;
        std::vector<spill::Writer> writers;
        writers.reserve(static_cast<std::size_t>(FANOUT));
        std::vector<int> ids(static_cast<std::size_t>(FANOUT));
        for (int p = 0; p < FANOUT; ++p) {
            ids[static_cast<std::size_t>(p)] = next_run_id_++;
            writers.emplace_back(
                dir_.run_path(ids[static_cast<std::size_t>(p)]));
        }
        std::vector<std::size_t> bytes(static_cast<std::size_t>(FANOUT), 0);
        std::vector<std::int64_t> rows(static_cast<std::size_t>(FANOUT), 0);

        while (auto m = co_await in_->next(max_rows)) {
            const std::int64_t n = m->rows;
            std::vector<std::int64_t> rowid(static_cast<std::size_t>(n));
            for (std::int64_t i = 0; i < n; ++i)
                rowid[static_cast<std::size_t>(i)] = next_row_id_ + i;
            next_row_id_ += n;
            std::vector<Series> tagged;
            tagged.reserve(m->columns.size() + 1);
            tagged.push_back(Series::flat_i64(rowid.data(), n));
            for (Series& c : m->columns) tagged.push_back(std::move(c));
            partition_rows(tagged, drop_first_column(tagged), n, 0, FANOUT,
                           writers, bytes, rows);
        }
        for (spill::Writer& w : writers) w.close();

        for (int p = 0; p < FANOUT; ++p)
            co_await finalize_partition(
                dir_.run_path(ids[static_cast<std::size_t>(p)]),
                bytes[static_cast<std::size_t>(p)],
                rows[static_cast<std::size_t>(p)], 0);

        cur_.resize(survivor_runs_.size());
        pos_.assign(survivor_runs_.size(), 0);
        for (std::size_t k = 0; k < survivor_runs_.size(); ++k)
            cur_[k] = co_await survivor_runs_[k]->next(max_rows);
        drained_ = true;
    }

    // K-way merges the row-id-sorted survivor runs into ascending row-id
    // order (rows are globally unique row-ids, so a one-row-at-a-time pick is
    // fine here - this spill-finalize path is rare, not the streaming fast
    // path), dropping the row-id helper column before emitting.
    coro::CoroTask<std::optional<Morsel>> merge_next(std::int64_t max_rows) {
        std::vector<std::vector<Series>> pieces;
        std::int64_t out_rows = 0;
        while (out_rows < max_rows) {
            int winner = -1;
            for (std::size_t k = 0; k < cur_.size(); ++k) {
                if (!cur_[k] || pos_[k] >= cur_[k]->rows) continue;
                if (winner < 0) {
                    winner = static_cast<int>(k);
                    continue;
                }
                const std::size_t wk = static_cast<std::size_t>(winner);
                const std::int64_t cand =
                    cur_[k]->columns[0].data<std::int64_t>()[pos_[k]];
                const std::int64_t best =
                    cur_[wk]->columns[0].data<std::int64_t>()[pos_[wk]];
                if (cand < best) winner = static_cast<int>(k);
            }
            if (winner < 0) break;
            const std::size_t wk = static_cast<std::size_t>(winner);
            Morsel piece = slice_morsel(*cur_[wk], pos_[wk], 1);
            piece.columns.erase(piece.columns.begin());
            pieces.push_back(std::move(piece.columns));
            ++out_rows;
            ++pos_[wk];
            if (pos_[wk] >= cur_[wk]->rows)
                cur_[wk] = co_await survivor_runs_[wk]->next(max_rows);
        }
        if (pieces.empty()) co_return std::nullopt;
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
        co_return out;
    }

    std::unique_ptr<Cursor> in_;
    std::vector<std::string> sch_;
    std::uint64_t budget_;
    ankerl::unordered_dense::set<std::string> seen_;
    std::vector<ankerl::unordered_dense::set<std::string>> seen_p_;
    std::size_t fast_bytes_ = 0;
    std::int64_t next_row_id_ = 0;
    bool spilling_ = false;
    bool drained_ = false;
    spill::Dir dir_;
    int next_run_id_ = 0;
    std::vector<std::unique_ptr<Cursor>> survivor_runs_;
    std::vector<std::optional<Morsel>> cur_;
    std::vector<std::int64_t> pos_;
};

// Streaming per-column summary statistics, matching DataFrame::describe. One
// pass with O(numeric columns) state: count/null_count and running min/max plus
// Welford (mean, M2) for mean/sample-std. Output columns are data-dependent
// (one per numeric input column), reported via out_names().
class DescribeCursor : public Cursor {
   public:
    DescribeCursor(std::unique_ptr<Cursor> in, std::vector<std::string> sch)
        : in_(std::move(in)), sch_(std::move(sch)) {}

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
        done_ = true;
        // Per-morsel SIMD reduction into one mergeable FieldStat per numeric
        // column (the engine's shared aggregation atom), so the numeric work is
        // vectorized and bounded by the column count.
        std::vector<int> num_idx;
        std::vector<FieldStat> acc;
        std::int64_t total_rows = 0;
        bool first = true;
        while (auto m = co_await in_->next(max_rows)) {
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
        co_return morsel_of(std::move(out));
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
//
// Pass 1 is DISTINCT == GROUP BY all columns: it reuses the mergeable agg IR
// (AggOp::Count keyed by the row-key column) through the same bounded parallel
// sink as GroupByCursor - a batch of morsels, one partial AggState per morsel
// in parallel, merged serially - instead of a single hash map fed one row at a
// time. Pass 2's per-row lookup against the finalized (key, count) map is
// independent per row, so it fans out too; the output Bool column is bit-
// packed, so each parallel task owns whole bytes (8 rows) to avoid a shared-
// byte write race.
class IsDupCursor : public Cursor {
   public:
    IsDupCursor(std::unique_ptr<Cursor> in, std::uint64_t budget, bool unique)
        : first_(std::move(in)), spool_(budget), unique_(unique) {}

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (!counted_) co_await count(max_rows);
        while (auto m = co_await pass2_->next(max_rows)) {
            const std::int64_t n = m->rows;
            const std::int64_t nbytes = (n + 7) / 8;
            std::vector<std::uint8_t> bits(static_cast<std::size_t>(nbytes), 0);
            const std::vector<Series>& cols = m->columns;
            parallel_for(nbytes, std::int64_t{1} << 10,
                         [&](std::int64_t bb, std::int64_t be) {
                             for (std::int64_t byte = bb; byte < be; ++byte) {
                                 const std::int64_t base = byte * 8;
                                 const std::int64_t lim = std::min(base + 8, n);
                                 std::uint8_t v = 0;
                                 for (std::int64_t i = base; i < lim; ++i) {
                                     auto it = counts_.find(row_key(cols, i));
                                     const std::int64_t c =
                                         it != counts_.end() ? it->second : 0;
                                     if (unique_ ? c == 1 : c > 1)
                                         v |= static_cast<std::uint8_t>(
                                             1u << (i - base));
                                 }
                                 bits[static_cast<std::size_t>(byte)] = v;
                             }
                         });
            Morsel out;
            out.rows = n;
            out.columns.push_back(Series::flat(TypeId::Bool, bits.data(), n));
            co_return out;
        }
        co_return std::nullopt;
    }

   private:
    coro::CoroTask<void> count(std::int64_t max_rows) {
        std::vector<AggSpec> specs(1);
        specs[0].op = AggOp::Count;
        specs[0].out = "count";
        AggStatePtr state = agg_new(specs);
        constexpr std::size_t BATCH = 32;
        std::vector<Morsel> batch;
        batch.reserve(BATCH);
        auto key_series = [](const Morsel& m) {
            std::vector<std::string> keys(static_cast<std::size_t>(m.rows));
            for (std::int64_t i = 0; i < m.rows; ++i)
                keys[static_cast<std::size_t>(i)] = row_key(m.columns, i);
            return Series::strings(keys);
        };
        bool eof = false;
        while (!eof) {
            batch.clear();
            for (std::size_t b = 0; b < BATCH; ++b) {
                auto m = co_await first_->next(max_rows);
                if (!m) {
                    eof = true;
                    break;
                }
                batch.push_back(std::move(*m));
            }
            if (batch.empty()) break;
            if (batch.size() == 1) {
                Series key = key_series(batch[0]);
                agg_accumulate(*state, key, {});
            } else {
                std::vector<AggStatePtr> partials(batch.size());
                parallel_for(static_cast<std::int64_t>(batch.size()), 1,
                             [&](std::int64_t bi, std::int64_t ei) {
                                 for (std::int64_t j = bi; j < ei; ++j) {
                                     auto st = agg_new(specs);
                                     Series key = key_series(
                                         batch[static_cast<std::size_t>(j)]);
                                     agg_accumulate(*st, key, {});
                                     partials[static_cast<std::size_t>(j)] =
                                         std::move(st);
                                 }
                             });
                for (auto& p : partials)
                    if (p) agg_merge(*state, *p);
            }
            for (auto& m : batch) spool_.add(std::move(m.columns), m.rows);
        }
        first_.reset();
        DataFrame r = agg_finalize(*state, "key");
        const Series& kc = r.columns[0];
        const Series& cc = r.columns[1];
        const std::int64_t d = r.num_rows();
        counts_.reserve(static_cast<std::size_t>(d));
        for (std::int64_t i = 0; i < d; ++i)
            counts_.emplace(std::string(kc.string_at(i)),
                            cc.data<std::int64_t>()[i]);
        pass2_ = spool_.reader();
        counted_ = true;
    }

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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (!built_) co_await build(max_rows);
        auto m = co_await pass2_->next(max_rows);
        if (!m) co_return std::nullopt;
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
        co_return out;
    }

    std::optional<std::vector<std::string>> out_names() const override {
        return produced_;
    }

   private:
    coro::CoroTask<void> build(std::int64_t max_rows) {
        ci_ = static_cast<int>(std::distance(
            sch_.begin(), std::find(sch_.begin(), sch_.end(), column_)));
        if (ci_ >= static_cast<int>(sch_.size()))
            throw std::out_of_range("to_dummies: no column named " + column_);

        ankerl::unordered_dense::set<std::string> seen;
        std::vector<Series> chunks;
        while (auto m = co_await first_->next(max_rows)) {
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

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        if (done_) co_return std::nullopt;
        done_ = true;
        const int ii = idx_of(index_), ci = idx_of(on_), vi = idx_of(values_);
        if (ii < 0) throw std::out_of_range("pivot: no column named " + index_);
        if (ci < 0) throw std::out_of_range("pivot: no column named " + on_);
        if (vi < 0)
            throw std::out_of_range("pivot: no column named " + values_);

        // Pass 1: distinct index and `on` values (ascending, matching eager).
        ankerl::unordered_dense::set<std::string> seen_i, seen_c;
        std::vector<Series> ich, cch;
        while (auto m = co_await first_->next(max_rows)) {
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
        while (auto m = co_await p2->next(max_rows)) {
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
        co_return morsel_of(std::move(out));
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
    std::vector<std::string> keys;
    std::vector<GroupAgg> aggs;
    std::vector<AggDynSpec> dyn;
    std::string dyn_prefix;
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

AggOp to_agg_op(Agg a) {
    switch (a) {
#define DFTU_AGG_OP(id, code, name) \
    case Agg::id:                   \
        return AggOp::id;
#include <dftracer/utils/dataframe/agg_ops.def>
#undef DFTU_AGG_OP
    }
    return AggOp::Count;
}

Agg from_agg_op(AggOp a) {
    switch (a) {
#define DFTU_AGG_OP(id, code, name) \
    case AggOp::id:                 \
        return Agg::id;
#include <dftracer/utils/dataframe/agg_ops.def>
#undef DFTU_AGG_OP
    }
    return Agg::Count;
}

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
                // Dyn column names are discovered at run time: data-dependent
                // schema, signalled empty like pivot (collect() relabels).
                if (!o.dyn.empty()) return std::vector<std::string>{};
                std::vector<std::string> s = o.keys;
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
            [](const GroupByOp& o) { return "group_by " + join_names(o.keys); },
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

    // Hoist a filter past a preceding op that keeps its columns' positions and
    // rows: with_column (unless the predicate reads the written column),
    // sort_by, rename. filter-before-sort is the big win - the sort runs on
    // survivors.
    bool changed = true;
    while (changed) {
        changed = false;
        for (std::size_t i = 1; i < nodes.size(); ++i) {
            const auto* filt = std::get_if<FilterOp>(&nodes[i].op->node);
            if (!filt) continue;
            const LazyOp& prev = *nodes[i - 1].op;
            bool hoist = false;
            if (std::holds_alternative<WithColumnOp>(prev.node)) {
                hoist = !expr_references(filt->pred, nodes[i - 1].write_idx);
            } else if (std::holds_alternative<SortByOp>(prev.node) ||
                       std::holds_alternative<RenameOp>(prev.node)) {
                hoist = true;
            }
            if (hoist) {
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

// Projection pushdown: for a schema-preserving plan (filter/sort_by/slice/tail/
// topk/sample) that ends in a select, insert a projection after the source
// keeping only the columns the output and ops read, renumbering the filters
// into it. Any other op leaves the plan unchanged; pushdown is an optimization,
// so bailing is always correct. Biggest payoff is a scan source that then reads
// only the kept columns.
std::vector<std::shared_ptr<const LazyOp>> pushdown_projections(
    const std::vector<std::string>& source_names,
    const std::vector<std::shared_ptr<const LazyOp>>& ops) {
    if (ops.size() < 2) return ops;  // need a select plus something before it
    const std::size_t last = ops.size() - 1;
    if (!std::holds_alternative<SelectOp>(ops[last]->node)) return ops;
    for (std::size_t i = 0; i < last; ++i) {
        const auto& n = ops[i]->node;
        if (!(std::holds_alternative<FilterOp>(n) ||
              std::holds_alternative<SortByOp>(n) ||
              std::holds_alternative<SliceOp>(n) ||
              std::holds_alternative<TailOp>(n) ||
              std::holds_alternative<TopkOp>(n) ||
              std::holds_alternative<SampleOp>(n)))
            return ops;  // changes the schema or reads whole rows: bail
    }

    const int nsrc = static_cast<int>(source_names.size());
    std::vector<char> need(static_cast<std::size_t>(nsrc), 0);
    for (const std::string& nm : std::get<SelectOp>(ops[last]->node).names) {
        int c = col_index(source_names, nm);
        if (c < 0) return ops;
        need[static_cast<std::size_t>(c)] = 1;
    }
    // These ops preserve the source schema by position, so predicate indices
    // are source indices.
    for (std::size_t i = 0; i < last; ++i) {
        const auto& n = ops[i]->node;
        if (const auto* f = std::get_if<FilterOp>(&n)) {
            for (int c = 0; c < nsrc; ++c)
                if (expr_references(f->pred, c))
                    need[static_cast<std::size_t>(c)] = 1;
        } else if (const auto* s = std::get_if<SortByOp>(&n)) {
            int c = col_index(source_names, s->name);
            if (c >= 0) need[static_cast<std::size_t>(c)] = 1;
        } else if (const auto* t = std::get_if<TopkOp>(&n)) {
            int c = col_index(source_names, t->name);
            if (c >= 0) need[static_cast<std::size_t>(c)] = 1;
        }
    }

    std::vector<std::string> live;
    std::vector<std::int32_t> old_to_new(static_cast<std::size_t>(nsrc), -1);
    for (int c = 0; c < nsrc; ++c)
        if (need[static_cast<std::size_t>(c)]) {
            old_to_new[static_cast<std::size_t>(c)] =
                static_cast<std::int32_t>(live.size());
            live.push_back(source_names[static_cast<std::size_t>(c)]);
        }
    if (static_cast<int>(live.size()) == nsrc) return ops;  // nothing to prune

    std::vector<std::shared_ptr<const LazyOp>> out;
    out.reserve(ops.size() + 1);
    out.push_back(std::make_shared<LazyOp>(LazyOp{SelectOp{live}}));
    for (const auto& op : ops) {
        if (const auto* f = std::get_if<FilterOp>(&op->node)) {
            out.push_back(std::make_shared<LazyOp>(
                LazyOp{FilterOp{expr_remap_cols(f->pred, old_to_new)}}));
        } else {
            out.push_back(op);  // sort/slice/tail/topk/sample/select: by name
        }
    }
    return out;
}

// Fuse [filter]* [with_column]* [select] into one pass: one AND-ed mask, gather
// only the columns the outputs read, one CSE-fused eval_many - no intermediate
// frame. nullopt for any other shape, or a with_column that replaces a column
// or reads another with_column; the caller then runs op-by-op.
std::optional<DataFrame> try_fuse_map(
    const DataFrame& src,
    const std::vector<std::shared_ptr<const LazyOp>>& ops) {
    std::vector<const FilterOp*> filters;
    std::vector<const WithColumnOp*> withs;
    const SelectOp* sel = nullptr;
    for (const auto& op : ops) {
        const auto& n = op->node;
        if (const auto* f = std::get_if<FilterOp>(&n)) {
            if (!withs.empty() || sel) return std::nullopt;
            filters.push_back(f);
        } else if (const auto* w = std::get_if<WithColumnOp>(&n)) {
            if (sel) return std::nullopt;
            withs.push_back(w);
        } else if (const auto* s = std::get_if<SelectOp>(&n)) {
            if (sel) return std::nullopt;
            sel = s;
        } else {
            return std::nullopt;
        }
    }
    if (!sel) return std::nullopt;

    const std::int32_t nsrc = static_cast<std::int32_t>(src.columns.size());
    // Only append-new with_columns that read source columns; see the contract.
    for (const WithColumnOp* w : withs) {
        if (col_index(src.names, w->name) >= 0) return std::nullopt;
        for (std::int32_t j = nsrc;
             j < nsrc + static_cast<std::int32_t>(withs.size()); ++j)
            if (expr_references(w->expr, j)) return std::nullopt;
    }

    std::vector<Expr> outs;
    outs.reserve(sel->names.size());
    for (const std::string& nm : sel->names) {
        const Expr* we = nullptr;
        for (const WithColumnOp* w : withs)
            if (w->name == nm) we = &w->expr;
        if (we) {
            outs.push_back(*we);
        } else {
            int c = col_index(src.names, nm);
            if (c < 0) return std::nullopt;
            outs.push_back(expr_col(c));
        }
    }

    std::vector<char> need(static_cast<std::size_t>(nsrc), 0);
    for (const Expr& e : outs)
        for (std::int32_t c = 0; c < nsrc; ++c)
            if (expr_references(e, c)) need[static_cast<std::size_t>(c)] = 1;
    std::vector<std::int32_t> old_to_new(static_cast<std::size_t>(nsrc), -1);
    DataFrame sub;
    for (std::int32_t c = 0; c < nsrc; ++c)
        if (need[static_cast<std::size_t>(c)]) {
            old_to_new[static_cast<std::size_t>(c)] =
                static_cast<std::int32_t>(sub.columns.size());
            sub.names.push_back(src.names[static_cast<std::size_t>(c)]);
            sub.columns.push_back(
                src.columns[static_cast<std::size_t>(c)].share());
        }

    if (!filters.empty()) {
        // A trivial `col <cmp> scalar` filter runs as a direct Series kernel;
        // only a compound predicate goes through the expression compiler.
        auto mask_of = [&](const Expr& p) -> Series {
            std::int32_t c;
            CmpOp op;
            Scalar rhs;
            if (expr_as_col_cmp(p, &c, &op, &rhs))
                return src.columns[static_cast<std::size_t>(c)].compare(op,
                                                                        rhs);
            return eval(p, column_ptrs(src.columns));
        };
        Series mask = mask_of(filters[0]->pred);
        for (std::size_t i = 1; i < filters.size(); ++i)
            mask = mask & mask_of(filters[i]->pred);
        sub = sub.filter(mask);
    }

    // Bare columns pass through zero-copy, `col <op> col` runs a direct Series
    // kernel, and only the rest go through the CSE-fused compiler.
    std::vector<Expr> computed;
    for (const Expr& e : outs) {
        std::int32_t a, b;
        BinaryOp op;
        if (expr_col_index(e) < 0 && !expr_as_col_binary(e, &op, &a, &b))
            computed.push_back(expr_remap_cols(e, old_to_new));
    }
    std::vector<Series> comp =
        computed.empty() ? std::vector<Series>{}
                         : eval_many(computed, column_ptrs(sub.columns));
    DataFrame out;
    out.names = sel->names;
    out.columns.reserve(outs.size());
    std::size_t ci = 0;
    for (const Expr& e : outs) {
        std::int32_t c = expr_col_index(e), a, b;
        BinaryOp op;
        if (c >= 0) {
            out.columns.push_back(
                sub.columns[static_cast<std::size_t>(old_to_new[c])].share());
        } else if (expr_as_col_binary(e, &op, &a, &b)) {
            const Series& x =
                sub.columns[static_cast<std::size_t>(old_to_new[a])];
            const Series& y =
                sub.columns[static_cast<std::size_t>(old_to_new[b])];
            switch (op) {
                case BinaryOp::Add:
                    out.columns.push_back(x.add(y));
                    break;
                case BinaryOp::Sub:
                    out.columns.push_back(x.sub(y));
                    break;
                case BinaryOp::Mul:
                    out.columns.push_back(x.mul(y));
                    break;
                case BinaryOp::Div:
                    out.columns.push_back(x.div(y));
                    break;
            }
        } else {
            out.columns.push_back(std::move(comp[ci++]));
        }
    }
    return out;
}

// Whole-column execution over a resident frame (map plans fuse via
// try_fuse_map). nullopt for an op with no whole-column form; caller streams.
std::optional<DataFrame> run_ops_in_memory(
    DataFrame df, const std::vector<std::shared_ptr<const LazyOp>>& ops) {
    if (auto fused = try_fuse_map(df, ops)) return fused;
    for (const auto& op : ops) {
        bool ok = true;
        DataFrame next = std::visit(
            overloaded{
                [&](const FilterOp& o) {
                    return df.filter(eval(o.pred, column_ptrs(df.columns)));
                },
                [&](const WithColumnOp& o) {
                    return df.with_column(
                        o.name, eval(o.expr, column_ptrs(df.columns)));
                },
                [&](const SelectOp& o) { return df.select(o.names); },
                [&](const RenameOp& o) { return df.rename(o.names); },
                [&](const SliceOp& o) { return df.slice(o.offset, o.len); },
                [&](const TailOp& o) { return df.tail(o.n); },
                [&](const DropNullsOp&) { return df.drop_nulls(); },
                [&](const FillNullOp& o) { return df.fill_null(o.value); },
                [&](const SortByOp& o) {
                    return df.sort_by(o.name, o.descending);
                },
                [&](const UniqueOp&) { return df.unique(); },
                [&](const SampleOp& o) { return df.sample(o.n, o.seed); },
                [&](const TopkOp& o) {
                    return df.topk(o.name, o.k, o.largest);
                },
                [&](const WithRowIndexOp& o) {
                    return df.with_row_index(o.name);
                },
                [&](const NullCountOp&) { return df.null_count(); },
                [&](const GroupByOp& o) {
                    if (o.dyn.empty()) return df.group_by(o.keys, o.aggs);
                    // DataFrame::group_by drops the dyn side-channel; run the
                    // same AggState path the streaming cursor uses.
                    LoweredGroupAggs lowered = lower_group_aggs(o.aggs);
                    AggStatePtr state = agg_new(lowered.specs, o.dyn);
                    agg_accumulate_chunk(*state, df, o.keys,
                                         lowered.value_names, o.dyn_prefix);
                    return agg_finalize(*state, o.keys);
                },
                [&](const auto&) {
                    ok = false;
                    return DataFrame{};
                }},
            op->node);
        if (!ok) return std::nullopt;
        df = std::move(next);
    }
    return df;
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
                                                       o.keys, o.aggs, budget,
                                                       o.dyn, o.dyn_prefix);
            },
            [&](const SortByOp& o) -> std::unique_ptr<Cursor> {
                return std::make_unique<SortMergeCursor>(
                    std::move(in), sch, o.name, o.descending, budget);
            },
            [&](const UniqueOp&) -> std::unique_ptr<Cursor> {
                return std::make_unique<UniqueCursor>(std::move(in), sch,
                                                      budget);
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

Schema InMemorySource::schema() const {
    Schema s;
    s.names = frame_->names;
    s.types.reserve(frame_->columns.size());
    for (const Series& c : frame_->columns) s.types.push_back(c.type());
    return s;
}

ScanResult InMemorySource::scan(const ScanRequest& req) const {
    // Honor projection zero-copy: share only the requested columns, in the
    // requested order. Every filter stays No - the whole-column engine applies
    // them (a resident source usually takes the as_frame() fast path anyway).
    std::shared_ptr<const DataFrame> src = frame_;
    if (!req.projection.empty()) {
        DataFrame proj;
        proj.names = req.projection;
        proj.columns.reserve(req.projection.size());
        for (const std::string& nm : req.projection) {
            const int c = col_index(frame_->names, nm);
            proj.columns.push_back(
                frame_->columns[static_cast<std::size_t>(c)].share());
        }
        src = std::make_shared<const DataFrame>(std::move(proj));
    }
    ScanResult r;
    r.cursor = std::make_unique<InMemoryCursor>(std::move(src));
    r.filters.assign(req.filters.size(), Pushed::No);
    return r;
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

LazyFrame LazyFrame::fill_null(Scalar value) const {
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
    return group_by(std::vector<std::string>{std::move(key)}, std::move(aggs));
}

LazyFrame LazyFrame::group_by(std::vector<std::string> keys,
                              std::vector<GroupAgg> aggs,
                              std::vector<AggDynSpec> dyn,
                              std::string dyn_prefix) const {
    auto ops = ops_;
    ops.push_back(std::make_shared<LazyOp>(
        LazyOp{GroupByOp{std::move(keys), std::move(aggs), std::move(dyn),
                         std::move(dyn_prefix)}}));
    return with_ops(std::move(ops));
}

namespace {
// A computed KEY becomes the output key column, so it must be named as the
// eager DataFrame::group_by(Expr) path does ("key" for one key, "key<k>" for N)
// or lazy and eager schemas diverge; value/by temps stay hidden ("__gb_").
LazyFrame desugar_expr_group_by(const LazyFrame& self, std::vector<Expr> keys,
                                std::vector<AggExprSpec> aggs,
                                bool single_key) {
    LazyFrame lf = self;
    std::vector<std::string> sch = lf.schema();
    int tmp = 0;
    auto resolve_val = [&](const Expr& e, const char* prefix) -> std::string {
        const std::int32_t idx = expr_col_index(e);
        if (idx >= 0 && static_cast<std::size_t>(idx) < sch.size())
            return sch[static_cast<std::size_t>(idx)];
        std::string name =
            std::string("__gb_") + prefix + std::to_string(tmp++);
        lf = lf.with_column(name, e);
        sch.push_back(name);
        return name;
    };

    std::vector<std::string> key_names;
    key_names.reserve(keys.size());
    for (std::size_t k = 0; k < keys.size(); ++k) {
        const std::int32_t idx = expr_col_index(keys[k]);
        if (idx >= 0 && static_cast<std::size_t>(idx) < sch.size()) {
            key_names.push_back(sch[static_cast<std::size_t>(idx)]);
            continue;
        }
        std::string name = single_key ? "key" : "key" + std::to_string(k);
        lf = lf.with_column(name, keys[k]);
        sch.push_back(name);
        key_names.push_back(std::move(name));
    }

    std::vector<GroupAgg> gaggs;
    gaggs.reserve(aggs.size());
    for (const AggExprSpec& a : aggs) {
        GroupAgg g;
        g.op = from_agg_op(a.op);
        g.out = a.out;
        g.param = a.param;
        if (a.op != AggOp::Count) g.column = resolve_val(a.value, "v");
        if (a.op == AggOp::ArgMax) g.by = resolve_val(a.by, "by");
        gaggs.push_back(std::move(g));
    }
    return lf.group_by(std::move(key_names), std::move(gaggs));
}
}  // namespace

LazyFrame LazyFrame::group_by(Expr key, std::vector<AggExprSpec> aggs) const {
    return desugar_expr_group_by(*this, std::vector<Expr>{std::move(key)},
                                 std::move(aggs), /*single_key=*/true);
}

LazyFrame LazyFrame::group_by(std::vector<Expr> keys,
                              std::vector<AggExprSpec> aggs) const {
    return desugar_expr_group_by(*this, std::move(keys), std::move(aggs),
                                 /*single_key=*/false);
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
    for (const auto& op : pushdown_projections(
             source_->names(), pushdown_predicates(source_->names(), ops_)))
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

coro::AsyncGenerator<DataFrame> LazyFrame::stream(
    std::int64_t morsel_rows) const {
    const std::vector<std::string> names = source_->names();
    auto ops = pushdown_projections(names, pushdown_predicates(names, ops_));
    // 0 resolves to auto (~1/3 RAM), same policy as View.
    const std::uint64_t budget = resolve_spill_budget(memory_budget_);
    const std::int64_t eff_rows =
        morsel_rows > 0 ? morsel_rows : DEFAULT_MORSEL_ROWS;

    // A leading Select is a pure source projection: pushdown_projections emits
    // one with the filter col-refs already remapped into it, and a user's own
    // leading select is one too. Push it into the scan so the source harvests
    // only those columns; the source returns them in this exact order, so the
    // now-identity Select and the remapped filters stay positionally aligned.
    std::vector<std::string> projection;
    std::size_t first = 0;
    if (!ops.empty())
        if (const auto* s = std::get_if<SelectOp>(&ops.front()->node)) {
            projection = s->names;
            first = 1;
        }

    // Candidate filters: the contiguous run right after the optional
    // projection. Their col-refs are positional against the scan's column set
    // (projection when present, else the source schema), which is exactly what
    // scan() resolves them against.
    ScanRequest req;
    req.projection = projection;
    req.memory_budget = budget;
    std::vector<std::size_t> cand_pos;
    for (std::size_t i = first; i < ops.size(); ++i) {
        const auto* f = std::get_if<FilterOp>(&ops[i]->node);
        if (!f) break;
        req.filters.push_back(f->pred);
        cand_pos.push_back(i);
    }

    ScanResult r = source_->scan(req);

    // Drop every candidate the source applied exactly; the engine re-applies
    // No/Inexact (and any filter deeper in the plan).
    std::vector<bool> drop(ops.size(), false);
    for (std::size_t k = 0; k < cand_pos.size() && k < r.filters.size(); ++k)
        if (r.filters[k] == Pushed::Exact) drop[cand_pos[k]] = true;

    std::unique_ptr<Cursor> cur = std::move(r.cursor);
    std::vector<std::string> sch = projection.empty() ? names : projection;
    for (std::size_t i = 0; i < ops.size(); ++i) {
        if (drop[i]) continue;
        cur = make_cursor(*ops[i], std::move(cur), sch, budget);
        sch = out_schema(*ops[i], std::move(sch));
    }
    while (auto m = co_await cur->next(eff_rows)) {
        DataFrame chunk =
            frame_from_morsel(std::move(*m), sch, cur->out_names());
        co_yield std::move(chunk);
    }
}

coro::CoroTask<DataFrame> LazyFrame::collect(std::int64_t morsel_rows) const {
    // A resident source runs whole-column, matching eager. An explicit
    // morsel_rows or memory_budget asks to stream/spill instead.
    if (morsel_rows <= 0 && memory_budget_ == 0) {
        if (const DataFrame* f = source_->as_frame()) {
            auto ops = pushdown_projections(
                source_->names(), pushdown_predicates(source_->names(), ops_));
            DataFrame start;
            start.names = f->names;
            start.columns.reserve(f->columns.size());
            for (const Series& c : f->columns)
                start.columns.push_back(c.share());  // zero-copy, move-only
            if (auto out = run_ops_in_memory(std::move(start), ops))
                co_return std::move(*out);
        }
    }
    co_return co_await drain_stream(stream(morsel_rows));
}

LoweredGroupAggs lower_group_aggs(const std::vector<GroupAgg>& aggs) {
    LoweredGroupAggs out;
    ankerl::unordered_dense::map<std::string, std::int32_t> dedup;
    auto resolve = [&](const std::string& name) -> std::int32_t {
        auto it = dedup.find(name);
        if (it != dedup.end()) return it->second;
        const std::int32_t idx =
            static_cast<std::int32_t>(out.value_names.size());
        out.value_names.push_back(name);
        dedup.emplace(name, idx);
        return idx;
    };
    out.specs.reserve(aggs.size());
    for (const GroupAgg& a : aggs) {
        AggSpec sp;
        sp.op = to_agg_op(a.op);
        sp.out = a.out;
        sp.param = a.param;
        sp.value_col = sp.op == AggOp::Count ? -1 : resolve(a.column);
        if (agg_uses_by_col(sp.op)) sp.by_col = resolve(a.by);
        out.specs.push_back(std::move(sp));
    }
    return out;
}

void agg_accumulate_chunk(AggState& state, const DataFrame& frame,
                          const std::vector<std::string>& keys,
                          const std::vector<std::string>& value_names,
                          const std::string& dyn_prefix) {
    std::vector<const Series*> kcols;
    kcols.reserve(keys.size());
    for (const std::string& k : keys) {
        const std::int64_t ki = frame.column_index(k);
        if (ki < 0) throw std::out_of_range("group_by: no column named " + k);
        kcols.push_back(&frame.columns[static_cast<std::size_t>(ki)]);
    }
    std::vector<const Series*> vcols;
    vcols.reserve(value_names.size());
    for (const std::string& v : value_names) {
        const std::int64_t vi = frame.column_index(v);
        if (vi < 0) throw std::out_of_range("group_by: no column named " + v);
        vcols.push_back(&frame.columns[static_cast<std::size_t>(vi)]);
    }
    // Pass the row count explicitly: an empty key list (a global reduce to one
    // group) carries no key column to infer the length from.
    if (dyn_prefix.empty()) {
        agg_accumulate(state, kcols, vcols, 0, frame.num_rows());
        return;
    }
    std::vector<AggDynInput> dcols;
    for (std::size_t i = 0; i < frame.names.size(); ++i)
        if (frame.names[i].rfind(dyn_prefix, 0) == 0)
            dcols.push_back(
                {frame.names[i].substr(dyn_prefix.size()), &frame.columns[i]});
    agg_accumulate(state, kcols, vcols, dcols, 0, frame.num_rows());
}

coro::CoroTask<AggStatePtr> LazyFrame::collect_group_state(
    std::vector<std::string> keys, std::vector<GroupAgg> aggs,
    std::vector<AggDynSpec> dyn, std::string dyn_prefix,
    std::int64_t morsel_rows) const {
    LoweredGroupAggs lowered = lower_group_aggs(aggs);
    AggStatePtr state = agg_new(lowered.specs, dyn);
    auto gen = stream(morsel_rows);
    while (auto df = co_await gen.next())
        agg_accumulate_chunk(*state, *df, keys, lowered.value_names,
                             dyn.empty() ? std::string() : dyn_prefix);
    co_return state;
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
