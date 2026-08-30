#ifndef DFTRACER_UTILS_DATAFRAME_LAZYFRAME_H
#define DFTRACER_UTILS_DATAFRAME_LAZYFRAME_H

#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dftracer::utils::dataframe {

/// A chunk of columns flowing through the lazy pipeline. Carries no names - the
/// schema lives in the plan. Columns are FLAT.
struct Morsel {
    std::vector<Series> columns;
    std::int64_t rows = 0;
};

/// A stateful reader over one Source. next() returns the next morsel, or
/// nullopt at end; `max_rows` is a size hint. The engine pulls it
/// synchronously. A producer that wants async/prefetch/backpressure does that
/// inside next() (e.g. hand back a morsel a background reader prepared) - the
/// engine stays synchronous.
class Cursor {
   public:
    virtual ~Cursor() = default;
    virtual std::optional<Morsel> next(std::int64_t max_rows) = 0;
    /// Output column names, when they are only known after producing (a
    /// data-dependent schema like pivot/to_dummies). nullopt means the plan's
    /// static schema is authoritative. Valid only after the cursor is drained.
    virtual std::optional<std::vector<std::string>> out_names() const {
        return std::nullopt;
    }
};

/// A data source for a lazy query. Immutable: names() reports the schema and
/// open() hands out a fresh Cursor, so one Source can back many collect()s.
/// Implement these two to plug any producer (a file, another engine, a trace
/// scan) into LazyFrame.
class Source {
   public:
    virtual ~Source() = default;
    virtual std::vector<std::string> names() const = 0;
    virtual std::unique_ptr<Cursor> open() const = 0;
};

/// A Source over an already-materialized in-memory frame.
class InMemorySource : public Source {
   public:
    explicit InMemorySource(DataFrame frame);
    std::vector<std::string> names() const override;
    std::unique_ptr<Cursor> open() const override;

   private:
    std::shared_ptr<const DataFrame> frame_;
};

/// A deferred query over a Source. Builder methods record ops and return a new
/// LazyFrame; nothing runs until collect(). A filter/with_column expr's col(i)
/// refers to the i-th column of the frame at that point.
class LazyOp;

class LazyFrame {
   public:
    static LazyFrame scan(std::shared_ptr<const Source> source);

    LazyFrame select(std::vector<std::string> names) const;
    LazyFrame filter(Expr predicate) const;
    LazyFrame with_column(std::string name, Expr expr) const;
    LazyFrame rename(std::vector<std::string> names) const;
    LazyFrame slice(std::int64_t offset, std::int64_t len) const;
    LazyFrame head(std::int64_t n) const;
    LazyFrame tail(std::int64_t n) const;
    LazyFrame drop_nulls() const;
    LazyFrame fill_null(dftu_scalar value) const;
    /// Fill nulls with a natural C++ value; converts to each column's type.
    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    LazyFrame fill_null(T value) const {
        return fill_null(to_scalar(value));
    }
    LazyFrame with_row_index(std::string name) const;
    /// A one-row frame of each column's null count.
    LazyFrame null_count() const;
    /// Expand a List `column`: each element becomes its own row.
    LazyFrame explode(std::string column) const;
    /// Reshape wide -> long: keep `id_vars`, stack `value_vars` into a
    /// variable/value column pair.
    LazyFrame unpivot(std::vector<std::string> id_vars,
                      std::vector<std::string> value_vars) const;
    /// The `k` rows with the largest (or smallest) `name` values.
    LazyFrame topk(std::string name, std::int64_t k, bool largest = true) const;
    /// Group by `key` and compute each aggregate (streaming: one partial state,
    /// bounded by the group count).
    LazyFrame group_by(std::string key, std::vector<GroupAgg> aggs) const;
    /// Sort by `name`. Buffers input (external-merge spill is a follow-up).
    LazyFrame sort_by(std::string name, bool descending = false) const;
    /// Distinct rows, first occurrence. Buffers input (spill is a follow-up).
    LazyFrame unique() const;
    /// Alias of unique().
    LazyFrame drop_duplicates() const;
    /// A deterministic n-row sample. Buffers input (reservoir is a follow-up).
    LazyFrame sample(std::int64_t n, std::uint64_t seed = 0) const;
    /// One Bool column: true where the whole row is duplicated. Buffers input.
    LazyFrame is_duplicated() const;
    /// One Bool column: true where the whole row is unique. Buffers input.
    LazyFrame is_unique() const;
    /// Tumbling/sliding time-window aggregation. Buffers input.
    LazyFrame group_by_dynamic(std::string time_col, std::int64_t every,
                               std::int64_t period, std::vector<GroupAgg> aggs,
                               std::int64_t origin = 0,
                               bool origin_min = false) const;
    /// unpivot alias.
    LazyFrame melt(std::vector<std::string> id_vars,
                   std::vector<std::string> value_vars) const;
    /// Reshape long -> wide. Buffers input; output columns are data-dependent
    /// (one per distinct `on` value), so schema() is empty until collect().
    LazyFrame pivot(std::string index, std::string on, std::string values,
                    std::string agg = "first") const;
    /// One-hot encode `column`. Buffers input; output columns are
    /// data-dependent, so schema() is empty until collect().
    LazyFrame to_dummies(std::string column) const;
    /// Per-column summary statistics. Buffers input; output columns are
    /// data-dependent, so schema() is empty until collect().
    LazyFrame describe() const;

    /// Output column names without running the query. Empty for a plan ending
    /// in a data-dependent op (pivot/to_dummies/describe).
    std::vector<std::string> schema() const;

    /// The optimized plan as text (source then one op per line), for
    /// introspection and tests.
    std::string explain() const;

    /// Run the pipeline and materialize the surviving rows; `morsel_rows` is
    /// the scan chunk size.
    DataFrame collect(std::int64_t morsel_rows = 65536) const;

   private:
    LazyFrame(std::shared_ptr<const Source> source,
              std::vector<std::shared_ptr<const LazyOp>> ops)
        : source_(std::move(source)), ops_(std::move(ops)) {}
    std::shared_ptr<const Source> source_;
    std::vector<std::shared_ptr<const LazyOp>> ops_;
};

/// Free-function form of DataFrame::lazy(), for `lazy(df)` call sites.
LazyFrame lazy(DataFrame frame);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_LAZYFRAME_H
