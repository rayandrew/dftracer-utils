#ifndef DFTRACER_UTILS_DATAFRAME_DATAFRAME_H
#define DFTRACER_UTILS_DATAFRAME_DATAFRAME_H

#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/coro/async_generator.h>
#include <dftracer/utils/dataframe/agg_expr.h>
#include <dftracer/utils/dataframe/series.h>
#include <dftracer/utils/dataframe/types.h>
#include <dftracer/utils/query/query.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
struct ArrowSchema;
struct ArrowArray;
#endif

namespace dftracer::utils::dataframe {

#ifdef DFTRACER_UTILS_ENABLE_ARROW
class OwnedArrow;
#endif

/// Aggregate function selector for group_by / group_by_dynamic / pivot. First
/// and Last take the group's first / last non-null value in row order (an
/// order-independent merge, so the parallel/distributed path is exact).
enum class Agg {
    Sum,
    Min,
    Max,
    Count,
    Mean,
    Var,
    Std,
    Skew,
    Kurt,
    First,
    Last,
    Pct,
    Hist,
    ArgMax,
    SumSq,
    SetUnion
};

/// Canonical lowercase name of `agg` (the string the C ABI accepts).
const char* to_string(Agg agg) noexcept;

/// Parse a canonical aggregate name (as `to_string` produces) into an Agg.
/// Throws std::out_of_range on an unknown name.
Agg agg_from_string(std::string_view name);

/// One aggregate in a group_by: `op` selects the function (First/Last are pivot
/// only); `column` is the value column (ignored for Count, which is the group
/// size); `out` is the result column name.
struct GroupAgg {
    Agg op = Agg::Count;
    std::string column;
    std::string out;
    double param = 0.0;  ///< Pct: the quantile level q in [0, 1]
    std::string by;      ///< ArgMax: the column maximized; unused otherwise
};

class LazyFrame;         // dataframe/lazyframe.h

/// A named ordered set of columns (the RecordBatch / DataChunk analogue). The
/// eager methods return a new DataFrame (move-only); projection ops share the
/// underlying column buffers zero-copy, row ops build new columns.
struct DataFrame {
    std::vector<std::string> names;
    std::vector<Series> columns;

    std::size_t num_columns() const noexcept { return columns.size(); }
    std::int64_t num_rows() const noexcept {
        return columns.empty() ? 0 : columns.front().length();
    }

    /// Index of the column named `name`, or -1 if absent.
    std::int64_t column_index(std::string_view name) const;
    /// A column by name, sharing its buffers zero-copy; invalid if absent.
    Series column(std::string_view name) const;

    /// Begin a deferred query over this frame; include dataframe/lazyframe.h.
    LazyFrame lazy() const;

    /// Yield fixed-size row slices of this frame (0 = DEFAULT_MORSEL_ROWS).
    /// Zero-copy: each chunk shares this frame's column buffers.
    coro::AsyncGenerator<DataFrame> stream(std::int64_t morsel_rows = 0) const;

    DataFrame take(const std::vector<std::int64_t>& indices) const;
    DataFrame filter(const Series& mask) const;
    DataFrame slice(std::int64_t offset, std::int64_t len) const;
    DataFrame head(std::int64_t n) const;
    DataFrame tail(std::int64_t n) const;
    DataFrame reverse() const;
    DataFrame sort_by(const std::string& name, bool descending = false) const;
    /// Stable lexicographic sort by several key columns (nulls last).
    DataFrame sort_by_multi(const std::vector<std::string>& by,
                            bool descending = false) const;
    /// Per-column direction form: `descending[i]` applies to `by[i]`; a single
    /// flag broadcasts to every key.
    DataFrame sort_by_multi(const std::vector<std::string>& by,
                            const std::vector<bool>& descending) const;
    DataFrame topk(const std::string& name, std::int64_t k,
                   bool largest = true) const;

    /// Drop every row that is null in any column.
    DataFrame drop_nulls() const;
    /// Fill nulls in every column with `value` (columns whose type rejects it
    /// are left unchanged).
    DataFrame fill_null(dftu_scalar value) const;
    /// Fill nulls with a natural C++ value (fill_null(0)); the value converts
    /// to each column's element type.
    template <class T, class = std::enable_if_t<std::is_arithmetic_v<T>>>
    DataFrame fill_null(T value) const {
        return fill_null(to_scalar(value));
    }
    /// Distinct rows, keeping the first occurrence (all columns as the key).
    DataFrame unique() const;
    /// Alias of unique().
    DataFrame drop_duplicates() const;
    /// A deterministic n-row sample (mix64(index + seed) bottom-n).
    DataFrame sample(std::int64_t n, std::uint64_t seed = 0) const;
    /// Prepend an Int64 row-index column named `name`.
    DataFrame with_row_index(const std::string& name) const;
    /// Per-column summary statistics (one row per statistic).
    DataFrame describe() const;
    /// A one-row frame of each column's null count.
    DataFrame null_count() const;
    /// Bool mask (per row): true where the whole row is duplicated.
    Series is_duplicated() const;
    /// Bool mask (per row): true where the whole row is unique.
    Series is_unique() const;

    /// Evaluate a compiled query as a per-row bit-packed Bool mask (for
    /// filter()). Throws std::runtime_error if the predicate has no columnar
    /// lowering for this frame (pattern match, ordered string compare, or a
    /// referenced field absent here).
    Series mask(const query::Query& q) const;

    DataFrame select(const std::vector<std::string>& names) const;
    DataFrame rename(const std::vector<std::string>& new_names) const;
    DataFrame with_column(const std::string& name, const Series& col) const;

    /// Group rows by column `key` (first-seen order) and compute each
    /// aggregate.
    DataFrame group_by(const std::string& key,
                       const std::vector<GroupAgg>& aggs) const;
    /// Group by N key columns (a composite key: hashed and compared
    /// column-by-column, each keeping its own type) and compute each
    /// aggregate.
    DataFrame group_by(const std::vector<std::string>& keys,
                       const std::vector<GroupAgg>& aggs) const;
    /// Group by an expression key and compute each expression aggregate in one
    /// CSE'd, pruned pass (see group_agg_expr). A bare column-ref key uses that
    /// column's name for the result; a computed key is named "key".
    DataFrame group_by(const Expr& key,
                       const std::vector<AggExprSpec>& aggs) const;
    /// Group by N expression keys; each bare column-ref key uses that column's
    /// name, a computed key is named "key<i>".
    DataFrame group_by(const std::vector<Expr>& keys,
                       const std::vector<AggExprSpec>& aggs) const;

    /// Reshape wide -> long: keep `id_vars`, stack `value_vars` into a
    /// `variable`/`value` column pair. `melt` is an alias.
    DataFrame unpivot(const std::vector<std::string>& id_vars,
                      const std::vector<std::string>& value_vars) const;
    DataFrame melt(const std::vector<std::string>& id_vars,
                   const std::vector<std::string>& value_vars) const;

    /// Expand a List `column`: each list element becomes its own row (other
    /// columns repeated); an empty/null list yields one null row.
    DataFrame explode(const std::string& column) const;

    /// One-hot encode `column`: replace it with one Int8 column per distinct
    /// value (sorted), named `<column>_<value>`.
    DataFrame to_dummies(const std::string& column) const;

    /// Reshape long -> wide: rows are the distinct `index` values (sorted), one
    /// value column per distinct `columns` value (sorted), each cell `values`
    /// aggregated over the matching rows. `agg` is the collision reducer
    /// (first|last|sum|min|max|mean; mean yields Float64).
    DataFrame pivot(const std::string& index, const std::string& on,
                    const std::string& values,
                    const std::string& agg = "first") const;
    /// pivot with a typed collision reducer (any Agg, including First/Last).
    DataFrame pivot(const std::string& index, const std::string& on,
                    const std::string& values, Agg agg) const;

    /// Tumbling/sliding time-window aggregation over an ascending Int64
    /// `time_col`: windows start at the first time floored to a multiple of
    /// `every`, stride by `every`, and each covers `[start, start + period)`
    /// (`period <= 0` means "= every"). One row per non-empty window, a leading
    /// Int64 `time_col` window-start column plus each aggregate.
    DataFrame group_by_dynamic(const std::string& time_col, std::int64_t every,
                               std::int64_t period,
                               const std::vector<GroupAgg>& aggs,
                               std::int64_t origin = 0,
                               bool origin_min = false) const;

#ifdef DFTRACER_UTILS_ENABLE_ARROW
    /// Export this frame through the Arrow C Data Interface as a struct array
    /// with one child per column (zero copy: the exported buffers alias the
    /// columns' and are kept alive by the returned owner's release callbacks).
    /// Include <dftracer/utils/dataframe/arrow.h> for the OwnedArrow
    /// definition.
    OwnedArrow to_arrow() const;

#ifdef DFTRACER_UTILS_ENABLE_ARROW_IPC
    /// Serialize this frame to an Arrow IPC stream (schema message + one record
    /// batch + EOS) as bytes any Arrow IPC reader can open - no pyarrow. Throws
    /// std::runtime_error on an encode failure.
    std::vector<std::uint8_t> to_ipc() const;
#endif

    /// Import a STRUCT Arrow array (viewed through `schema`) as a DataFrame:
    /// one column per struct child, named by its child schema, wrapped zero
    /// copy. ADOPTS `array` (ownership of its buffers moves into the frame and
    /// the passed array is marked released, so the caller must not release it
    /// again); `schema` is only read. An empty frame on a non-struct or
    /// unsupported child type.
    static DataFrame from_arrow(const ArrowSchema* schema,
                                const ArrowArray* array);
#endif
};

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_DATAFRAME_H
