#ifndef DFTRACER_UTILS_DATAFRAME_BATCH_OPS_H
#define DFTRACER_UTILS_DATAFRAME_BATCH_OPS_H

#include <dftracer/utils/dataframe/dataframe.h>

#include <cstdint>
#include <string>
#include <vector>

// Frame-level operations on a DataFrame: the columnar primitives a native
// (distributed) dataframe is built from. Row ops (take/filter/slice/sort) build
// new columns; projection ops (select/rename/with_column) share buffers
// zero-copy. Pure columnar - no knowledge of any domain schema.
namespace dftracer::utils::dataframe {

/// Gather the rows of every column at `indices` into a new DataFrame (same
/// names).
DataFrame take(const DataFrame& b, const std::vector<std::int64_t>& indices);

/// Keep the rows where the bit-packed Bool `mask` (length == b.num_rows()) is
/// true, across every column.
DataFrame filter(const DataFrame& b, const Series& mask);

/// The `len` rows starting at `offset` (clamped to the row count), across every
/// column.
DataFrame slice(const DataFrame& b, std::int64_t offset, std::int64_t len);

/// The first `n` rows (clamped), across every column.
DataFrame head(const DataFrame& b, std::int64_t n);

/// Project the named columns, in the given order, sharing buffers zero-copy.
/// Throws std::out_of_range if a name is absent.
DataFrame select(const DataFrame& b, const std::vector<std::string>& names);

/// Rename columns positionally (`new_names` aligns to the existing columns),
/// sharing buffers zero-copy. Throws std::invalid_argument on a count mismatch.
DataFrame rename(const DataFrame& b, const std::vector<std::string>& new_names);

/// Add `col` under `name` (replacing an existing column of that name), sharing
/// the other columns zero-copy. `col` must match the batch row count.
DataFrame with_column(const DataFrame& b, const std::string& name,
                      const Series& col);

/// Order every column by the values of column `name` (stable; nulls last),
/// ascending or `descending`. Throws std::out_of_range if `name` is absent.
DataFrame sort_by(const DataFrame& b, const std::string& name, bool descending);

/// The k rows with the largest (or smallest) values of column `name`, ordered
/// best-first. Throws std::out_of_range if `name` is absent.
DataFrame topk(const DataFrame& b, const std::string& name, std::int64_t k,
               bool largest);

/// How concat aligns the parts' columns.
enum class ConcatHow {
    Vertical,  ///< Parts must share a schema (same names/types/order); throws
               ///< std::invalid_argument on a mismatch.
    Diagonal   ///< Union the parts' columns: a column absent from a part is
               ///< null-filled, and a column whose type differs across parts is
               ///< promoted (mixed numeric -> Float64). A numeric/String or
               ///< Bool/other-type clash throws.
};

/// Vertically concatenate batches into one DataFrame. `how` picks strict
/// (Vertical) or schema-union (Diagonal) column alignment. An empty input
/// yields an empty DataFrame. Column order follows first appearance across the
/// parts.
DataFrame concat(const std::vector<const DataFrame*>& parts,
                 ConcatHow how = ConcatHow::Vertical);

/// Vertically concatenate columns of the same type into one FLAT column.
Series concat_columns(const std::vector<const Series*>& parts);

/// Group the rows of `b` by the values of column `key` (first-seen order) and
/// compute each aggregate. Re-aggregation over a materialized batch (e.g.
/// merging partial results); the scan-time path is the View. Throws
/// std::out_of_range on an unknown column/op.
DataFrame group_by(const DataFrame& b, const std::string& key,
                   const std::vector<GroupAgg>& aggs);
/// N-key form: a composite key over `keys` (hashed and compared
/// column-by-column, each keeping its own type).
DataFrame group_by(const DataFrame& b, const std::vector<std::string>& keys,
                   const std::vector<GroupAgg>& aggs);

/// Partition the rows into `n_parts` batches by a stable hash of the `keys`
/// columns (rows with equal keys always land in the same part). The shuffle
/// primitive for distributed group_by/join: hash-partition, ship parts, then
/// group_by/join each part locally. Throws std::out_of_range on an unknown
/// column; std::invalid_argument if `keys` is empty or `n_parts < 1`.
std::vector<DataFrame> hash_partition(const DataFrame& b,
                                      const std::vector<std::string>& keys,
                                      std::int64_t n_parts);

/// Drop every row that is null in ANY column (a combined not-null mask, then
/// filter).
DataFrame drop_nulls(const DataFrame& b);

/// Fill nulls in every column with `value`, converting it to each column's
/// element type; columns whose type does not accept the scalar (string/bool)
/// are shared unchanged.
DataFrame fill_null(const DataFrame& b, dftu_scalar value);

/// Distinct ROWS (keep first), hashing all columns as the composite key.
/// `drop_duplicates` is an alias.
DataFrame unique(const DataFrame& b);

/// Stable lexicographic sort by several key columns, ascending or `descending`
/// (nulls last in both directions). Throws std::out_of_range if a name is
/// absent.
DataFrame sort_by_multi(const DataFrame& b,
                        const std::vector<std::string>& names, bool descending);

/// The last `n` rows (clamped), across every column.
DataFrame tail(const DataFrame& b, std::int64_t n);

/// The rows in reverse order, across every column.
DataFrame reverse(const DataFrame& b);

/// A DETERMINISTIC sample of `n` rows (the rows whose mix64(index + seed) hash
/// is smallest, in ascending row order), across every column.
DataFrame sample(const DataFrame& b, std::int64_t n, std::uint64_t seed);

/// Prepend an Int64 column [0, num_rows) named `name`.
DataFrame with_row_index(const DataFrame& b, const std::string& name);

/// Per-column summary statistics: one row per statistic (count, null_count,
/// mean, std, min, max) and one Float64 column per numeric input column, plus a
/// leading "statistic" string column.
DataFrame describe(const DataFrame& b);

/// A one-row frame with one Int64 column per input column giving that column's
/// null count.
DataFrame null_count(const DataFrame& b);

/// Bit-packed Bool mask (length num_rows): true where the whole ROW is
/// duplicated (occurs more than once), hashing all columns.
Series is_duplicated(const DataFrame& b);

/// Bit-packed Bool mask (length num_rows): true where the whole ROW is unique
/// (occurs exactly once), hashing all columns.
Series is_unique(const DataFrame& b);

/// The distinct values of `v` and their counts, as a two-column frame `value`
/// (v's type) and `count` (Int64), most-frequent first.
DataFrame value_counts(const Series& v);

/// Reshape wide -> long: keep the `id_vars` columns and stack the `value_vars`
/// columns into two new columns `variable` (String, the source column name) and
/// `value` (the stacked values). The result has num_rows * value_vars rows. If
/// the value columns do not all share a type they must all be numeric and are
/// cast to a common type (Float64 if any is float, else Int64). `melt` is an
/// alias. Throws std::out_of_range on an unknown column; std::invalid_argument
/// on an empty `value_vars` or an unmixable set of value types.
DataFrame unpivot(const DataFrame& b, const std::vector<std::string>& id_vars,
                  const std::vector<std::string>& value_vars);

/// Expand a List column: every element of each row's list becomes its own row,
/// with the other columns repeated per element. An empty or null list yields
/// one row whose exploded value is null (polars semantics). Throws
/// std::out_of_range if `column` is absent; std::invalid_argument if it is not
/// a List column.
DataFrame explode(const DataFrame& b, const std::string& column);

/// One-hot encode `column`: replace it in place with one Int8 column per
/// distinct non-null value (sorted ascending for a stable column order), named
/// `<column>_<value>` and holding 1 where the row equals that value else 0. The
/// other columns pass through unchanged. Throws std::out_of_range if `column`
/// is absent.
DataFrame to_dummies(const DataFrame& b, const std::string& column);

/// Reshape long -> wide. Group rows by the value of `index_name`; the distinct
/// values of `columns_name` become new value columns (named by their value via
/// cell_to_string); each cell is `values_name` aggregated over the rows sharing
/// that (index, column) pair. Output rows are the distinct index values sorted
/// ascending (non-null); value columns are the distinct column values sorted
/// ascending, filled with the aggregate or null where a pair is absent. `agg`
/// is the collision reducer: "first" / "last" keep that row's value, while
/// "sum"/"min"/"max"/"mean" reduce numerically (mean yields Float64). Throws
/// std::out_of_range on an unknown column; std::out_of_range on an unknown agg.
DataFrame pivot(const DataFrame& b, const std::string& index_name,
                const std::string& columns_name, const std::string& values_name,
                const std::string& agg);

/// Tumbling/sliding time-window aggregation over an ASCENDING Int64 `time_col`.
/// Windows start at the first time value floored to a multiple of `every` and
/// stride by `every`; each covers `[start, start + period)`. `period <= 0`
/// means
/// "= every" (tumbling); `period > every` overlaps (sliding). Emits one row per
/// non-empty window: a leading Int64 `time_col` = the window start, then one
/// column per aggregate (the same sum|min|max|count|mean vocabulary as
/// group_by), named by GroupAgg.out. Throws std::invalid_argument if `every <=
/// 0` or `time_col` is not Int64; std::out_of_range on an unknown column/op.
/// `origin` anchors the window grid at `origin + k*every` (default 0 = the
/// classic ts-floored grid); pass a window's begin to align buckets to it.
/// `origin_min` overrides `origin` with the minimum time value (buckets begin
/// exactly at min ts), the frame-native analogue of time_bucket("min").
DataFrame group_by_dynamic(const DataFrame& b, const std::string& time_col,
                           std::int64_t every, std::int64_t period,
                           const std::vector<GroupAgg>& aggs,
                           std::int64_t origin = 0, bool origin_min = false);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_BATCH_OPS_H
