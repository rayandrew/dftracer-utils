#ifndef DFTRACER_UTILS_DATAFRAME_LAZYFRAME_H
#define DFTRACER_UTILS_DATAFRAME_LAZYFRAME_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/coro/async_generator.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/dataframe/agg.h>
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
    /// Per-morsel schema for a streaming source whose morsels may differ in
    /// columns or types. Empty means positional alignment with the plan schema
    /// (the common case). One interned id per column, resolved via `intern`.
    std::vector<std::uint32_t> name_ids;
    std::shared_ptr<const dftracer::utils::StringIntern> intern;
    /// Name-keyed dyn value columns, carried out of band from `columns` so the
    /// positional plan schema stays fixed while the dyn set varies per morsel.
    /// `dyn_names[i]` (producer-tagged) labels `dyn_columns[i]`; a dyn group_by
    /// folds these through agg_accumulate's dyn feed. Empty for a non-dyn
    /// morsel.
    std::vector<std::string> dyn_names;
    std::vector<Series> dyn_columns;
};

/// A stateful reader over one Source. next() returns the next morsel, or
/// nullopt at end; `max_rows` is a size hint. The engine pulls it async
/// (co_await), so a producer can suspend on real I/O inside next() instead of
/// blocking a worker thread.
class Cursor {
   public:
    virtual ~Cursor() = default;
    virtual coro::CoroTask<std::optional<Morsel>> next(
        std::int64_t max_rows) = 0;
    /// Output column names, when they are only known after producing (a
    /// data-dependent schema like pivot/to_dummies). nullopt means the plan's
    /// static schema is authoritative. Valid only after the cursor is drained.
    virtual std::optional<std::vector<std::string>> out_names() const {
        return std::nullopt;
    }
};

/// The static, scan-free schema of a Source: one Field per column, in order.
/// A column whose type is not knowable without scanning gets a `Field` whose
/// `type.id == TypeId::Unknown`, never a guessed concrete type.
struct Schema {
    std::vector<Field> fields;
};

/// How completely a source applied a pushed-down filter, per ScanRequest
/// filter. Guides whether the engine must re-apply it over the survivors.
/// Mirrors dftu_pushed (dataframe/abi.h).
enum class Pushed {
    No = DFTU_PUSHED_NO,      ///< Not applied by the source; engine applies it.
    Inexact =
        DFTU_PUSHED_INEXACT,  ///< Pruned I/O but did not filter; re-apply.
    Exact =
        DFTU_PUSHED_EXACT,    ///< Fully applied by the source; engine drops it.
};
static_assert(static_cast<int>(Pushed::No) == DFTU_PUSHED_NO);
static_assert(static_cast<int>(Pushed::Inexact) == DFTU_PUSHED_INEXACT);
static_assert(static_cast<int>(Pushed::Exact) == DFTU_PUSHED_EXACT);

/// A pushdown request the optimizer hands a Source at scan time.
struct ScanRequest {
    /// Columns the plan needs, in the order the scan must return them. Empty
    /// means all source columns. A source that accepts a non-empty projection
    /// MUST return exactly these columns, in this order.
    std::vector<std::string> projection;
    /// Candidate predicates, each positional against the request's column set
    /// (projection when non-empty, else schema()). A source translates what it
    /// can and reports the rest No.
    std::vector<Expr> filters;
    std::int64_t limit = -1;  ///< Slice pushdown hint; -1 means no limit.
    /// The resolved LazyFrame budget (bytes); a streaming source may use it to
    /// bound its own in-flight buffering. Most sources ignore it.
    std::uint64_t memory_budget = 0;
};

/// The result of Source::scan: a fresh async Cursor plus, per ScanRequest
/// filter, how completely the source applied it.
struct ScanResult {
    std::unique_ptr<Cursor> cursor;
    std::vector<Pushed> filters;
};

/// A pushdown-aware data source for a lazy query. Immutable: schema() reports
/// the columns without scanning and scan() hands out a fresh Cursor honoring
/// the pushed projection/filters, so one Source can back many collect()s.
/// Implement these two to plug any producer (a file, another engine, a trace
/// scan) into LazyFrame.
class Source {
   public:
    virtual ~Source() = default;
    /// The column names and types, without a scan (index/catalog/footer).
    virtual Schema schema() const = 0;
    /// Open a Cursor honoring `req`. When req.projection is non-empty the
    /// cursor's morsels carry exactly those columns in that order; the returned
    /// ScanResult.filters reports, per req.filter, how completely it was
    /// applied.
    virtual ScanResult scan(const ScanRequest& req) const = 0;
    /// The resident frame when this source is already in memory, else nullptr
    /// (rows produced only by streaming). collect() runs a resident source
    /// whole-column, matching the eager path instead of paying the morsel tax.
    virtual const DataFrame* as_frame() const { return nullptr; }

    /// Convenience: the schema's column names. Non-virtual; a caller that only
    /// needs names reads this instead of building a full scan.
    std::vector<std::string> names() const {
        std::vector<std::string> out;
        Schema s = schema();
        out.reserve(s.fields.size());
        for (const Field& f : s.fields) out.push_back(f.name);
        return out;
    }
};

/// A Source over an already-materialized in-memory frame.
class InMemorySource : public Source {
   public:
    explicit InMemorySource(DataFrame frame);
    Schema schema() const override;
    ScanResult scan(const ScanRequest& req) const override;
    const DataFrame* as_frame() const override { return frame_.get(); }

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
    LazyFrame fill_null(Scalar value) const;
    /// Keeps the rows at `indices` (arbitrary order, repeats allowed), as
    /// DataFrame::take. Not streaming: the whole frame must be resident before
    /// the indices can be applied.
    LazyFrame take(std::vector<std::int64_t> indices) const;
    /// Keeps rows where the precomputed `mask` is true, positionally aligned to
    /// this frame's rows. Streams: each morsel consumes the matching mask
    /// slice. Named apart from filter(Expr) - a mask column is data, not a
    /// predicate to compile - so the two never collide at a call site.
    LazyFrame filter_mask(Series mask) const;
    /// Reverses row order, as DataFrame::reverse. Not streaming: every row
    /// must be resident before it can be reordered.
    LazyFrame reverse() const;
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
    /// Group by N key columns (a composite key: hashed and compared
    /// column-by-column, each keeping its own type). Streaming, as the
    /// single-key overload. `dyn` (optional) enables the name-keyed dyn
    /// side-table: each morsel's dyn columns (its own out-of-band dyn set, or a
    /// resident source's columns whose name starts with `dyn_prefix`) are
    /// folded through agg_accumulate's dyn feed, with `dyn_prefix` stripped
    /// from each name. The dyn output columns are appended after the fixed
    /// aggregates.
    LazyFrame group_by(std::vector<std::string> keys,
                       std::vector<GroupAgg> aggs,
                       std::vector<AggDynSpec> dyn = {},
                       std::string dyn_prefix = {}) const;
    /// Group by an expression key and compute each expression aggregate.
    /// Desugars to with_column + the string group_by: a bare column-ref key or
    /// aggregate value is used directly, a computed one is materialized into a
    /// hidden temp column first. Fully streaming, same as the string overload.
    LazyFrame group_by(Expr key, std::vector<AggExprSpec> aggs) const;
    /// Group by N expression keys; each desugars like the single-key overload.
    LazyFrame group_by(std::vector<Expr> keys,
                       std::vector<AggExprSpec> aggs) const;
    /// Sort by `name`. External merge sort: spills sorted runs past the memory
    /// budget and k-way merges them, so peak memory stays bounded.
    LazyFrame sort_by(std::string name, bool descending = false) const;
    /// Stable lexicographic sort by several key columns (nulls last), as
    /// DataFrame::sort_by_multi. Not streaming: needs every row to compare
    /// across the whole frame.
    LazyFrame sort_by_multi(std::vector<std::string> by,
                            bool descending = false) const;
    /// Per-column direction form: `descending[i]` applies to `by[i]`.
    LazyFrame sort_by_multi(std::vector<std::string> by,
                            std::vector<bool> descending) const;
    /// Distinct rows, first occurrence, in original order. Streams input and
    /// output; holds only the distinct set (the result itself).
    LazyFrame unique() const;
    /// Alias of unique().
    LazyFrame drop_duplicates() const;
    /// A deterministic n-row sample (mix64 min-hash). Streaming: bounded to n
    /// rows regardless of input size.
    LazyFrame sample(std::int64_t n, std::uint64_t seed = 0) const;
    /// One Bool column: true where the whole row is duplicated. Two-pass
    /// (count, then per-row mask in input order); state is the count map.
    LazyFrame is_duplicated() const;
    /// One Bool column: true where the whole row is unique. Two-pass, as
    /// is_duplicated.
    LazyFrame is_unique() const;
    /// Tumbling/sliding time-window aggregation over an ascending Int64 time
    /// column. Streaming: holds one agg state per open window (bounded by the
    /// window count, not the input). Requires ascending time.
    LazyFrame group_by_dynamic(std::string time_col, std::int64_t every,
                               std::int64_t period, std::vector<GroupAgg> aggs,
                               std::int64_t origin = 0,
                               bool origin_min = false) const;
    /// unpivot alias.
    LazyFrame melt(std::vector<std::string> id_vars,
                   std::vector<std::string> value_vars) const;
    /// Reshape long -> wide. Two-pass (distinct index+on values, then cell
    /// aggregation via the agg IR); state is bounded by the output (index rows
    /// x on-values). Output columns are data-dependent (one per distinct `on`
    /// value), so schema() is empty until collect().
    LazyFrame pivot(std::string index, std::string on, std::string values,
                    std::string agg = "first") const;
    /// One-hot encode `column`. Two-pass (distinct values, then stream-encode);
    /// state is bounded by the value cardinality. Output columns are
    /// data-dependent, so schema() is empty until collect().
    LazyFrame to_dummies(std::string column) const;
    /// Per-column summary statistics (count/null_count/mean/std/min/max).
    /// Streaming, constant state per column; output columns are data-dependent
    /// (one per numeric input column), so schema() is empty until collect().
    LazyFrame describe() const;

    /// Out-of-core budget for the pipeline breakers (sort/unique/group_by):
    /// when a sink's in-memory state grows past this many bytes it spills to a
    /// sorted temp run, k-way merged at the end, so peak memory stays bounded.
    /// 0 (the default) means "auto": ~1/3 of available memory. Pass
    /// NO_SPILL_BUDGET to disable spilling. Same knob and policy as View.
    LazyFrame memory_budget(std::uint64_t bytes) const;
    /// Explicitly set the budget to ~1/3 of available memory (same as the
    /// default); sugar for readers who want spilling stated at the call site.
    LazyFrame auto_spill() const;

    /// Output column names without running the query. Empty for a plan ending
    /// in a data-dependent op (pivot/to_dummies/describe).
    std::vector<std::string> schema() const;

    /// The optimized plan as text (source then one op per line), for
    /// introspection and tests.
    std::string explain() const;

    /// Pull-based chunk generator: the streaming terminal. Each yielded
    /// DataFrame is standalone, carrying its own schema. One-shot: re-run
    /// from the LazyFrame to restart. `morsel_rows` is the scan chunk size;
    /// <= 0 means auto. collect() drains this.
    coro::AsyncGenerator<DataFrame> stream(std::int64_t morsel_rows = 0) const;

    /// Run the pipeline and materialize the surviving rows. `morsel_rows` is
    /// the scan chunk size; <= 0 (the default) means auto; one pass over a
    /// resident source, the bounded streaming default otherwise.
    coro::CoroTask<DataFrame> collect(std::int64_t morsel_rows = 0) const;

    /// Group `keys` with `aggs` and return the mergeable partial instead of a
    /// finalized frame: streams this pipeline (no group op appended) and folds
    /// every morsel into one AggState. The caller finalizes (agg_finalize),
    /// coarsens (agg_regroup), or persists it. State is bounded by the distinct
    /// group count, not the input; used by the rollup materialize path.
    coro::CoroTask<AggStatePtr> collect_group_state(
        std::vector<std::string> keys, std::vector<GroupAgg> aggs,
        std::vector<AggDynSpec> dyn = {}, std::string dyn_prefix = {},
        std::int64_t morsel_rows = 0) const;

   private:
    LazyFrame(std::shared_ptr<const Source> source,
              std::vector<std::shared_ptr<const LazyOp>> ops,
              std::uint64_t memory_budget = 0)
        : source_(std::move(source)),
          ops_(std::move(ops)),
          memory_budget_(memory_budget) {}
    // Clone with a new op list, preserving the source and budget.
    LazyFrame with_ops(std::vector<std::shared_ptr<const LazyOp>> ops) const {
        return LazyFrame(source_, std::move(ops), memory_budget_);
    }
    std::shared_ptr<const Source> source_;
    std::vector<std::shared_ptr<const LazyOp>> ops_;
    std::uint64_t memory_budget_ = 0;
};

/// Free-function form of DataFrame::lazy(), for `lazy(df)` call sites.
LazyFrame lazy(DataFrame frame);

/// Lower name-based GroupAggs to index-based AggSpecs plus the deduped list of
/// value column names they reference (Count references none); the i-th spec's
/// value_col/by_col index into `value_names`. The gagg lowering that
/// collect_group_state and any external chunk driver share, so a Fold feeding
/// its own chunks accumulates into a byte-identical AggState.
struct LoweredGroupAggs {
    std::vector<AggSpec> specs;
    std::vector<std::string> value_names;
};
LoweredGroupAggs lower_group_aggs(const std::vector<GroupAgg>& aggs);

/// Accumulate one already-built chunk `frame` into `state`: resolve `keys` and
/// `value_names` (from lower_group_aggs) by name against `frame`, feeding every
/// column whose name starts with `dyn_prefix` as a dyn input (prefix stripped).
/// The per-morsel step collect_group_state runs, exposed for a non-LazyFrame
/// chunk driver (a trace Fold accumulating over a shared scan).
void agg_accumulate_chunk(AggState& state, const DataFrame& frame,
                          const std::vector<std::string>& keys,
                          const std::vector<std::string>& value_names,
                          const std::string& dyn_prefix);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_LAZYFRAME_H
