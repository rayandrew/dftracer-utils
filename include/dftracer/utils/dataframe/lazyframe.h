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

    /// Output column names without running the query.
    std::vector<std::string> schema() const;

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
