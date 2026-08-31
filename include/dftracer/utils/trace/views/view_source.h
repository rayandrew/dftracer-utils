#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_SOURCE_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_SOURCE_H

#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/trace/views/view.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace dftracer::utils::trace::views {

/// Cursor over a View's buffered scan result: hands out the whole buffer as
/// one morsel, zero-copy (no max_rows chunking - concat_columns can't rejoin
/// a nested/List column, which a histogram agg produces).
class ViewCursor : public dftracer::utils::dataframe::Cursor {
   public:
    explicit ViewCursor(
        std::shared_ptr<const dftracer::utils::dataframe::DataFrame> buf)
        : buf_(std::move(buf)) {}

    coro::CoroTask<std::optional<dftracer::utils::dataframe::Morsel>> next(
        std::int64_t max_rows) override;

   private:
    std::shared_ptr<const dftracer::utils::dataframe::DataFrame> buf_;
    bool done_ = false;
};

/// Adapts a View as a LazyFrame Source, so View::collect() can build a
/// LazyFrame without running the scan. An aggregation buffers one cached run
/// of View::collect_frame(); a plain row query (see can_stream_rows) streams
/// instead, so collect() never buffers the whole matching set.
class ViewSource : public dftracer::utils::dataframe::Source {
   public:
    explicit ViewSource(View view) : view_(std::move(view)) {}

    std::vector<std::string> names() const override;
    const dftracer::utils::dataframe::DataFrame* as_frame() const override;
    std::unique_ptr<dftracer::utils::dataframe::Cursor> open(
        std::uint64_t memory_budget) const override;

   private:
    /// A row query with no sort/topk/pagination/select - the post-scan ops
    /// run_collect_rows applies after the fuse, which streaming cannot.
    bool can_stream_rows() const;

    /// Best-effort column list for the streaming source's names(), from index
    /// metadata rather than a scan.
    std::vector<std::string> row_schema() const;

    std::shared_ptr<const dftracer::utils::dataframe::DataFrame> buffer()
        const {
        if (!buf_)
            buf_ =
                std::make_shared<const dftracer::utils::dataframe::DataFrame>(
                    view_.collect_frame().get());
        return buf_;
    }

    View view_;
    mutable std::shared_ptr<const dftracer::utils::dataframe::DataFrame> buf_;
};

}  // namespace dftracer::utils::trace::views

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_SOURCE_H
