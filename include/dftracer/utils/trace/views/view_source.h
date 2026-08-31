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
/// LazyFrame without running the scan. A plan's real result schema (an
/// aggregation's group_by/agg columns are not the trace's raw column
/// universe) is only known after running it, so names()/as_frame()/open()
/// share one cached run of View::collect_frame().
class ViewSource : public dftracer::utils::dataframe::Source {
   public:
    explicit ViewSource(View view) : view_(std::move(view)) {}

    std::vector<std::string> names() const override { return buffer()->names; }
    const dftracer::utils::dataframe::DataFrame* as_frame() const override {
        return buffer().get();
    }
    std::unique_ptr<dftracer::utils::dataframe::Cursor> open() const override {
        return std::make_unique<ViewCursor>(buffer());
    }

   private:
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
