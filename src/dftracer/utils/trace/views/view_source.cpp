#include <dftracer/utils/trace/views/view_source.h>

#include <algorithm>
#include <utility>

namespace dftracer::utils::trace::views {

coro::CoroTask<std::optional<dftracer::utils::dataframe::Morsel>>
ViewCursor::next(std::int64_t /*max_rows*/) {
    if (done_) co_return std::nullopt;
    done_ = true;
    dftracer::utils::dataframe::Morsel m;
    m.rows = buf_->num_rows();
    m.columns.reserve(buf_->columns.size());
    for (const dftracer::utils::dataframe::Series& c : buf_->columns)
        m.columns.push_back(c.share());
    co_return m;
}

}  // namespace dftracer::utils::trace::views
