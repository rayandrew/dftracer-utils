#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/trace/views/result_batch.h>
#include <doctest/doctest.h>

namespace dataframe = dftracer::utils::dataframe;

namespace views = dftracer::utils::trace::views;

TEST_SUITE("result_batch") {
    TEST_CASE(
        "collect_batch runs a View aggregation into a dataframe::DataFrame") {
        // An empty view aggregates to a result with no rows. This exercises the
        // coroutine terminal wiring (View::collect -> dataframe::DataFrame).
        views::View empty;
        dataframe::DataFrame b = views::collect_batch(empty).get();
        CHECK(b.num_rows() == 0);
    }
}
