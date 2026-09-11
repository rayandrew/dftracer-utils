#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

using dftracer::utils::coro::CoroTask;
using dftracer::utils::dataframe::Cursor;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::InMemorySource;
using dftracer::utils::dataframe::LazyFrame;
using dftracer::utils::dataframe::Morsel;
using dftracer::utils::dataframe::Ordering;
using dftracer::utils::dataframe::Series;

namespace {

DataFrame make_df(std::int64_t n) {
    std::vector<std::int64_t> a(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) a[static_cast<std::size_t>(i)] = i;
    DataFrame df;
    df.names = {"a"};
    df.columns.push_back(Series::flat_i64(a.data(), n));
    return df;
}

struct DrainResult {
    std::vector<std::int64_t> batch_indices;
    std::vector<Ordering> orderings;
    std::int64_t total_rows = 0;
};

CoroTask<DrainResult> drain(std::unique_ptr<Cursor> cur,
                            std::int64_t max_rows) {
    DrainResult out;
    while (auto m = co_await cur->next(max_rows)) {
        out.batch_indices.push_back(m->batch_index);
        out.orderings.push_back(m->ordering);
        out.total_rows += m->rows;
    }
    co_return out;
}

DrainResult run_drain(std::unique_ptr<Cursor> cur, std::int64_t max_rows) {
    return dftracer::utils::default_runtime()
        .submit(drain(std::move(cur), max_rows))
        .get();
}

}  // namespace

TEST_CASE(
    "InMemorySource morsels are Sequence-ordered with a contiguous "
    "batch_index") {
    auto source = std::make_shared<InMemorySource>(make_df(10));
    LazyFrame lf = LazyFrame::scan(source);
    DrainResult r = run_drain(lf.open_cursor(), 3);

    REQUIRE(r.batch_indices.size() > 1);
    for (std::size_t i = 0; i < r.orderings.size(); ++i)
        CHECK(r.orderings[i] == Ordering::Sequence);
    for (std::size_t i = 0; i < r.batch_indices.size(); ++i)
        CHECK(r.batch_indices[i] == static_cast<std::int64_t>(i));
    CHECK(r.total_rows == 10);
}

TEST_CASE(
    "select/with_column/fill_null/with_row_index preserve Sequence ordering") {
    auto source = std::make_shared<InMemorySource>(make_df(10));
    LazyFrame lf =
        LazyFrame::scan(source)
            .select({"a"})
            .with_column("b",
                         dftracer::utils::dataframe::col(0) *
                             dftracer::utils::dataframe::lit(std::int64_t{2}))
            .fill_null(0)
            .with_row_index("idx");
    DrainResult r = run_drain(lf.open_cursor(), 3);

    REQUIRE(r.batch_indices.size() > 1);
    for (std::size_t i = 0; i < r.orderings.size(); ++i)
        CHECK(r.orderings[i] == Ordering::Sequence);
    for (std::size_t i = 0; i < r.batch_indices.size(); ++i)
        CHECK(r.batch_indices[i] == static_cast<std::int64_t>(i));
}

TEST_CASE("sort_by reports Unordered downstream") {
    auto source = std::make_shared<InMemorySource>(make_df(10));
    LazyFrame lf = LazyFrame::scan(source).sort_by("a", /*descending=*/true);
    DrainResult r = run_drain(lf.open_cursor(), 1 << 20);

    REQUIRE(!r.orderings.empty());
    for (Ordering o : r.orderings) CHECK(o == Ordering::Unordered);
}

TEST_CASE("group_by reports Unordered downstream") {
    auto source = std::make_shared<InMemorySource>(make_df(10));
    dftracer::utils::dataframe::GroupAgg count_agg;
    count_agg.op = dftracer::utils::dataframe::Agg::Count;
    count_agg.out = "n";
    LazyFrame lf = LazyFrame::scan(source).group_by("a", {count_agg});
    DrainResult r = run_drain(lf.open_cursor(), 1 << 20);

    REQUIRE(!r.orderings.empty());
    for (Ordering o : r.orderings) CHECK(o == Ordering::Unordered);
}

TEST_CASE("unique reports Unordered downstream") {
    auto source = std::make_shared<InMemorySource>(make_df(10));
    LazyFrame lf = LazyFrame::scan(source).unique();
    DrainResult r = run_drain(lf.open_cursor(), 1 << 20);

    REQUIRE(!r.orderings.empty());
    for (Ordering o : r.orderings) CHECK(o == Ordering::Unordered);
}
