#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/runtime.h>
#include <doctest/doctest.h>

#include <algorithm>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include "test_view_common.h"

namespace {

template <class T>
T run(coro::CoroTask<T> task) {
    return dftracer::utils::default_runtime().submit(std::move(task)).get();
}

// A multi-member trace whose members carry different arg keys (x on the
// first half, y on the second), so a per-batch morsel's columns differ and
// the streaming path must exercise name-based reconciliation.
std::string create_mixed_arg_trace(TestEnvironment& env) {
    std::string pfw = env.get_dir() + "/mixed_args.pfw";
    std::ofstream ofs(pfw);
    for (int i = 0; i < 25; ++i)
        ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":)"
            << (1000 + i * 100) << R"(,"dur":)" << (10 + i)
            << R"(,"args":{"x":)" << i << R"(}})" << "\n";
    for (int i = 0; i < 25; ++i)
        ofs << R"({"ph":"X","name":"write","cat":"POSIX","pid":1,"tid":1,"ts":)"
            << (4000 + i * 100) << R"(,"dur":)" << (10 + i)
            << R"(,"args":{"y":)" << i << R"(}})" << "\n";
    ofs.close();
    std::string gz = pfw + ".gz";
    // A small member size forces the x-events and y-events into separate
    // gzip members, so the scan splits them into separate batches/morsels.
    dftu_utils_test::compress_file_to_gzip_multimember(pfw, gz, 512);
    fs::remove(pfw);
    return gz;
}

// (name, x, y) tuples, sorted for order-independent comparison across the
// streaming (fan-out order) and buffered (accumulate-then-build) paths.
std::vector<std::tuple<std::string, double, double>> row_key_set(
    const dataframe::DataFrame& df) {
    std::vector<std::tuple<std::string, double, double>> rows;
    const bool has_x = bhas(df, "x");
    const bool has_y = bhas(df, "y");
    for (std::int64_t i = 0; i < df.num_rows(); ++i) {
        const double x =
            has_x && !df.columns[static_cast<std::size_t>(bcol(df, "x"))]
                          .is_null(i)
                ? bnum(df, i, "x")
                : -1.0;
        const double y =
            has_y && !df.columns[static_cast<std::size_t>(bcol(df, "y"))]
                          .is_null(i)
                ? bnum(df, i, "y")
                : -1.0;
        rows.emplace_back(bstr(df, i, "name"), x, y);
    }
    std::sort(rows.begin(), rows.end());
    return rows;
}

}  // namespace

TEST_SUITE("View - streaming row query") {
    TEST_CASE(
        "collect() streaming row query matches the buffered collect_frame()") {
        TestEnvironment env(200);
        std::string gz = create_mixed_arg_trace(env);
        std::string idx = determine_index_path(gz, "");
        View v = View::from_file(gz, idx).metadata(false);

        dataframe::DataFrame via_lazy = run(v.collect().collect());
        dataframe::DataFrame via_eager = run(v.collect_frame());

        REQUIRE(via_lazy.num_rows() == 50);
        REQUIRE(via_lazy.num_rows() == via_eager.num_rows());
        CHECK(bhas(via_lazy, "x"));
        CHECK(bhas(via_lazy, "y"));
        CHECK(bhas(via_eager, "x"));
        CHECK(bhas(via_eager, "y"));

        CHECK(row_key_set(via_lazy) == row_key_set(via_eager));
    }

    TEST_CASE("View::stream() drained matches View::collect().collect()") {
        TestEnvironment env(200);
        std::string gz = create_mixed_arg_trace(env);
        std::string idx = determine_index_path(gz, "");
        View v = View::from_file(gz, idx).metadata(false);

        auto drain = [](const View& view) {
            return dftracer::utils::default_runtime()
                .submit([](const View& vv)
                            -> coro::CoroTask<std::vector<
                                std::tuple<std::string, double, double>>> {
                    std::vector<std::tuple<std::string, double, double>> rows;
                    auto gen = vv.stream();
                    while (auto chunk = co_await gen.next()) {
                        auto part = row_key_set(*chunk);
                        rows.insert(rows.end(), part.begin(), part.end());
                    }
                    co_return rows;
                }(view))
                .get();
        };

        std::vector<std::tuple<std::string, double, double>> via_stream =
            drain(v);
        std::sort(via_stream.begin(), via_stream.end());
        dataframe::DataFrame via_collect = run(v.collect().collect());

        REQUIRE(static_cast<std::int64_t>(via_stream.size()) ==
                via_collect.num_rows());
        CHECK(via_stream == row_key_set(via_collect));
    }

    TEST_CASE("collect() falls back to buffered collect_frame() with select") {
        const auto& s = shared_trace();
        View v = View::from_file(s.gz, s.idx)
                     .metadata(false)
                     .select({"cat", "name"});

        dataframe::DataFrame via_lazy = run(v.collect().collect());
        dataframe::DataFrame via_eager = run(v.collect_frame());

        REQUIRE(via_lazy.num_rows() == via_eager.num_rows());
        CHECK(via_lazy.columns.size() == 2);
        CHECK(row_key_set(via_lazy) == row_key_set(via_eager));
    }

    TEST_CASE(
        "View::stream() bounds a buffered select-plan into multiple chunks") {
        const auto& s = shared_trace();  // 50 rows
        View v = View::from_file(s.gz, s.idx)
                     .metadata(false)
                     .select({"cat", "name"});

        auto [chunks, rows] =
            dftracer::utils::default_runtime()
                .submit([](const View& vv)
                            -> coro::CoroTask<
                                std::pair<std::size_t, std::int64_t>> {
                    std::size_t n = 0;
                    std::int64_t rows = 0;
                    auto gen = vv.stream(10);
                    while (auto chunk = co_await gen.next()) {
                        ++n;
                        rows += chunk->num_rows();
                    }
                    co_return std::make_pair(n, rows);
                }(v))
                .get();

        dataframe::DataFrame via_collect = run(v.collect().collect());
        CHECK(chunks > 1);
        CHECK(rows == via_collect.num_rows());
    }

    TEST_CASE("View::stream() over a histogram aggregation yields one chunk") {
        TestEnvironment env(200);
        std::string gz = create_mixed_arg_trace(env);
        std::string idx = determine_index_path(gz, "");
        View v = View::from_file(gz, idx)
                     .group_by({GroupKey::cat()})
                     .agg({{AggOp::Hist, "dur", "h"}});

        std::size_t chunks =
            dftracer::utils::default_runtime()
                .submit([](const View& vv) -> coro::CoroTask<std::size_t> {
                    std::size_t n = 0;
                    auto gen = vv.stream(1);
                    while (auto chunk = co_await gen.next()) ++n;
                    co_return n;
                }(v))
                .get();

        CHECK(chunks == 1);
    }
}
