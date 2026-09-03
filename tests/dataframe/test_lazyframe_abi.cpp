// The dftu_lazyframe C ABI: wrap a materialized frame as a lazy plan, chain a
// couple of builder ops, and collect back to the eager equivalent. Handles are
// freed explicitly so asan flags any leak.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/expr.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

dftu_series* i64_col(const std::int64_t* v, std::int64_t n) {
    return dftu_series_new_flat(DFTU_TYPE_INT64, v, n, nullptr);
}

const std::int64_t* i64_of(const dftu_series* c) {
    return static_cast<const std::int64_t*>(dftu_series_data(c));
}

dftu_scalar i64_scalar(std::int64_t v) {
    dftu_scalar s;
    s.kind = DFTU_SCALAR_TAG_I64;
    s.value.i = v;
    return s;
}

// A two-column frame: "k" keys, "v" values.
dftu_dataframe* make_frame(const std::vector<std::int64_t>& k,
                           const std::vector<std::int64_t>& v) {
    dftu_series* cols[2] = {
        i64_col(k.data(), static_cast<std::int64_t>(k.size())),
        i64_col(v.data(), static_cast<std::int64_t>(v.size()))};
    const char* names[2] = {"k", "v"};
    return dftu_dataframe_new(names, cols, 2);
}

}  // namespace

TEST_SUITE("lazyframe_abi") {
    TEST_CASE("dftu_dataframe_lazy then filter collects the eager equivalent") {
        dftu_dataframe* df = make_frame({1, 1, 2, 3}, {5, 15, 20, 8});
        REQUIRE(df != nullptr);

        dftu_lazyframe* lf = dftu_dataframe_lazy(df);
        REQUIRE(lf != nullptr);

        // Keep rows where v (column 1) > 10.
        dftu_expr* pred =
            dftu_expr_cmp(DFTU_CMP_GT, dftu_expr_col(1), i64_scalar(10));
        REQUIRE(pred != nullptr);
        dftu_lazyframe* filtered = dftu_lazyframe_filter(lf, pred);
        REQUIRE(filtered != nullptr);

        dftu_dataframe* out = dftu_lazyframe_collect(filtered, 0);
        REQUIRE(out != nullptr);

        CHECK(dftu_dataframe_num_rows(out) == 2);
        dftu_series* vc = dftu_dataframe_column(out, "v");
        REQUIRE(vc != nullptr);
        const std::int64_t* vd = i64_of(vc);
        CHECK(vd[0] == 15);
        CHECK(vd[1] == 20);

        // collect does not consume the plan: re-running yields the same rows.
        dftu_dataframe* again = dftu_lazyframe_collect(filtered, 0);
        REQUIRE(again != nullptr);
        CHECK(dftu_dataframe_num_rows(again) == 2);

        dftu_series_free(vc);
        dftu_dataframe_free(again);
        dftu_dataframe_free(out);
        dftu_expr_free(pred);
        dftu_lazyframe_free(filtered);
        dftu_lazyframe_free(lf);
        dftu_dataframe_free(df);
    }

    TEST_CASE("dftu_lazyframe_group_by sums per key") {
        dftu_dataframe* df = make_frame({1, 1, 2, 3}, {5, 15, 20, 8});
        REQUIRE(df != nullptr);

        dftu_lazyframe* lf = dftu_dataframe_lazy(df);
        REQUIRE(lf != nullptr);

        const char* keys[1] = {"k"};
        dftu_group_agg aggs[1] = {{"sum", "v", "v_sum"}};
        dftu_lazyframe* grouped = dftu_lazyframe_group_by(lf, keys, 1, aggs, 1);
        REQUIRE(grouped != nullptr);

        dftu_lazyframe* sorted = dftu_lazyframe_sort_by(grouped, "k", 0);
        REQUIRE(sorted != nullptr);

        dftu_dataframe* out = dftu_lazyframe_collect(sorted, 0);
        REQUIRE(out != nullptr);
        CHECK(dftu_dataframe_num_rows(out) == 3);

        dftu_series* kc = dftu_dataframe_column(out, "k");
        dftu_series* sc = dftu_dataframe_column(out, "v_sum");
        REQUIRE(kc != nullptr);
        REQUIRE(sc != nullptr);
        const std::int64_t* kd = i64_of(kc);
        const std::int64_t* sd = i64_of(sc);
        CHECK(kd[0] == 1);
        CHECK(sd[0] == 20);
        CHECK(kd[1] == 2);
        CHECK(sd[1] == 20);
        CHECK(kd[2] == 3);
        CHECK(sd[2] == 8);

        dftu_series_free(kc);
        dftu_series_free(sc);
        dftu_dataframe_free(out);
        dftu_lazyframe_free(sorted);
        dftu_lazyframe_free(grouped);
        dftu_lazyframe_free(lf);
        dftu_dataframe_free(df);
    }

    TEST_CASE("a null handle is rejected, not crashed") {
        CHECK(dftu_dataframe_lazy(nullptr) == nullptr);
        CHECK(dftu_lazyframe_collect(nullptr, 0) == nullptr);
        CHECK(dftu_lazyframe_filter(nullptr, nullptr) == nullptr);
        dftu_lazyframe_free(nullptr);
    }
}
