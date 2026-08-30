#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using dftracer::utils::dataframe::col;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::eval;
using dftracer::utils::dataframe::Series;

namespace {

DataFrame make_df() {
    std::vector<std::int64_t> a{1, 2, 3, 4, 5, 6};
    std::vector<std::int64_t> b{10, 20, 30, 40, 50, 60};
    DataFrame df;
    df.names = {"a", "b"};
    df.columns.push_back(Series::flat_i64(a.data(), 6));
    df.columns.push_back(Series::flat_i64(b.data(), 6));
    return df;
}

std::vector<const Series*> ptrs(const DataFrame& df) {
    std::vector<const Series*> in;
    for (const Series& c : df.columns) in.push_back(&c);
    return in;
}

}  // namespace

TEST_SUITE("lazyframe") {
    TEST_CASE("lazy filter+with_column+select matches the eager path") {
        auto lf = make_df()
                      .lazy()
                      .filter(col(0) > std::int64_t{3})
                      .with_column("c", col(0) + col(1))
                      .select({"a", "c"});

        CHECK(lf.schema() == std::vector<std::string>{"a", "c"});

        // Small morsel size to exercise multi-morsel scan + concat.
        DataFrame lazy = lf.collect(2);

        DataFrame e = make_df();
        DataFrame ef = e.filter(eval(col(0) > std::int64_t{3}, ptrs(e)));
        DataFrame ew = ef.with_column("c", eval(col(0) + col(1), ptrs(ef)));
        DataFrame eager = ew.select({"a", "c"});

        REQUIRE(lazy.num_rows() == eager.num_rows());
        REQUIRE(lazy.num_columns() == eager.num_columns());
        CHECK(lazy.names == eager.names);
        CHECK(lazy.num_rows() == 3);  // a in {4,5,6}
        for (std::size_t c = 0; c < lazy.num_columns(); ++c) {
            const std::int64_t* lp = lazy.columns[c].data<std::int64_t>();
            const std::int64_t* ep = eager.columns[c].data<std::int64_t>();
            for (std::int64_t i = 0; i < lazy.num_rows(); ++i)
                CHECK(lp[i] == ep[i]);
        }
        const std::int64_t* cc = lazy.column("c").data<std::int64_t>();
        CHECK(cc[0] == 44);
        CHECK(cc[2] == 66);
    }

    TEST_CASE("lazy(df) free function and full-scan roundtrip") {
        DataFrame all = dftracer::utils::dataframe::lazy(make_df()).collect();
        CHECK(all.num_rows() == 6);
        CHECK(all.num_columns() == 2);
        CHECK(all.column("b").data<std::int64_t>()[5] == 60);
    }
}
