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
using dftracer::utils::dataframe::LazyFrame;
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

    TEST_CASE(
        "predicate pushdown moves an independent filter before with_column") {
        // Filter on 'a' (col 0) is independent of the added 'c', so it moves
        // up.
        auto lf = make_df()
                      .lazy()
                      .with_column("c", col(0) + col(1))
                      .filter(col(0) > std::int64_t{3});
        const std::string plan = lf.explain();
        const auto fpos = plan.find("filter");
        const auto wpos = plan.find("with_column");
        CHECK(fpos != std::string::npos);
        CHECK(wpos != std::string::npos);
        CHECK(fpos < wpos);  // filter reordered before with_column

        DataFrame r = lf.collect();
        CHECK(r.num_rows() == 3);
        CHECK(r.column("c").data<std::int64_t>()[0] == 44);  // 4 + 40
    }

    TEST_CASE("streaming row ops: head / slice / tail / rename") {
        LazyFrame base = make_df().lazy();      // a=1..6, b=10..60

        DataFrame h = base.head(3).collect(2);  // morsel 2 -> multi-morsel
        CHECK(h.num_rows() == 3);
        CHECK(h.column("a").data<std::int64_t>()[0] == 1);
        CHECK(h.column("a").data<std::int64_t>()[2] == 3);

        DataFrame s = base.slice(2, 3).collect(2);  // rows a=3,4,5
        CHECK(s.num_rows() == 3);
        CHECK(s.column("a").data<std::int64_t>()[0] == 3);
        CHECK(s.column("a").data<std::int64_t>()[2] == 5);

        DataFrame t = base.tail(2).collect(2);  // a=5,6
        CHECK(t.num_rows() == 2);
        CHECK(t.column("a").data<std::int64_t>()[0] == 5);
        CHECK(t.column("a").data<std::int64_t>()[1] == 6);

        DataFrame r = base.rename({"x", "y"}).collect();
        CHECK(r.names == std::vector<std::string>{"x", "y"});
        CHECK(r.column("x").data<std::int64_t>()[5] == 6);
    }

    TEST_CASE("predicate pushdown keeps a dependent filter after with_column") {
        // Filter on 'c' (col 2, the added column) must NOT move up.
        auto lf = make_df()
                      .lazy()
                      .with_column("c", col(0) + col(1))
                      .filter(col(2) > std::int64_t{50});  // c > 50
        const std::string plan = lf.explain();
        CHECK(plan.find("with_column") < plan.find("filter"));

        DataFrame r = lf.collect();
        // c = a+b in {11,22,33,44,55,66}; c > 50 -> rows 5,6 (c=55,66).
        CHECK(r.num_rows() == 2);
        CHECK(r.column("c").data<std::int64_t>()[0] == 55);
    }
}
