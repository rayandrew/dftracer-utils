#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using dftracer::utils::dataframe::Agg;
using dftracer::utils::dataframe::col;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::eval;
using dftracer::utils::dataframe::GroupAgg;
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

    TEST_CASE("streaming fill_null / drop_nulls / with_row_index") {
        std::vector<std::int64_t> a{1, 2, 3, 4};
        std::uint8_t bm = 0b1011;  // rows 0,1,3 valid; row 2 null
        DataFrame df;
        df.names = {"a"};
        df.columns.push_back(Series::flat_i64(a.data(), 4, &bm));

        DataFrame f = df.lazy().fill_null(std::int64_t{-1}).collect();
        CHECK(f.num_rows() == 4);
        CHECK(f.column("a").data<std::int64_t>()[2] == -1);

        DataFrame d = df.lazy().drop_nulls().collect();
        CHECK(d.num_rows() == 3);

        DataFrame w = make_df().lazy().with_row_index("idx").collect(2);
        CHECK(w.names[0] == "idx");
        CHECK(w.column("idx").data<std::int64_t>()[0] == 0);
        CHECK(w.column("idx").data<std::int64_t>()[5] == 5);

        // null_count over the nullable column (1 null), streamed at morsel 2.
        DataFrame nc = df.lazy().null_count().collect(2);
        CHECK(nc.num_rows() == 1);
        CHECK(nc.column("a").data<std::int64_t>()[0] == 1);
    }

    TEST_CASE("lazy topk and unpivot") {
        DataFrame tk = make_df().lazy().topk("a", 2).collect(2);  // largest 2 a
        CHECK(tk.num_rows() == 2);
        const std::int64_t* a = tk.column("a").data<std::int64_t>();
        CHECK(((a[0] == 6 && a[1] == 5) || (a[0] == 5 && a[1] == 6)));

        DataFrame up = make_df().lazy().unpivot({"a"}, {"b"}).collect(2);
        CHECK(up.names == std::vector<std::string>{"a", "variable", "value"});
        CHECK(up.num_rows() == 6);
    }

    TEST_CASE("streaming group_by matches eager (mergeable across morsels)") {
        std::vector<std::int64_t> g{0, 1, 0, 1, 0, 1};
        std::vector<std::int64_t> v{1, 2, 3, 4, 5, 6};
        DataFrame df;
        df.names = {"g", "v"};
        df.columns.push_back(Series::flat_i64(g.data(), 6));
        df.columns.push_back(Series::flat_i64(v.data(), 6));
        std::vector<GroupAgg> aggs{{Agg::Sum, "v", "sum", 0.0},
                                   {Agg::Mean, "v", "mean", 0.0},
                                   {Agg::Count, "", "count", 0.0}};

        // Small morsel size: mean must NOT be a mean-of-means.
        DataFrame r = df.lazy().group_by("g", aggs).collect(2);
        REQUIRE(r.num_rows() == 2);
        const std::int64_t* gk = r.column("g").data<std::int64_t>();
        const std::int64_t* sum = r.column("sum").data<std::int64_t>();
        const double* mean = r.column("mean").data<double>();
        const std::int64_t* cnt = r.column("count").data<std::int64_t>();
        for (std::int64_t i = 0; i < 2; ++i) {
            if (gk[i] == 0) {
                CHECK(sum[i] == 9);  // 1+3+5
                CHECK(mean[i] == doctest::Approx(3.0));
                CHECK(cnt[i] == 3);
            } else {
                CHECK(sum[i] == 12);  // 2+4+6
                CHECK(mean[i] == doctest::Approx(4.0));
                CHECK(cnt[i] == 3);
            }
        }
    }

    TEST_CASE("sort_by and unique (buffering sinks)") {
        DataFrame s = make_df().lazy().sort_by("a", true).collect(2);
        CHECK(s.num_rows() == 6);
        CHECK(s.column("a").data<std::int64_t>()[0] == 6);
        CHECK(s.column("a").data<std::int64_t>()[5] == 1);

        std::vector<std::int64_t> d{1, 1, 2, 2, 3};
        DataFrame df;
        df.names = {"x"};
        df.columns.push_back(Series::flat_i64(d.data(), 5));
        DataFrame u = df.lazy().unique().collect(2);
        CHECK(u.num_rows() == 3);
    }

    TEST_CASE("sample / is_duplicated / group_by_dynamic") {
        DataFrame s = make_df().lazy().sample(3, 42).collect(2);
        CHECK(s.num_rows() == 3);

        std::vector<std::int64_t> d{1, 1, 2};
        DataFrame df;
        df.names = {"x"};
        df.columns.push_back(Series::flat_i64(d.data(), 3));
        DataFrame du = df.lazy().is_duplicated().collect();
        CHECK(du.names == std::vector<std::string>{"is_duplicated"});
        CHECK(du.num_rows() == 3);

        std::vector<std::int64_t> t{0, 1, 2, 3}, v{1, 1, 1, 1};
        DataFrame tf;
        tf.names = {"t", "v"};
        tf.columns.push_back(Series::flat_i64(t.data(), 4));
        tf.columns.push_back(Series::flat_i64(v.data(), 4));
        std::vector<GroupAgg> aggs{{Agg::Sum, "v", "sum", 0.0}};
        DataFrame gd = tf.lazy().group_by_dynamic("t", 2, 2, aggs).collect(2);
        CHECK(gd.names == std::vector<std::string>{"t", "sum"});
        CHECK(gd.num_rows() == 2);  // windows [0,2), [2,4)
        CHECK(gd.column("sum").data<std::int64_t>()[0] == 2);
    }

    TEST_CASE("data-dependent schema: pivot / to_dummies / describe") {
        // to_dummies: one Int8 column per distinct value of "g".
        std::vector<std::int64_t> g{0, 1, 0};
        DataFrame gf;
        gf.names = {"g"};
        gf.columns.push_back(Series::flat_i64(g.data(), 3));
        LazyFrame dl = gf.lazy().to_dummies("g");
        CHECK(dl.schema().empty());   // unknown until run
        DataFrame d = dl.collect(2);
        CHECK(d.num_rows() == 3);
        CHECK(d.num_columns() == 2);  // g_0, g_1
        CHECK(!d.names.empty());

        // pivot: rows = distinct index, one value column per distinct "on".
        std::vector<std::int64_t> i{0, 0, 1, 1}, k{10, 20, 10, 20},
            v{1, 2, 3, 4};
        DataFrame pf;
        pf.names = {"i", "k", "v"};
        pf.columns.push_back(Series::flat_i64(i.data(), 4));
        pf.columns.push_back(Series::flat_i64(k.data(), 4));
        pf.columns.push_back(Series::flat_i64(v.data(), 4));
        DataFrame p = pf.lazy().pivot("i", "k", "v").collect(2);
        CHECK(p.num_rows() == 2);     // i in {0,1}
        CHECK(p.num_columns() == 3);  // i + two k-values

        // describe: some stats rows, non-empty schema after collect.
        DataFrame ds = make_df().lazy().describe().collect();
        CHECK(ds.num_rows() > 0);
        CHECK(!ds.names.empty());
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
