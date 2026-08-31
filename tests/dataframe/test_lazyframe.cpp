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

    TEST_CASE("fused map (in-memory engine) matches the eager path") {
        // collect() with no morsel size runs the whole-column in-memory engine,
        // which fuses filter+with_column+select into one pass. Result must
        // equal the eager chain.
        auto lf = make_df()
                      .lazy()
                      .filter(col(1) > std::int64_t{20})
                      .with_column("c", col(0) + col(1))
                      .select({"a", "c"});
        DataFrame r = lf.collect();  // in-memory, fused
        CHECK(r.names == std::vector<std::string>{"a", "c"});
        CHECK(r.num_rows() == 4);    // b in {30,40,50,60} -> a in {3,4,5,6}
        const std::int64_t* a = r.column("a").data<std::int64_t>();
        const std::int64_t* c = r.column("c").data<std::int64_t>();
        CHECK(a[0] == 3);
        CHECK(c[0] == 33);  // 3 + 30
        CHECK(a[3] == 6);
        CHECK(c[3] == 66);  // 6 + 60
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

    TEST_CASE("predicate pushdown moves a filter before sort_by") {
        // sort_by only reorders rows, so a row-local filter hoists ahead of it
        // and the sort runs on the survivors only; the result is those rows in
        // sorted order.
        auto lf = make_df().lazy().sort_by("a", true).filter(col(0) >
                                                             std::int64_t{3});
        const std::string plan = lf.explain();
        const auto fpos = plan.find("filter");
        const auto spos = plan.find("sort");
        CHECK(fpos != std::string::npos);
        CHECK(spos != std::string::npos);
        CHECK(fpos < spos);  // filter reordered before sort_by

        DataFrame r = lf.collect();
        // survivors a>3 => {4,5,6}, sorted descending => {6,5,4}.
        CHECK(r.num_rows() == 3);
        const std::int64_t* a = r.column("a").data<std::int64_t>();
        CHECK(a[0] == 6);
        CHECK(a[1] == 5);
        CHECK(a[2] == 4);
    }

    TEST_CASE("projection pushdown drops unread source columns") {
        // 'b' is neither read by the filter nor in the output, so a projection
        // to [a] is inserted right after the source, ahead of the filter.
        auto lf =
            make_df().lazy().filter(col(0) > std::int64_t{2}).select({"a"});
        const std::string plan = lf.explain();
        const auto proj = plan.find("select [a]");
        const auto filt = plan.find("filter");
        CHECK(proj != std::string::npos);
        CHECK(filt != std::string::npos);
        CHECK(proj < filt);  // projection pushed ahead of the filter

        DataFrame r = lf.collect();
        CHECK(r.names == std::vector<std::string>{"a"});
        CHECK(r.num_rows() == 4);  // a in {3,4,5,6}
        const std::int64_t* a = r.column("a").data<std::int64_t>();
        CHECK(a[0] == 3);
        CHECK(a[3] == 6);
    }

    TEST_CASE("projection pushdown remaps a filter on a surviving column") {
        // Filter reads 'b' (col 1) but output is 'a'; both are live, and the
        // remapped predicate must still select the right rows after the source
        // projection renumbers columns.
        auto lf =
            make_df().lazy().filter(col(1) > std::int64_t{30}).select({"a"});
        DataFrame r = lf.collect();
        CHECK(r.names == std::vector<std::string>{"a"});
        CHECK(r.num_rows() == 3);  // b in {40,50,60} -> a in {4,5,6}
        const std::int64_t* a = r.column("a").data<std::int64_t>();
        CHECK(a[0] == 4);
        CHECK(a[2] == 6);
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

    TEST_CASE("sort_by (in-memory) and unique") {
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

    TEST_CASE("streaming unique matches eager (first occurrence, order)") {
        // Duplicates spread across morsels; keep-first order must be preserved.
        std::vector<std::int64_t> x{5, 3, 5, 1, 3, 5, 2, 1};
        DataFrame df;
        df.names = {"x"};
        df.columns.push_back(Series::flat_i64(x.data(), 8));
        DataFrame lz = df.lazy().unique().collect(3);  // multi-morsel scan
        DataFrame eg = df.unique();
        REQUIRE(lz.num_rows() == eg.num_rows());
        const std::int64_t* lp = lz.column("x").data<std::int64_t>();
        const std::int64_t* ep = eg.column("x").data<std::int64_t>();
        for (std::int64_t i = 0; i < lz.num_rows(); ++i) CHECK(lp[i] == ep[i]);
        CHECK(lz.num_rows() == 4);  // {5,3,1,2}
    }

    TEST_CASE("sort_by external merge (spilling) matches eager") {
        // Scrambled keys + a payload column, tiny budget + tiny morsels so the
        // sort spills several runs and k-way merges them back.
        std::vector<std::int64_t> k(200), v(200);
        for (std::int64_t i = 0; i < 200; ++i) {
            k[static_cast<std::size_t>(i)] =
                (i * 73 + 11) % 200;  // permutation
            v[static_cast<std::size_t>(i)] = i;
        }
        DataFrame df;
        df.names = {"k", "v"};
        df.columns.push_back(Series::flat_i64(k.data(), 200));
        df.columns.push_back(Series::flat_i64(v.data(), 200));

        for (bool desc : {false, true}) {
            DataFrame lz = df.lazy()
                               .memory_budget(1024)  // force spilling
                               .sort_by("k", desc)
                               .collect(16);
            DataFrame eg = df.sort_by("k", desc);
            REQUIRE(lz.num_rows() == 200);
            const std::int64_t* lk = lz.column("k").data<std::int64_t>();
            const std::int64_t* ek = eg.column("k").data<std::int64_t>();
            for (std::int64_t i = 0; i < 200; ++i) CHECK(lk[i] == ek[i]);
            // Keys are a permutation (all distinct), so the payload aligns too.
            const std::int64_t* lv = lz.column("v").data<std::int64_t>();
            const std::int64_t* ev = eg.column("v").data<std::int64_t>();
            for (std::int64_t i = 0; i < 200; ++i) CHECK(lv[i] == ev[i]);
        }
    }

    TEST_CASE("sample / is_duplicated / group_by_dynamic") {
        DataFrame s = make_df().lazy().sample(3, 42).collect(2);
        CHECK(s.num_rows() == 3);

        // Streaming min-hash sample must match eager DataFrame::sample exactly,
        // across a multi-morsel scan (bounded state, same survivors + order).
        std::vector<std::int64_t> big(100);
        for (std::int64_t i = 0; i < 100; ++i)
            big[static_cast<std::size_t>(i)] = i;
        DataFrame bf;
        bf.names = {"x"};
        bf.columns.push_back(Series::flat_i64(big.data(), 100));
        DataFrame lz = bf.lazy().sample(7, 123).collect(8);  // morsel 8
        DataFrame eg = bf.sample(7, 123);
        REQUIRE(lz.num_rows() == eg.num_rows());
        REQUIRE(lz.num_rows() == 7);
        const std::int64_t* lp = lz.column("x").data<std::int64_t>();
        const std::int64_t* ep = eg.column("x").data<std::int64_t>();
        for (std::int64_t i = 0; i < 7; ++i) CHECK(lp[i] == ep[i]);

        // is_duplicated / is_unique: two-pass, per-row mask in input order.
        // Must match eager across a multi-morsel scan.
        std::vector<std::int64_t> d{1, 1, 2, 3, 2, 1};
        DataFrame df;
        df.names = {"x"};
        df.columns.push_back(Series::flat_i64(d.data(), 6));
        auto bit = [](const Series& s, std::int64_t i) {
            return (s.data<std::uint8_t>()[i >> 3] >> (i & 7)) & 1;
        };
        DataFrame du = df.lazy().is_duplicated().collect(2);
        CHECK(du.names == std::vector<std::string>{"is_duplicated"});
        REQUIRE(du.num_rows() == 6);
        Series ed = df.is_duplicated();
        for (std::int64_t i = 0; i < 6; ++i)
            CHECK(bit(du.column("is_duplicated"), i) == bit(ed, i));

        DataFrame uq = df.lazy().is_unique().collect(2);
        CHECK(uq.names == std::vector<std::string>{"is_unique"});
        Series eu = df.is_unique();
        for (std::int64_t i = 0; i < 6; ++i)
            CHECK(bit(uq.column("is_unique"), i) == bit(eu, i));

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

        // Streaming windowed agg must match eager across a multi-morsel scan,
        // including a sliding window (period > every). Ascending time.
        std::vector<std::int64_t> bt(60), bv(60);
        for (std::int64_t i = 0; i < 60; ++i) {
            bt[static_cast<std::size_t>(i)] = i;
            bv[static_cast<std::size_t>(i)] = i * 2;
        }
        DataFrame wf;
        wf.names = {"t", "v"};
        wf.columns.push_back(Series::flat_i64(bt.data(), 60));
        wf.columns.push_back(Series::flat_i64(bv.data(), 60));
        std::vector<GroupAgg> wa{{Agg::Sum, "v", "s", 0.0},
                                 {Agg::Count, "", "c", 0.0}};
        DataFrame wl =
            wf.lazy().group_by_dynamic("t", 5, 12, wa).collect(7);  // sliding
        DataFrame we = wf.group_by_dynamic("t", 5, 12, wa);
        REQUIRE(wl.num_rows() == we.num_rows());
        const std::int64_t* lt = wl.column("t").data<std::int64_t>();
        const std::int64_t* et = we.column("t").data<std::int64_t>();
        const std::int64_t* ls = wl.column("s").data<std::int64_t>();
        const std::int64_t* es = we.column("s").data<std::int64_t>();
        const std::int64_t* lc = wl.column("c").data<std::int64_t>();
        const std::int64_t* ec = we.column("c").data<std::int64_t>();
        for (std::int64_t i = 0; i < wl.num_rows(); ++i) {
            CHECK(lt[i] == et[i]);
            CHECK(ls[i] == es[i]);
            CHECK(lc[i] == ec[i]);
        }
    }

    TEST_CASE("data-dependent schema: pivot / to_dummies / describe") {
        // to_dummies: one Int8 column per distinct value of "g".
        std::vector<std::int64_t> g{0, 1, 0};
        DataFrame gf;
        gf.names = {"g"};
        gf.columns.push_back(Series::flat_i64(g.data(), 3));
        LazyFrame dl = gf.lazy().to_dummies("g");
        CHECK(dl.schema().empty());    // unknown until run
        DataFrame d = dl.collect(2);
        DataFrame dd = gf.to_dummies("g");
        REQUIRE(d.names == dd.names);  // g_0, g_1 in ascending order
        REQUIRE(d.num_rows() == dd.num_rows());
        for (const std::string& cn : d.names) {
            const std::int8_t* a = d.column(cn).data<std::int8_t>();
            const std::int8_t* b = dd.column(cn).data<std::int8_t>();
            for (std::int64_t r = 0; r < d.num_rows(); ++r) CHECK(a[r] == b[r]);
        }

        // pivot: rows = distinct index, one value column per distinct "on".
        std::vector<std::int64_t> i{0, 0, 1, 1}, k{10, 20, 10, 20},
            v{1, 2, 3, 4};
        DataFrame pf;
        pf.names = {"i", "k", "v"};
        pf.columns.push_back(Series::flat_i64(i.data(), 4));
        pf.columns.push_back(Series::flat_i64(k.data(), 4));
        pf.columns.push_back(Series::flat_i64(v.data(), 4));
        DataFrame p = pf.lazy().pivot("i", "k", "v", "sum").collect(2);
        DataFrame pe = pf.pivot("i", "k", "v", "sum");
        REQUIRE(p.names == pe.names);  // i, 10, 20 (ascending on-values)
        REQUIRE(p.num_rows() == pe.num_rows());
        for (const std::string& cn : p.names) {
            const std::int64_t* a = p.column(cn).data<std::int64_t>();
            const std::int64_t* b = pe.column(cn).data<std::int64_t>();
            for (std::int64_t r = 0; r < p.num_rows(); ++r) CHECK(a[r] == b[r]);
        }

        // describe: streaming stats must match eager, across small morsels.
        DataFrame ds = make_df().lazy().describe().collect(2);
        DataFrame de = make_df().describe();
        REQUIRE(ds.names == de.names);
        REQUIRE(ds.num_rows() == de.num_rows());  // 6 statistics
        for (const std::string& cn : {std::string("a"), std::string("b")}) {
            const double* s = ds.column(cn).data<double>();
            const double* e = de.column(cn).data<double>();
            for (std::int64_t r = 0; r < ds.num_rows(); ++r)
                CHECK(s[r] == doctest::Approx(e[r]));
        }
    }

    TEST_CASE("two-pass sinks spill the input under a tiny budget") {
        // memory_budget(1) forces the input spool to disk between passes; the
        // results must still match the in-memory path.
        std::vector<std::int64_t> x{5, 3, 5, 1, 3, 5, 2, 1};
        DataFrame df;
        df.names = {"x"};
        df.columns.push_back(Series::flat_i64(x.data(), 8));
        auto bit = [](const Series& s, std::int64_t i) {
            return (s.data<std::uint8_t>()[i >> 3] >> (i & 7)) & 1;
        };
        DataFrame du = df.lazy().memory_budget(1).is_duplicated().collect(2);
        Series ed = df.is_duplicated();
        REQUIRE(du.num_rows() == 8);
        for (std::int64_t i = 0; i < 8; ++i)
            CHECK(bit(du.column("is_duplicated"), i) == bit(ed, i));

        std::vector<std::int64_t> i2{0, 0, 1, 1}, k2{10, 20, 10, 20},
            v2{1, 2, 3, 4};
        DataFrame pf;
        pf.names = {"i", "k", "v"};
        pf.columns.push_back(Series::flat_i64(i2.data(), 4));
        pf.columns.push_back(Series::flat_i64(k2.data(), 4));
        pf.columns.push_back(Series::flat_i64(v2.data(), 4));
        DataFrame p =
            pf.lazy().memory_budget(1).pivot("i", "k", "v", "sum").collect(2);
        DataFrame pe = pf.pivot("i", "k", "v", "sum");
        REQUIRE(p.names == pe.names);
        for (const std::string& cn : p.names) {
            const std::int64_t* a = p.column(cn).data<std::int64_t>();
            const std::int64_t* b = pe.column(cn).data<std::int64_t>();
            for (std::int64_t r = 0; r < p.num_rows(); ++r) CHECK(a[r] == b[r]);
        }
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
