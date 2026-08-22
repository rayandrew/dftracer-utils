#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/query/query.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

using dftracer::utils::dataframe::Agg;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::GroupAgg;
using dftracer::utils::dataframe::Series;

namespace {
Series i64(std::vector<std::int64_t> v) {
    return Series::flat_i64(v.data(), static_cast<std::int64_t>(v.size()));
}
// An Int64 column with an Arrow validity bitmap; `valid[i]==0` marks a null.
Series i64_nullable(std::vector<std::int64_t> v, std::vector<int> valid) {
    const std::int64_t n = static_cast<std::int64_t>(v.size());
    std::vector<std::uint8_t> bitmap(static_cast<std::size_t>((n + 7) / 8), 0);
    for (std::int64_t i = 0; i < n; ++i)
        if (valid[static_cast<std::size_t>(i)])
            bitmap[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    return Series::flat_i64(v.data(), n, bitmap.data());
}
}  // namespace

TEST_CASE("Series reductions via methods") {
    Series a = i64({3, 1, 4, 1, 5});
    CHECK(a.count() == 5);
    CHECK(a.sum().f64() == doctest::Approx(14.0));
    CHECK(a.min().f64() == doctest::Approx(1.0));
    CHECK(a.max().f64() == doctest::Approx(5.0));
    CHECK(a.mean() == doctest::Approx(2.8));
    CHECK(a.nunique() == 4);
    CHECK(a.median() == doctest::Approx(3.0));
}

TEST_CASE("Series arithmetic + operators") {
    Series a = i64({1, 2, 3});
    Series b = i64({10, 20, 30});
    Series sum = a + b;
    CHECK(sum.data<std::int64_t>()[0] == 11);
    CHECK(sum.data<std::int64_t>()[2] == 33);
    Series prod = a.mul(b);
    CHECK(prod.data<std::int64_t>()[1] == 40);
}

TEST_CASE("Series comparison + logical produce masks, drive filter") {
    Series a = i64({5, 15, 25, 35});
    Series big = a > 10;                         // mask: 0,1,1,1
    Series small = a.lt(30);                     // mask: 1,1,1,0
    Series both = big & small;                   // mask: 0,1,1,0
    Series kept = a.filter(both).materialize();  // selection -> flat
    REQUIRE(kept.length() == 2);
    CHECK(kept.data<std::int64_t>()[0] == 15);
    CHECK(kept.data<std::int64_t>()[1] == 25);
}

TEST_CASE("DataFrame select / filter / sort_by / column") {
    DataFrame df;
    df.names = {"id", "val"};
    df.columns.push_back(i64({1, 2, 3, 4}));
    df.columns.push_back(i64({40, 10, 30, 20}));
    CHECK(df.num_rows() == 4);
    CHECK(df.column_index("val") == 1);
    CHECK(df.column_index("missing") == -1);

    DataFrame just_val = df.select({"val"});
    CHECK(just_val.num_columns() == 1);

    DataFrame sorted = df.sort_by("val");
    CHECK(sorted.column("val").data<std::int64_t>()[0] == 10);
    CHECK(sorted.column("id").data<std::int64_t>()[0] == 2);

    DataFrame top2 = df.filter(df.column("val").gt(25));
    CHECK(top2.num_rows() == 2);
}

TEST_CASE("DataFrame group_by aggregates") {
    DataFrame df;
    df.names = {"k", "v"};
    df.columns.push_back(i64({1, 2, 1, 2, 1}));
    df.columns.push_back(i64({10, 5, 20, 7, 30}));
    DataFrame g = df.group_by(
        "k", {GroupAgg{Agg::Sum, "v", "total"}, GroupAgg{Agg::Count, "", "n"}});
    CHECK(g.num_rows() == 2);
    // key 1 -> sum 60 count 3 ; key 2 -> sum 12 count 2 (first-seen order)
    const std::int64_t* total = g.column("total").data<std::int64_t>();
    const std::int64_t* n = g.column("n").data<std::int64_t>();
    CHECK(total[0] == 60);
    CHECK(n[0] == 3);
    CHECK(total[1] == 12);
    CHECK(n[1] == 2);
}

TEST_CASE("dftu_dataframe opaque C ABI: build, inspect, frame ops") {
    Series id = i64({1, 2, 3, 4});
    Series val = i64({40, 10, 30, 20});
    const char* names[] = {"id", "val"};
    dftu_series* cols[] = {id.release(), val.release()};  // ownership moves in
    dftu_dataframe* df = dftu_dataframe_new(names, cols, 2);
    REQUIRE(df != nullptr);
    CHECK(dftu_dataframe_num_rows(df) == 4);
    CHECK(dftu_dataframe_num_columns(df) == 2);
    CHECK(std::string(dftu_dataframe_column_name(df, 1)) == "val");
    CHECK(dftu_dataframe_column_name(df, 5) == nullptr);

    // sort_by "val" ascending -> id column reorders to [2,4,3,1]
    dftu_dataframe* sorted = dftu_dataframe_sort_by(df, "val", 0);
    dftu_series* sid = dftu_dataframe_column(sorted, "id");
    REQUIRE(sid != nullptr);
    CHECK(Series{sid}.data<std::int64_t>()[0] == 2);

    const char* keep[] = {"val"};
    dftu_dataframe* proj = dftu_dataframe_select(df, keep, 1);
    CHECK(dftu_dataframe_num_columns(proj) == 1);
    CHECK(dftu_dataframe_column(proj, "id") == nullptr);

    // filter val > 25 -> 2 rows (40, 30)
    Series mask = Series{dftu_dataframe_column(df, "val")}.gt(std::int64_t(25));
    dftu_dataframe* kept = dftu_dataframe_filter(df, mask.handle());
    CHECK(dftu_dataframe_num_rows(kept) == 2);

    dftu_dataframe_free(kept);
    dftu_dataframe_free(proj);
    dftu_dataframe_free(sorted);
    dftu_dataframe_free(df);
}

TEST_CASE("Expr operators build and evaluate a fused expression") {
    namespace df = dftracer::utils::dataframe;
    Series a = i64({1, 2, 3, 4});
    Series b = i64({10, 20, 30, 40});
    // (a + b) evaluated over [a, b]
    df::Expr e = df::col(0) + df::col(1);
    Series r = df::eval(e, {&a, &b});
    REQUIRE(r.length() == 4);
    CHECK(r.data<std::int64_t>()[0] == 11);
    CHECK(r.data<std::int64_t>()[3] == 44);
    // a comparison expression yields a Bool mask
    df::Expr mask = df::col(0) > std::int64_t(2);
    Series m = df::eval(mask, {&a});
    CHECK(((m.data<std::uint8_t>()[0] >> 0) & 1) == 0);  // 1 > 2 = false
    CHECK(((m.data<std::uint8_t>()[0] >> 3) & 1) == 1);  // 4 > 2 = true
}

TEST_CASE("DataFrame tail / reverse / with_row_index / sample") {
    DataFrame df;
    df.names = {"id", "val"};
    df.columns.push_back(i64({1, 2, 3, 4, 5}));
    df.columns.push_back(i64({10, 20, 30, 40, 50}));

    DataFrame t = df.tail(2);
    REQUIRE(t.num_rows() == 2);
    CHECK(t.column("id").data<std::int64_t>()[0] == 4);
    CHECK(t.column("id").data<std::int64_t>()[1] == 5);

    DataFrame r = df.reverse();
    REQUIRE(r.num_rows() == 5);
    CHECK(r.column("id").data<std::int64_t>()[0] == 5);
    CHECK(r.column("val").data<std::int64_t>()[0] == 50);

    DataFrame wi = df.with_row_index("row");
    REQUIRE(wi.num_columns() == 3);
    CHECK(wi.names[0] == "row");
    CHECK(wi.column("row").data<std::int64_t>()[0] == 0);
    CHECK(wi.column("row").data<std::int64_t>()[4] == 4);

    // Deterministic: same seed -> same rows, and the sampled rows stay aligned.
    DataFrame s1 = df.sample(3, 42);
    DataFrame s2 = df.sample(3, 42);
    REQUIRE(s1.num_rows() == 3);
    for (std::int64_t i = 0; i < 3; ++i) {
        std::int64_t id = s1.column("id").data<std::int64_t>()[i];
        CHECK(s2.column("id").data<std::int64_t>()[i] == id);
        CHECK(s1.column("val").data<std::int64_t>()[i] == id * 10);
    }
}

TEST_CASE("DataFrame sort_by_multi lexicographic") {
    DataFrame df;
    df.names = {"a", "b"};
    df.columns.push_back(i64({2, 1, 2, 1}));
    df.columns.push_back(i64({9, 8, 7, 6}));
    DataFrame s = df.sort_by_multi({"a", "b"}, false);
    // a asc then b asc: (1,6),(1,8),(2,7),(2,9)
    const std::int64_t* a = s.column("a").data<std::int64_t>();
    const std::int64_t* b = s.column("b").data<std::int64_t>();
    CHECK(a[0] == 1);
    CHECK(b[0] == 6);
    CHECK(a[1] == 1);
    CHECK(b[1] == 8);
    CHECK(a[2] == 2);
    CHECK(b[2] == 7);
    CHECK(a[3] == 2);
    CHECK(b[3] == 9);
}

TEST_CASE("DataFrame unique / is_duplicated / is_unique") {
    DataFrame df;
    df.names = {"k", "v"};
    df.columns.push_back(i64({1, 2, 1, 3, 2}));
    df.columns.push_back(i64({7, 8, 7, 9, 8}));
    // rows: (1,7) dup, (2,8) dup, (1,7) dup, (3,9) unique, (2,8) dup
    DataFrame u = df.drop_duplicates();
    REQUIRE(u.num_rows() == 3);  // (1,7),(2,8),(3,9) keep first
    CHECK(u.column("k").data<std::int64_t>()[0] == 1);
    CHECK(u.column("k").data<std::int64_t>()[2] == 3);

    Series dup = df.is_duplicated();
    const std::uint8_t* db = dup.data<std::uint8_t>();
    auto bit = [](const std::uint8_t* p, std::int64_t i) {
        return (p[i >> 3] >> (i & 7)) & 1;
    };
    CHECK(bit(db, 0) == 1);
    CHECK(bit(db, 3) == 0);  // (3,9) not duplicated
    Series uniq = df.is_unique();
    const std::uint8_t* ub = uniq.data<std::uint8_t>();
    CHECK(bit(ub, 3) == 1);
    CHECK(bit(ub, 0) == 0);
}

TEST_CASE("DataFrame drop_nulls / fill_null / null_count") {
    DataFrame df;
    df.names = {"a", "b"};
    df.columns.push_back(i64_nullable({1, 2, 3, 4}, {1, 0, 1, 1}));
    df.columns.push_back(i64_nullable({5, 6, 7, 8}, {1, 1, 1, 0}));

    DataFrame nc = df.null_count();
    REQUIRE(nc.num_rows() == 1);
    CHECK(nc.column("a").data<std::int64_t>()[0] == 1);
    CHECK(nc.column("b").data<std::int64_t>()[0] == 1);

    DataFrame dn = df.drop_nulls();
    REQUIRE(dn.num_rows() == 2);  // rows 0 and 2 fully non-null
    CHECK(dn.column("a").data<std::int64_t>()[0] == 1);
    CHECK(dn.column("a").data<std::int64_t>()[1] == 3);

    DataFrame f = df.fill_null(-1);
    CHECK(f.column("a").null_count() == 0);
    CHECK(f.column("a").data<std::int64_t>()[1] == -1);
    CHECK(f.column("b").data<std::int64_t>()[3] == -1);
}

TEST_CASE("Ergonomic scalar API: no raw dftu_scalar at the call site") {
    // A bare literal 0 must resolve unambiguously against the dftu_scalar
    // overload.
    DataFrame df;
    df.names = {"a"};
    df.columns.push_back(i64_nullable({1, 2, 3, 4}, {1, 0, 1, 1}));
    DataFrame f = df.fill_null(0);
    CHECK(f.column("a").null_count() == 0);
    CHECK(f.column("a").data<std::int64_t>()[1] == 0);

    Series ints = i64({-5, 5, 50, 500});
    Series bounded = ints.clip(0, 100);
    CHECK(bounded.data<std::int64_t>()[0] == 0);
    CHECK(bounded.data<std::int64_t>()[1] == 5);
    CHECK(bounded.data<std::int64_t>()[3] == 100);

    std::vector<double> raw = {1.0, 2.0, 3.0};
    Series floats = Series::flat_f64(raw.data(), 3);
    Series filled = floats.fillna(1.5);
    CHECK(filled.length() == 3);

    // Factory helpers cover anywhere a bare dftu_scalar is still needed.
    namespace df_ns = dftracer::utils::dataframe;
    Series still = ints.clip(df_ns::i64(0), df_ns::i64(100));
    CHECK(still.data<std::int64_t>()[0] == 0);
    Series fnan = floats.fillna(df_ns::f64(2.5));
    CHECK(fnan.length() == 3);
}

TEST_CASE("DataFrame describe") {
    DataFrame df;
    df.names = {"x"};
    df.columns.push_back(i64({2, 4, 6, 8}));
    DataFrame d = df.describe();
    REQUIRE(d.num_rows() == 6);   // count,null_count,mean,std,min,max
    CHECK(d.num_columns() == 2);  // statistic + x
    CHECK(d.names[0] == "statistic");
    CHECK(d.names[1] == "x");
    const double* x = d.column("x").data<double>();
    CHECK(x[0] == doctest::Approx(4.0));  // count
    CHECK(x[1] == doctest::Approx(0.0));  // null_count
    CHECK(x[2] == doctest::Approx(5.0));  // mean
    CHECK(x[4] == doctest::Approx(2.0));  // min
    CHECK(x[5] == doctest::Approx(8.0));  // max
}

TEST_CASE("Series value_counts most-frequent first") {
    Series v = i64({1, 2, 1, 3, 1, 2});
    dftu_dataframe* vc = dftu_series_value_counts(v.handle());
    REQUIRE(vc != nullptr);
    CHECK(dftu_dataframe_num_rows(vc) == 3);
    dftu_series* value = dftu_dataframe_column(vc, "value");
    dftu_series* count = dftu_dataframe_column(vc, "count");
    REQUIRE(value != nullptr);
    REQUIRE(count != nullptr);
    // most frequent is 1 (x3)
    CHECK(Series{value}.data<std::int64_t>()[0] == 1);
    CHECK(Series{count}.data<std::int64_t>()[0] == 3);
    dftu_dataframe_free(vc);
}

TEST_CASE("dftu_dataframe new frame ops via C ABI") {
    Series id = i64({1, 2, 3, 3});
    Series val = i64({10, 20, 30, 30});
    const char* names[] = {"id", "val"};
    dftu_series* cols[] = {id.release(), val.release()};
    dftu_dataframe* df = dftu_dataframe_new(names, cols, 2);
    REQUIRE(df != nullptr);

    dftu_dataframe* uniq = dftu_dataframe_unique(df);
    CHECK(dftu_dataframe_num_rows(uniq) == 3);  // (3,30) deduped

    dftu_dataframe* rev = dftu_dataframe_reverse(df);
    dftu_series* rid = dftu_dataframe_column(rev, "id");
    CHECK(Series{rid}.data<std::int64_t>()[0] == 3);

    dftu_dataframe* tl = dftu_dataframe_tail(df, 1);
    CHECK(dftu_dataframe_num_rows(tl) == 1);

    dftu_series* dupmask = dftu_dataframe_is_duplicated(df);
    REQUIRE(dupmask != nullptr);
    dftu_series_free(dupmask);

    dftu_dataframe_free(tl);
    dftu_dataframe_free(rev);
    dftu_dataframe_free(uniq);
    dftu_dataframe_free(df);
}

TEST_CASE("DataFrame unpivot / melt wide -> long") {
    DataFrame df;
    df.names = {"id", "a", "b"};
    df.columns.push_back(i64({1, 2}));
    df.columns.push_back(i64({10, 20}));
    df.columns.push_back(i64({30, 40}));

    DataFrame m = df.unpivot({"id"}, {"a", "b"});
    REQUIRE(m.num_rows() == 4);  // 2 rows * 2 value_vars
    REQUIRE(m.num_columns() == 3);
    CHECK(m.names[0] == "id");
    CHECK(m.names[1] == "variable");
    CHECK(m.names[2] == "value");
    // Block order: all of "a" then all of "b".
    const std::int64_t* id = m.column("id").data<std::int64_t>();
    const std::int64_t* val = m.column("value").data<std::int64_t>();
    CHECK(id[0] == 1);
    CHECK(id[1] == 2);
    CHECK(id[2] == 1);
    CHECK(id[3] == 2);
    CHECK(val[0] == 10);
    CHECK(val[1] == 20);
    CHECK(val[2] == 30);
    CHECK(val[3] == 40);
    Series var = m.column("variable");
    CHECK(var.string_at(0) == "a");
    CHECK(var.string_at(3) == "b");

    // melt is an alias.
    DataFrame m2 = df.melt({"id"}, {"a", "b"});
    CHECK(m2.num_rows() == 4);
}

TEST_CASE("DataFrame explode a List column") {
    DataFrame df;
    df.names = {"id", "vals"};
    df.columns.push_back(i64({1, 2, 3}));
    // Lists: [10,20], [], [30]  (row 1 empty -> one null row)
    std::vector<std::int32_t> offsets = {0, 2, 2, 3};
    Series child = i64({10, 20, 30});
    df.columns.push_back(Series::list(offsets, std::move(child)));

    DataFrame e = df.explode("vals");
    REQUIRE(e.num_rows() == 4);  // 2 + 1(null) + 1
    const std::int64_t* id = e.column("id").data<std::int64_t>();
    CHECK(id[0] == 1);
    CHECK(id[1] == 1);
    CHECK(id[2] == 2);  // empty-list row repeated once
    CHECK(id[3] == 3);
    Series v = e.column("vals");
    CHECK(v.data<std::int64_t>()[0] == 10);
    CHECK(v.data<std::int64_t>()[1] == 20);
    CHECK(v.is_null(2));  // empty list -> null value
    CHECK(v.data<std::int64_t>()[3] == 30);
}

TEST_CASE("DataFrame to_dummies one-hot") {
    DataFrame df;
    df.names = {"id", "k"};
    df.columns.push_back(i64({1, 2, 3, 4}));
    df.columns.push_back(i64({7, 8, 7, 9}));  // distinct sorted: 7,8,9

    DataFrame oh = df.to_dummies("k");
    // id + three dummy columns (7,8,9)
    REQUIRE(oh.num_columns() == 4);
    CHECK(oh.names[0] == "id");
    CHECK(oh.names[1] == "k_7");
    CHECK(oh.names[2] == "k_8");
    CHECK(oh.names[3] == "k_9");
    const std::int8_t* d7 = oh.column("k_7").data<std::int8_t>();
    const std::int8_t* d8 = oh.column("k_8").data<std::int8_t>();
    const std::int8_t* d9 = oh.column("k_9").data<std::int8_t>();
    CHECK(d7[0] == 1);
    CHECK(d7[2] == 1);
    CHECK(d7[1] == 0);
    CHECK(d8[1] == 1);
    CHECK(d9[3] == 1);
    CHECK(d9[0] == 0);
}

TEST_CASE("DataFrame pivot long->wide") {
    using dftracer::utils::dataframe::TypeId;
    DataFrame df;
    df.names = {"idx", "col", "val"};
    df.columns.push_back(i64({1, 1, 2, 1}));
    df.columns.push_back(Series::strings({"a", "b", "a", "a"}));
    df.columns.push_back(i64({10, 20, 30, 40}));
    // (idx=1, col=a) collides across rows 0 (10) and 3 (40).

    SUBCASE("first keeps the earliest, null where a pair is absent") {
        DataFrame p = df.pivot("idx", "col", "val", "first");
        REQUIRE(p.num_rows() == 2);     // idx 1, 2 (sorted)
        REQUIRE(p.num_columns() == 3);  // idx + col a, b (sorted)
        CHECK(p.names[0] == "idx");
        CHECK(p.names[1] == "a");
        CHECK(p.names[2] == "b");
        Series a = p.column("a");
        Series b = p.column("b");
        CHECK(a.data<std::int64_t>()[0] == 10);  // idx1: first of (10,40)
        CHECK(a.data<std::int64_t>()[1] == 30);  // idx2
        CHECK(b.data<std::int64_t>()[0] == 20);  // idx1 col b
        CHECK(b.is_null(1));                     // idx2 has no col b
    }
    SUBCASE("typed Agg overload matches the string form") {
        DataFrame p = df.pivot("idx", "col", "val", Agg::First);
        REQUIRE(p.num_rows() == 2);
        CHECK(p.column("a").data<std::int64_t>()[0] == 10);
        CHECK(p.column("a").data<std::int64_t>()[1] == 30);
    }
    SUBCASE("mean reduces collisions to Float64") {
        DataFrame p = df.pivot("idx", "col", "val", "mean");
        Series a = p.column("a");
        CHECK(a.type() == TypeId::Float64);
        CHECK(a.data<double>()[0] == doctest::Approx(25.0));  // (10+40)/2
        CHECK(a.data<double>()[1] == doctest::Approx(30.0));
    }
}

TEST_CASE("DataFrame group_by_dynamic tumbling and sliding") {
    DataFrame df;
    df.names = {"t", "v"};
    df.columns.push_back(i64({0, 1, 2, 3, 4, 5}));
    df.columns.push_back(i64({10, 20, 30, 40, 50, 60}));

    SUBCASE("tumbling (period defaults to every)") {
        DataFrame g = df.group_by_dynamic(
            "t", 2, 0,
            {GroupAgg{Agg::Sum, "v", "sv"}, GroupAgg{Agg::Count, "", "n"}});
        REQUIRE(g.num_rows() == 3);
        const std::int64_t* start = g.column("t").data<std::int64_t>();
        const std::int64_t* sv = g.column("sv").data<std::int64_t>();
        const std::int64_t* n = g.column("n").data<std::int64_t>();
        CHECK(start[0] == 0);
        CHECK(sv[0] == 30);   // [0,2): 10+20
        CHECK(n[0] == 2);
        CHECK(start[1] == 2);
        CHECK(sv[1] == 70);   // [2,4): 30+40
        CHECK(start[2] == 4);
        CHECK(sv[2] == 110);  // [4,6): 50+60
    }
    SUBCASE("sliding (period > every overlaps)") {
        DataFrame g =
            df.group_by_dynamic("t", 2, 4, {GroupAgg{Agg::Sum, "v", "sv"}});
        REQUIRE(g.num_rows() == 3);
        const std::int64_t* start = g.column("t").data<std::int64_t>();
        const std::int64_t* sv = g.column("sv").data<std::int64_t>();
        CHECK(start[0] == 0);
        CHECK(sv[0] == 100);  // [0,4): 10+20+30+40
        CHECK(start[1] == 2);
        CHECK(sv[1] == 180);  // [2,6): 30+40+50+60
        CHECK(start[2] == 4);
        CHECK(sv[2] == 110);  // [4,8): 50+60
    }
}

TEST_CASE("dftu_dataframe reshaping ops via C ABI") {
    Series id = i64({1, 2});
    Series a = i64({10, 20});
    Series bcol = i64({30, 40});
    const char* names[] = {"id", "a", "b"};
    dftu_series* cols[] = {id.release(), a.release(), bcol.release()};
    dftu_dataframe* df = dftu_dataframe_new(names, cols, 3);
    REQUIRE(df != nullptr);

    const char* ids[] = {"id"};
    const char* vals[] = {"a", "b"};
    dftu_dataframe* m = dftu_dataframe_unpivot(df, ids, 1, vals, 2);
    REQUIRE(m != nullptr);
    CHECK(dftu_dataframe_num_rows(m) == 4);
    CHECK(dftu_dataframe_num_columns(m) == 3);

    dftu_dataframe* oh = dftu_dataframe_to_dummies(df, "a");
    REQUIRE(oh != nullptr);
    // a has distinct 10,20 -> id, a_10, a_20, b
    CHECK(dftu_dataframe_num_columns(oh) == 4);

    // absent column -> NULL
    CHECK(dftu_dataframe_explode(df, "missing") == nullptr);
    // non-list column -> NULL
    CHECK(dftu_dataframe_explode(df, "a") == nullptr);

    // pivot via the C ABI: index=id, columns=id, values=a, "first".
    dftu_dataframe* pv = dftu_dataframe_pivot(df, "id", "id", "a", "first");
    REQUIRE(pv != nullptr);
    CHECK(dftu_dataframe_num_rows(pv) == 2);  // distinct id 1,2
    CHECK(dftu_dataframe_pivot(df, "missing", "id", "a", "first") == nullptr);

    // group_by_dynamic via the C ABI over the "id" time column (1,2).
    dftu_group_agg gaggs[] = {{"sum", "a", "sa"}, {"count", nullptr, "n"}};
    dftu_dataframe* gd =
        dftu_dataframe_group_by_dynamic(df, "id", 5, 0, gaggs, 2);
    REQUIRE(gd != nullptr);
    CHECK(dftu_dataframe_num_rows(gd) == 1);  // one window [0,5) holds id 1,2
    CHECK(dftu_dataframe_group_by_dynamic(df, "id", 0, 0, gaggs, 2) == nullptr);

    dftu_dataframe_free(gd);
    dftu_dataframe_free(pv);
    dftu_dataframe_free(oh);
    dftu_dataframe_free(m);
    dftu_dataframe_free(df);
}

TEST_CASE("DataFrame mask from a compiled query") {
    DataFrame df;
    df.names = {"cat", "dur"};
    df.columns.push_back(
        Series::strings({"POSIX", "STDIO", "POSIX", "MPI", "POSIX"}));
    df.columns.push_back(i64({10, 20, 30, 40, 50}));

    auto q = dftracer::utils::query::parse_or_throw("cat == \"POSIX\"");
    Series m = df.mask(q);
    REQUIRE(m.length() == 5);
    const std::uint8_t* bits = m.data<std::uint8_t>();
    auto set = [&](std::int64_t i) { return (bits[i >> 3] >> (i & 7)) & 1; };
    CHECK(set(0));
    CHECK_FALSE(set(1));
    CHECK(set(2));
    CHECK_FALSE(set(3));
    CHECK(set(4));

    DataFrame kept = df.filter(m);
    REQUIRE(kept.num_rows() == 3);
    const std::int64_t* dur = kept.column("dur").data<std::int64_t>();
    CHECK(dur[0] == 10);
    CHECK(dur[1] == 30);
    CHECK(dur[2] == 50);
}

TEST_CASE("DataFrame mask throws when a predicate cannot be lowered") {
    DataFrame df;
    df.names = {"cat"};
    df.columns.push_back(Series::strings({"POSIX", "STDIO"}));
    // References a column absent from the frame; no columnar lowering exists.
    auto q = dftracer::utils::query::parse_or_throw("dur > 5");
    CHECK_THROWS_AS((void)df.mask(q), std::runtime_error);
}
