#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/utilities/common/arrow/array_view.h>
#include <dftracer/utils/utilities/common/arrow/arrow.h>
#include <doctest/doctest.h>
#include <nanoarrow/nanoarrow.h>

#include <limits>
#include <optional>
#include <vector>

using namespace dftracer::utils::utilities::common::arrow;

namespace {

struct Reader {
    ArrowArrayView av;
    explicit Reader(ArrowExportResult& r) {
        REQUIRE(init_array_view(av, r.get_schema(), r.get_array()) ==
                NANOARROW_OK);
    }
    ~Reader() { ArrowArrayViewReset(&av); }
    std::int64_t rows() const { return av.length; }
    std::int64_t i64(int c, std::int64_t r) const {
        return ArrowArrayViewGetIntUnsafe(av.children[c], r);
    }
    double dbl(int c, std::int64_t r) const {
        return ArrowArrayViewGetDoubleUnsafe(av.children[c], r);
    }
    bool null(int c, std::int64_t r) const {
        return ArrowArrayViewIsNull(av.children[c], r) != 0;
    }
};

// {t:INT64, v:INT64, x:INT64}; a nullopt v becomes a null cell, x = 900 + i.
ArrowExportResult make_tvx(const std::vector<std::int64_t>& t,
                           const std::vector<std::optional<std::int64_t>>& v) {
    RecordBatchBuilder b;
    b.declare_schema({{"t", ColumnType::INT64},
                      {"v", ColumnType::INT64},
                      {"x", ColumnType::INT64}});
    for (std::size_t i = 0; i < t.size(); ++i) {
        b.append_int64(0, t[i]);
        if (v[i])
            b.append_int64(1, *v[i]);
        else
            b.append_null(1);
        b.append_int64(2, 900 + static_cast<std::int64_t>(i));
        b.end_row();
    }
    return b.finish();
}

// {t:INT64, v:DOUBLE}.
ArrowExportResult make_tvd(const std::vector<std::int64_t>& t,
                           const std::vector<double>& v) {
    RecordBatchBuilder b;
    b.declare_schema({{"t", ColumnType::INT64}, {"v", ColumnType::DOUBLE}});
    for (std::size_t i = 0; i < t.size(); ++i) {
        b.append_int64(0, t[i]);
        b.append_double(1, v[i]);
        b.end_row();
    }
    return b.finish();
}

// {t:DOUBLE, v:DOUBLE}.
ArrowExportResult make_dtv(const std::vector<double>& t,
                           const std::vector<double>& v) {
    RecordBatchBuilder b;
    b.declare_schema({{"t", ColumnType::DOUBLE}, {"v", ColumnType::DOUBLE}});
    for (std::size_t i = 0; i < t.size(); ++i) {
        b.append_double(0, t[i]);
        b.append_double(1, v[i]);
        b.end_row();
    }
    return b.finish();
}

// {p:INT64, t:INT64, v:INT64}.
ArrowExportResult make_ptv(const std::vector<std::int64_t>& p,
                           const std::vector<std::int64_t>& t,
                           const std::vector<std::int64_t>& v) {
    RecordBatchBuilder b;
    b.declare_schema({{"p", ColumnType::INT64},
                      {"t", ColumnType::INT64},
                      {"v", ColumnType::INT64}});
    for (std::size_t i = 0; i < p.size(); ++i) {
        b.append_int64(0, p[i]);
        b.append_int64(1, t[i]);
        b.append_int64(2, v[i]);
        b.end_row();
    }
    return b.finish();
}

ArrowExportResult gf(ArrowExportResult& in, std::vector<std::uint32_t> part,
                     std::uint32_t tcol, std::int64_t width,
                     std::vector<std::uint32_t> vcols, GapFillMode mode,
                     bool has_range = false, std::int64_t rs = 0,
                     std::int64_t re = 0) {
    return gap_fill(in.get_schema(), in.get_array(), part.data(),
                    static_cast<std::uint32_t>(part.size()), tcol, width,
                    vcols.data(), static_cast<std::uint32_t>(vcols.size()),
                    mode, has_range, rs, re);
}

ArrowExportResult gfd(ArrowExportResult& in, std::vector<std::uint32_t> part,
                      std::uint32_t tcol, double width,
                      std::vector<std::uint32_t> vcols, GapFillMode mode,
                      bool has_range = false, double rs = 0, double re = 0) {
    return gap_fill(in.get_schema(), in.get_array(), part.data(),
                    static_cast<std::uint32_t>(part.size()), tcol, width,
                    vcols.data(), static_cast<std::uint32_t>(vcols.size()),
                    mode, has_range, rs, re);
}

}  // namespace

TEST_CASE("gap_fill - NONE leaves a missing bucket null, preserves real rows") {
    auto in = make_tvx({0, 10, 30}, {100, 200, 400});
    auto out = gf(in, {}, 0, 10, {1}, GapFillMode::NONE);
    Reader rd(out);
    REQUIRE(rd.rows() == 4);
    std::vector<std::int64_t> tt = {0, 10, 20, 30};
    for (std::int64_t i = 0; i < 4; ++i) CHECK(rd.i64(0, i) == tt[i]);
    CHECK(rd.i64(1, 0) == 100);
    CHECK(rd.i64(1, 1) == 200);
    CHECK(rd.null(1, 2));  // generated bucket 20 -> null value
    CHECK(rd.i64(1, 3) == 400);
    // non-value column x is null on the generated row, preserved on real rows.
    CHECK(rd.i64(2, 0) == 900);
    CHECK(rd.i64(2, 1) == 901);
    CHECK(rd.null(2, 2));
    CHECK(rd.i64(2, 3) == 902);
}

TEST_CASE("gap_fill - LOCF carries the last real value forward") {
    auto in = make_tvx({0, 10, 30}, {100, 200, 400});
    auto out = gf(in, {}, 0, 10, {1}, GapFillMode::LOCF);
    Reader rd(out);
    REQUIRE(rd.rows() == 4);
    CHECK(rd.i64(1, 2) == 200);  // bucket 20 carries bucket 10's value
    CHECK(rd.null(2, 2));        // x still null on generated row
}

TEST_CASE("gap_fill - LOCF leading gap before first real is null") {
    auto in = make_tvx({0, 10}, {100, 200});
    auto out = gf(in, {}, 0, 10, {1}, GapFillMode::LOCF, true, -20, 10);
    Reader rd(out);
    REQUIRE(rd.rows() == 4);  // grid -20,-10,0,10
    CHECK(rd.i64(0, 0) == -20);
    CHECK(rd.null(1, 0));     // leading gap -> null
    CHECK(rd.null(1, 1));     // leading gap -> null
    CHECK(rd.i64(1, 2) == 100);
    CHECK(rd.i64(1, 3) == 200);
}

TEST_CASE("gap_fill - LINEAR interpolates the missing bucket as double") {
    auto in = make_tvd({0, 20}, {100.0, 200.0});
    auto out = gf(in, {}, 0, 10, {1}, GapFillMode::LINEAR);
    Reader rd(out);
    REQUIRE(rd.rows() == 3);
    CHECK(rd.dbl(1, 0) == doctest::Approx(100.0));
    CHECK(rd.dbl(1, 1) == doctest::Approx(150.0));  // interpolated
    CHECK(rd.dbl(1, 2) == doctest::Approx(200.0));
}

TEST_CASE("gap_fill - LINEAR does not extrapolate past the real endpoints") {
    auto in = make_tvd({0, 20}, {100.0, 200.0});
    auto out = gf(in, {}, 0, 10, {1}, GapFillMode::LINEAR, true, -10, 30);
    Reader rd(out);
    REQUIRE(rd.rows() == 5);  // grid -10,0,10,20,30
    CHECK(rd.null(1, 0));     // before first -> null
    CHECK(rd.dbl(1, 1) == doctest::Approx(100.0));
    CHECK(rd.dbl(1, 2) == doctest::Approx(150.0));
    CHECK(rd.dbl(1, 3) == doctest::Approx(200.0));
    CHECK(rd.null(1, 4));  // after last -> null
}

TEST_CASE("gap_fill - double time column over a double grid") {
    SUBCASE("LOCF carries the last real value across a generated bucket") {
        auto in = make_dtv({0.0, 10.0, 30.0}, {100.0, 200.0, 400.0});
        auto out = gfd(in, {}, 0, 10.0, {1}, GapFillMode::LOCF);
        Reader rd(out);
        REQUIRE(rd.rows() == 4);  // grid 0,10,20,30
        std::vector<double> tt = {0.0, 10.0, 20.0, 30.0};
        for (std::int64_t i = 0; i < 4; ++i)
            CHECK(rd.dbl(0, i) == doctest::Approx(tt[i]));
        CHECK(rd.dbl(1, 0) == doctest::Approx(100.0));
        CHECK(rd.dbl(1, 1) == doctest::Approx(200.0));
        CHECK(rd.dbl(1, 2) == doctest::Approx(200.0));  // bucket 20 -> LOCF
        CHECK(rd.dbl(1, 3) == doctest::Approx(400.0));
    }
    SUBCASE("LINEAR interpolates the missing bucket") {
        auto in = make_dtv({0.0, 20.0}, {100.0, 200.0});
        auto out = gfd(in, {}, 0, 10.0, {1}, GapFillMode::LINEAR);
        Reader rd(out);
        REQUIRE(rd.rows() == 3);                        // grid 0,10,20
        CHECK(rd.dbl(1, 0) == doctest::Approx(100.0));
        CHECK(rd.dbl(1, 1) == doctest::Approx(150.0));  // interpolated
        CHECK(rd.dbl(1, 2) == doctest::Approx(200.0));
    }
    SUBCASE("fractional width aligns the grid via floor(t/width)*width") {
        auto in = make_dtv({0.0, 1.0}, {10.0, 20.0});
        auto out = gfd(in, {}, 0, 0.5, {1}, GapFillMode::LOCF);
        Reader rd(out);
        REQUIRE(rd.rows() == 3);                       // grid 0.0,0.5,1.0
        CHECK(rd.dbl(0, 1) == doctest::Approx(0.5));
        CHECK(rd.dbl(1, 1) == doctest::Approx(10.0));  // 0.5 -> LOCF of 0.0
    }
}

TEST_CASE("gap_fill - partitions are filled independently") {
    auto in = make_ptv({1, 1, 2, 2}, {0, 20, 0, 10}, {1, 3, 5, 7});
    auto out = gf(in, {0}, 1, 10, {2}, GapFillMode::NONE);
    Reader rd(out);
    // p1 grid 0,10,20 (10 generated); p2 grid 0,10 (no gap).
    REQUIRE(rd.rows() == 5);
    // p1 rows
    CHECK((rd.i64(0, 0) == 1 && rd.i64(1, 0) == 0 && rd.i64(2, 0) == 1));
    CHECK((rd.i64(0, 1) == 1 && rd.i64(1, 1) == 10 && rd.null(2, 1)));
    CHECK((rd.i64(0, 2) == 1 && rd.i64(1, 2) == 20 && rd.i64(2, 2) == 3));
    // p2 rows: grid bounded by p2's own min/max, no cross-partition fill.
    CHECK((rd.i64(0, 3) == 2 && rd.i64(1, 3) == 0 && rd.i64(2, 3) == 5));
    CHECK((rd.i64(0, 4) == 2 && rd.i64(1, 4) == 10 && rd.i64(2, 4) == 7));
}

TEST_CASE("gap_fill - explicit range unifies grids across partitions") {
    auto in = make_ptv({1, 1, 2, 2}, {0, 10, 20, 30}, {1, 2, 3, 4});
    SUBCASE("per-partition min/max keeps each grid tight") {
        auto out = gf(in, {0}, 1, 10, {2}, GapFillMode::NONE);
        Reader rd(out);
        CHECK(rd.rows() == 4);  // p1 {0,10}, p2 {20,30}, no generated rows
    }
    SUBCASE("explicit range gives every partition the same span") {
        auto out = gf(in, {0}, 1, 10, {2}, GapFillMode::NONE, true, 0, 30);
        Reader rd(out);
        REQUIRE(rd.rows() == 8);  // each partition -> 0,10,20,30
        CHECK((rd.i64(0, 0) == 1 && rd.i64(1, 0) == 0 && rd.i64(2, 0) == 1));
        CHECK((rd.i64(1, 2) == 20 && rd.null(2, 2)));  // p1 generated
        CHECK((rd.i64(0, 4) == 2 && rd.i64(1, 4) == 0 && rd.null(2, 4)));
    }
}

TEST_CASE("gap_fill - deterministic regardless of input row order") {
    auto a = make_ptv({1, 1, 2}, {0, 20, 10}, {1, 3, 5});
    auto b = make_ptv({2, 1, 1}, {10, 20, 0}, {5, 3, 1});
    auto oa = gf(a, {0}, 1, 10, {2}, GapFillMode::LOCF);
    auto ob = gf(b, {0}, 1, 10, {2}, GapFillMode::LOCF);
    Reader ra(oa), rb(ob);
    REQUIRE(ra.rows() == rb.rows());
    for (std::int64_t i = 0; i < ra.rows(); ++i) {
        CHECK(ra.i64(0, i) == rb.i64(0, i));
        CHECK(ra.i64(1, i) == rb.i64(1, i));
        CHECK(ra.null(2, i) == rb.null(2, i));
        if (!ra.null(2, i)) CHECK(ra.i64(2, i) == rb.i64(2, i));
    }
}

TEST_CASE("gap_fill - duplicate real times in one bucket keep the first") {
    auto in = make_tvx({0, 0, 20}, {100, 111, 200});
    auto out = gf(in, {}, 0, 10, {1}, GapFillMode::NONE);
    Reader rd(out);
    REQUIRE(rd.rows() == 3);     // grid 0,10,20
    CHECK(rd.i64(1, 0) == 100);  // first row at bucket 0 kept
    CHECK(rd.null(1, 1));        // bucket 10 generated
    CHECK(rd.i64(1, 2) == 200);
}

TEST_CASE("gap_fill - rejects a non-positive bucket width") {
    auto in = make_tvx({0, 10}, {1, 2});
    CHECK_THROWS_AS(gf(in, {}, 0, 0, {1}, GapFillMode::NONE),
                    dftracer::utils::DFTUtilsException);
}

TEST_CASE("gap_fill - runaway grid guard trips on a huge range") {
    auto in = make_tvx({0, 10}, {1, 2});
    CHECK_THROWS_AS(
        gf(in, {}, 0, 1, {1}, GapFillMode::NONE, true, 0, 1'000'000'000),
        dftracer::utils::DFTUtilsException);
}

TEST_CASE("gap_fill - grid guard trips before the span overflows int64") {
    // range spanning the full int64 domain: the point-count must be computed
    // without a signed subtraction overflow, then the guard rejects it.
    auto in = make_tvx({0, 10}, {1, 2});
    const std::int64_t lo = std::numeric_limits<std::int64_t>::min();
    const std::int64_t hi = std::numeric_limits<std::int64_t>::max();
    CHECK_THROWS_AS(gf(in, {}, 0, 1, {1}, GapFillMode::NONE, true, lo, hi),
                    dftracer::utils::DFTUtilsException);
}

#endif  // DFTRACER_UTILS_ENABLE_ARROW
