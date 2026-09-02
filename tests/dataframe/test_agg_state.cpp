#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/agg.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace df = dftracer::utils::dataframe;
using df::AggOp;
using df::AggSpec;
using df::DataFrame;
using df::Series;

namespace {

// Two-key sample: k1 in {0,1}, k2 in {0,1,2}, plus a numeric value, a tag
// string (SetUnion), a name string + a numeric by-column (ArgMax).
struct Sample {
    std::vector<std::int64_t> k1, k2, v, by;
    std::vector<std::string> tag, name;
    Series sk1, sk2, sv, sby, stag, sname;
    std::vector<const Series*> keys2, keys1, values;

    explicit Sample(std::int64_t n) {
        for (std::int64_t i = 0; i < n; ++i) {
            k1.push_back(i % 2);
            k2.push_back(i % 3);
            v.push_back(i);
            by.push_back(i);  // unique so ArgMax has no tie ambiguity
            tag.push_back("t" + std::to_string(i % 5));
            name.push_back("n" + std::to_string(i));
        }
        sk1 = Series::flat_i64(k1.data(), n);
        sk2 = Series::flat_i64(k2.data(), n);
        sv = Series::flat_i64(v.data(), n);
        sby = Series::flat_i64(by.data(), n);
        stag = Series::strings(tag);
        sname = Series::strings(name);
        keys2 = {&sk1, &sk2};
        keys1 = {&sk1};
        values = {&sv, &stag, &sname, &sby};
    }
};

std::vector<AggSpec> mixed_specs() {
    return {
        AggSpec{AggOp::Sum, 0, "sum"},
        AggSpec{AggOp::Count, -1, "cnt"},
        AggSpec{AggOp::Mean, 0, "mean"},
        AggSpec{AggOp::Pct, 0, "p50", 0.5},
        AggSpec{AggOp::Hist, 0, "hist"},
        AggSpec{AggOp::SetUnion, 1, "tags"},
        AggSpec{AggOp::ArgMax, 2, "argmax", 0.0, 3},
    };
}

}  // namespace

TEST_CASE("group_agg_state + agg_finalize matches group_agg") {
    Sample s(4096);
    auto st = df::group_agg_state(s.keys1, s.values, mixed_specs());
    DataFrame a = df::agg_finalize(*st, "k1");
    DataFrame b = df::group_agg(s.keys1, s.values, mixed_specs(),
                                std::vector<std::string>{"k1"});
    REQUIRE(a.num_rows() == b.num_rows());
    for (std::int64_t r = 0; r < a.num_rows(); ++r) {
        CHECK(a.column("sum").data<std::int64_t>()[r] ==
              b.column("sum").data<std::int64_t>()[r]);
        CHECK(a.column("cnt").data<std::int64_t>()[r] ==
              b.column("cnt").data<std::int64_t>()[r]);
    }
}

TEST_CASE("agg_regroup coarsens (k1,k2)->k1 matching a direct k1 group_agg") {
    Sample s(6000);
    auto fine = df::group_agg_state(s.keys2, s.values, mixed_specs());

    auto coarse = df::agg_regroup(*fine, {0});  // keep k1 only
    df::agg_sort_groups(*coarse);
    DataFrame got = df::agg_finalize(*coarse, "k1");

    auto direct_st = df::group_agg_state(s.keys1, s.values, mixed_specs());
    df::agg_sort_groups(*direct_st);
    DataFrame want = df::agg_finalize(*direct_st, "k1");

    REQUIRE(got.num_rows() == want.num_rows());
    for (std::int64_t r = 0; r < got.num_rows(); ++r) {
        CHECK(got.column("k1").data<std::int64_t>()[r] ==
              want.column("k1").data<std::int64_t>()[r]);
        CHECK(got.column("sum").data<std::int64_t>()[r] ==
              want.column("sum").data<std::int64_t>()[r]);
        CHECK(got.column("cnt").data<std::int64_t>()[r] ==
              want.column("cnt").data<std::int64_t>()[r]);
        CHECK(got.column("mean").data<double>()[r] ==
              doctest::Approx(want.column("mean").data<double>()[r]));
        CHECK(got.column("p50").data<double>()[r] ==
              doctest::Approx(want.column("p50").data<double>()[r]));
        CHECK(std::string(got.column("tags").string_at(r)) ==
              std::string(want.column("tags").string_at(r)));
        CHECK(std::string(got.column("argmax").string_at(r)) ==
              std::string(want.column("argmax").string_at(r)));
    }
    // Hist survives regroup as a list column with a row per group.
    CHECK(got.column("hist").length() == want.column("hist").length());
}

TEST_CASE("agg_regroup preserves occupancy busy/active exactly") {
    // ts/dur events across two keys; coarsen to one key and compare busy/active
    // against a direct single-key occupancy aggregation.
    const std::int64_t n = 2000;
    std::vector<std::int64_t> k1(n), k2(n), ts(n), dur(n);
    for (std::int64_t i = 0; i < n; ++i) {
        k1[static_cast<std::size_t>(i)] = i % 2;
        k2[static_cast<std::size_t>(i)] = i % 4;
        ts[static_cast<std::size_t>(i)] = i * 3;
        dur[static_cast<std::size_t>(i)] = 5;
    }
    Series sk1 = Series::flat_i64(k1.data(), n);
    Series sk2 = Series::flat_i64(k2.data(), n);
    Series sts = Series::flat_i64(ts.data(), n);
    Series sdur = Series::flat_i64(dur.data(), n);
    std::vector<const Series*> keys2{&sk1, &sk2};
    std::vector<const Series*> keys1{&sk1};
    std::vector<const Series*> values{&sts, &sdur};
    std::vector<AggSpec> specs{AggSpec{AggOp::Busy, 0, "busy", 0.0, 1},
                               AggSpec{AggOp::Active, 0, "active", 0.0, 1}};

    auto fine = df::group_agg_state(keys2, values, specs);
    auto coarse = df::agg_regroup(*fine, {0});
    df::agg_sort_groups(*coarse);
    DataFrame got = df::agg_finalize(*coarse, "k1");

    auto direct = df::group_agg_state(keys1, values, specs);
    df::agg_sort_groups(*direct);
    DataFrame want = df::agg_finalize(*direct, "k1");

    REQUIRE(got.num_rows() == want.num_rows());
    for (std::int64_t r = 0; r < got.num_rows(); ++r) {
        CHECK(got.column("busy").data<double>()[r] ==
              doctest::Approx(want.column("busy").data<double>()[r]));
        CHECK(got.column("active").data<double>()[r] ==
              doctest::Approx(want.column("active").data<double>()[r]));
    }
}

TEST_CASE("agg_regroup with bucket_recut coarsens the time grain") {
    // key 0 is a time bucket at grain 10; key 1 is a group. Recut to grain 20.
    const std::int64_t n = 4000;
    std::vector<std::int64_t> bucket(n), grp(n), v(n);
    for (std::int64_t i = 0; i < n; ++i) {
        bucket[static_cast<std::size_t>(i)] = (i % 8) * 10;  // 0,10,...,70
        grp[static_cast<std::size_t>(i)] = i % 2;
        v[static_cast<std::size_t>(i)] = i;
    }
    Series sb = Series::flat_i64(bucket.data(), n);
    Series sg = Series::flat_i64(grp.data(), n);
    Series sv = Series::flat_i64(v.data(), n);
    std::vector<const Series*> keys{&sb, &sg};
    std::vector<const Series*> values{&sv};
    std::vector<AggSpec> specs{AggSpec{AggOp::Sum, 0, "sum"},
                               AggSpec{AggOp::Count, -1, "cnt"}};

    auto fine = df::group_agg_state(keys, values, specs);
    auto coarse = df::agg_regroup(*fine, {0, 1}, /*bucket_recut=*/20);
    df::agg_sort_groups(*coarse);
    DataFrame got =
        df::agg_finalize(*coarse, std::vector<std::string>{"bucket", "grp"});

    // Direct: floor each bucket to grain 20 up front, then group.
    std::vector<std::int64_t> cb(n);
    for (std::int64_t i = 0; i < n; ++i)
        cb[static_cast<std::size_t>(i)] =
            (bucket[static_cast<std::size_t>(i)] / 20) * 20;
    Series scb = Series::flat_i64(cb.data(), n);
    std::vector<const Series*> ckeys{&scb, &sg};
    auto direct = df::group_agg_state(ckeys, values, specs);
    df::agg_sort_groups(*direct);
    DataFrame want =
        df::agg_finalize(*direct, std::vector<std::string>{"bucket", "grp"});

    REQUIRE(got.num_rows() == want.num_rows());
    for (std::int64_t r = 0; r < got.num_rows(); ++r) {
        CHECK(got.column("bucket").data<std::int64_t>()[r] ==
              want.column("bucket").data<std::int64_t>()[r]);
        CHECK(got.column("grp").data<std::int64_t>()[r] ==
              want.column("grp").data<std::int64_t>()[r]);
        CHECK(got.column("sum").data<std::int64_t>()[r] ==
              want.column("sum").data<std::int64_t>()[r]);
        CHECK(got.column("cnt").data<std::int64_t>()[r] ==
              want.column("cnt").data<std::int64_t>()[r]);
    }
}

TEST_CASE("agg_regroup over serialize/deserialize round-trip is exact") {
    Sample s(3000);
    auto fine = df::group_agg_state(s.keys2, s.values, mixed_specs());
    // Persist each group as a single-group blob, then rebuild one merged state
    // (the rollup read path), and coarsen that.
    auto rebuilt = df::agg_new(mixed_specs());
    const std::int64_t ng = df::agg_num_groups(*fine);
    for (std::int64_t g = 0; g < ng; ++g) {
        std::string blob = df::agg_serialize(*df::agg_extract_group(*fine, g));
        auto one = df::agg_deserialize(blob);
        df::agg_merge(*rebuilt, *one);
    }
    auto coarse = df::agg_regroup(*rebuilt, {0});
    df::agg_sort_groups(*coarse);
    DataFrame got = df::agg_finalize(*coarse, "k1");

    auto direct = df::group_agg_state(s.keys1, s.values, mixed_specs());
    df::agg_sort_groups(*direct);
    DataFrame want = df::agg_finalize(*direct, "k1");

    REQUIRE(got.num_rows() == want.num_rows());
    for (std::int64_t r = 0; r < got.num_rows(); ++r) {
        CHECK(got.column("sum").data<std::int64_t>()[r] ==
              want.column("sum").data<std::int64_t>()[r]);
        CHECK(got.column("p50").data<double>()[r] ==
              doctest::Approx(want.column("p50").data<double>()[r]));
        CHECK(std::string(got.column("tags").string_at(r)) ==
              std::string(want.column("tags").string_at(r)));
    }
}
