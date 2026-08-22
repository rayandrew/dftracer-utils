// MonoidAccumulator serialize/deserialize contract: every kind roundtrips its
// state, a deserialized partial merges identically to the live one, and a
// concatenated product-value row walks back component by component via the
// consumed counts.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/monoid.h>
#include <doctest/doctest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace plugins = dftracer::utils::plugins;
using plugins::minmax_kind;
using plugins::MinMaxFamily;
using plugins::MonoidAccumulator;

namespace {

std::string ser(const MonoidAccumulator& m) {
    std::string out;
    m.serialize(out);
    return out;
}

std::size_t deser(MonoidAccumulator& m, const std::string& bytes) {
    return m.deserialize(reinterpret_cast<const std::byte*>(bytes.data()),
                         bytes.size());
}

bool is_f64_readout(dftu_monoid_kind k) {
    return minmax_kind(k).family == MinMaxFamily::FLOAT ||
           k == DFTU_MONOID_SUM_F64;
}

bool uses_f64_add(dftu_monoid_kind k) {
    return is_f64_readout(k) || k == DFTU_MONOID_SKETCH;
}

void add_reps(MonoidAccumulator& m) {
    if (uses_f64_add(m.kind())) {
        m.add_f64(3.5, 1.0);
        m.add_f64(7.25, 1.0);
    } else {
        m.add_u64(7);
        m.add_u64(42);
    }
}

void check_scalar_eq(const MonoidAccumulator& a, const MonoidAccumulator& b) {
    if (is_f64_readout(a.kind()))
        CHECK(a.to_value().as.f64 == doctest::Approx(b.to_value().as.f64));
    else
        CHECK(a.to_value().as.u64 == b.to_value().as.u64);
}

void check_quantiles_eq(const MonoidAccumulator& a,
                        const MonoidAccumulator& b) {
    const dftu_quantiles qa = a.to_value().as.quant;
    const dftu_quantiles qb = b.to_value().as.quant;
    CHECK(qa.count == qb.count);
    CHECK(qa.min == doctest::Approx(qb.min));
    CHECK(qa.max == doctest::Approx(qb.max));
    CHECK(qa.mean == doctest::Approx(qb.mean));
    CHECK(qa.p50 == doctest::Approx(qb.p50).epsilon(0.02));
    CHECK(qa.p90 == doctest::Approx(qb.p90).epsilon(0.02));
    CHECK(qa.p95 == doctest::Approx(qb.p95).epsilon(0.02));
    CHECK(qa.p99 == doctest::Approx(qb.p99).epsilon(0.02));
}

const dftu_monoid_kind SCALAR_KINDS[] = {
    DFTU_MONOID_COUNTER,  DFTU_MONOID_SUM_F64, DFTU_MONOID_MIN_F64,
    DFTU_MONOID_MAX_F64,  DFTU_MONOID_MIN_U64, DFTU_MONOID_MAX_U64,
    DFTU_MONOID_BOOL_AND, DFTU_MONOID_BOOL_OR, DFTU_MONOID_BITSET_OR,
    DFTU_MONOID_DISTINCT, DFTU_MONOID_MIN_I8,  DFTU_MONOID_MIN_I16,
    DFTU_MONOID_MIN_I32,  DFTU_MONOID_MIN_I64, DFTU_MONOID_MIN_U8,
    DFTU_MONOID_MIN_U16,  DFTU_MONOID_MIN_U32, DFTU_MONOID_MIN_F32,
    DFTU_MONOID_MAX_I8,   DFTU_MONOID_MAX_I16, DFTU_MONOID_MAX_I32,
    DFTU_MONOID_MAX_I64,  DFTU_MONOID_MAX_U8,  DFTU_MONOID_MAX_U16,
    DFTU_MONOID_MAX_U32,  DFTU_MONOID_MAX_F32};

}  // namespace

TEST_SUITE("MonoidSerialize") {
    TEST_CASE("roundtrip: every fixed-scalar kind") {
        for (dftu_monoid_kind k : SCALAR_KINDS) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            add_reps(a);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            check_scalar_eq(a, b);
        }
    }

    TEST_CASE("roundtrip: SET_I64 and SET_STR") {
        for (dftu_monoid_kind k : {DFTU_MONOID_SET_I64, DFTU_MONOID_SET_STR}) {
            MonoidAccumulator a(k);
            a.add_u64(9);
            a.add_u64(4);
            a.add_u64(9);
            a.add_u64(100);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.sorted_elements() == b.sorted_elements());
        }
    }

    TEST_CASE("roundtrip: LIST_I64 and LIST_STR") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_LIST_I64, DFTU_MONOID_LIST_STR}) {
            MonoidAccumulator a(k);
            a.add_ordered(30, 3);
            a.add_ordered(10, 1);
            a.add_ordered(20, 2);
            a.add_ordered(10, 5);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.ordered_elements() == b.ordered_elements());
        }
    }

    TEST_CASE("roundtrip: DISTINCT (sparse and dense state)") {
        SUBCASE("sparse: a handful of distinct values") {
            MonoidAccumulator a(DFTU_MONOID_DISTINCT);
            for (std::uint64_t v = 0; v < 50; ++v) a.add_u64(v);
            const std::string bytes = ser(a);
            MonoidAccumulator b(DFTU_MONOID_DISTINCT);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.to_value().as.u64 == b.to_value().as.u64);
        }
        SUBCASE("dense: enough distinct values to promote past sparse") {
            MonoidAccumulator a(DFTU_MONOID_DISTINCT);
            for (std::uint64_t v = 0; v < 100000; ++v) a.add_u64(v);
            const std::string bytes = ser(a);
            MonoidAccumulator b(DFTU_MONOID_DISTINCT);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.to_value().as.u64 == b.to_value().as.u64);
        }
    }

    TEST_CASE("roundtrip: SKETCH within relative accuracy") {
        MonoidAccumulator a(DFTU_MONOID_SKETCH);
        for (std::uint64_t v = 1; v <= 1000; ++v)
            a.add_f64(static_cast<double>(v), 1.0);
        const std::string bytes = ser(a);
        MonoidAccumulator b(DFTU_MONOID_SKETCH);
        const std::size_t consumed = deser(b, bytes);
        CHECK(consumed == bytes.size());
        check_quantiles_eq(a, b);
    }

    TEST_CASE(
        "merge-equivalence: deserialize(serialize(a)).merge(b) == a.merge(b)") {
        SUBCASE("scalar COUNTER") {
            MonoidAccumulator a(DFTU_MONOID_COUNTER);
            a.add_u64(5);
            a.add_u64(3);
            MonoidAccumulator b(DFTU_MONOID_COUNTER);
            b.add_u64(11);
            MonoidAccumulator revived(DFTU_MONOID_COUNTER);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            CHECK(revived.to_value().as.u64 == live.to_value().as.u64);
        }
        SUBCASE("SET_I64 union") {
            MonoidAccumulator a(DFTU_MONOID_SET_I64);
            a.add_u64(1);
            a.add_u64(2);
            a.add_u64(3);
            MonoidAccumulator b(DFTU_MONOID_SET_I64);
            b.add_u64(3);
            b.add_u64(4);
            MonoidAccumulator revived(DFTU_MONOID_SET_I64);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            CHECK(revived.sorted_elements() == live.sorted_elements());
        }
        SUBCASE("LIST_I64 concat") {
            MonoidAccumulator a(DFTU_MONOID_LIST_I64);
            a.add_ordered(2, 20);
            a.add_ordered(1, 10);
            MonoidAccumulator b(DFTU_MONOID_LIST_I64);
            b.add_ordered(3, 30);
            b.add_ordered(0, 5);
            MonoidAccumulator revived(DFTU_MONOID_LIST_I64);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            CHECK(revived.ordered_elements() == live.ordered_elements());
        }
        SUBCASE("SKETCH merge") {
            MonoidAccumulator a(DFTU_MONOID_SKETCH);
            for (std::uint64_t v = 1; v <= 500; ++v)
                a.add_f64(static_cast<double>(v), 1.0);
            MonoidAccumulator b(DFTU_MONOID_SKETCH);
            for (std::uint64_t v = 501; v <= 1000; ++v)
                b.add_f64(static_cast<double>(v), 1.0);
            MonoidAccumulator revived(DFTU_MONOID_SKETCH);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            check_quantiles_eq(revived, live);
        }
    }

    TEST_CASE("roundtrip: ARGMIN/ARGMAX (i64 and str payloads)") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_ARGMIN_I64, DFTU_MONOID_ARGMAX_I64,
              DFTU_MONOID_ARGMIN_STR, DFTU_MONOID_ARGMAX_STR}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            a.add_argby(3.0, 100);
            a.add_argby(1.0, 200);
            a.add_argby(5.0, 300);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.argby_payload() == b.argby_payload());
            CHECK(a.to_value().as.u64 == b.to_value().as.u64);
        }
    }

    TEST_CASE("roundtrip: ARGMIN_ROW/ARGMAX_ROW (payload row survives)") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_ARGMIN_ROW, DFTU_MONOID_ARGMAX_ROW}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            auto add = [&](double by, std::vector<std::int64_t> pl) {
                a.add_argrow(by, pl.data(),
                             static_cast<std::uint32_t>(pl.size()));
            };
            // Heterogeneous 3-slot payload row; the extreme `by` keeps one row.
            add(3.0, {100, 7, -5});
            add(1.0, {200, 8, -6});
            add(5.0, {300, 9, -7});
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.argrow_has() == b.argrow_has());
            CHECK(a.argrow_by() == b.argrow_by());
            CHECK(a.argrow_payload() == b.argrow_payload());
        }
    }

    TEST_CASE(
        "merge-equivalence: revived ARGMAX_ROW merges like the live one") {
        auto add = [](MonoidAccumulator& m, double by,
                      std::vector<std::int64_t> pl) {
            m.add_argrow(by, pl.data(), static_cast<std::uint32_t>(pl.size()));
        };
        MonoidAccumulator a(DFTU_MONOID_ARGMAX_ROW);
        add(a, 2.0, {20, 2});
        add(a, 1.0, {10, 1});
        MonoidAccumulator b(DFTU_MONOID_ARGMAX_ROW);
        add(b, 3.0, {30, 3});
        MonoidAccumulator revived(DFTU_MONOID_ARGMAX_ROW);
        deser(revived, ser(a));
        revived.merge(b);
        MonoidAccumulator live = a;
        live.merge(b);
        // by 3 wins; the whole row {30,3} is kept, row-consistent.
        CHECK(revived.argrow_payload() == std::vector<std::int64_t>{30, 3});
        CHECK(revived.argrow_payload() == live.argrow_payload());
        // Order-independence: b.merge(a) == a.merge(b).
        MonoidAccumulator other = b;
        other.merge(a);
        CHECK(other.argrow_payload() == live.argrow_payload());
        CHECK(other.argrow_by() == live.argrow_by());
    }

    TEST_CASE("roundtrip: TOPK/BOTTOMK (i64 and str payloads)") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_TOPK_I64, DFTU_MONOID_TOPK_STR,
              DFTU_MONOID_BOTTOMK_I64, DFTU_MONOID_BOTTOMK_STR}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            // More than k=3 distinct entries so the bound is exercised.
            a.add_topk(3, 3.0, 100);
            a.add_topk(3, 1.0, 200);
            a.add_topk(3, 5.0, 300);
            a.add_topk(3, 2.0, 400);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.topk_elements() == b.topk_elements());
        }
    }

    TEST_CASE(
        "merge-equivalence: revived TOPK/BOTTOMK merge like the live one") {
        SUBCASE("TOPK_STR keeps the k largest across a revived merge") {
            MonoidAccumulator a(DFTU_MONOID_TOPK_STR);
            a.add_topk(2, 2.0, 20);
            a.add_topk(2, 1.0, 10);
            MonoidAccumulator b(DFTU_MONOID_TOPK_STR);
            b.add_topk(2, 3.0, 30);
            MonoidAccumulator revived(DFTU_MONOID_TOPK_STR);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            // Top-2 by desc: 30 (by 3), 20 (by 2); the by-1 entry is dropped.
            CHECK(revived.topk_elements() == std::vector<std::int64_t>{30, 20});
            CHECK(revived.topk_elements() == live.topk_elements());
        }
        SUBCASE("BOTTOMK_I64 keeps the k smallest across a revived merge") {
            MonoidAccumulator a(DFTU_MONOID_BOTTOMK_I64);
            a.add_topk(2, 9.0, 90);
            a.add_topk(2, 4.0, 40);
            MonoidAccumulator b(DFTU_MONOID_BOTTOMK_I64);
            b.add_topk(2, 1.0, 10);
            MonoidAccumulator revived(DFTU_MONOID_BOTTOMK_I64);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            // Bottom-2 by asc: 10 (by 1), 40 (by 4); the by-9 entry is dropped.
            CHECK(revived.topk_elements() == std::vector<std::int64_t>{10, 40});
            CHECK(revived.topk_elements() == live.topk_elements());
        }
    }

    TEST_CASE("roundtrip: APPROX_TOPK (i64 and str values)") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_APPROX_TOPK_I64, DFTU_MONOID_APPROX_TOPK_STR}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            // k=3 with >3 distinct plus a repeat, so both the fill and the
            // eviction (with its error) paths are serialized.
            a.add_approx_topk(3, 100);
            a.add_approx_topk(3, 100);
            a.add_approx_topk(3, 200);
            a.add_approx_topk(3, 300);
            a.add_approx_topk(3, 400);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.approx_topk_entries() == b.approx_topk_entries());
        }
    }

    TEST_CASE(
        "merge-equivalence: revived APPROX_TOPK merges like the live one") {
        // k=8 >= distinct, so counts are exact and the merged heavy hitters are
        // the true frequencies regardless of split or merge order.
        MonoidAccumulator a(DFTU_MONOID_APPROX_TOPK_I64);
        a.add_approx_topk(8, 10);
        a.add_approx_topk(8, 10);
        a.add_approx_topk(8, 20);
        MonoidAccumulator b(DFTU_MONOID_APPROX_TOPK_I64);
        b.add_approx_topk(8, 20);
        b.add_approx_topk(8, 30);
        MonoidAccumulator revived(DFTU_MONOID_APPROX_TOPK_I64);
        deser(revived, ser(a));
        revived.merge(b);
        MonoidAccumulator live = a;
        live.merge(b);
        using P = std::vector<std::pair<std::int64_t, std::uint64_t>>;
        // Counts by desc: 10 x2, 20 x2, 30 x1; ties (10 vs 20) order by value.
        CHECK(revived.approx_topk_entries() == P{{10, 2}, {20, 2}, {30, 1}});
        CHECK(revived.approx_topk_entries() == live.approx_topk_entries());
    }

    TEST_CASE("roundtrip: SAMPLE (i64 and str items)") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_SAMPLE_I64, DFTU_MONOID_SAMPLE_STR}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            // More than k=3 distinct items plus a repeat, so both the evict
            // path and the distinct-dedup path are serialized.
            a.add_sample(3, 100);
            a.add_sample(3, 200);
            a.add_sample(3, 300);
            a.add_sample(3, 400);
            a.add_sample(3, 100);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.sample_items().size() == 3);
            CHECK(a.sample_items() == b.sample_items());
        }
    }

    TEST_CASE("merge-equivalence: revived SAMPLE merges like the live one") {
        // k=5 >= distinct, so the sample keeps the full distinct set and merge
        // is order-independent: revived(a).merge(b) == a.merge(b) ==
        // b.merge(a).
        MonoidAccumulator a(DFTU_MONOID_SAMPLE_I64);
        a.add_sample(5, 10);
        a.add_sample(5, 20);
        a.add_sample(5, 20);
        MonoidAccumulator b(DFTU_MONOID_SAMPLE_I64);
        b.add_sample(5, 20);
        b.add_sample(5, 30);
        MonoidAccumulator revived(DFTU_MONOID_SAMPLE_I64);
        deser(revived, ser(a));
        revived.merge(b);
        MonoidAccumulator live = a;
        live.merge(b);
        using I = std::vector<std::int64_t>;
        CHECK(revived.sample_items() == I{10, 20, 30});
        CHECK(revived.sample_items() == live.sample_items());
        MonoidAccumulator other = b;
        other.merge(a);
        CHECK(other.sample_items() == live.sample_items());
    }

    TEST_CASE("roundtrip: MEAN/VARIANCE/STDDEV FieldStat state") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_MEAN, DFTU_MONOID_VARIANCE, DFTU_MONOID_STDDEV}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            for (double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0})
                a.add_f64(x, 1.0);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.to_value().as.f64 == doctest::Approx(b.to_value().as.f64));
        }
    }

    TEST_CASE("roundtrip: SKEWNESS/KURTOSIS FieldStat state") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_SKEWNESS, DFTU_MONOID_KURTOSIS}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            for (double x : {2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0})
                a.add_f64(x, 1.0);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.to_value().as.f64 == doctest::Approx(b.to_value().as.f64));
        }
    }

    TEST_CASE("roundtrip: co-moment CORR/COVAR/REGR power sums") {
        for (dftu_monoid_kind k :
             {DFTU_MONOID_CORR, DFTU_MONOID_COVAR_POP, DFTU_MONOID_COVAR_SAMP,
              DFTU_MONOID_REGR_SLOPE, DFTU_MONOID_REGR_INTERCEPT,
              DFTU_MONOID_REGR_R2}) {
            CAPTURE(static_cast<int>(k));
            MonoidAccumulator a(k);
            a.add_xy(1.0, 2.9);
            a.add_xy(2.0, 5.1);
            a.add_xy(3.0, 8.2);
            a.add_xy(4.0, 10.7);
            const std::string bytes = ser(a);
            MonoidAccumulator b(k);
            const std::size_t consumed = deser(b, bytes);
            CHECK(consumed == bytes.size());
            CHECK(a.to_value().as.f64 == doctest::Approx(b.to_value().as.f64));
        }
    }

    TEST_CASE("merge-equivalence: revived co-moment merges like the live one") {
        // regr of y = 3x + 5 split across two slices: slope 3, intercept 5,
        // r2 1, corr 1; the co-moment power sums merge order-independently.
        MonoidAccumulator a(DFTU_MONOID_REGR_SLOPE);
        a.add_xy(1.0, 8.0);
        a.add_xy(2.0, 11.0);
        MonoidAccumulator b(DFTU_MONOID_REGR_SLOPE);
        b.add_xy(3.0, 14.0);
        b.add_xy(4.0, 17.0);
        MonoidAccumulator revived(DFTU_MONOID_REGR_SLOPE);
        deser(revived, ser(a));
        revived.merge(b);
        MonoidAccumulator live = a;
        live.merge(b);
        CHECK(revived.to_value().as.f64 ==
              doctest::Approx(live.to_value().as.f64));
        CHECK(revived.to_value().as.f64 == doctest::Approx(3.0));
        MonoidAccumulator other = b;
        other.merge(a);
        CHECK(other.to_value().as.f64 ==
              doctest::Approx(live.to_value().as.f64));
    }

    TEST_CASE(
        "merge-equivalence: revived argby/moments merge like the live one") {
        SUBCASE("ARGMAX_STR keeps the payload at the extreme") {
            MonoidAccumulator a(DFTU_MONOID_ARGMAX_STR);
            a.add_argby(2.0, 20);
            a.add_argby(1.0, 10);
            MonoidAccumulator b(DFTU_MONOID_ARGMAX_STR);
            b.add_argby(3.0, 30);
            MonoidAccumulator revived(DFTU_MONOID_ARGMAX_STR);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            CHECK(revived.argby_payload() == 30);
            CHECK(revived.argby_payload() == live.argby_payload());
        }
        SUBCASE("VARIANCE merges the power sums") {
            MonoidAccumulator a(DFTU_MONOID_VARIANCE);
            for (double x : {2.0, 4.0, 4.0, 4.0}) a.add_f64(x, 1.0);
            MonoidAccumulator b(DFTU_MONOID_VARIANCE);
            for (double x : {5.0, 5.0, 7.0, 9.0}) b.add_f64(x, 1.0);
            MonoidAccumulator revived(DFTU_MONOID_VARIANCE);
            deser(revived, ser(a));
            revived.merge(b);
            MonoidAccumulator live = a;
            live.merge(b);
            CHECK(revived.to_value().as.f64 ==
                  doctest::Approx(live.to_value().as.f64));
            // Whole set {2,4,4,4,5,5,7,9}: sample variance 32/7.
            CHECK(revived.to_value().as.f64 == doctest::Approx(32.0 / 7.0));
        }
    }

    TEST_CASE("concatenation: walk a product-value row via consumed counts") {
        const dftu_monoid_kind row[] = {
            DFTU_MONOID_COUNTER, DFTU_MONOID_SUM_F64, DFTU_MONOID_SET_I64,
            DFTU_MONOID_SKETCH};
        std::vector<MonoidAccumulator> components;
        std::string buf;
        for (dftu_monoid_kind k : row) {
            MonoidAccumulator m(k);
            if (uses_f64_add(k)) {
                m.add_f64(2.0, 1.0);
                m.add_f64(4.0, 1.0);
            } else {
                m.add_u64(6);
                m.add_u64(9);
            }
            m.serialize(buf);
            components.push_back(m);
        }

        std::size_t off = 0;
        for (std::size_t i = 0; i < components.size(); ++i) {
            MonoidAccumulator revived(row[i]);
            const std::size_t consumed = revived.deserialize(
                reinterpret_cast<const std::byte*>(buf.data()) + off,
                buf.size() - off);
            CHECK(consumed > 0);
            off += consumed;
            const MonoidAccumulator& a = components[i];
            if (row[i] == DFTU_MONOID_SET_I64)
                CHECK(a.sorted_elements() == revived.sorted_elements());
            else if (row[i] == DFTU_MONOID_SKETCH)
                check_quantiles_eq(a, revived);
            else
                check_scalar_eq(a, revived);
        }
        CHECK(off == buf.size());
    }

    TEST_CASE("deserialize returns 0 on a short buffer") {
        MonoidAccumulator a(DFTU_MONOID_COUNTER);
        a.add_u64(123);
        const std::string bytes = ser(a);
        MonoidAccumulator b(DFTU_MONOID_COUNTER);
        const std::size_t consumed = b.deserialize(
            reinterpret_cast<const std::byte*>(bytes.data()), bytes.size() - 1);
        CHECK(consumed == 0);
    }
}
