#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

// Confirms the new Arrow logical types compute, not just round-trip: Float16
// arithmetic/reduction promote to Float32 and produce real numbers,
// FixedSizeBinary/Decimal128 compare/sort/group_by/is_in on their exact
// bytes, and Decimal128 arithmetic goes through the documented (lossy)
// Float64 cast.
#ifdef DFTRACER_UTILS_ENABLE_ARROW
// clang-format off
// nanoarrow.h must precede arrow.h: arrow.h (via plugins/arrow_abi.h) defines
// ArrowSchema/ArrowArray under the ARROW_C_DATA_INTERFACE guard but not
// ArrowArrayStream, which nanoarrow.h's inline helpers then need complete.
#include <nanoarrow/nanoarrow.h>
#include <dftracer/utils/dataframe/arrow.h>
#include <dftracer/utils/dataframe/scalar.h>
// clang-format on

using dftracer::utils::dataframe::Agg;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::DataType;
using dftracer::utils::dataframe::fixed_size_binary;
using dftracer::utils::dataframe::GroupAgg;
using dftracer::utils::dataframe::OwnedArrow;
using dftracer::utils::dataframe::Scalar;
using dftracer::utils::dataframe::Series;
using dftracer::utils::dataframe::TypeId;

namespace {

struct BuiltArrow {
    ArrowSchema schema{};
    ArrowArray array{};
    ~BuiltArrow() {
        if (array.release) array.release(&array);
        if (schema.release) schema.release(&schema);
    }
};

void finish(ArrowArray* a) {
    ArrowError err;
    REQUIRE(ArrowArrayFinishBuildingDefault(a, &err) == NANOARROW_OK);
}

Series make_float16(const std::vector<float>& values) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetType(&b.schema, NANOARROW_TYPE_HALF_FLOAT) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (float v : values)
        REQUIRE(ArrowArrayAppendDouble(&b.array, static_cast<double>(v)) ==
                NANOARROW_OK);
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

Series make_fixed_size_binary(const std::vector<std::string>& values,
                              std::int32_t width) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetTypeFixedSize(&b.schema,
                                        NANOARROW_TYPE_FIXED_SIZE_BINARY,
                                        width) == NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (const std::string& v : values) {
        REQUIRE(static_cast<std::int32_t>(v.size()) == width);
        ArrowBufferView bv;
        bv.data.data = v.data();
        bv.size_bytes = width;
        REQUIRE(ArrowArrayAppendBytes(&b.array, bv) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

Series make_timestamp(const std::vector<std::int64_t>& values) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetTypeDateTime(&b.schema, NANOARROW_TYPE_TIMESTAMP,
                                       NANOARROW_TIME_UNIT_MICRO,
                                       "UTC") == NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (std::int64_t v : values)
        REQUIRE(ArrowArrayAppendInt(&b.array, v) == NANOARROW_OK);
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

Series make_decimal128(const std::vector<std::int64_t>& values) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetTypeDecimal(&b.schema, NANOARROW_TYPE_DECIMAL128, 38,
                                      0) == NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (std::int64_t v : values) {
        struct ArrowDecimal dec;
        ArrowDecimalInit(&dec, 128, 38, 0);
        ArrowDecimalSetInt(&dec, v);
        REQUIRE(ArrowArrayAppendDecimal(&b.array, &dec) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

}  // namespace

TEST_SUITE("dataframe_arrow_types_compute") {
    TEST_CASE("Float16 arithmetic promotes to Float32 and computes exactly") {
        Series a = make_float16({1.5f, 2.5f, -3.0f});
        Series b = make_float16({1.0f, 1.0f, 1.0f});
        REQUIRE(a.valid());
        REQUIRE(b.valid());

        Series sum = a.add(b);
        REQUIRE(sum.valid());
        CHECK(sum.type() == TypeId::Float32);
        const float* d = sum.data<float>();
        REQUIRE(d != nullptr);
        CHECK(d[0] == doctest::Approx(2.5f));
        CHECK(d[1] == doctest::Approx(3.5f));
        CHECK(d[2] == doctest::Approx(-2.0f));
    }

    TEST_CASE("Float16 reduction (sum) promotes to Float32 and computes") {
        Series a = make_float16({1.0f, 2.0f, 3.0f});
        REQUIRE(a.valid());
        Scalar total = a.sum();
        CHECK(total.f64() == doctest::Approx(6.0));
    }

    TEST_CASE("FixedSizeBinary sorts by exact byte order") {
        Series s = make_fixed_size_binary({"bbbb", "aaaa", "cccc", "aaaa"}, 4);
        REQUIRE(s.valid());
        Series sorted = s.sort(false);
        REQUIRE(sorted.valid());
        REQUIRE(sorted.length() == 4);
        const char* d = sorted.data<char>();
        REQUIRE(d != nullptr);
        CHECK(std::string(d, 4) == "aaaa");
        CHECK(std::string(d + 4, 4) == "aaaa");
        CHECK(std::string(d + 8, 4) == "bbbb");
        CHECK(std::string(d + 12, 4) == "cccc");
    }

    TEST_CASE("FixedSizeBinary is_in matches on exact bytes") {
        Series s = make_fixed_size_binary({"aaaa", "bbbb", "cccc"}, 4);
        Series needles = make_fixed_size_binary({"bbbb"}, 4);
        REQUIRE(s.valid());
        REQUIRE(needles.valid());
        Series mask = s.is_in(needles);
        REQUIRE(mask.valid());
        REQUIRE(mask.length() == 3);
        CHECK(mask.is_null(0) == false);
        const std::uint8_t* bits = mask.data<std::uint8_t>();
        REQUIRE(bits != nullptr);
        auto bit = [&](std::int64_t i) {
            return ((bits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(bit(0) == false);
        CHECK(bit(1) == true);
        CHECK(bit(2) == false);
    }

    TEST_CASE("FixedSizeBinary groups by exact byte equality") {
        Series keys = make_fixed_size_binary(
            {"aaaa", "bbbb", "aaaa", "cccc", "bbbb", "aaaa"}, 4);
        std::vector<std::int64_t> ones = {1, 1, 1, 1, 1, 1};
        Series values = Series::flat_i64(ones.data(), 6);
        REQUIRE(keys.valid());
        REQUIRE(values.valid());

        dftu_series* out_keys = nullptr;
        dftu_series* out_values[5] = {};
        std::int32_t n =
            dftu_series_group_by(keys.handle(), values.handle(),
                                 DFTU_REDUCE_COUNT, &out_keys, out_values, 5);
        REQUIRE(n == 1);
        REQUIRE(out_keys != nullptr);
        CHECK(dftu_series_length(out_keys) == 3);  // aaaa, bbbb, cccc
        Series counts{out_values[0]};
        REQUIRE(counts.valid());
        REQUIRE(counts.length() == 3);
        const std::int64_t* c = counts.data<std::int64_t>();
        REQUIRE(c != nullptr);
        CHECK(c[0] == 3);  // aaaa first-seen
        CHECK(c[1] == 2);  // bbbb
        CHECK(c[2] == 1);  // cccc
        dftu_series_free(out_keys);
    }

    TEST_CASE("Decimal128 sorts by exact numeric order, not byte order") {
        // -1 and 300 both have a nonzero low byte; only a signed 128-bit
        // compare (not memcmp) orders them correctly.
        Series s = make_decimal128({300, -1, 100});
        REQUIRE(s.valid());
        Series sorted = s.sort(false);
        REQUIRE(sorted.valid());
        REQUIRE(sorted.length() == 3);
        const std::uint8_t* d = sorted.data<std::uint8_t>();
        REQUIRE(d != nullptr);
        std::int64_t v0, v1, v2;
        std::memcpy(&v0, d, 8);
        std::memcpy(&v1, d + 16, 8);
        std::memcpy(&v2, d + 32, 8);
        CHECK(v0 == -1);
        CHECK(v1 == 100);
        CHECK(v2 == 300);
    }

    TEST_CASE("Decimal128 groups by exact value") {
        Series keys = make_decimal128({100, 300, 100});
        std::vector<std::int64_t> ones = {1, 1, 1};
        Series values = Series::flat_i64(ones.data(), 3);
        REQUIRE(keys.valid());

        dftu_series* out_keys = nullptr;
        dftu_series* out_values[5] = {};
        std::int32_t n =
            dftu_series_group_by(keys.handle(), values.handle(),
                                 DFTU_REDUCE_COUNT, &out_keys, out_values, 5);
        REQUIRE(n == 1);
        REQUIRE(dftu_series_length(out_keys) == 2);
        Series counts{out_values[0]};
        REQUIRE(counts.length() == 2);
        const std::int64_t* c = counts.data<std::int64_t>();
        CHECK(c[0] == 2);  // 100 first-seen
        CHECK(c[1] == 1);  // 300
        dftu_series_free(out_keys);
    }

    TEST_CASE(
        "Decimal128 arithmetic converts to Float64 (documented precision "
        "loss)") {
        Series a = make_decimal128({100, 200});
        Series b = make_decimal128({1, 1});
        REQUIRE(a.valid());
        Series sum = a.add(b);
        REQUIRE(sum.valid());
        CHECK(sum.type() == TypeId::Float64);
        const double* d = sum.data<double>();
        REQUIRE(d != nullptr);
        CHECK(d[0] == doctest::Approx(101.0));
        CHECK(d[1] == doctest::Approx(201.0));
    }

    TEST_CASE("Casting INTO a decimal type is refused, not silently wrong") {
        std::vector<double> vals = {1.5, 2.5};
        Series a = Series::flat_f64(vals.data(), 2);
        Series casted = a.cast(TypeId::Decimal128);
        CHECK_FALSE(casted.valid());
    }

    TEST_CASE("Timestamp sorts on its exact physical Int64 value") {
        Series s = make_timestamp({300, -1, 100});
        REQUIRE(s.valid());
        Series sorted = s.sort(false);
        REQUIRE(sorted.valid());
        REQUIRE(sorted.type() ==
                TypeId::Timestamp);  // stays Timestamp, not Int64
        const std::int64_t* d = sorted.data<std::int64_t>();
        REQUIRE(d != nullptr);
        CHECK(d[0] == -1);
        CHECK(d[1] == 100);
        CHECK(d[2] == 300);
    }

    TEST_CASE("Timestamp min/max compute on the exact value") {
        Series s = make_timestamp({300, -1, 100});
        REQUIRE(s.valid());
        Scalar mn = s.min();
        Scalar mx = s.max();
        CHECK(mn.i64() == -1);
        CHECK(mx.i64() == 300);
    }

    TEST_CASE("Timestamp comparison works against a scalar threshold") {
        Series s = make_timestamp({300, -1, 100});
        REQUIRE(s.valid());
        dftu_scalar rhs{};
        rhs.kind = DFTU_SCALAR_TAG_I64;
        rhs.value.i = 100;
        dftu_series* mask = dftu_series_compare(s.handle(), DFTU_CMP_GT, rhs);
        REQUIRE(mask != nullptr);
        Series m{mask};
        REQUIRE(m.length() == 3);
        const std::uint8_t* bits = m.data<std::uint8_t>();
        REQUIRE(bits != nullptr);
        auto bit = [&](std::int64_t i) {
            return ((bits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(bit(0) == true);   // 300 > 100
        CHECK(bit(1) == false);  // -1 > 100
        CHECK(bit(2) == false);  // 100 > 100
    }

    TEST_CASE("Timestamp is_in matches on the exact value") {
        Series s = make_timestamp({300, -1, 100});
        Series needles = make_timestamp({100});
        REQUIRE(s.valid());
        Series mask = s.is_in(needles);
        REQUIRE(mask.valid());
        REQUIRE(mask.length() == 3);
        const std::uint8_t* bits = mask.data<std::uint8_t>();
        REQUIRE(bits != nullptr);
        auto bit = [&](std::int64_t i) {
            return ((bits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(bit(0) == false);
        CHECK(bit(1) == false);
        CHECK(bit(2) == true);
    }

    TEST_CASE("Timestamp groups on the exact value, via both group_by paths") {
        Series keys = make_timestamp({100, 300, 100});
        std::vector<std::int64_t> ones = {1, 1, 1};
        Series values = Series::flat_i64(ones.data(), 3);
        REQUIRE(keys.valid());

        // Low-level ABI primitive (kernels/group_by.cpp).
        dftu_series* out_keys = nullptr;
        dftu_series* out_values[5] = {};
        std::int32_t n =
            dftu_series_group_by(keys.handle(), values.handle(),
                                 DFTU_REDUCE_COUNT, &out_keys, out_values, 5);
        REQUIRE(n == 1);
        REQUIRE(dftu_series_length(out_keys) == 2);
        Series counts{out_values[0]};
        const std::int64_t* c = counts.data<std::int64_t>();
        CHECK(c[0] == 2);  // 100 first-seen
        CHECK(c[1] == 1);  // 300
        dftu_series_free(out_keys);

        // DataFrame::group_by (AggState/group_agg engine).
        DataFrame df;
        df.names = {"ts", "v"};
        df.columns.push_back(keys.share());
        df.columns.push_back(values.share());
        DataFrame grouped = df.group_by("ts", {GroupAgg{Agg::Count, "v", "n"}});
        REQUIRE(grouped.num_rows() == 2);
        Series gkeys = grouped.column("ts");
        Series gn = grouped.column("n");
        REQUIRE(gkeys.type() == TypeId::Timestamp);
        const std::int64_t* kp = gkeys.data<std::int64_t>();
        const std::int64_t* np = gn.data<std::int64_t>();
        REQUIRE(kp != nullptr);
        REQUIRE(np != nullptr);
        for (std::int64_t i = 0; i < 2; ++i) {
            if (kp[i] == 100)
                CHECK(np[i] == 2);
            else if (kp[i] == 300)
                CHECK(np[i] == 1);
            else
                FAIL("unexpected group key");
        }
    }

    TEST_CASE("Timestamp arithmetic is refused, not silently an integer") {
        Series a = make_timestamp({100, 200});
        Series b = make_timestamp({1, 1});
        REQUIRE(a.valid());
        Series sum = a.add(b);
        CHECK_FALSE(sum.valid());
    }

    TEST_CASE(
        "DataFrame::group_by refuses a FixedSizeBinary key loudly, not "
        "wrong groups") {
        Series keys = make_fixed_size_binary({"aaaa", "bbbb", "aaaa"}, 4);
        std::vector<std::int64_t> ones = {1, 1, 1};
        Series values = Series::flat_i64(ones.data(), 3);
        DataFrame df;
        df.names = {"k", "v"};
        df.columns.push_back(keys.share());
        df.columns.push_back(values.share());
        CHECK_THROWS_AS(df.group_by("k", {GroupAgg{Agg::Count, "v", "n"}}),
                        std::invalid_argument);
    }

    TEST_CASE(
        "DataFrame::group_by refuses a Decimal128 key loudly, not wrong "
        "groups") {
        Series keys = make_decimal128({100, 300, 100});
        std::vector<std::int64_t> ones = {1, 1, 1};
        Series values = Series::flat_i64(ones.data(), 3);
        DataFrame df;
        df.names = {"k", "v"};
        df.columns.push_back(keys.share());
        df.columns.push_back(values.share());
        CHECK_THROWS_AS(df.group_by("k", {GroupAgg{Agg::Count, "v", "n"}}),
                        std::invalid_argument);
    }
}
#else
TEST_SUITE("dataframe_arrow_types_compute") {
    TEST_CASE("Arrow disabled: nothing to exercise") { CHECK(true); }
}
#endif
