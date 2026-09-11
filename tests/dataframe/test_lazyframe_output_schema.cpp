#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
// clang-format off
#include <nanoarrow/nanoarrow.h>
#include <dftracer/utils/dataframe/arrow.h>
// clang-format on
#endif

using namespace dftracer::utils::dataframe;

namespace {

DataFrame run(dftracer::utils::coro::CoroTask<DataFrame> t) {
    return dftracer::utils::default_runtime().submit(std::move(t)).get();
}

// The strongest oracle: what collect() actually produced, as a Schema, so it
// can be compared field-for-field against output_schema()'s claim.
Schema collected_schema(const DataFrame& df) {
    Schema s;
    s.fields.reserve(df.columns.size());
    for (std::size_t i = 0; i < df.columns.size(); ++i)
        s.fields.push_back(Field{df.names[i], df.columns[i].data_type(), true});
    return s;
}

void check_matches_collect(const LazyFrame& lf) {
    Schema claimed = lf.output_schema();
    DataFrame collected = run(lf.collect());
    Schema actual = collected_schema(collected);
    REQUIRE(claimed.fields.size() == actual.fields.size());
    for (std::size_t i = 0; i < claimed.fields.size(); ++i) {
        CAPTURE(i);
        CAPTURE(claimed.fields[i].name);
        CHECK(claimed.fields[i].name == actual.fields[i].name);
        CHECK(claimed.fields[i].type == actual.fields[i].type);
    }
}

Series i64(std::vector<std::int64_t> v) {
    return Series::flat_i64(v.data(), static_cast<std::int64_t>(v.size()));
}
Series f64(std::vector<double> v) {
    return Series::flat_f64(v.data(), static_cast<std::int64_t>(v.size()));
}

std::shared_ptr<const Source> src(DataFrame df) {
    return std::make_shared<const InMemorySource>(std::move(df));
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW

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

Series make_timestamp(const std::vector<std::int64_t>& values, TimeUnit unit,
                      const char* tz) {
    ArrowTimeUnit au = unit == TimeUnit::Second  ? NANOARROW_TIME_UNIT_SECOND
                       : unit == TimeUnit::Milli ? NANOARROW_TIME_UNIT_MILLI
                       : unit == TimeUnit::Micro ? NANOARROW_TIME_UNIT_MICRO
                                                 : NANOARROW_TIME_UNIT_NANO;
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetTypeDateTime(&b.schema, NANOARROW_TYPE_TIMESTAMP, au,
                                       tz) == NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (std::int64_t v : values)
        REQUIRE(ArrowArrayAppendInt(&b.array, v) == NANOARROW_OK);
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

Series make_decimal128(std::int32_t precision, std::int32_t scale,
                       const std::vector<std::int64_t>& values) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetTypeDecimal(&b.schema, NANOARROW_TYPE_DECIMAL128,
                                      precision, scale) == NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (std::int64_t v : values) {
        ArrowDecimal dec;
        ArrowDecimalInit(&dec, 128, precision, scale);
        ArrowDecimalSetInt(&dec, v);
        REQUIRE(ArrowArrayAppendDecimal(&b.array, &dec) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

#endif  // DFTRACER_UTILS_ENABLE_ARROW

DataFrame basic_frame() {
    DataFrame df;
    df.names = {"a", "b", "cat"};
    df.columns.push_back(i64({1, 2, 3, 4, 5}));
    df.columns.push_back(f64({1.5, 2.5, 3.5, 4.5, 5.5}));
    df.columns.push_back(Series::strings({"x", "y", "x", "z", "y"}));
    return df;
}

}  // namespace

TEST_SUITE("LazyFrame output_schema") {
    TEST_CASE("select subsets in order, types preserved") {
        LazyFrame lf = LazyFrame::scan(src(basic_frame())).select({"cat", "a"});
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 2);
        CHECK(s.fields[0] == Field{"cat", scalar(TypeId::String), true});
        CHECK(s.fields[1] == Field{"a", scalar(TypeId::Int64), true});
        check_matches_collect(lf);
    }

    TEST_CASE("rename keeps types, renames positionally") {
        LazyFrame lf =
            LazyFrame::scan(src(basic_frame())).rename({"x", "y", "z"});
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 3);
        CHECK(s.fields[0].name == "x");
        CHECK(s.fields[0].type == scalar(TypeId::Int64));
        check_matches_collect(lf);
    }

    TEST_CASE(
        "filter/sort_by/head/tail/slice/reverse/unique/drop_nulls keep types") {
        LazyFrame base = LazyFrame::scan(src(basic_frame()));
        check_matches_collect(base.filter(col(0) > std::int64_t{1}));
        check_matches_collect(base.sort_by("a"));
        check_matches_collect(base.head(2));
        check_matches_collect(base.tail(2));
        check_matches_collect(base.slice(1, 2));
        check_matches_collect(base.reverse());
        check_matches_collect(base.unique());
        check_matches_collect(base.drop_duplicates());
        check_matches_collect(base.drop_nulls());
    }

    TEST_CASE("with_column types the new column from its Expr") {
        LazyFrame lf = LazyFrame::scan(src(basic_frame()))
                           .with_column("d", col(0) + col(1));
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 4);
        CHECK(s.fields[3] == Field{"d", scalar(TypeId::Float64), true});
        check_matches_collect(lf);
    }

    TEST_CASE("with_column replacing an existing name keeps its position") {
        LazyFrame lf = LazyFrame::scan(src(basic_frame()))
                           .with_column("a", col(0) > std::int64_t{1});
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 3);
        CHECK(s.fields[0] == Field{"a", scalar(TypeId::Bool), true});
        check_matches_collect(lf);
    }

    TEST_CASE("with_row_index prepends an Int64 field") {
        LazyFrame lf =
            LazyFrame::scan(src(basic_frame())).with_row_index("idx");
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 4);
        CHECK(s.fields[0] == Field{"idx", scalar(TypeId::Int64), true});
        check_matches_collect(lf);
    }

    TEST_CASE("explode reports the List element type") {
        DataFrame df;
        df.names = {"k", "vals"};
        df.columns.push_back(i64({1, 2}));
        std::vector<std::int32_t> off{0, 2, 3};
        df.columns.push_back(Series::list(off, i64({10, 20, 30})));
        LazyFrame lf = LazyFrame::scan(src(std::move(df))).explode("vals");
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 2);
        CHECK(s.fields[1] == Field{"vals", scalar(TypeId::Int64), true});
        check_matches_collect(lf);
    }

    TEST_CASE("fill_null keeps the column's type") {
        LazyFrame lf =
            LazyFrame::scan(src(basic_frame())).fill_null(std::int64_t{0});
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 3);
        CHECK(s.fields[0].type == scalar(TypeId::Int64));
        CHECK(s.fields[1].type == scalar(TypeId::Float64));
        check_matches_collect(lf);
    }

#ifdef DFTRACER_UTILS_ENABLE_ARROW
    TEST_CASE(
        "fill_null leaves a Timestamp column (and its timezone) untouched") {
        // dftu_series_fillna only runs on a numeric-dispatchable column, so a
        // Timestamp column is passed through unchanged, tz included.
        DataFrame df;
        df.names = {"ts"};
        df.columns.push_back(
            make_timestamp({1, 2, 3}, TimeUnit::Micro, "America/Chicago"));
        LazyFrame lf =
            LazyFrame::scan(src(std::move(df))).fill_null(std::int64_t{0});
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 1);
        CHECK(s.fields[0].type ==
              timestamp(TimeUnit::Micro, "America/Chicago"));
        check_matches_collect(lf);
    }

    TEST_CASE(
        "select/rename/filter/sort_by preserve a Timestamp's tz and a "
        "Decimal128's precision/scale") {
        DataFrame df;
        df.names = {"ts", "dec", "n"};
        df.columns.push_back(make_timestamp({1, 2, 3}, TimeUnit::Nano, "UTC"));
        df.columns.push_back(make_decimal128(38, 9, {1, 2, 3}));
        df.columns.push_back(i64({3, 1, 2}));

        LazyFrame lf = LazyFrame::scan(src(std::move(df)))
                           .select({"ts", "dec", "n"})
                           .rename({"ts2", "dec2", "n2"})
                           .filter(col(2) > std::int64_t{0})
                           .sort_by("n2");
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 3);
        CHECK(s.fields[0] ==
              Field{"ts2", timestamp(TimeUnit::Nano, "UTC"), true});
        CHECK(s.fields[1] == Field{"dec2", decimal128(38, 9), true});
        check_matches_collect(lf);
    }
#endif  // DFTRACER_UTILS_ENABLE_ARROW

    TEST_CASE("group_by types Count/Sum/Mean/Min/Max from the value column") {
        LazyFrame lf = LazyFrame::scan(src(basic_frame()))
                           .group_by("cat", {GroupAgg{Agg::Count, "", "n"},
                                             GroupAgg{Agg::Sum, "a", "sum_a"},
                                             GroupAgg{Agg::Mean, "a", "mean_a"},
                                             GroupAgg{Agg::Min, "b", "min_b"},
                                             GroupAgg{Agg::Max, "b", "max_b"}});
        Schema s = lf.output_schema();
        REQUIRE(s.fields.size() == 6);
        CHECK(s.fields[0] == Field{"cat", scalar(TypeId::String), true});
        CHECK(s.fields[1] == Field{"n", scalar(TypeId::Int64), true});
        CHECK(s.fields[2] == Field{"sum_a", scalar(TypeId::Int64), true});
        CHECK(s.fields[3] == Field{"mean_a", scalar(TypeId::Float64), true});
        CHECK(s.fields[4] == Field{"min_b", scalar(TypeId::Float64), true});
        CHECK(s.fields[5] == Field{"max_b", scalar(TypeId::Float64), true});
        check_matches_collect(lf);
    }

    TEST_CASE("data-dependent ops report an empty schema, not a guess") {
        LazyFrame base = LazyFrame::scan(src(basic_frame()));
        CHECK(base.pivot("cat", "cat", "a").output_schema().fields.empty());
        CHECK(base.to_dummies("cat").output_schema().fields.empty());
        CHECK(base.describe().output_schema().fields.empty());
    }

    TEST_CASE("output_schema over a chain still matches collect()") {
        LazyFrame lf = LazyFrame::scan(src(basic_frame()))
                           .filter(col(0) > std::int64_t{0})
                           .with_column("d", col(0) * col(1))
                           .sort_by("d", true)
                           .head(3);
        check_matches_collect(lf);
    }
}
