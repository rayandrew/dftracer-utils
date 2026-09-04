#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/utilities/common/arrow/array_view.h>
#include <dftracer/utils/utilities/common/arrow/arrow.h>
#include <doctest/doctest.h>
#include <nanoarrow/nanoarrow.h>

#include <string>
#include <vector>

using namespace dftracer::utils::utilities::common::arrow;

static std::string child_schema_name(const ArrowExportResult& r, int64_t i) {
    return r.get_schema()->children[i]->name;
}

static std::string child_schema_format(const ArrowExportResult& r, int64_t i) {
    return r.get_schema()->children[i]->format;
}

static int64_t child_null_count(const ArrowExportResult& r, int64_t i) {
    return r.get_array()->children[i]->null_count;
}

TEST_CASE("RecordBatchBuilder - Static schema mode") {
    SUBCASE("builds batch with int64, double, string columns") {
        RecordBatchBuilder b;
        b.declare_schema({{"id", ColumnType::INT64},
                          {"value", ColumnType::DOUBLE},
                          {"name", ColumnType::STRING}});

        std::string s0 = "alice", s1 = "bob", s2 = "carol";

        b.append_int64(0, 1);
        b.append_double(1, 1.1);
        b.append_string(2, s0);
        b.end_row();

        b.append_int64(0, 2);
        b.append_double(1, 2.2);
        b.append_string(2, s1);
        b.end_row();

        b.append_int64(0, 3);
        b.append_double(1, 3.3);
        b.append_string(2, s2);
        b.end_row();

        auto result = b.finish();

        REQUIRE(result.valid());
        CHECK(result.num_rows() == 3);
        CHECK(result.num_columns() == 3);

        CHECK(child_schema_name(result, 0) == "id");
        CHECK(child_schema_name(result, 1) == "value");
        CHECK(child_schema_name(result, 2) == "name");

        // nanoarrow format strings: "l"=int64, "g"=float64, "u"=utf8 string
        CHECK(child_schema_format(result, 0) == std::string("l"));
        CHECK(child_schema_format(result, 1) == std::string("g"));
        CHECK(child_schema_format(result, 2) == std::string("u"));
    }

    SUBCASE("reserve and reset cycle for streaming") {
        RecordBatchBuilder b;
        b.declare_schema(
            {{"ts", ColumnType::INT64}, {"dur", ColumnType::DOUBLE}});
        b.reserve(100);

        for (int i = 0; i < 5; ++i) {
            b.append_int64(0, i);
            b.append_double(1, static_cast<double>(i) * 0.5);
            b.end_row();
        }

        auto first = b.finish();
        REQUIRE(first.valid());
        CHECK(first.num_rows() == 5);

        b.reset(true);  // keep schema

        for (int i = 0; i < 3; ++i) {
            b.append_int64(0, i + 100);
            b.append_double(1, static_cast<double>(i) * 1.5);
            b.end_row();
        }

        auto second = b.finish();
        REQUIRE(second.valid());
        CHECK(second.num_rows() == 3);
        CHECK(second.num_columns() == 2);
    }
}

TEST_CASE("RecordBatchBuilder - Dynamic schema mode") {
    SUBCASE("discovers columns from data") {
        RecordBatchBuilder b;

        size_t x = b.add_or_get_column("x", ColumnType::INT64);
        CHECK(x == 0);

        b.append_int64(x, 42);
        b.end_row();

        size_t y = b.add_or_get_column("y", ColumnType::STRING);
        CHECK(y == 1);

        std::string hello = "hello";
        b.append_int64(x, 43);
        b.append_string(y, hello);
        b.end_row();

        auto result = b.finish();
        REQUIRE(result.valid());
        CHECK(result.num_rows() == 2);
        CHECK(result.num_columns() == 2);

        CHECK(child_schema_name(result, 0) == "x");
        CHECK(child_schema_name(result, 1) == "y");
    }

    SUBCASE("backfills nulls for missing columns") {
        RecordBatchBuilder b;

        size_t a = b.add_or_get_column("a", ColumnType::INT64);
        b.append_int64(a, 1);
        b.end_row();  // row 0: a=1, b not yet known

        size_t bv = b.add_or_get_column("b", ColumnType::DOUBLE);
        b.append_int64(a, 2);
        b.append_double(bv, 3.14);
        b.end_row();  // row 1: a=2, b=3.14

        auto result = b.finish();
        REQUIRE(result.valid());
        CHECK(result.num_rows() == 2);
        CHECK(result.num_columns() == 2);

        // Column "b" must have a null in row 0.
        CHECK(child_null_count(result, 1) > 0);
    }
}

TEST_CASE("RecordBatchBuilder - Null handling") {
    RecordBatchBuilder b;
    b.declare_schema({{"v", ColumnType::INT64}});

    b.append_int64(0, 10);
    b.end_row();

    b.append_null(0);
    b.end_row();

    b.append_int64(0, 30);
    b.end_row();

    auto result = b.finish();
    REQUIRE(result.valid());
    CHECK(result.num_rows() == 3);
    CHECK(child_null_count(result, 0) == 1);
}

TEST_CASE("RecordBatchBuilder - append_null under a locked schema") {
    // append_null must gate first-touch marking on schema_locked_ like every
    // other append_*; a locked schema uses explicit backfill, not first-touch
    // row detection. This exercises that mixed real/null appends after
    // lock_schema() still produce a well-formed batch.
    RecordBatchBuilder b;
    b.add_or_get_column("a", ColumnType::INT64);
    b.add_or_get_column("b", ColumnType::INT64);
    b.lock_schema();
    CHECK(b.is_schema_locked());

    b.append_int64(0, 1);
    b.append_null(1);
    b.end_row();

    b.append_int64(0, 2);
    b.append_int64(1, 3);
    b.end_row();

    auto result = b.finish();
    REQUIRE(result.valid());
    CHECK(result.num_rows() == 2);
    CHECK(result.num_columns() == 2);
    CHECK(child_null_count(result, 0) == 0);
    CHECK(child_null_count(result, 1) == 1);
}

TEST_CASE("RecordBatchBuilder - Empty batch") {
    RecordBatchBuilder b;
    b.declare_schema({{"a", ColumnType::INT64}, {"b", ColumnType::STRING}});

    auto result = b.finish();
    REQUIRE(result.valid());
    CHECK(result.num_rows() == 0);
    CHECK(result.num_columns() == 2);
}

TEST_CASE("RecordBatchBuilder - Bool column") {
    RecordBatchBuilder b;
    b.declare_schema({{"flag", ColumnType::BOOL}});

    b.append_bool(0, true);
    b.end_row();
    b.append_bool(0, false);
    b.end_row();
    b.append_bool(0, true);
    b.end_row();

    auto result = b.finish();
    REQUIRE(result.valid());
    CHECK(result.num_rows() == 3);
    CHECK(result.num_columns() == 1);
}

TEST_CASE("RecordBatchBuilder - HIST list<struct> column") {
    RecordBatchBuilder b;
    b.declare_schema({{"name", ColumnType::STRING}, {"h", ColumnType::HIST}});

    b.append_string(0, "read");
    b.append_hist(1, {{0.0, 1.0, 3}, {1.0, 2.0, 5}});
    b.end_row();

    b.append_string(0, "write");
    b.append_hist(1, {});  // empty histogram
    b.end_row();

    auto result = b.finish();
    REQUIRE(result.valid());
    CHECK(result.num_rows() == 2);
    CHECK(result.num_columns() == 2);
    // list<item: struct<lo,hi,count>>: format "+l" with a struct child.
    CHECK(child_schema_format(result, 1) == std::string("+l"));
    CHECK(std::string(result.get_schema()->children[1]->children[0]->format) ==
          std::string("+s"));
}

namespace {

struct ResultView {
    ArrowArrayView view{};
    explicit ResultView(ArrowExportResult& r) {
        REQUIRE(init_array_view(view, r.get_schema(), r.get_array()) ==
                NANOARROW_OK);
    }
    ~ResultView() { ArrowArrayViewReset(&view); }
    const ArrowArrayView* col(int64_t i) const { return view.children[i]; }
};

int64_t col_int(const ResultView& rv, int64_t c, int64_t r) {
    return ArrowArrayViewGetIntUnsafe(rv.col(c), r);
}

std::string col_str(const ResultView& rv, int64_t c, int64_t r) {
    ArrowStringView s = ArrowArrayViewGetStringUnsafe(rv.col(c), r);
    return std::string(s.data, static_cast<std::size_t>(s.size_bytes));
}

}  // namespace

TEST_CASE("explode - STRING_LIST one row per element") {
    RecordBatchBuilder b;
    b.declare_schema(
        {{"pid", ColumnType::INT64}, {"files", ColumnType::STRING_LIST}});
    b.append_int64(0, 1);
    b.append_string_list(1, {"a", "b", "c"});
    b.end_row();
    b.append_int64(0, 2);
    b.append_string_list(1, {"x"});
    b.end_row();
    auto in = b.finish();

    auto out =
        explode(in.get_schema(), in.get_array(), 1, /*keep_empty=*/false);
    REQUIRE(out.valid());
    CHECK(out.num_rows() == 4);
    CHECK(out.num_columns() == 2);
    CHECK(child_schema_name(out, 0) == "pid");
    CHECK(child_schema_name(out, 1) == "files");

    ResultView rv(out);
    CHECK(col_int(rv, 0, 0) == 1);
    CHECK(col_int(rv, 0, 1) == 1);
    CHECK(col_int(rv, 0, 2) == 1);
    CHECK(col_int(rv, 0, 3) == 2);
    CHECK(col_str(rv, 1, 0) == "a");
    CHECK(col_str(rv, 1, 1) == "b");
    CHECK(col_str(rv, 1, 2) == "c");
    CHECK(col_str(rv, 1, 3) == "x");
}

TEST_CASE("explode - INT64_LIST one row per element") {
    RecordBatchBuilder b;
    b.declare_schema(
        {{"pid", ColumnType::INT64}, {"durs", ColumnType::INT64_LIST}});
    b.append_int64(0, 1);
    b.append_int64_list(1, {20, 5, 8});
    b.end_row();
    b.append_int64(0, 2);
    b.append_int64_list(1, {3});
    b.end_row();
    auto in = b.finish();

    auto out = explode(in.get_schema(), in.get_array(), 1, false);
    REQUIRE(out.valid());
    CHECK(out.num_rows() == 4);
    CHECK(child_schema_name(out, 1) == "durs");

    ResultView rv(out);
    CHECK(col_int(rv, 0, 0) == 1);
    CHECK(col_int(rv, 0, 3) == 2);
    CHECK(col_int(rv, 1, 0) == 20);
    CHECK(col_int(rv, 1, 1) == 5);
    CHECK(col_int(rv, 1, 2) == 8);
    CHECK(col_int(rv, 1, 3) == 3);
}

TEST_CASE("explode - STRUCT_LIST flattens fields into columns") {
    RecordBatchBuilder b;
    b.declare_schema(
        {{"pid", ColumnType::INT64},
         {"tk",
          ColumnType::STRUCT_LIST,
          {{"value", ColumnType::STRING}, {"count", ColumnType::INT64}}}});
    b.append_int64(0, 1);
    b.append_struct_list(1,
                         {{StructCell{.str = "read"}, StructCell{.i64 = 5}},
                          {StructCell{.str = "write"}, StructCell{.i64 = 2}}});
    b.end_row();
    b.append_int64(0, 2);
    b.append_struct_list(1,
                         {{StructCell{.str = "open"}, StructCell{.i64 = 9}}});
    b.end_row();
    auto in = b.finish();

    auto out = explode(in.get_schema(), in.get_array(), 1, false);
    REQUIRE(out.valid());
    CHECK(out.num_rows() == 3);
    CHECK(out.num_columns() == 3);
    CHECK(child_schema_name(out, 0) == "pid");
    CHECK(child_schema_name(out, 1) == "value");
    CHECK(child_schema_name(out, 2) == "count");

    ResultView rv(out);
    CHECK(col_int(rv, 0, 0) == 1);
    CHECK(col_str(rv, 1, 0) == "read");
    CHECK(col_int(rv, 2, 0) == 5);
    CHECK(col_int(rv, 0, 1) == 1);
    CHECK(col_str(rv, 1, 1) == "write");
    CHECK(col_int(rv, 2, 1) == 2);
    CHECK(col_int(rv, 0, 2) == 2);
    CHECK(col_str(rv, 1, 2) == "open");
    CHECK(col_int(rv, 2, 2) == 9);
}

TEST_CASE("explode - empty list drop vs keep_empty") {
    RecordBatchBuilder b;
    b.declare_schema(
        {{"pid", ColumnType::INT64}, {"files", ColumnType::STRING_LIST}});
    b.append_int64(0, 1);
    b.append_string_list(1, {"a"});
    b.end_row();
    b.append_int64(0, 2);
    b.append_string_list(1, {});  // empty list
    b.end_row();
    b.append_int64(0, 3);
    b.append_string_list(1, {"b"});
    b.end_row();
    auto in = b.finish();

    SUBCASE("default drops the empty-list row") {
        auto out = explode(in.get_schema(), in.get_array(), 1, false);
        REQUIRE(out.valid());
        CHECK(out.num_rows() == 2);
        ResultView rv(out);
        CHECK(col_int(rv, 0, 0) == 1);
        CHECK(col_str(rv, 1, 0) == "a");
        CHECK(col_int(rv, 0, 1) == 3);
        CHECK(col_str(rv, 1, 1) == "b");
    }

    SUBCASE("keep_empty emits one null-exploded row") {
        auto out = explode(in.get_schema(), in.get_array(), 1, true);
        REQUIRE(out.valid());
        CHECK(out.num_rows() == 3);
        ResultView rv(out);
        CHECK(col_int(rv, 0, 1) == 2);
        CHECK(ArrowArrayViewIsNull(rv.col(1), 1) != 0);
        CHECK(col_str(rv, 1, 0) == "a");
        CHECK(col_str(rv, 1, 2) == "b");
    }
}

TEST_CASE("explode - list/struct passthrough columns survive") {
    RecordBatchBuilder b;
    b.declare_schema(
        {{"pid", ColumnType::INT64},
         {"files", ColumnType::STRING_LIST},
         {"tags", ColumnType::STRING_LIST},
         {"tk",
          ColumnType::STRUCT_LIST,
          {{"value", ColumnType::STRING}, {"count", ColumnType::INT64}}}});
    b.append_int64(0, 1);
    b.append_string_list(1, {"a", "b"});
    b.append_string_list(2, {"t1", "t2"});
    b.append_struct_list(3,
                         {{StructCell{.str = "read"}, StructCell{.i64 = 5}}});
    b.end_row();
    b.append_int64(0, 2);
    b.append_string_list(1, {"x"});
    b.append_string_list(2, {"t3"});
    b.append_struct_list(3,
                         {{StructCell{.str = "open"}, StructCell{.i64 = 9}}});
    b.end_row();
    auto in = b.finish();

    // Explode "files"; "tags" (list) and "tk" (struct-list) must pass through.
    auto out = explode(in.get_schema(), in.get_array(), 1, false);
    REQUIRE(out.valid());
    CHECK(out.num_rows() == 3);
    CHECK(out.num_columns() == 4);
    CHECK(child_schema_name(out, 2) == "tags");
    CHECK(child_schema_format(out, 2) == std::string("+l"));
    CHECK(child_schema_name(out, 3) == "tk");
    CHECK(child_schema_format(out, 3) == std::string("+l"));

    ResultView rv(out);
    auto list_at = [&](int64_t c, int64_t r) {
        const ArrowArrayView* lv = rv.col(c);
        int64_t s = ArrowArrayViewListChildOffset(lv, r);
        int64_t e = ArrowArrayViewListChildOffset(lv, r + 1);
        std::vector<std::string> v;
        for (int64_t p = s; p < e; ++p) {
            ArrowStringView sv =
                ArrowArrayViewGetStringUnsafe(lv->children[0], p);
            v.emplace_back(sv.data, static_cast<std::size_t>(sv.size_bytes));
        }
        return v;
    };
    // Row 0 of "files" exploded into two rows; its "tags" list repeats whole.
    CHECK(col_str(rv, 1, 0) == "a");
    CHECK(col_str(rv, 1, 1) == "b");
    CHECK(list_at(2, 0) == std::vector<std::string>{"t1", "t2"});
    CHECK(list_at(2, 1) == std::vector<std::string>{"t1", "t2"});
    CHECK(list_at(2, 2) == std::vector<std::string>{"t3"});
    // Struct-list passthrough retains one element per row.
    const ArrowArrayView* tk = rv.col(3);
    CHECK(ArrowArrayViewListChildOffset(tk, 1) -
              ArrowArrayViewListChildOffset(tk, 0) ==
          1);
}

#endif  // DFTRACER_UTILS_ENABLE_ARROW
