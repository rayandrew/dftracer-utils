#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

// Confirms LargeString/LargeBinary/LargeList compute, not just round-trip:
// they keep their own 64-bit-offset physical layout end to end (no narrowing
// copy on import), and every kernel that already worked on String/Binary/List
// works on them directly - equality, comparison, sort, group_by, a string op,
// is_in, and take/gather (which sort's reordering relies on).
#ifdef DFTRACER_UTILS_ENABLE_ARROW
// clang-format off
#include <nanoarrow/nanoarrow.h>
#include <dftracer/utils/dataframe/arrow.h>
#include <dftracer/utils/dataframe/abi.h>
// clang-format on

using dftracer::utils::dataframe::is_wide_offset_type;
using dftracer::utils::dataframe::narrow_varwidth_type;
using dftracer::utils::dataframe::OwnedArrow;
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

Series make_large_utf8(const std::vector<std::string>& values) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetType(&b.schema, NANOARROW_TYPE_LARGE_STRING) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (const std::string& v : values) {
        ArrowStringView sv{v.data(), static_cast<int64_t>(v.size())};
        REQUIRE(ArrowArrayAppendString(&b.array, sv) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

Series make_large_binary(const std::vector<std::string>& values) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetType(&b.schema, NANOARROW_TYPE_LARGE_BINARY) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (const std::string& v : values) {
        ArrowBufferView bv;
        bv.data.data = v.data();
        bv.size_bytes = static_cast<int64_t>(v.size());
        REQUIRE(ArrowArrayAppendBytes(&b.array, bv) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

// large_list<int64>: rows = lists of int64 values.
Series make_large_list_i64(const std::vector<std::vector<std::int64_t>>& rows) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetType(&b.schema, NANOARROW_TYPE_LARGE_LIST) ==
            NANOARROW_OK);
    REQUIRE(ArrowSchemaSetType(b.schema.children[0], NANOARROW_TYPE_INT64) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (const auto& row : rows) {
        for (std::int64_t v : row)
            REQUIRE(ArrowArrayAppendInt(b.array.children[0], v) ==
                    NANOARROW_OK);
        REQUIRE(ArrowArrayFinishElement(&b.array) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

// large_list<large_utf8>: rows = lists of strings, nested Large both levels.
Series make_large_list_large_utf8(
    const std::vector<std::vector<std::string>>& rows) {
    BuiltArrow b;
    ArrowSchemaInit(&b.schema);
    REQUIRE(ArrowSchemaSetType(&b.schema, NANOARROW_TYPE_LARGE_LIST) ==
            NANOARROW_OK);
    REQUIRE(ArrowSchemaSetType(b.schema.children[0],
                               NANOARROW_TYPE_LARGE_STRING) == NANOARROW_OK);
    REQUIRE(ArrowArrayInitFromSchema(&b.array, &b.schema, nullptr) ==
            NANOARROW_OK);
    REQUIRE(ArrowArrayStartAppending(&b.array) == NANOARROW_OK);
    for (const auto& row : rows) {
        for (const std::string& s : row) {
            ArrowStringView sv{s.data(), static_cast<int64_t>(s.size())};
            REQUIRE(ArrowArrayAppendString(b.array.children[0], sv) ==
                    NANOARROW_OK);
        }
        REQUIRE(ArrowArrayFinishElement(&b.array) == NANOARROW_OK);
    }
    finish(&b.array);
    return Series::from_arrow(&b.schema, &b.array);
}

}  // namespace

TEST_SUITE("dataframe_arrow_large_types_compute") {
    TEST_CASE(
        "large_utf8 imports as LargeString with its own 64-bit "
        "offsets, not narrowed to String") {
        Series s = make_large_utf8({"alpha", "beta", "gamma"});
        REQUIRE(s.valid());
        CHECK(s.type() == TypeId::LargeString);
        CHECK(is_wide_offset_type(s.type()));
        CHECK(narrow_varwidth_type(s.type()) == TypeId::String);
        REQUIRE(s.offsets64() != nullptr);
        CHECK(s.offsets() == nullptr);  // no int32 offsets buffer at all
        CHECK(s.string_at(0) == "alpha");
        CHECK(s.string_at(1) == "beta");
        CHECK(s.string_at(2) == "gamma");
    }

    TEST_CASE(
        "LargeString: equality and comparison (!=) run directly on the "
        "wide-offset column") {
        Series s = make_large_utf8({"foo", "bar", "foo", "baz"});
        REQUIRE(s.valid());

        Series eq = s.str_eq("foo");
        REQUIRE(eq.valid());
        REQUIRE(eq.length() == 4);
        const std::uint8_t* bits = eq.data<std::uint8_t>();
        REQUIRE(bits != nullptr);
        auto bit = [&](std::int64_t i) {
            return ((bits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(bit(0) == true);
        CHECK(bit(1) == false);
        CHECK(bit(2) == true);
        CHECK(bit(3) == false);

        dftu_series* ne = dftu_series_logical_not(eq.handle());
        REQUIRE(ne != nullptr);
        Series neq{ne};
        const std::uint8_t* nbits = neq.data<std::uint8_t>();
        REQUIRE(nbits != nullptr);
        auto nbit = [&](std::int64_t i) {
            return ((nbits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(nbit(0) == false);
        CHECK(nbit(1) == true);
        CHECK(nbit(2) == false);
        CHECK(nbit(3) == true);
    }

    TEST_CASE(
        "LargeString: str_contains runs directly on the wide-offset "
        "column") {
        Series s = make_large_utf8({"hello world", "goodbye", "worldly"});
        REQUIRE(s.valid());
        Series mask = s.str_contains("world");
        REQUIRE(mask.valid());
        REQUIRE(mask.length() == 3);
        const std::uint8_t* bits = mask.data<std::uint8_t>();
        REQUIRE(bits != nullptr);
        auto bit = [&](std::int64_t i) {
            return ((bits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(bit(0) == true);
        CHECK(bit(1) == false);
        CHECK(bit(2) == true);
    }

    TEST_CASE("LargeString: is_in runs directly on the wide-offset column") {
        Series s = make_large_utf8({"a", "b", "c", "d"});
        Series needles = Series::strings({"b", "d"});
        REQUIRE(s.valid());
        REQUIRE(needles.valid());
        Series mask = s.is_in(needles);
        REQUIRE(mask.valid());
        REQUIRE(mask.length() == 4);
        const std::uint8_t* bits = mask.data<std::uint8_t>();
        REQUIRE(bits != nullptr);
        auto bit = [&](std::int64_t i) {
            return ((bits[i >> 3] >> (i & 7)) & 1) != 0;
        };
        CHECK(bit(0) == false);
        CHECK(bit(1) == true);
        CHECK(bit(2) == false);
        CHECK(bit(3) == true);
    }

    TEST_CASE(
        "LargeString: sort (argsort + take) reorders correctly and "
        "stays LargeString, offsets still 64-bit") {
        Series s = make_large_utf8({"banana", "apple", "cherry"});
        REQUIRE(s.valid());
        Series sorted = s.sort(false);
        REQUIRE(sorted.valid());
        CHECK(sorted.type() == TypeId::LargeString);
        CHECK(is_wide_offset_type(sorted.type()));
        REQUIRE(sorted.offsets64() != nullptr);
        REQUIRE(sorted.length() == 3);
        CHECK(sorted.string_at(0) == "apple");
        CHECK(sorted.string_at(1) == "banana");
        CHECK(sorted.string_at(2) == "cherry");
    }

    TEST_CASE("LargeString: dftu_series_group_by groups correctly") {
        Series keys = make_large_utf8({"x", "y", "x", "z", "y", "x"});
        std::vector<std::int64_t> ones(6, 1);
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
        CHECK(dftu_series_length(out_keys) == 3);  // x, y, z
        Series counts{out_values[0]};
        REQUIRE(counts.valid());
        REQUIRE(counts.length() == 3);
        const std::int64_t* c = counts.data<std::int64_t>();
        REQUIRE(c != nullptr);
        CHECK(c[0] == 3);  // x first-seen
        CHECK(c[1] == 2);  // y
        CHECK(c[2] == 1);  // z
        dftu_series_free(out_keys);
    }

    TEST_CASE(
        "LargeBinary: element data survives take/gather at 64-bit "
        "offsets") {
        std::string b0(3, '\x00');
        b0[1] = '\x01';
        Series s = make_large_binary({b0, "abc", "de"});
        REQUIRE(s.valid());
        CHECK(s.type() == TypeId::LargeBinary);
        CHECK(is_wide_offset_type(s.type()));

        Series taken = s.take(std::vector<std::int64_t>{2, 0, 1});
        REQUIRE(taken.valid());
        CHECK(taken.type() == TypeId::LargeBinary);
        REQUIRE(taken.offsets64() != nullptr);
        REQUIRE(taken.length() == 3);
        CHECK(taken.string_at(0) == "de");
        CHECK(taken.string_at(1) == b0);
        CHECK(taken.string_at(2) == "abc");
    }

    TEST_CASE(
        "LargeList<int64>: child data survives materialize/take at "
        "64-bit offsets") {
        Series s = make_large_list_i64({{1, 2, 3}, {}, {4, 5}});
        REQUIRE(s.valid());
        CHECK(s.type() == TypeId::LargeList);
        CHECK(is_wide_offset_type(s.type()));
        REQUIRE(s.offsets64() != nullptr);
        REQUIRE(s.length() == 3);

        Series taken = s.take(std::vector<std::int64_t>{2, 0});
        REQUIRE(taken.valid());
        CHECK(taken.type() == TypeId::LargeList);
        REQUIRE(taken.offsets64() != nullptr);
        REQUIRE(taken.length() == 2);
        Series child = taken.child(0);
        REQUIRE(child.valid());
        REQUIRE(child.type() == TypeId::Int64);
        const std::int64_t* vals = child.data<std::int64_t>();
        REQUIRE(vals != nullptr);
        // Row 0 of `taken` is source row 2 ({4, 5}); row 1 is source row 0
        // ({1, 2, 3}).
        REQUIRE(child.length() == 5);
        CHECK(vals[0] == 4);
        CHECK(vals[1] == 5);
        CHECK(vals[2] == 1);
        CHECK(vals[3] == 2);
        CHECK(vals[4] == 3);
    }

    TEST_CASE(
        "Nested large_list<large_utf8>: both levels are wide-offset "
        "and the string data survives") {
        Series s = make_large_list_large_utf8({{"a", "bb"}, {"ccc"}, {}});
        REQUIRE(s.valid());
        CHECK(s.type() == TypeId::LargeList);
        CHECK(is_wide_offset_type(s.type()));
        REQUIRE(s.length() == 3);

        Series child = s.child(0);
        REQUIRE(child.valid());
        CHECK(child.type() == TypeId::LargeString);
        CHECK(is_wide_offset_type(child.type()));
        REQUIRE(child.length() == 3);
        CHECK(child.string_at(0) == "a");
        CHECK(child.string_at(1) == "bb");
        CHECK(child.string_at(2) == "ccc");

        // take/gather on the outer LargeList recurses into the LargeString
        // child, exercising the nested wide-offset gather path.
        Series taken = s.take(std::vector<std::int64_t>{1, 0});
        REQUIRE(taken.valid());
        CHECK(taken.type() == TypeId::LargeList);
        Series tchild = taken.child(0);
        REQUIRE(tchild.valid());
        CHECK(tchild.type() == TypeId::LargeString);
        REQUIRE(tchild.length() == 3);  // "ccc", "a", "bb"
        CHECK(tchild.string_at(0) == "ccc");
        CHECK(tchild.string_at(1) == "a");
        CHECK(tchild.string_at(2) == "bb");
    }

    TEST_CASE(
        "large_utf8 round-trips through export as utf8 with intact "
        "values, and its offsets were never copied down to int32") {
        Series s = make_large_utf8({"one", "two", "three"});
        REQUIRE(s.valid());
        REQUIRE(s.type() == TypeId::LargeString);
        const std::int64_t* wide_off = s.offsets64();
        REQUIRE(wide_off != nullptr);
        CHECK(s.offsets() == nullptr);  // never materialized an int32 buffer

        OwnedArrow out = s.to_arrow();
        REQUIRE(static_cast<bool>(out));
        ArrowSchemaView view;
        ArrowError err;
        REQUIRE(ArrowSchemaViewInit(&view, out.schema(), &err) == NANOARROW_OK);
        // A faithful representation of the same data: our LargeString
        // exports as Arrow's large_utf8, not narrowed to utf8.
        CHECK(view.type == NANOARROW_TYPE_LARGE_STRING);

        Series back = Series::from_arrow(out.schema(), out.array());
        REQUIRE(back.valid());
        CHECK(back.type() == TypeId::LargeString);
        REQUIRE(back.length() == 3);
        CHECK(back.string_at(0) == "one");
        CHECK(back.string_at(1) == "two");
        CHECK(back.string_at(2) == "three");
    }

    TEST_CASE(
        "DataFrame::group_by refuses a LargeString key loudly, not "
        "wrong groups") {
        using dftracer::utils::dataframe::Agg;
        using dftracer::utils::dataframe::DataFrame;
        using dftracer::utils::dataframe::GroupAgg;
        Series keys = make_large_utf8({"a", "b", "a"});
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
TEST_SUITE("dataframe_arrow_large_types_compute") {
    TEST_CASE("Arrow disabled: nothing to exercise") { CHECK(true); }
}
#endif
