// Arrow EXPORT half of the native unnest: materialize_exploded turns an
// ExplodedRows into one record batch [key cols, kept value cols, the exploded
// element column]. These cases only build under DFTRACER_UTILS_ENABLE_ARROW;
// unnest itself is proved Arrow-free in test_map_unnest.cpp. Read the batch
// back through the Arrow C data buffers and assert the element column (dup
// preserved, utf8), the opaque kept value, and a keep_empty null element via
// validity.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <doctest/doctest.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/map_unnest.h>
#include <dftracer/utils/plugins/map_unnest_arrow.h>
#include <dftracer/utils/plugins/monoid.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using dftracer::utils::StringIntern;
using dftracer::utils::plugins::ExplodedRows;
using dftracer::utils::plugins::MapAccum;
using dftracer::utils::plugins::materialize_exploded;
using dftracer::utils::plugins::unnest_map;
namespace arr = dftracer::utils::utilities::common::arrow;

namespace {

using Key = std::vector<std::int64_t>;

MapAccum make_map(const char* name, std::vector<dftu_type> kt,
                  std::vector<dftu_monoid_kind> vals) {
    MapAccum m;
    m.name = name;
    m.key_n = static_cast<std::uint32_t>(kt.size());
    m.key_types = std::move(kt);
    m.value_kinds = std::move(vals);
    return m;
}

const std::int64_t* i64_col(const ArrowArray* a, int c) {
    const ArrowArray* ch = a->children[c];
    return static_cast<const std::int64_t*>(ch->buffers[1]) + ch->offset;
}

bool cell_is_null(const ArrowArray* a, int c, std::int64_t r) {
    const ArrowArray* ch = a->children[c];
    const auto* v = static_cast<const std::uint8_t*>(ch->buffers[0]);
    if (!v) return false;
    const std::int64_t idx = ch->offset + r;
    return ((v[idx / 8] >> (idx % 8)) & 1) == 0;
}

// One utf8 scalar cell of a string column.
std::string string_at(const ArrowArray* a, int c, std::int64_t r) {
    const ArrowArray* ch = a->children[c];
    const auto* off =
        static_cast<const std::int32_t*>(ch->buffers[1]) + ch->offset;
    const char* chars = static_cast<const char*>(ch->buffers[2]);
    return std::string(chars + off[r], chars + off[r + 1]);
}

}  // namespace

TEST_SUITE("MapUnnestArrow") {
    TEST_CASE("LIST_STR element column, dup preserved, counter kept opaque") {
        StringIntern intern;
        const dftu_str foo = intern.get_or_insert("foo");
        const dftu_str bar = intern.get_or_insert("bar");
        const dftu_str baz = intern.get_or_insert("baz");

        MapAccum m = make_map("f", {DFTU_T_I64},
                              {DFTU_MONOID_LIST_STR, DFTU_MONOID_COUNTER});
        m.touch({1})[0].add_ordered(0, foo);
        m.touch({1})[0].add_ordered(1, foo);  // dup
        m.touch({1})[0].add_ordered(2, bar);
        m.touch({1})[1].add_u64(7);
        m.touch({2})[0].add_ordered(0, baz);
        m.touch({2})[1].add_u64(3);

        ExplodedRows e = unnest_map(m, 0, false);
        arr::ArrowExportResult res = materialize_exploded(e, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 4);
        REQUIRE(a->n_children == 3);  // k0, value (counter), v0 (element)
        const ArrowSchema* s = res.get_schema();
        CHECK(std::string_view(s->children[0]->name) == "k0");
        CHECK(std::string_view(s->children[1]->name) == "value");
        CHECK(std::string_view(s->children[2]->name) == "v0");
        CHECK(std::string_view(s->children[2]->format) == "u");

        const std::int64_t* k0 = i64_col(a, 0);
        const std::int64_t* cnt = i64_col(a, 1);
        CHECK(k0[0] == 1);
        CHECK(string_at(a, 2, 0) == "foo");
        CHECK(k0[1] == 1);
        CHECK(string_at(a, 2, 1) == "foo");  // dup preserved
        CHECK(k0[2] == 1);
        CHECK(string_at(a, 2, 2) == "bar");
        CHECK(k0[3] == 2);
        CHECK(string_at(a, 2, 3) == "baz");
        for (int r = 0; r < 3; ++r) CHECK(cnt[r] == 7);  // kept counter carried
        CHECK(cnt[3] == 3);
    }

    TEST_CASE("keep_empty emits a null element cell via validity bits") {
        StringIntern intern;
        const dftu_str foo = intern.get_or_insert("foo");

        MapAccum m = make_map("f", {DFTU_T_I64},
                              {DFTU_MONOID_LIST_STR, DFTU_MONOID_COUNTER});
        m.touch({1})[0].add_ordered(0, foo);
        m.touch({1})[1].add_u64(5);
        m.touch({2})[1].add_u64(9);  // empty list for key 2

        ExplodedRows e = unnest_map(m, 0, true);
        arr::ArrowExportResult res = materialize_exploded(e, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 2);
        REQUIRE(a->n_children == 3);

        CHECK(i64_col(a, 0)[0] == 1);
        CHECK_FALSE(cell_is_null(a, 2, 0));
        CHECK(string_at(a, 2, 0) == "foo");
        // key 2: element null, kept counter still present.
        CHECK(i64_col(a, 0)[1] == 2);
        CHECK(cell_is_null(a, 2, 1));
        CHECK(i64_col(a, 1)[1] == 9);
    }
}

#else
TEST_CASE("map unnest arrow export requires DFTRACER_UTILS_ENABLE_ARROW") {}
#endif  // DFTRACER_UTILS_ENABLE_ARROW
