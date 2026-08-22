// Arrow EXPORT half of GROUPING SETS: materialize_grouping_sets turns the
// per-set regrouped MapAccums into one record batch [nullable key cols k0..,
// value cols, grouping_id]. These cases only build under
// DFTRACER_UTILS_ENABLE_ARROW; the core is proved Arrow-free in
// test_map_grouping.cpp. Read the batch back through the Arrow C data buffers
// and assert rolled-up values, dropped-dim nulls (via the validity bitmap), and
// the grouping_id column.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <doctest/doctest.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/map_grouping.h>
#include <dftracer/utils/plugins/map_grouping_arrow.h>
#include <dftracer/utils/plugins/monoid.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using dftracer::utils::StringIntern;
using dftracer::utils::plugins::cube_sets;
using dftracer::utils::plugins::grouping_sets;
using dftracer::utils::plugins::MapAccum;
using dftracer::utils::plugins::materialize_grouping_sets;
namespace arr = dftracer::utils::utilities::common::arrow;

namespace {

using Key = std::vector<std::int64_t>;
using KeepSet = std::vector<std::uint32_t>;

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

MapAccum build_ab() {
    MapAccum m =
        make_map("ab", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
    m.touch({1, 10})[0].add_u64(1);
    m.touch({1, 20})[0].add_u64(2);
    m.touch({2, 10})[0].add_u64(4);
    m.touch({2, 20})[0].add_u64(8);
    return m;
}

}  // namespace

TEST_SUITE("MapGroupingSetsArrow") {
    TEST_CASE("unified schema: nullable key cols, value col, grouping_id") {
        StringIntern intern;
        MapAccum m = build_ab();
        std::vector<KeepSet> sets = {{0, 1}, {0}, {1}, {}};
        std::vector<MapAccum> g = grouping_sets(m, sets);
        arr::ArrowExportResult res =
            materialize_grouping_sets(g, sets, m, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        const ArrowSchema* s = res.get_schema();
        REQUIRE(a->n_children == 4);  // k0, k1, value, grouping_id
        CHECK(std::string_view(s->children[0]->name) == "k0");
        CHECK(std::string_view(s->children[1]->name) == "k1");
        CHECK(std::string_view(s->children[2]->name) == "value");
        CHECK(std::string_view(s->children[3]->name) == "grouping_id");
        // 4 (base) + 2 (per a) + 2 (per b) + 1 (grand total) = 9 rows.
        REQUIRE(a->length == 9);
    }

    TEST_CASE("dropped dims are null; rolled-up values and grouping_id right") {
        StringIntern intern;
        MapAccum m = build_ab();
        std::vector<KeepSet> sets = {{0, 1}, {0}, {1}, {}};
        std::vector<MapAccum> g = grouping_sets(m, sets);
        arr::ArrowExportResult res =
            materialize_grouping_sets(g, sets, m, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        const std::int64_t* gid = i64_col(a, 3);
        const std::int64_t* val = i64_col(a, 2);

        // Rows 0..3: base grouping, both keys present, grouping_id 0.
        for (int r = 0; r < 4; ++r) {
            CHECK(gid[r] == 0);
            CHECK_FALSE(cell_is_null(a, 0, r));
            CHECK_FALSE(cell_is_null(a, 1, r));
        }

        // Rows 4..5: {0} (per a), k1 null, sorted by a: (1)->3, (2)->12.
        CHECK(gid[4] == 1);
        CHECK_FALSE(cell_is_null(a, 0, 4));
        CHECK(cell_is_null(a, 1, 4));
        CHECK(i64_col(a, 0)[4] == 1);
        CHECK(val[4] == 3);
        CHECK(i64_col(a, 0)[5] == 2);
        CHECK(val[5] == 12);

        // Rows 6..7: {1} (per b), k0 null, sorted by b: (10)->5, (20)->10.
        CHECK(gid[6] == 2);
        CHECK(cell_is_null(a, 0, 6));
        CHECK_FALSE(cell_is_null(a, 1, 6));
        CHECK(i64_col(a, 1)[6] == 10);
        CHECK(val[6] == 5);
        CHECK(i64_col(a, 1)[7] == 20);
        CHECK(val[7] == 10);

        // Row 8: grand total {}, both keys null, value 15, grouping_id 3.
        CHECK(gid[8] == 3);
        CHECK(cell_is_null(a, 0, 8));
        CHECK(cell_is_null(a, 1, 8));
        CHECK(val[8] == 15);
    }

    TEST_CASE("cube_sets drives the same unified batch") {
        StringIntern intern;
        MapAccum m = build_ab();
        std::vector<KeepSet> c = cube_sets(2);
        std::vector<MapAccum> g = grouping_sets(m, c);
        arr::ArrowExportResult res = materialize_grouping_sets(g, c, m, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        // c = {{},{0},{1},{0,1}}: 1 + 2 + 2 + 4 = 9 rows.
        REQUIRE(a->length == 9);
        // Row 0 is the grand total (empty set first), both keys null.
        CHECK(i64_col(a, 3)[0] == 0);
        CHECK(cell_is_null(a, 0, 0));
        CHECK(cell_is_null(a, 1, 0));
        CHECK(i64_col(a, 2)[0] == 15);
    }
}

#else
TEST_CASE("map grouping arrow export requires DFTRACER_UTILS_ENABLE_ARROW") {}
#endif  // DFTRACER_UTILS_ENABLE_ARROW
