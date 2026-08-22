// Native UNNEST/EXPLODE over a MapAccum: explodes a list-valued collection
// value component into one row per element, in pure C++ over the
// MonoidAccumulator structures, no Arrow. These cases must build and pass with
// DFTRACER_UTILS_ENABLE_ARROW OFF - the proof unnest is a native op and not an
// export-only feature. The other value components ride through opaquely.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/map_unnest.h>
#include <dftracer/utils/plugins/monoid.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <string>
#include <vector>

using dftracer::utils::plugins::ExplodedRow;
using dftracer::utils::plugins::ExplodedRows;
using dftracer::utils::plugins::MapAccum;
using dftracer::utils::plugins::MonoidAccumulator;
using dftracer::utils::plugins::unnest_map;

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

// pid -> (LIST_STR files, COUNTER hits). files preserves insertion order and
// duplicates; the counter is the opaque kept value.
void add_list(MapAccum& m, const Key& key, std::int64_t order,
              std::int64_t id) {
    m.touch(key)[0].add_ordered(order, id);
}
void add_hit(MapAccum& m, const Key& key, std::uint64_t v) {
    m.touch(key)[1].add_u64(v);
}

std::uint64_t counter_of(const MonoidAccumulator& m) {
    return m.to_value().as.u64;
}

}  // namespace

TEST_SUITE("MapUnnest") {
    TEST_CASE("LIST_STR explodes per element, duplicates preserved in order") {
        MapAccum m = make_map("f", {DFTU_T_I64},
                              {DFTU_MONOID_LIST_STR, DFTU_MONOID_COUNTER});
        add_list(m, {1}, 0, 10);
        add_list(m, {1}, 1, 10);  // dup file id
        add_list(m, {1}, 2, 11);
        add_hit(m, {1}, 7);
        add_list(m, {2}, 0, 20);
        add_hit(m, {2}, 3);

        ExplodedRows e = unnest_map(m, 0, false);
        REQUIRE(e.valid);
        CHECK(e.elem_type == DFTU_T_STR);
        CHECK(e.elem_name == "v0");
        REQUIRE(e.value_kinds.size() == 1);  // counter kept
        CHECK(e.value_kinds[0] == DFTU_MONOID_COUNTER);
        REQUIRE(e.rows.size() == 4);         // pid1: 10,10,11 ; pid2: 20

        // pid 1 first (sorted key), element order 10,10,11 (dup kept).
        CHECK(e.rows[0].key == Key{1});
        CHECK_FALSE(e.rows[0].elem_null);
        CHECK(e.rows[0].elem == 10);
        CHECK(e.rows[1].key == Key{1});
        CHECK(e.rows[1].elem == 10);
        CHECK(e.rows[2].key == Key{1});
        CHECK(e.rows[2].elem == 11);
        CHECK(e.rows[3].key == Key{2});
        CHECK(e.rows[3].elem == 20);

        // Kept COUNTER carried opaquely, readout intact and key repeated.
        for (std::size_t i = 0; i < 3; ++i) {
            REQUIRE(e.rows[i].kept_values.size() == 1);
            CHECK(counter_of(e.rows[i].kept_values[0]) == 7);
        }
        CHECK(counter_of(e.rows[3].kept_values[0]) == 3);
    }

    TEST_CASE("SET_STR explodes distinct elements only, sorted") {
        MapAccum m = make_map("s", {DFTU_T_I64}, {DFTU_MONOID_SET_STR});
        m.touch({1})[0].add_u64(21);
        m.touch({1})[0].add_u64(20);
        m.touch({1})[0].add_u64(21);  // dup collapses in a set

        ExplodedRows e = unnest_map(m, 0, false);
        REQUIRE(e.valid);
        CHECK(e.elem_type == DFTU_T_STR);
        CHECK(e.elem_name == "value");
        CHECK(e.value_kinds.empty());
        REQUIRE(e.rows.size() == 2);  // {20,21}
        CHECK(e.rows[0].elem == 20);
        CHECK(e.rows[1].elem == 21);
    }

    TEST_CASE("un-reloaded spilled runs make the unnest invalid, not partial") {
        MapAccum m = make_map("f", {DFTU_T_I64},
                              {DFTU_MONOID_LIST_STR, DFTU_MONOID_COUNTER});
        add_list(m, {1}, 0, 10);
        add_hit(m, {1}, 7);
        REQUIRE(unnest_map(m, 0, false).valid);
        m.runs.push_back({"pending-run"});
        ExplodedRows e = unnest_map(m, 0, false);
        CHECK_FALSE(e.valid);
        CHECK(e.rows.empty());
    }

    TEST_CASE(
        "empty collection: drops by default, keep_empty emits a null row") {
        MapAccum m = make_map("f", {DFTU_T_I64},
                              {DFTU_MONOID_LIST_STR, DFTU_MONOID_COUNTER});
        add_list(m, {1}, 0, 10);
        add_hit(m, {1}, 5);
        add_hit(m, {2}, 9);  // key 2 has an empty list

        ExplodedRows drop = unnest_map(m, 0, false);
        REQUIRE(drop.valid);
        REQUIRE(drop.rows.size() == 1);  // only pid 1's one element
        CHECK(drop.rows[0].key == Key{1});

        ExplodedRows keep = unnest_map(m, 0, true);
        REQUIRE(keep.valid);
        REQUIRE(keep.rows.size() == 2);
        CHECK(keep.rows[0].key == Key{1});
        CHECK_FALSE(keep.rows[0].elem_null);
        CHECK(keep.rows[1].key == Key{2});
        CHECK(keep.rows[1].elem_null);  // empty list -> null element
        CHECK(counter_of(keep.rows[1].kept_values[0]) == 9);
    }

    TEST_CASE("two build orders yield identical exploded rows") {
        auto build = [](bool reversed) {
            MapAccum m = make_map("f", {DFTU_T_I64},
                                  {DFTU_MONOID_LIST_STR, DFTU_MONOID_COUNTER});
            if (reversed) {
                add_list(m, {3}, 0, 30);
                add_hit(m, {3}, 3);
                add_list(m, {1}, 1, 11);
                add_list(m, {1}, 0, 10);
                add_hit(m, {1}, 1);
            } else {
                add_list(m, {1}, 0, 10);
                add_list(m, {1}, 1, 11);
                add_hit(m, {1}, 1);
                add_list(m, {3}, 0, 30);
                add_hit(m, {3}, 3);
            }
            return unnest_map(m, 0, false);
        };
        ExplodedRows a = build(false);
        ExplodedRows b = build(true);
        REQUIRE(a.rows.size() == b.rows.size());
        for (std::size_t i = 0; i < a.rows.size(); ++i) {
            CHECK(a.rows[i].key == b.rows[i].key);
            CHECK(a.rows[i].elem == b.rows[i].elem);
            CHECK(a.rows[i].elem_null == b.rows[i].elem_null);
        }
    }

    TEST_CASE("BOTTOMK_I64 explodes its kept elements, i64 element type") {
        MapAccum m = make_map("b", {DFTU_T_I64}, {DFTU_MONOID_BOTTOMK_I64});
        m.touch({1})[0].add_topk(2, 5.0, 500);
        m.touch({1})[0].add_topk(2, 1.0, 100);  // smaller by kept
        m.touch({1})[0].add_topk(2, 3.0, 300);

        ExplodedRows e = unnest_map(m, 0, false);
        REQUIRE(e.valid);
        CHECK(e.elem_type == DFTU_T_I64);
        REQUIRE(e.rows.size() == 2);  // bottom-2 by `by`: 100 then 300
        CHECK(e.rows[0].elem == 100);
        CHECK(e.rows[1].elem == 300);
    }

    TEST_CASE("non-collection and out-of-range components are rejected") {
        MapAccum scalar = make_map("c", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        scalar.touch({1})[0].add_u64(1);
        CHECK_FALSE(unnest_map(scalar, 0, false).valid);

        MapAccum approx =
            make_map("a", {DFTU_T_I64}, {DFTU_MONOID_APPROX_TOPK_STR});
        approx.touch({1})[0].add_approx_topk(4, 7);
        CHECK_FALSE(
            unnest_map(approx, 0, false).valid);       // struct-list deferred

        MapAccum lst = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_LIST_I64});
        CHECK_FALSE(unnest_map(lst, 1, false).valid);  // out of range
    }
}
