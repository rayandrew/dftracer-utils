// Native GROUPING SETS / CUBE / ROLLUP: composes regroup_map over several
// keep-subsets of a MapAccum key, no Arrow. These cases must build and pass
// with DFTRACER_UTILS_ENABLE_ARROW OFF - the proof the core is a native op and
// not an export-only feature.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/map_grouping.h>
#include <dftracer/utils/plugins/monoid.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using dftracer::utils::plugins::cube_sets;
using dftracer::utils::plugins::grouping_sets;
using dftracer::utils::plugins::MapAccum;
using dftracer::utils::plugins::MonoidAccumulator;
using dftracer::utils::plugins::rollup_sets;

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

void add_counter(MapAccum& m, const Key& key, std::uint64_t v) {
    m.touch(key)[0].add_u64(v);
}

const std::vector<MonoidAccumulator>* find_entry(const MapAccum& m,
                                                 const Key& key) {
    for (const MapAccum::EntriesMap& part : m.parts) {
        auto it = part.find(key);
        if (it != part.end()) return &it->second;
    }
    return nullptr;
}

std::uint64_t counter_of(const MonoidAccumulator& m) {
    return m.to_value().as.u64;
}

// (a,b) -> COUNTER. a in {1,2}, b in {10,20}. Per-cell counts distinct so every
// rolled-up total is checkable.
MapAccum build_ab() {
    MapAccum m =
        make_map("ab", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
    add_counter(m, {1, 10}, 1);
    add_counter(m, {1, 20}, 2);
    add_counter(m, {2, 10}, 4);
    add_counter(m, {2, 20}, 8);
    return m;
}

}  // namespace

TEST_SUITE("MapGroupingSets") {
    TEST_CASE("keep-set list yields one rollup per subset, incl grand total") {
        MapAccum m = build_ab();
        std::vector<KeepSet> sets = {{0, 1}, {0}, {1}, {}};
        std::vector<MapAccum> g = grouping_sets(m, sets);
        REQUIRE(g.size() == 4);

        // {0,1}: the base grouping, unchanged (4 cells).
        CHECK(g[0].key_n == 2);
        CHECK(g[0].total_entries() == 4);
        CHECK(counter_of((*find_entry(g[0], {1, 10}))[0]) == 1);
        CHECK(counter_of((*find_entry(g[0], {2, 20}))[0]) == 8);

        // {0}: rolled up over b, count summed per a.
        CHECK(g[1].key_n == 1);
        CHECK(g[1].total_entries() == 2);
        CHECK(counter_of((*find_entry(g[1], {1}))[0]) == 3);   // 1 + 2
        CHECK(counter_of((*find_entry(g[1], {2}))[0]) == 12);  // 4 + 8

        // {1}: rolled up over a, count summed per b.
        CHECK(g[2].key_n == 1);
        CHECK(g[2].total_entries() == 2);
        CHECK(counter_of((*find_entry(g[2], {10}))[0]) == 5);   // 1 + 4
        CHECK(counter_of((*find_entry(g[2], {20}))[0]) == 10);  // 2 + 8

        // {}: grand total, one entry with the empty key.
        CHECK(g[3].key_n == 0);
        REQUIRE(g[3].total_entries() == 1);
        CHECK(counter_of((*find_entry(g[3], {}))[0]) == 15);  // 1+2+4+8
    }

    TEST_CASE("out-of-range keep index yields that set's empty map") {
        MapAccum m = build_ab();
        std::vector<KeepSet> sets = {{0}, {5}};
        std::vector<MapAccum> g = grouping_sets(m, sets);
        REQUIRE(g.size() == 2);
        CHECK(g[0].total_entries() == 2);
        CHECK(g[1].key_n == 0);
        CHECK(g[1].total_entries() == 0);
    }

    TEST_CASE("un-reloaded spilled runs yield no grouping sets, not partial") {
        MapAccum m = build_ab();
        std::vector<KeepSet> sets = {{0}, {}};
        REQUIRE(grouping_sets(m, sets).size() == 2);
        m.runs.push_back({"pending-run"});
        CHECK(grouping_sets(m, sets).empty());
    }

    TEST_CASE("determinism across build orders") {
        MapAccum fwd = build_ab();
        MapAccum rev =
            make_map("ab", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(rev, {2, 20}, 8);
        add_counter(rev, {2, 10}, 4);
        add_counter(rev, {1, 20}, 2);
        add_counter(rev, {1, 10}, 1);

        std::vector<KeepSet> sets = {{0}, {}};
        std::vector<MapAccum> a = grouping_sets(fwd, sets);
        std::vector<MapAccum> b = grouping_sets(rev, sets);
        REQUIRE(a.size() == b.size());
        for (std::size_t i = 0; i < a.size(); ++i) {
            CHECK(a[i].total_entries() == b[i].total_entries());
            for (const MapAccum::EntriesMap& part : a[i].parts)
                for (const auto& e : part) {
                    const std::vector<MonoidAccumulator>* ob =
                        find_entry(b[i], e.first);
                    REQUIRE(ob != nullptr);
                    CHECK(counter_of(e.second[0]) == counter_of((*ob)[0]));
                }
        }
    }
}

TEST_SUITE("MapCubeRollup") {
    TEST_CASE("cube_sets(2) enumerates all subsets by ascending mask") {
        std::vector<KeepSet> c = cube_sets(2);
        std::vector<KeepSet> want = {{}, {0}, {1}, {0, 1}};
        CHECK(c == want);
    }

    TEST_CASE("cube_sets(0) is a single empty set; cube_sets(3) has 8") {
        CHECK(cube_sets(0) == std::vector<KeepSet>{{}});
        CHECK(cube_sets(3).size() == 8);
    }

    TEST_CASE("cube_sets caps oversized key_n") {
        CHECK(cube_sets(21).empty());
    }

    TEST_CASE("rollup_sets(2) is the key prefixes down to empty") {
        std::vector<KeepSet> r = rollup_sets(2);
        std::vector<KeepSet> want = {{0, 1}, {0}, {}};
        CHECK(r == want);
    }

    TEST_CASE("rollup_sets(0) is a single empty set") {
        CHECK(rollup_sets(0) == std::vector<KeepSet>{{}});
    }

    TEST_CASE("cube composed with grouping_sets covers every subset total") {
        MapAccum m = build_ab();
        std::vector<KeepSet> c = cube_sets(2);
        std::vector<MapAccum> g = grouping_sets(m, c);
        REQUIRE(g.size() == 4);
        // c = {{},{0},{1},{0,1}}: grand total, per-a, per-b, base.
        CHECK(counter_of((*find_entry(g[0], {}))[0]) == 15);
        CHECK(g[1].total_entries() == 2);
        CHECK(g[2].total_entries() == 2);
        CHECK(g[3].total_entries() == 4);
    }
}
