// Native map-to-map equi join: builds two MapAccum results over a shared key
// space and joins them in pure C++ over the MonoidAccumulator structures, no
// Arrow. These cases must build and pass with DFTRACER_UTILS_ENABLE_ARROW OFF -
// that is the proof the join is a native op and not an export-only feature. The
// join only compares keys and copies value Monoids opaquely, so it is generic
// over any value schema (a COUNTER, a SUM_F64, a SET_STR all ride through
// unread).

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/map_join.h>
#include <dftracer/utils/plugins/monoid.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <set>
#include <string>
#include <vector>

using dftracer::utils::plugins::fk_join;
using dftracer::utils::plugins::join_maps;
using dftracer::utils::plugins::JoinedMap;
using dftracer::utils::plugins::JoinedRow;
using dftracer::utils::plugins::JoinType;
using dftracer::utils::plugins::MapAccum;
using dftracer::utils::plugins::MonoidAccumulator;
using dftracer::utils::plugins::regroup_map;

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

void add_counter(MapAccum& m, const Key& key, std::uint64_t v) {
    m.touch(key)[0].add_u64(v);
}

void add_sum(MapAccum& m, const Key& key, double v) {
    m.touch(key)[0].add_f64(v, 1.0);
}

void add_set(MapAccum& m, const Key& key, std::int64_t id) {
    m.touch(key)[0].add_u64(static_cast<std::uint64_t>(id));
}

const std::vector<MonoidAccumulator>* find_entry(const MapAccum& m,
                                                 const Key& key) {
    for (const MapAccum::EntriesMap& part : m.parts) {
        auto it = part.find(key);
        if (it != part.end()) return &it->second;
    }
    return nullptr;
}

const JoinedRow* find_row(const JoinedMap& j, const Key& key) {
    for (const JoinedRow& r : j.rows)
        if (r.key == key) return &r;
    return nullptr;
}

std::uint64_t counter_of(const MonoidAccumulator& m) {
    return m.to_value().as.u64;
}
double sum_of(const MonoidAccumulator& m) { return m.to_value().as.f64; }

std::set<std::int64_t> keys_of(const JoinedMap& j) {
    std::set<std::int64_t> out;
    for (const JoinedRow& r : j.rows) out.insert(r.key[0]);
    return out;
}

// left = pid -> COUNTER over {1,2,3}; right = pid -> SUM_F64 over {2,3,4}.
// Intersection is {2,3}; pid 1 is left-only, pid 4 is right-only.
MapAccum build_left() {
    MapAccum m = make_map("left", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
    add_counter(m, {1}, 10);
    add_counter(m, {2}, 20);
    add_counter(m, {2}, 5);  // pid 2 counter -> 25
    add_counter(m, {3}, 30);
    return m;
}

MapAccum build_right() {
    MapAccum m = make_map("right", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
    add_sum(m, {2}, 2.5);
    add_sum(m, {3}, 3.0);
    add_sum(m, {3}, 0.5);  // pid 3 sum -> 3.5
    add_sum(m, {4}, 4.0);
    return m;
}

}  // namespace

TEST_SUITE("MapJoin") {
    TEST_CASE("INNER keeps only the key intersection, both sides present") {
        JoinedMap j = join_maps(build_left(), build_right(), JoinType::INNER);
        REQUIRE(j.valid);
        CHECK(keys_of(j) == std::set<std::int64_t>{2, 3});
        for (const JoinedRow& r : j.rows) {
            CHECK(r.left_present);
            CHECK(r.right_present);
        }
        const JoinedRow* r2 = find_row(j, {2});
        REQUIRE(r2 != nullptr);
        CHECK(counter_of(r2->left_values[0]) == 25);
        CHECK(sum_of(r2->right_values[0]) == doctest::Approx(2.5));
        const JoinedRow* r3 = find_row(j, {3});
        REQUIRE(r3 != nullptr);
        CHECK(counter_of(r3->left_values[0]) == 30);
        CHECK(sum_of(r3->right_values[0]) == doctest::Approx(3.5));
    }

    TEST_CASE("LEFT keeps all left keys; unmatched null the right side") {
        JoinedMap j = join_maps(build_left(), build_right(), JoinType::LEFT);
        REQUIRE(j.valid);
        CHECK(keys_of(j) == std::set<std::int64_t>{1, 2, 3});
        const JoinedRow* r1 = find_row(j, {1});
        REQUIRE(r1 != nullptr);
        CHECK(r1->left_present);
        CHECK(counter_of(r1->left_values[0]) == 10);
        CHECK_FALSE(r1->right_present);
        CHECK(r1->right_values.empty());
        const JoinedRow* r2 = find_row(j, {2});
        REQUIRE(r2 != nullptr);
        CHECK(r2->left_present);
        CHECK(r2->right_present);
        CHECK(sum_of(r2->right_values[0]) == doctest::Approx(2.5));
    }

    TEST_CASE("RIGHT keeps all right keys; unmatched null the left side") {
        JoinedMap j = join_maps(build_left(), build_right(), JoinType::RIGHT);
        REQUIRE(j.valid);
        CHECK(keys_of(j) == std::set<std::int64_t>{2, 3, 4});
        const JoinedRow* r4 = find_row(j, {4});
        REQUIRE(r4 != nullptr);
        CHECK_FALSE(r4->left_present);
        CHECK(r4->left_values.empty());
        CHECK(r4->right_present);
        CHECK(sum_of(r4->right_values[0]) == doctest::Approx(4.0));
        const JoinedRow* r3 = find_row(j, {3});
        REQUIRE(r3 != nullptr);
        CHECK(r3->left_present);
        CHECK(counter_of(r3->left_values[0]) == 30);
    }

    TEST_CASE(
        "FULL keeps the key union; each side present iff it had the key") {
        JoinedMap j = join_maps(build_left(), build_right(), JoinType::FULL);
        REQUIRE(j.valid);
        CHECK(keys_of(j) == std::set<std::int64_t>{1, 2, 3, 4});
        const JoinedRow* r1 = find_row(j, {1});
        REQUIRE(r1 != nullptr);
        CHECK(r1->left_present);
        CHECK_FALSE(r1->right_present);
        const JoinedRow* r4 = find_row(j, {4});
        REQUIRE(r4 != nullptr);
        CHECK_FALSE(r4->left_present);
        CHECK(r4->right_present);
        const JoinedRow* r2 = find_row(j, {2});
        REQUIRE(r2 != nullptr);
        CHECK(r2->left_present);
        CHECK(r2->right_present);
    }

    TEST_CASE("LEFT_SEMI keeps matched left keys once, left-only") {
        JoinedMap j =
            join_maps(build_left(), build_right(), JoinType::LEFT_SEMI);
        REQUIRE(j.valid);
        CHECK(j.left_only);
        CHECK(keys_of(j) == std::set<std::int64_t>{2, 3});
        const JoinedRow* r2 = find_row(j, {2});
        REQUIRE(r2 != nullptr);
        CHECK(r2->left_present);
        CHECK(counter_of(r2->left_values[0]) == 25);
        CHECK_FALSE(r2->right_present);
        CHECK(r2->right_values.empty());
    }

    TEST_CASE("LEFT_ANTI keeps unmatched left keys, left-only") {
        JoinedMap j =
            join_maps(build_left(), build_right(), JoinType::LEFT_ANTI);
        REQUIRE(j.valid);
        CHECK(j.left_only);
        CHECK(keys_of(j) == std::set<std::int64_t>{1});
        const JoinedRow* r1 = find_row(j, {1});
        REQUIRE(r1 != nullptr);
        CHECK(r1->left_present);
        CHECK(counter_of(r1->left_values[0]) == 10);
        CHECK_FALSE(r1->right_present);
        CHECK(r1->right_values.empty());
    }

    TEST_CASE("multi-column key joins on the full tuple") {
        MapAccum l =
            make_map("l", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(l, {1, 100}, 1);
        add_counter(l, {1, 200}, 2);
        add_counter(l, {2, 100}, 3);
        MapAccum r =
            make_map("r", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        add_sum(r, {1, 200}, 9.0);  // matches (1,200)
        add_sum(r, {2, 100}, 8.0);  // matches (2,100)
        add_sum(r, {2, 999}, 7.0);  // right-only

        JoinedMap j = join_maps(l, r, JoinType::INNER);
        REQUIRE(j.valid);
        REQUIRE(j.rows.size() == 2);
        // Sorted by the tuple: (1,200) then (2,100).
        CHECK(j.rows[0].key == Key{1, 200});
        CHECK(j.rows[1].key == Key{2, 100});
        CHECK(counter_of(j.rows[0].left_values[0]) == 2);
        CHECK(sum_of(j.rows[0].right_values[0]) == doctest::Approx(9.0));

        JoinedMap jf = join_maps(l, r, JoinType::FULL);
        CHECK(jf.rows.size() == 4);  // (1,100) (1,200) (2,100) (2,999)
        const JoinedRow* only_left = find_row(jf, {1, 100});
        REQUIRE(only_left != nullptr);
        CHECK(only_left->left_present);
        CHECK_FALSE(only_left->right_present);
        const JoinedRow* only_right = find_row(jf, {2, 999});
        REQUIRE(only_right != nullptr);
        CHECK_FALSE(only_right->left_present);
        CHECK(only_right->right_present);
    }

    TEST_CASE("rows are sorted by key and two build orders match") {
        MapAccum r = build_right();

        MapAccum fwd = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(fwd, {1}, 10);
        add_counter(fwd, {2}, 25);
        add_counter(fwd, {3}, 30);

        MapAccum rev = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(rev, {3}, 30);
        add_counter(rev, {2}, 25);
        add_counter(rev, {1}, 10);

        JoinedMap a = join_maps(fwd, r, JoinType::FULL);
        JoinedMap b = join_maps(rev, r, JoinType::FULL);
        REQUIRE(a.rows.size() == b.rows.size());
        // Emitted in ascending key order regardless of insertion order.
        for (std::size_t i = 0; i + 1 < a.rows.size(); ++i)
            CHECK(a.rows[i].key < a.rows[i + 1].key);
        for (std::size_t i = 0; i < a.rows.size(); ++i) {
            CHECK(a.rows[i].key == b.rows[i].key);
            CHECK(a.rows[i].left_present == b.rows[i].left_present);
            CHECK(a.rows[i].right_present == b.rows[i].right_present);
        }
    }

    TEST_CASE("collection value monoid survives the join opaquely") {
        // left value is a SET_STR (interned ids); the join never reads it, so
        // the carried MonoidAccumulator's element set must be intact on the
        // joined row.
        MapAccum l = make_map("files", {DFTU_T_I64}, {DFTU_MONOID_SET_STR});
        add_set(l, {1}, 100);
        add_set(l, {1}, 101);
        add_set(l, {1}, 100);  // dup, set keeps {100,101}
        add_set(l, {2}, 200);
        MapAccum r = make_map("cnt", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(r, {1}, 7);

        JoinedMap j = join_maps(l, r, JoinType::LEFT);
        REQUIRE(j.valid);
        const JoinedRow* r1 = find_row(j, {1});
        REQUIRE(r1 != nullptr);
        CHECK(r1->left_present);
        CHECK(r1->right_present);
        std::vector<std::int64_t> elems = r1->left_values[0].sorted_elements();
        CHECK(elems == std::vector<std::int64_t>{100, 101});
        CHECK(counter_of(r1->right_values[0]) == 7);

        const JoinedRow* r2 = find_row(j, {2});
        REQUIRE(r2 != nullptr);
        CHECK(r2->left_present);
        CHECK_FALSE(r2->right_present);
        CHECK(r2->left_values[0].sorted_elements() ==
              std::vector<std::int64_t>{200});
    }

    TEST_CASE("mismatched key schema yields an invalid result") {
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(l, {1}, 1);
        MapAccum r =
            make_map("r", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        add_sum(r, {1, 2}, 1.0);
        JoinedMap j = join_maps(l, r, JoinType::INNER);
        CHECK_FALSE(j.valid);
        CHECK(j.rows.empty());
    }

    TEST_CASE("un-reloaded spilled runs make the join invalid, not partial") {
        MapAccum l = build_left();
        MapAccum r = build_right();
        REQUIRE(join_maps(l, r, JoinType::INNER).valid);
        // A live spilled run left in place (not reloaded into parts) must not
        // silently drop that data: the join refuses rather than emit a partial.
        l.runs.push_back({"pending-run"});
        JoinedMap j = join_maps(l, r, JoinType::INNER);
        CHECK_FALSE(j.valid);
        CHECK(j.rows.empty());
        MapAccum l2 = build_left();
        MapAccum r2 = build_right();
        r2.runs.push_back({"pending-run"});
        CHECK_FALSE(join_maps(l2, r2, JoinType::INNER).valid);
    }
}

TEST_SUITE("MapRegroup") {
    TEST_CASE("rollup to a key subset sums the collapsed monoids") {
        MapAccum m =
            make_map("pf", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(m, {1, 100}, 3);
        add_counter(m, {1, 200}, 4);  // same pid, different file
        add_counter(m, {2, 100}, 5);
        add_counter(m, {2, 100}, 1);  // (2,100) -> 6

        std::uint32_t keep[] = {0};
        MapAccum g = regroup_map(m, keep, 1);
        REQUIRE(g.key_n == 1);
        REQUIRE(g.key_types.size() == 1);
        CHECK(g.key_types[0] == DFTU_T_I64);
        CHECK(g.value_kinds ==
              std::vector<dftu_monoid_kind>{DFTU_MONOID_COUNTER});
        CHECK(g.total_entries() == 2);

        const std::vector<MonoidAccumulator>* p1 = find_entry(g, {1});
        REQUIRE(p1 != nullptr);
        CHECK(counter_of((*p1)[0]) == 7);  // 3 + 4 summed over files
        const std::vector<MonoidAccumulator>* p2 = find_entry(g, {2});
        REQUIRE(p2 != nullptr);
        CHECK(counter_of((*p2)[0]) == 6);
    }

    TEST_CASE("keep-all permutation preserves values, reorders the key") {
        MapAccum m =
            make_map("ab", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(m, {1, 100}, 3);
        add_counter(m, {2, 200}, 5);

        std::uint32_t keep[] = {1, 0};  // swap the two components
        MapAccum g = regroup_map(m, keep, 2);
        REQUIRE(g.key_n == 2);
        CHECK(g.total_entries() == 2);
        const std::vector<MonoidAccumulator>* a = find_entry(g, {100, 1});
        REQUIRE(a != nullptr);
        CHECK(counter_of((*a)[0]) == 3);
        const std::vector<MonoidAccumulator>* b = find_entry(g, {200, 2});
        REQUIRE(b != nullptr);
        CHECK(counter_of((*b)[0]) == 5);
    }

    TEST_CASE("collection value unions on collapse (opaque monoid merge)") {
        MapAccum m =
            make_map("sets", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_SET_STR});
        add_set(m, {1, 100}, 10);
        add_set(m, {1, 100}, 11);
        add_set(m, {1, 200}, 11);  // overlaps id 11 across files
        add_set(m, {1, 200}, 12);

        std::uint32_t keep[] = {0};
        MapAccum g = regroup_map(m, keep, 1);
        REQUIRE(g.total_entries() == 1);
        const std::vector<MonoidAccumulator>* p1 = find_entry(g, {1});
        REQUIRE(p1 != nullptr);
        CHECK((*p1)[0].sorted_elements() ==
              std::vector<std::int64_t>{10, 11, 12});
    }

    TEST_CASE("out-of-range keep index yields an empty map") {
        MapAccum m = make_map("m", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(m, {1}, 1);
        std::uint32_t keep[] = {5};
        MapAccum g = regroup_map(m, keep, 1);
        CHECK(g.key_n == 0);
        CHECK(g.total_entries() == 0);
    }

    TEST_CASE(
        "un-reloaded spilled runs yield an empty regroup, not a partial") {
        MapAccum m =
            make_map("pf", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(m, {1, 100}, 3);
        std::uint32_t keep[] = {0};
        REQUIRE(regroup_map(m, keep, 1).key_n == 1);
        m.runs.push_back({"pending-run"});
        MapAccum g = regroup_map(m, keep, 1);
        CHECK(g.key_n == 0);
        CHECK(g.total_entries() == 0);
    }
}

TEST_SUITE("FkJoin") {
    // left keyed on (pid, file) COUNTER; right keyed on (pid) SUM_F64. The FK
    // join regroups left to pid (counts summed over files) then equi joins.
    static MapAccum fk_left() {
        MapAccum m =
            make_map("pf", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(m, {1, 100}, 3);
        add_counter(m, {1, 200}, 4);  // pid 1 -> 7
        add_counter(m, {2, 100}, 5);
        add_counter(m, {2, 300}, 1);  // pid 2 -> 6
        add_counter(m, {9, 100}, 2);  // pid 9 left-only
        return m;
    }

    static MapAccum fk_right() {
        MapAccum m = make_map("p", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        add_sum(m, {1}, 1.5);
        add_sum(m, {2}, 2.5);
        add_sum(m, {4}, 4.0);  // pid 4 right-only
        return m;
    }

    TEST_CASE("INNER matches pids on the regrouped left") {
        std::uint32_t lcols[] = {0};
        std::uint32_t rcols[] = {0};
        JoinedMap j =
            fk_join(fk_left(), lcols, fk_right(), rcols, 1, JoinType::INNER);
        REQUIRE(j.valid);
        CHECK(j.key_n == 1);
        CHECK(keys_of(j) == std::set<std::int64_t>{1, 2});
        const JoinedRow* r1 = find_row(j, {1});
        REQUIRE(r1 != nullptr);
        CHECK(counter_of(r1->left_values[0]) == 7);  // summed over files
        CHECK(sum_of(r1->right_values[0]) == doctest::Approx(1.5));
        const JoinedRow* r2 = find_row(j, {2});
        REQUIRE(r2 != nullptr);
        CHECK(counter_of(r2->left_values[0]) == 6);
        CHECK(sum_of(r2->right_values[0]) == doctest::Approx(2.5));
    }

    TEST_CASE("LEFT null-pads the regrouped left-only pid") {
        std::uint32_t lcols[] = {0};
        std::uint32_t rcols[] = {0};
        JoinedMap j =
            fk_join(fk_left(), lcols, fk_right(), rcols, 1, JoinType::LEFT);
        REQUIRE(j.valid);
        CHECK(keys_of(j) == std::set<std::int64_t>{1, 2, 9});
        const JoinedRow* r9 = find_row(j, {9});
        REQUIRE(r9 != nullptr);
        CHECK(r9->left_present);
        CHECK(counter_of(r9->left_values[0]) == 2);
        CHECK_FALSE(r9->right_present);
        CHECK(r9->right_values.empty());
    }

    TEST_CASE("no regroup when join key is a side's full key in order") {
        // right's join key {0} already is its full key, so it passes through;
        // left still regroups. Result matches the regrouped-both path.
        std::uint32_t lcols[] = {0};
        std::uint32_t rcols[] = {0};
        JoinedMap j =
            fk_join(fk_left(), lcols, fk_right(), rcols, 1, JoinType::INNER);
        REQUIRE(j.valid);
        const JoinedRow* r1 = find_row(j, {1});
        REQUIRE(r1 != nullptr);
        CHECK(counter_of(r1->left_values[0]) == 7);
    }

    TEST_CASE("both sides full key behaves as a plain equi join") {
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(l, {1}, 10);
        add_counter(l, {2}, 20);
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        add_sum(r, {2}, 2.0);
        std::uint32_t cols[] = {0};
        JoinedMap fk = fk_join(l, cols, r, cols, 1, JoinType::INNER);
        JoinedMap plain = join_maps(l, r, JoinType::INNER);
        REQUIRE(fk.valid);
        REQUIRE(fk.rows.size() == plain.rows.size());
        CHECK(keys_of(fk) == keys_of(plain));
    }

    TEST_CASE("mismatched join-key types are invalid") {
        MapAccum l =
            make_map("l", {DFTU_T_I64, DFTU_T_STR}, {DFTU_MONOID_COUNTER});
        add_counter(l, {1, 100}, 1);
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        add_sum(r, {1}, 1.0);
        std::uint32_t lcols[] = {1};  // STR
        std::uint32_t rcols[] = {0};  // I64
        JoinedMap j = fk_join(l, lcols, r, rcols, 1, JoinType::INNER);
        CHECK_FALSE(j.valid);
    }

    TEST_CASE("out-of-range join column is invalid") {
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(l, {1}, 1);
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        add_sum(r, {1}, 1.0);
        std::uint32_t lcols[] = {3};
        std::uint32_t rcols[] = {0};
        JoinedMap j = fk_join(l, lcols, r, rcols, 1, JoinType::INNER);
        CHECK_FALSE(j.valid);
    }

    TEST_CASE("determinism across left build orders") {
        MapAccum fwd =
            make_map("pf", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(fwd, {1, 100}, 3);
        add_counter(fwd, {1, 200}, 4);
        add_counter(fwd, {2, 100}, 6);
        MapAccum rev =
            make_map("pf", {DFTU_T_I64, DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        add_counter(rev, {2, 100}, 6);
        add_counter(rev, {1, 200}, 4);
        add_counter(rev, {1, 100}, 3);

        std::uint32_t lcols[] = {0};
        std::uint32_t rcols[] = {0};
        JoinedMap a = fk_join(fwd, lcols, fk_right(), rcols, 1, JoinType::FULL);
        JoinedMap b = fk_join(rev, lcols, fk_right(), rcols, 1, JoinType::FULL);
        REQUIRE(a.rows.size() == b.rows.size());
        for (std::size_t i = 0; i < a.rows.size(); ++i) {
            CHECK(a.rows[i].key == b.rows[i].key);
            CHECK(a.rows[i].left_present == b.rows[i].left_present);
            CHECK(a.rows[i].right_present == b.rows[i].right_present);
        }
    }

    TEST_CASE("un-reloaded spilled runs make the fk join invalid") {
        std::uint32_t lcols[] = {0};
        std::uint32_t rcols[] = {0};
        MapAccum l = fk_left();
        // Without the entry guard the spilled side would regroup to a valid
        // empty map and leak a silently partial join.
        l.runs.push_back({"pending-run"});
        JoinedMap j = fk_join(l, lcols, fk_right(), rcols, 1, JoinType::INNER);
        CHECK_FALSE(j.valid);
    }
}
