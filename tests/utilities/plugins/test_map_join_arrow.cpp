// Arrow EXPORT half of the native map-to-map join: materialize_joined turns a
// JoinedMap into one Arrow record batch [key cols, LEFT value cols (l_), RIGHT
// value cols (r_)]. These cases only build under DFTRACER_UTILS_ENABLE_ARROW;
// the join itself is proved Arrow-free in test_map_join.cpp. Read the batch
// back through the Arrow C data buffers and assert values, outer-side null
// padding (via the validity bitmap), collection and arg-row column expansion,
// and determinism across two build orders.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <doctest/doctest.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/map_join.h>
#include <dftracer/utils/plugins/map_join_arrow.h>
#include <dftracer/utils/plugins/monoid.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

using dftracer::utils::StringIntern;
using dftracer::utils::plugins::join_maps;
using dftracer::utils::plugins::JoinedMap;
using dftracer::utils::plugins::JoinType;
using dftracer::utils::plugins::MapAccum;
using dftracer::utils::plugins::materialize_joined;
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

const double* f64_col(const ArrowArray* a, int c) {
    const ArrowArray* ch = a->children[c];
    return static_cast<const double*>(ch->buffers[1]) + ch->offset;
}

bool cell_is_null(const ArrowArray* a, int c, std::int64_t r) {
    const ArrowArray* ch = a->children[c];
    const auto* v = static_cast<const std::uint8_t*>(ch->buffers[0]);
    if (!v) return false;
    const std::int64_t idx = ch->offset + r;
    return ((v[idx / 8] >> (idx % 8)) & 1) == 0;
}

// The utf8 elements of one STRING_LIST row (a list<utf8> column).
std::vector<std::string> string_list_at(const ArrowArray* a, int c,
                                        std::int64_t r) {
    const ArrowArray* listc = a->children[c];
    const auto* loff =
        static_cast<const std::int32_t*>(listc->buffers[1]) + listc->offset;
    const ArrowArray* strc = listc->children[0];
    const auto* soff = static_cast<const std::int32_t*>(strc->buffers[1]);
    const char* chars = static_cast<const char*>(strc->buffers[2]);
    std::vector<std::string> out;
    for (std::int32_t e = loff[r]; e < loff[r + 1]; ++e)
        out.emplace_back(chars + soff[e], chars + soff[e + 1]);
    return out;
}

}  // namespace

TEST_SUITE("MapJoinArrow") {
    TEST_CASE("INNER: key + both scalar value columns carry matched values") {
        StringIntern intern;
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        l.touch({2})[0].add_u64(20);
        l.touch({2})[0].add_u64(5);  // -> 25
        l.touch({3})[0].add_u64(30);
        l.touch({1})[0].add_u64(9);  // left-only, dropped by INNER
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        r.touch({2})[0].add_f64(2.5, 1.0);
        r.touch({3})[0].add_f64(3.5, 1.0);
        r.touch({4})[0].add_f64(4.0, 1.0);  // right-only, dropped by INNER

        JoinedMap j = join_maps(l, r, JoinType::INNER);
        arr::ArrowExportResult res = materialize_joined(j, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 2);
        REQUIRE(a->n_children == 3);  // k0, l_value, r_value
        // Schema column names / order.
        const ArrowSchema* s = res.get_schema();
        CHECK(std::string_view(s->children[0]->name) == "k0");
        CHECK(std::string_view(s->children[1]->name) == "l_value");
        CHECK(std::string_view(s->children[2]->name) == "r_value");

        const std::int64_t* k0 = i64_col(a, 0);
        const std::int64_t* lv = i64_col(a, 1);
        const double* rv = f64_col(a, 2);
        // Rows sorted by key: (2) then (3).
        CHECK(k0[0] == 2);
        CHECK(lv[0] == 25);
        CHECK(rv[0] == doctest::Approx(2.5));
        CHECK(k0[1] == 3);
        CHECK(lv[1] == 30);
        CHECK(rv[1] == doctest::Approx(3.5));
        for (int c = 0; c < 3; ++c) {
            CHECK_FALSE(cell_is_null(a, c, 0));
            CHECK_FALSE(cell_is_null(a, c, 1));
        }
    }

    TEST_CASE("FULL: absent side's value columns are null via validity bits") {
        StringIntern intern;
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        l.touch({1})[0].add_u64(10);        // left-only
        l.touch({2})[0].add_u64(20);        // both
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        r.touch({2})[0].add_f64(2.5, 1.0);  // both
        r.touch({4})[0].add_f64(4.0, 1.0);  // right-only

        JoinedMap j = join_maps(l, r, JoinType::FULL);
        arr::ArrowExportResult res = materialize_joined(j, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 3);  // keys 1, 2, 4

        const std::int64_t* k0 = i64_col(a, 0);
        CHECK(k0[0] == 1);
        CHECK(k0[1] == 2);
        CHECK(k0[2] == 4);
        // key 1: left present, right null.
        CHECK_FALSE(cell_is_null(a, 1, 0));
        CHECK(i64_col(a, 1)[0] == 10);
        CHECK(cell_is_null(a, 2, 0));
        // key 2: both present.
        CHECK_FALSE(cell_is_null(a, 1, 1));
        CHECK_FALSE(cell_is_null(a, 2, 1));
        // key 4: left null, right present.
        CHECK(cell_is_null(a, 1, 2));
        CHECK_FALSE(cell_is_null(a, 2, 2));
        CHECK(f64_col(a, 2)[2] == doctest::Approx(4.0));
    }

    TEST_CASE("LEFT_SEMI/LEFT_ANTI materialize a left-only batch") {
        StringIntern intern;
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        l.touch({1})[0].add_u64(10);  // left-only
        l.touch({2})[0].add_u64(20);  // matched
        l.touch({3})[0].add_u64(30);  // matched
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        r.touch({2})[0].add_f64(2.5, 1.0);
        r.touch({3})[0].add_f64(3.5, 1.0);
        r.touch({4})[0].add_f64(4.0, 1.0);  // right-only

        SUBCASE("semi: key + left value only, matched keys") {
            JoinedMap j = join_maps(l, r, JoinType::LEFT_SEMI);
            arr::ArrowExportResult res = materialize_joined(j, intern);
            REQUIRE(res.valid());
            const ArrowArray* a = res.get_array();
            REQUIRE(a->n_children == 2);  // k0, l_value; no r_value
            const ArrowSchema* s = res.get_schema();
            CHECK(std::string_view(s->children[0]->name) == "k0");
            CHECK(std::string_view(s->children[1]->name) == "l_value");
            REQUIRE(a->length == 2);  // keys 2, 3
            CHECK(i64_col(a, 0)[0] == 2);
            CHECK(i64_col(a, 1)[0] == 20);
            CHECK(i64_col(a, 0)[1] == 3);
            CHECK(i64_col(a, 1)[1] == 30);
        }
        SUBCASE("anti: key + left value only, unmatched keys") {
            JoinedMap j = join_maps(l, r, JoinType::LEFT_ANTI);
            arr::ArrowExportResult res = materialize_joined(j, intern);
            REQUIRE(res.valid());
            const ArrowArray* a = res.get_array();
            REQUIRE(a->n_children == 2);
            REQUIRE(a->length == 1);  // key 1
            CHECK(i64_col(a, 0)[0] == 1);
            CHECK(i64_col(a, 1)[0] == 10);
        }
    }

    TEST_CASE(
        "collection value (SET_STR) survives on both sides as list<utf8>") {
        StringIntern intern;
        const dftu_str foo = intern.get_or_insert("foo");
        const dftu_str bar = intern.get_or_insert("bar");
        const dftu_str baz = intern.get_or_insert("baz");

        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_SET_STR});
        l.touch({1})[0].add_u64(foo);
        l.touch({1})[0].add_u64(bar);
        l.touch({1})[0].add_u64(foo);  // dup
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SET_STR});
        r.touch({1})[0].add_u64(baz);

        JoinedMap j = join_maps(l, r, JoinType::INNER);
        arr::ArrowExportResult res = materialize_joined(j, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 1);
        REQUIRE(a->n_children == 3);
        const ArrowSchema* s = res.get_schema();
        // Both list<utf8> value columns present and named distinctly.
        CHECK(std::string_view(s->children[1]->name) == "l_value");
        CHECK(std::string_view(s->children[2]->name) == "r_value");
        CHECK(std::string_view(s->children[1]->children[0]->format) == "u");

        std::vector<std::string> left = string_list_at(a, 1, 0);
        std::vector<std::string> right = string_list_at(a, 2, 0);
        CHECK(left == std::vector<std::string>{"bar", "foo"});  // sorted label
        CHECK(right == std::vector<std::string>{"baz"});
    }

    TEST_CASE("arg-row value side expands to its payload columns") {
        StringIntern intern;
        // left = pid -> ARGMAX_ROW over {rank:i64, size:i64} payload.
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_ARGMAX_ROW});
        l.payload_types = {DFTU_T_I64, DFTU_T_I64};
        {
            std::vector<std::int64_t> p = {7, 100};
            l.touch({1})[0].add_argrow(1.0, p.data(), 2);
        }
        {
            std::vector<std::int64_t> p = {9, 200};  // larger by -> kept
            l.touch({1})[0].add_argrow(5.0, p.data(), 2);
        }
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        r.touch({1})[0].add_u64(3);

        JoinedMap j = join_maps(l, r, JoinType::INNER);
        arr::ArrowExportResult res = materialize_joined(j, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 1);
        REQUIRE(a->n_children == 4);  // k0, l_p0, l_p1, r_value
        const ArrowSchema* s = res.get_schema();
        CHECK(std::string_view(s->children[1]->name) == "l_p0");
        CHECK(std::string_view(s->children[2]->name) == "l_p1");
        CHECK(std::string_view(s->children[3]->name) == "r_value");
        CHECK(i64_col(a, 1)[0] == 9);
        CHECK(i64_col(a, 2)[0] == 200);
        CHECK(i64_col(a, 3)[0] == 3);
    }

    TEST_CASE(
        "arg-row null-pads on an outer row where the arg-row side absent") {
        StringIntern intern;
        MapAccum l = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_ARGMAX_ROW});
        l.payload_types = {DFTU_T_I64, DFTU_T_I64};
        {
            std::vector<std::int64_t> p = {9, 200};
            l.touch({1})[0].add_argrow(5.0, p.data(), 2);
        }
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        r.touch({2})[0].add_u64(3);  // right-only key

        JoinedMap j = join_maps(l, r, JoinType::FULL);
        arr::ArrowExportResult res = materialize_joined(j, intern);
        REQUIRE(res.valid());
        const ArrowArray* a = res.get_array();
        REQUIRE(a->length == 2);  // keys 1, 2
        REQUIRE(a->n_children == 4);
        // key 2 (right-only): both arg-row payload columns null, right present.
        CHECK(i64_col(a, 0)[1] == 2);
        CHECK(cell_is_null(a, 1, 1));
        CHECK(cell_is_null(a, 2, 1));
        CHECK_FALSE(cell_is_null(a, 3, 1));
        CHECK(i64_col(a, 3)[1] == 3);
    }

    TEST_CASE("materialize_joined is deterministic across build orders") {
        StringIntern intern;
        MapAccum r = make_map("r", {DFTU_T_I64}, {DFTU_MONOID_SUM_F64});
        r.touch({2})[0].add_f64(2.5, 1.0);
        r.touch({3})[0].add_f64(3.5, 1.0);
        r.touch({4})[0].add_f64(4.0, 1.0);

        MapAccum fwd = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        fwd.touch({1})[0].add_u64(10);
        fwd.touch({2})[0].add_u64(25);
        fwd.touch({3})[0].add_u64(30);
        MapAccum rev = make_map("l", {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        rev.touch({3})[0].add_u64(30);
        rev.touch({2})[0].add_u64(25);
        rev.touch({1})[0].add_u64(10);

        arr::ArrowExportResult ra =
            materialize_joined(join_maps(fwd, r, JoinType::FULL), intern);
        arr::ArrowExportResult rb =
            materialize_joined(join_maps(rev, r, JoinType::FULL), intern);
        const ArrowArray* a = ra.get_array();
        const ArrowArray* b = rb.get_array();
        REQUIRE(a->length == b->length);
        REQUIRE(a->length == 4);  // keys 1,2,3,4
        for (std::int64_t i = 0; i < a->length; ++i) {
            CHECK(i64_col(a, 0)[i] == i64_col(b, 0)[i]);
            CHECK(cell_is_null(a, 1, i) == cell_is_null(b, 1, i));
            CHECK(cell_is_null(a, 2, i) == cell_is_null(b, 2, i));
            if (!cell_is_null(a, 1, i))
                CHECK(i64_col(a, 1)[i] == i64_col(b, 1)[i]);
            if (!cell_is_null(a, 2, i))
                CHECK(f64_col(a, 2)[i] == doctest::Approx(f64_col(b, 2)[i]));
        }
    }
}

#else
TEST_CASE("map join arrow export requires DFTRACER_UTILS_ENABLE_ARROW") {}
#endif  // DFTRACER_UTILS_ENABLE_ARROW
