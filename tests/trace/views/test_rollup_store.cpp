#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/core/rocksdb/db_manager.h>
#include <dftracer/utils/trace/views/rollup_store.h>
#include <dftracer/utils/trace/views/view_spill.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <string>

using namespace dftracer::utils::trace::views::detail;
namespace codec = dftracer::utils::utilities::common::serialization;

namespace {

AggAccum make_accum(std::uint64_t count, const std::vector<double>& values) {
    AggAccum a;
    a.count = count;
    a.keys = {"grp"};
    FieldStat f;
    for (double v : values) f.add(v);
    a.fields = {f};
    return a;
}

// Round-trip an accum through the rollup value encoding - the exact bytes the
// merge operator deserializes and reserializes.
AggAccum roundtrip(const AggAccum& a) {
    std::string bytes;
    serialize_accum(bytes, "grp", a);
    codec::BinaryReader br(bytes);
    std::string key;
    AggAccum out;
    deserialize_accum(br, key, out);
    return out;
}

}  // namespace

TEST_SUITE("RollupStore") {
    TEST_CASE("key layout: tag, big-endian signature, group suffix") {
        const std::uint64_t sig = 0x0102030405060708ULL;

        std::string desc = rollup_desc_key(sig);
        REQUIRE(desc.size() == 9);
        CHECK(static_cast<unsigned char>(desc[0]) == 0x00);
        CHECK(static_cast<unsigned char>(desc[1]) == 0x01);  // big-endian
        CHECK(static_cast<unsigned char>(desc[8]) == 0x08);

        std::string row = rollup_row_key(sig, "abc");
        REQUIRE(row.size() == 12);
        CHECK(static_cast<unsigned char>(row[0]) == 0x01);
        CHECK(row.substr(9) == "abc");
        CHECK(row.substr(1, 8) == desc.substr(1, 8));  // same signature bytes
    }

    TEST_CASE("key ordering groups a view and separates descriptors") {
        const std::uint64_t s1 = 10, s2 = 11;
        // Descriptor sorts before any row of the same view (0x00 < 0x01).
        CHECK(rollup_desc_key(s1) < rollup_row_key(s1, ""));
        // Rows of one view sort by group key.
        CHECK(rollup_row_key(s1, "a") < rollup_row_key(s1, "b"));
        // Big-endian signature gives numeric ordering across views.
        CHECK(rollup_row_key(s1, "z") < rollup_row_key(s2, "a"));
    }

    TEST_CASE("merge_accum_free combines two partials positionally") {
        AggAccum a = make_accum(3, {10, 20, 30});  // n=3 sum=60 min=10 max=30
        AggAccum b = make_accum(5, {5, 15, 25, 35, 45});  // n=5 sum=125

        merge_accum_free(a, b);
        CHECK(a.count == 8);
        REQUIRE(a.fields.size() == 1);
        CHECK(a.fields[0].n == 8);
        CHECK(a.fields[0].sum == doctest::Approx(185));
        CHECK(a.fields[0].min == doctest::Approx(5));
        CHECK(a.fields[0].max == doctest::Approx(45));
    }

    TEST_CASE("merge into an empty accum adopts the source") {
        AggAccum empty;
        AggAccum b = make_accum(5, {5, 15, 25, 35, 45});
        merge_accum_free(empty, b);
        CHECK(empty.count == 5);
        REQUIRE(empty.fields.size() == 1);
        CHECK(empty.fields[0].sum == doctest::Approx(125));
        CHECK(empty.keys == std::vector<std::string>{"grp"});
    }

    TEST_CASE("merge is commutative (partials arrive in any rank order)") {
        AggAccum ab = make_accum(3, {10, 20, 30});
        AggAccum ba = make_accum(5, {5, 15, 25, 35, 45});
        AggAccum ab_copy = ab, ba_copy = ba;

        merge_accum_free(ab, ba);            // a then b
        merge_accum_free(ba_copy, ab_copy);  // b then a
        CHECK(ab.count == ba_copy.count);
        CHECK(ab.fields[0].sum == doctest::Approx(ba_copy.fields[0].sum));
        CHECK(ab.fields[0].min == doctest::Approx(ba_copy.fields[0].min));
        CHECK(ab.fields[0].max == doctest::Approx(ba_copy.fields[0].max));
    }

    TEST_CASE("serialized partials merge to the same result (operator path)") {
        AggAccum a = make_accum(3, {10, 20, 30});
        AggAccum b = make_accum(5, {5, 15, 25, 35, 45});

        // In memory.
        AggAccum direct = a;
        merge_accum_free(direct, b);

        // Through the on-disk encoding, as the merge operator does.
        AggAccum wire = roundtrip(a);
        merge_accum_free(wire, roundtrip(b));

        CHECK(wire.count == direct.count);
        CHECK(wire.fields[0].sum == doctest::Approx(direct.fields[0].sum));
        CHECK(wire.fields[0].n == direct.fields[0].n);
    }

    // The distributed path: two ranks' partials are reduced APP-SIDE with
    // merge_accum_free, then the merged result is persisted with plain Puts and
    // read back through a real RocksDB.
    TEST_CASE("app-side reduce of rank partials persists and reads back") {
        namespace rdb = dftracer::utils::rocksdb;
        const std::string dir =
            (fs::temp_directory_path() / "dftu_rollup_store_test").string();
        fs::remove_all(dir);
        rdb::RocksDBManager::instance().reset(dir);

        auto db = open_rollup_db(dir, rdb::RocksDatabase::OpenMode::ReadWrite);
        REQUIRE(db);
        const std::uint64_t sig = 0xABCDEF12;

        GroupMap rank_a;
        rank_a["x"] = make_accum(3, {10, 20, 30});
        GroupMap rank_b;
        rank_b["x"] = make_accum(5, {5, 15, 25, 35, 45});
        rank_b["y"] = make_accum(2, {1, 2});

        GroupMap merged = rank_a;
        for (const auto& [k, a] : rank_b) merge_accum_free(merged[k], a);
        persist_rollup(*db, sig, /*rest_sig=*/0, /*time_bucket_us=*/0,
                       /*group_by=*/{}, merged);

        CHECK(rollup_exists(*db, sig));
        GroupMap got = read_rollup(*db, sig);
        REQUIRE(got.count("x") == 1);
        REQUIRE(got.count("y") == 1);
        CHECK(got["x"].count == 8);  // 3 + 5, reduced across ranks
        CHECK(got["x"].fields[0].sum == doctest::Approx(185));
        CHECK(got["y"].count == 2);

        // A different signature is isolated and absent.
        CHECK_FALSE(rollup_exists(*db, 0x99999999));
        CHECK(read_rollup(*db, 0x99999999).empty());

        rdb::RocksDBManager::instance().reset(dir);
        fs::remove_all(dir);
    }
}
