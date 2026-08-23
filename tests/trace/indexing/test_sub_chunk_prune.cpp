#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/query/query.h>
#include <dftracer/utils/trace/indexing/sub_chunk_prune.h>
#include <doctest/doctest.h>

#include <vector>

using namespace dftracer::utils::trace::indexing;
using dftracer::utils::query::Query;

namespace {

SubChunkZoneMap bucket(std::uint64_t min_ts, std::uint64_t max_ts,
                       std::uint64_t min_dur = 0, std::uint64_t max_dur = 0) {
    SubChunkZoneMap z;
    z.event_count = 10;
    z.min_timestamp_us = min_ts;
    z.max_timestamp_us = max_ts;
    z.min_duration_us = min_dur;
    z.max_duration_us = max_dur;
    return z;
}

std::vector<char> mask(const std::vector<SubChunkZoneMap>& buckets,
                       const std::string& q) {
    auto parsed = Query::from_string(q);
    REQUIRE(parsed.has_value());
    return sub_chunk_keep_mask(buckets, parsed->root(), "ts", "dur");
}

}  // namespace

TEST_SUITE("sub_chunk_prune") {
    TEST_CASE("ts lower bound excludes early buckets") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100), bucket(100, 200),
                                          bucket(200, 300)};
        auto keep = mask(b, "ts >= 150");
        REQUIRE(keep.size() == 3);
        CHECK(keep[0] == 0);  // 0..100 all below 150
        CHECK(keep[1] == 1);  // 100..200 overlaps
        CHECK(keep[2] == 1);
    }

    TEST_CASE("ts window keeps only overlapping buckets") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100), bucket(100, 200),
                                          bucket(200, 300), bucket(300, 400)};
        auto keep = mask(b, "ts >= 150 and ts <= 250");
        REQUIRE(keep.size() == 4);
        CHECK(keep[0] == 0);
        CHECK(keep[1] == 1);
        CHECK(keep[2] == 1);
        CHECK(keep[3] == 0);
    }

    TEST_CASE("no range constraint keeps all (empty mask)") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100), bucket(100, 200)};
        CHECK(mask(b, "cat == \"POSIX\"").empty());
    }

    TEST_CASE("range under OR is unsound: no exclusion") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100), bucket(200, 300)};
        // ts under an OR must not drive skipping.
        CHECK(mask(b, "ts >= 500 or cat == \"POSIX\"").empty());
    }

    TEST_CASE("dur range excludes buckets") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100, 0, 5),
                                          bucket(100, 200, 50, 90)};
        auto keep = mask(b, "dur >= 40");
        REQUIRE(keep.size() == 2);
        CHECK(keep[0] == 0);  // dur 0..5 below 40
        CHECK(keep[1] == 1);
    }

    TEST_CASE("ts and dur together") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100, 0, 5),
                                          bucket(100, 200, 50, 90),
                                          bucket(200, 300, 0, 5)};
        auto keep = mask(b, "ts >= 150 and dur >= 40");
        REQUIRE(keep.size() == 3);
        CHECK(keep[0] == 0);  // fails both
        CHECK(keep[1] == 1);  // passes both
        CHECK(keep[2] == 0);  // ts ok but dur 0..5 below 40
    }

    TEST_CASE("all buckets survive -> empty mask") {
        std::vector<SubChunkZoneMap> b = {bucket(0, 100), bucket(100, 200)};
        CHECK(mask(b, "ts >= 0").empty());
    }
}
