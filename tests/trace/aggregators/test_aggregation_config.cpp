#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/trace/aggregators/aggregation_config.h>
#include <doctest/doctest.h>

using namespace dftracer::utils::trace::aggregators;

TEST_SUITE("AggregationConfig") {
    TEST_CASE("AggregationConfig - Default values") {
        AggregationConfig config;
        CHECK(config.time_interval_us == 1000000);
        CHECK(config.use_relative_time == false);
        CHECK(config.compute_statistics == true);
        CHECK(config.compute_percentiles == false);
        CHECK(config.output_format == std::string("json"));
    }

    TEST_CASE("AggregationConfig - Valid formats") {
        CHECK(AggregationConfig::is_valid_format("json"));
        CHECK_FALSE(AggregationConfig::is_valid_format("csv"));
    }

    TEST_CASE("AggregationConfig - group_by_file changes the hash") {
        AggregationConfig with_files;
        AggregationConfig without_files;
        without_files.group_by_file = false;

        REQUIRE(with_files.group_by_file);
        CHECK(with_files.compute_hash() != without_files.compute_hash());
    }

    TEST_CASE("AggregationConfig - grain fields change the hash") {
        const AggregationConfig base;

        AggregationConfig coarser = base;
        coarser.time_interval_us = base.time_interval_us * 10;
        CHECK(base.compute_hash() != coarser.compute_hash());

        AggregationConfig extra_keys = base;
        extra_keys.extra_group_keys.push_back("epoch");
        CHECK(base.compute_hash() != extra_keys.compute_hash());

        AggregationConfig pct = base;
        pct.compute_percentiles = !base.compute_percentiles;
        CHECK(base.compute_hash() != pct.compute_hash());

        AggregationConfig same = base;
        CHECK(base.compute_hash() == same.compute_hash());
    }
}
