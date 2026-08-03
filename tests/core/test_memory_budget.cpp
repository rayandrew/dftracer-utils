#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/memory_budget.h>
#include <doctest/doctest.h>

using dftracer::utils::format_memory_budget_warning;
using dftracer::utils::memory_budget_advice;
using dftracer::utils::PEAK_MEMORY_FACTOR;

TEST_SUITE("memory_budget_advice") {
    TEST_CASE("fits when peak is within available memory") {
        // 1 GB aggregated, 8 GB available: peak = 3 GB <= 8 GB.
        auto a = memory_budget_advice(1ULL << 30, 8ULL << 30);
        CHECK(a.fits);
        CHECK(a.peak_bytes == (3ULL << 30));
        CHECK(a.suggested_nodes == 1);
        CHECK(format_memory_budget_warning(a).empty());
    }

    TEST_CASE("does not fit when peak exceeds available, and advises nodes") {
        // 4 GB aggregated, 6 GB available: peak = 12 GB > 6 GB.
        auto a = memory_budget_advice(4ULL << 30, 6ULL << 30);
        CHECK_FALSE(a.fits);
        CHECK(a.peak_bytes == (12ULL << 30));
        // ceil(12 / 6) = 2 nodes.
        CHECK(a.suggested_nodes == 2);
        const std::string msg = format_memory_budget_warning(a);
        CHECK(msg.find("12.0 GB") != std::string::npos);
        CHECK(msg.find(">=2 nodes") != std::string::npos);
    }

    TEST_CASE("suggested_nodes rounds up") {
        // peak = 3 * 5 = 15 GB, available 4 GB: ceil(15/4) = 4.
        auto a = memory_budget_advice(5ULL << 30, 4ULL << 30);
        CHECK(a.suggested_nodes == 4);
    }

    TEST_CASE("peak is exactly PEAK_MEMORY_FACTOR times required") {
        auto a = memory_budget_advice(100, 1ULL << 30);
        CHECK(a.peak_bytes == 100 * PEAK_MEMORY_FACTOR);
    }
}
