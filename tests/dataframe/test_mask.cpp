#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/mask.h>
#include <dftracer/utils/dataframe/plan.h>
#include <dftracer/utils/query/abi.h>
#include <dftracer/utils/query/errc.h>
#include <dftracer/utils/query/query.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <vector>

using dftracer::utils::Condition;
using dftracer::utils::DFTUtilsException;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::Series;
namespace query = dftracer::utils::query;
namespace dataframe = dftracer::utils::dataframe;

namespace {
int bit_at(const Series& mask, int i) {
    const std::uint8_t* bits = mask.data<std::uint8_t>();
    return (bits[i >> 3] >> (i & 7)) & 1;
}
}  // namespace

TEST_SUITE("query vec_mask") {
    TEST_CASE("numeric + string predicate lowers to a SIMD mask") {
        std::vector<std::int64_t> dur{50, 150, 200, 80, 300};
        DataFrame b;
        b.names = {"dur", "cat"};
        b.columns.push_back(Series::flat_i64(dur.data(), 5));
        b.columns.push_back(Series::strings({"io", "cpu", "io", "io", "net"}));

        auto q = query::Query::from_string("dur > 100 and cat == \"io\"");
        REQUIRE(q.has_value());
        Series mask = dataframe::evaluate_mask(q->root(), b);
        REQUIRE(mask.type() == dftracer::utils::dataframe::TypeId::Bool);
        REQUIRE(mask.length() == 5);
        // dur>100: {1,2,4}; cat==io: {0,2,3}; AND -> only row 2.
        CHECK(bit_at(mask, 0) == 0);
        CHECK(bit_at(mask, 1) == 0);
        CHECK(bit_at(mask, 2) == 1);
        CHECK(bit_at(mask, 3) == 0);
        CHECK(bit_at(mask, 4) == 0);
    }

    TEST_CASE("in-list lowers to OR-of-equals") {
        std::vector<std::int64_t> x{1, 2, 3, 4};
        DataFrame b;
        b.names = {"x"};
        b.columns.push_back(Series::flat_i64(x.data(), 4));

        auto q = query::Query::from_string("x in [2, 4]");
        REQUIRE(q.has_value());
        Series mask = dataframe::evaluate_mask(q->root(), b);
        CHECK(bit_at(mask, 0) == 0);
        CHECK(bit_at(mask, 1) == 1);
        CHECK(bit_at(mask, 2) == 0);
        CHECK(bit_at(mask, 3) == 1);
    }

    TEST_CASE("unsupported predicate throws QueryErrc::Unsupported") {
        std::vector<std::int64_t> v{1};
        DataFrame b;
        b.names = {"name"};
        b.columns.push_back(Series::flat_i64(v.data(), 1));

        auto q = query::Query::from_string("name ~ \"^p\"");  // regex
        REQUIRE(q.has_value());
        bool threw = false;
        try {
            dataframe::evaluate_mask(q->root(), b);
        } catch (const DFTUtilsException& e) {
            threw = true;
            CHECK(e.condition() == Condition::Unsupported);
            CHECK(e.domain() == query::ERROR_DOMAIN.id);
        }
        CHECK(threw);
    }

    TEST_CASE("query plan: where + select + order by + limit") {
        std::vector<std::int64_t> dur{50, 150, 200, 80, 300};
        DataFrame b;
        b.names = {"dur", "cat"};
        b.columns.push_back(Series::flat_i64(dur.data(), 5));
        b.columns.push_back(Series::strings({"io", "cpu", "io", "io", "net"}));

        auto q = query::Query::from_string("dur > 60");
        REQUIRE(q.has_value());
        dataframe::QueryPlan plan;
        plan.where = &q->root();
        plan.select = {"dur", "cat"};
        plan.order_by = "dur";
        plan.descending = true;
        plan.limit = 2;
        DataFrame out = dataframe::execute(plan, b);
        // dur>60: {150,200,80,300}; sort desc -> 300,200,150,80; limit 2.
        REQUIRE(out.num_rows() == 2);
        REQUIRE(out.names == std::vector<std::string>{"dur", "cat"});
        CHECK(out.columns[0].data<std::int64_t>()[0] == 300);
        CHECK(out.columns[0].data<std::int64_t>()[1] == 200);
        CHECK(out.columns[1].string_at(0) == "net");
    }

    TEST_CASE("query plan: where + group by/agg + order by + limit") {
        std::vector<std::int64_t> dur{10, 5, 20, 15, 30};
        DataFrame b;
        b.names = {"cat", "dur"};
        b.columns.push_back(Series::strings({"io", "cpu", "io", "io", "net"}));
        b.columns.push_back(Series::flat_i64(dur.data(), 5));

        dataframe::QueryPlan plan;
        plan.group_by = "cat";
        plan.aggs = {{dataframe::Agg::Count, "", "count"},
                     {dataframe::Agg::Sum, "dur", "sum_dur"}};
        plan.order_by = "count";
        plan.descending = true;
        DataFrame out = dataframe::execute(plan, b);
        // groups io(3),cpu(1),net(1); order by count desc -> io first.
        REQUIRE(out.num_rows() == 3);
        REQUIRE(out.names ==
                std::vector<std::string>{"cat", "count", "sum_dur"});
        CHECK(out.columns[0].string_at(0) == "io");
        CHECK(out.columns[1].data<std::int64_t>()[0] == 3);
        CHECK(out.columns[2].data<std::int64_t>()[0] == 45);  // io dur 10+20+15
    }

    TEST_CASE("C ABI: parse / to_string / mask round-trip") {
        dftu_query* q = dftu_query_parse("dur >= 100 and cat == \"io\"");
        REQUIRE(q != nullptr);

        char* s = dftu_query_to_string(q);
        REQUIRE(s != nullptr);
        CHECK(std::strstr(s, "dur") != nullptr);
        CHECK(std::strstr(s, "io") != nullptr);
        dftu_query_string_free(s);

        std::vector<std::int64_t> dur{50, 150, 200};
        Series cdur = Series::flat_i64(dur.data(), 3);
        Series ccat = Series::strings({"io", "io", "net"});
        const dftu_series* cols[2] = {cdur.handle(), ccat.handle()};
        const char* names[2] = {"dur", "cat"};
        dftu_series* m = dftu_dataframe_mask(q, cols, names, 2);
        REQUIRE(m != nullptr);
        Series mask{m};
        // dur>=100: {1,2}; cat==io: {0,1}; AND -> row 1.
        CHECK(bit_at(mask, 0) == 0);
        CHECK(bit_at(mask, 1) == 1);
        CHECK(bit_at(mask, 2) == 0);

        // Unsupported predicate -> NULL (caller falls back to pushdown).
        dftu_query* rq = dftu_query_parse("cat ~ \"^i\"");
        REQUIRE(rq != nullptr);
        CHECK(dftu_dataframe_mask(rq, cols, names, 2) == nullptr);
        dftu_query_free(rq);
        dftu_query_free(q);
    }
}
