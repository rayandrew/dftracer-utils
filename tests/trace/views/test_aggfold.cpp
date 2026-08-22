#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/trace/views/aggfold.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/view_scan.h>
#include <doctest/doctest.h>

#include <array>
#include <map>
#include <string>
#include <vector>

#include "test_view_common.h"

using namespace dftracer::utils::trace::views::detail;
using dftracer::utils::StringIntern;

namespace dataframe = dftracer::utils::dataframe;

namespace {

// {group key -> value columns} from a columnar DataFrame, so two results
// compare regardless of row order. String columns are the group key, numeric
// columns the values. Both collect() and AggFold::result() yield this DataFrame
// form.
std::map<std::string, std::vector<double>> rows_by_key(
    const dftracer::utils::dataframe::DataFrame& b) {
    std::map<std::string, std::vector<double>> out;
    for (std::int64_t i = 0; i < b.num_rows(); ++i) {
        std::string k;
        std::vector<double> vals;
        for (const auto& c : b.columns) {
            if (c.type() == dataframe::TypeId::String) {
                k += std::string(c.string_at(i));
                k += '|';
            } else if (c.type() == dataframe::TypeId::Int64) {
                vals.push_back(static_cast<double>(c.data<std::int64_t>()[i]));
            } else if (c.type() == dataframe::TypeId::Uint64) {
                vals.push_back(static_cast<double>(c.data<std::uint64_t>()[i]));
            } else {
                vals.push_back(c.data<double>()[i]);
            }
        }
        out[k] = std::move(vals);
    }
    return out;
}

// POSIX reads carrying args.ret, so size (derived from ret) exercises the args
// path that the needs_args optimization must not drop.
std::string create_trace_with_ret(TestEnvironment& env, int n) {
    std::string pfw = env.get_dir() + "/ret.pfw";
    std::ofstream ofs(pfw);
    for (int i = 0; i < n; ++i)
        ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":)"
            << (1000 + i) << R"(,"dur":5,"args":{"ret":)" << (100 + i)
            << R"(}})" << "\n";
    ofs.close();
    std::string gz = pfw + ".gz";
    dftu_utils_test::compress_file_to_gzip(pfw, gz);
    fs::remove(pfw);
    return gz;
}

ViewPlan size_plan(const std::string& gz, const std::string& idx) {
    ViewPlan plan;
    ViewFile f;
    f.file_path = gz;
    f.index_path = idx;
    plan.files.push_back(f);
    plan.group_by = {GroupKey::cat()};
    plan.agg = {AggSpec(AggOp::Sum, "size", "sum_size")};
    return plan;
}

ViewPlan agg_plan(const std::string& gz, const std::string& idx) {
    ViewPlan plan;
    ViewFile f;
    f.file_path = gz;
    f.index_path = idx;
    plan.files.push_back(f);
    plan.group_by = {GroupKey::cat()};
    plan.agg = {AggSpec(AggOp::Count, "", "n"),
                AggSpec(AggOp::Sum, "dur", "sum_dur"),
                AggSpec(AggOp::Min, "dur", "min_dur"),
                AggSpec(AggOp::Max, "dur", "max_dur")};
    return plan;
}

}  // namespace

TEST_SUITE("AggFold") {
    // The whole point: aggregating through the fused scan (PodSource) must
    // equal aggregating through the scan-path fold (View::collect over
    // DomSource).
    TEST_CASE("AggFold over fuse equals View::collect") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);  // 30 POSIX, 20 STDIO
        std::string idx = determine_index_path(gz, "");
        {
            StringSink s;
            View::from_file(gz, idx).export_json(s).get();
        }

        // scan-path fold.
        auto expected = View::from_file(gz, idx)
                            .group_by({GroupKey::cat()})
                            .agg({{AggOp::Count, "", "n"},
                                  {AggOp::Sum, "dur", "sum_dur"},
                                  {AggOp::Min, "dur", "min_dur"},
                                  {AggOp::Max, "dur", "max_dur"}})
                            .collect()
                            .get();

        // fused fold: same plan, driven through PodSource.
        ViewPlan plan = agg_plan(gz, idx);
        ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);
        ensure_schema(plan);

        StringIntern intern;
        AggFold agg(plan, intern);
        std::array<Fold*, 1> folds{&agg};
        fuse(plan, vdef, folds, intern).get();
        dftracer::utils::dataframe::DataFrame got = agg.result();

        auto e = rows_by_key(expected);
        auto g = rows_by_key(got);
        REQUIRE(g.size() == e.size());
        for (const auto& [k, vals] : e) {
            CAPTURE(k);
            REQUIRE(g.count(k) == 1);
            REQUIRE(g[k].size() == vals.size());
            for (std::size_t i = 0; i < vals.size(); ++i)
                CHECK(g[k][i] == doctest::Approx(vals[i]));
        }

        // Anchor the values so a mutual bug cannot pass both paths.
        REQUIRE(g.count("posix|") == 1);
        CHECK(g["posix|"][0] == doctest::Approx(30));         // count
        CHECK(g["posix|"][1] == doctest::Approx(30 * 24.5));  // sum(10..39)
        CHECK(g["posix|"][2] == doctest::Approx(10));         // min
        CHECK(g["posix|"][3] == doctest::Approx(39));         // max
    }

    // The optimization: an arg-free query must not extract args, but a query
    // reading size (from ret/size_sum) must, or it silently aggregates nulls.
    TEST_CASE(
        "needs_args is off for arg-free, on for size, and size is right") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_trace_with_ret(env, 20);  // ret 100..119
        std::string idx = determine_index_path(gz, "");
        {
            StringSink s;
            View::from_file(gz, idx).export_json(s).get();
        }

        StringIntern i0;
        {
            ViewPlan p = agg_plan(gz, idx);  // count/sum/min/max over dur
            AggFold a(p, i0);
            CHECK_FALSE(a.needs_args());
        }
        StringIntern i1;
        {
            ViewPlan p = size_plan(gz, idx);
            AggFold a(p, i1);
            CHECK(a.needs_args());
        }

        auto expected = View::from_file(gz, idx)
                            .group_by({GroupKey::cat()})
                            .agg({{AggOp::Sum, "size", "sum_size"}})
                            .collect()
                            .get();

        ViewPlan plan = size_plan(gz, idx);
        ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);
        ensure_schema(plan);
        StringIntern intern;
        AggFold agg(plan, intern);
        REQUIRE(agg.needs_args());
        std::array<Fold*, 1> folds{&agg};
        fuse(plan, vdef, folds, intern).get();

        auto e = rows_by_key(expected);
        auto g = rows_by_key(agg.result());
        REQUIRE(g.count("posix|") == 1);
        REQUIRE(e.count("posix|") == 1);
        CHECK(g["posix|"][0] == doctest::Approx(e["posix|"][0]));
        // sum(ret 100..119) = 20 * 109.5 = 2190; must not be 0 (args dropped).
        CHECK(g["posix|"][0] == doctest::Approx(2190));
    }

    // A tiny budget forces the group map to spill to sorted runs mid-scan; the
    // k-way merge at result() must reproduce the in-memory aggregate exactly.
    TEST_CASE("AggFold spill matches in-memory") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 500, 300);
        std::string idx = determine_index_path(gz, "");
        {
            StringSink s;
            View::from_file(gz, idx).export_json(s).get();
        }

        auto run = [&](std::uint64_t budget) {
            ViewPlan plan = agg_plan(gz, idx);
            plan.memory_budget = budget;
            ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/true);
            ensure_schema(plan);
            StringIntern intern;
            AggFold agg(plan, intern);
            std::array<Fold*, 1> folds{&agg};
            fuse(plan, vdef, folds, intern).get();
            return rows_by_key(agg.result());
        };

        auto in_mem = run(0);   // no budget: pure in-memory
        auto spilled = run(1);  // budget 1: spill after every batch

        REQUIRE(spilled.size() == in_mem.size());
        for (const auto& [k, vals] : in_mem) {
            CAPTURE(k);
            REQUIRE(spilled.count(k) == 1);
            REQUIRE(spilled[k].size() == vals.size());
            for (std::size_t i = 0; i < vals.size(); ++i)
                CHECK(spilled[k][i] == doctest::Approx(vals[i]));
        }
    }
}
