#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/rocksdb/column_families.h>
#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/trace/comparator/compare_view.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/rollup_store.h>
#include <dftracer/utils/trace/views/view_agg_engine.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_executor.h>
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <functional>
#include <map>
#include <utility>

#include "test_view_common.h"

namespace {

// Minimal external Fold for the fold-factory seam: counts events over the
// shared scan.
struct CountFold : dftracer::utils::trace::views::detail::Fold {
    std::shared_ptr<std::atomic<std::uint64_t>> total;
    std::uint64_t local = 0;
    explicit CountFold(std::shared_ptr<std::atomic<std::uint64_t>> t)
        : total(std::move(t)) {}
    bool accepts(const dftracer::utils::trace::views::detail::ScanShape&)
        const override {
        return true;
    }
    std::unique_ptr<dftracer::utils::trace::views::detail::Fold> slice()
        const override {
        return std::make_unique<CountFold>(total);
    }
    void step(
        const dftracer::utils::trace::views::detail::FoldBatch& b) override {
        local += b.events.size();
    }
    void seal_unit(
        const dftracer::utils::trace::views::detail::ScanUnit&) override {}
    void drop_unit(
        const dftracer::utils::trace::views::detail::ScanUnit&) override {}
    void merge(dftracer::utils::trace::views::detail::Fold& o) override {
        local += static_cast<CountFold&>(o).local;
    }
    dftracer::utils::coro::CoroTask<bool> finalize(
        const dftracer::utils::trace::views::detail::CoverageSet&) override {
        total->fetch_add(local);
        co_return true;
    }
};

struct HistBin {
    double lower;
    double upper;
    std::uint64_t count;
};

// Bins of a list<struct<lo,hi,count>> hist column at `row`.
std::vector<HistBin> hist_bins(const dataframe::DataFrame& b, std::int64_t row,
                               std::string_view name) {
    const dataframe::Series& lc =
        b.columns[static_cast<std::size_t>(bcol(b, name))];
    const std::int32_t* off = lc.offsets();
    const dataframe::Series st = lc.child(0);  // struct<lo,hi,count>
    const dataframe::Series lo = st.child(0);
    const dataframe::Series hi = st.child(1);
    const dataframe::Series cnt = st.child(2);
    const double* lop = lo.data<double>();
    const double* hip = hi.data<double>();
    const std::uint64_t* cntp = cnt.data<std::uint64_t>();
    std::vector<HistBin> out;
    for (std::int32_t i = off[row]; i < off[row + 1]; ++i)
        out.push_back({lop[i], hip[i], cntp[i]});
    return out;
}

}  // namespace

TEST_SUITE("View") {
    TEST_CASE(
        "View - sort_by/topk match the vec kernels applied post-collect") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);  // posix=30, stdio=20
        std::string idx = determine_index_path(gz, "");
        auto make = [&] {
            return View::from_file(gz, idx)
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}});
        };

        dataframe::DataFrame base = make().collect().collect().get();

        // sort_by(n desc): the View plan must equal dataframe::sort_by on the
        // batch.
        dataframe::DataFrame on_view =
            make().sort_by("n", true).collect().collect().get();
        dataframe::DataFrame on_vec = dataframe::sort_by(base, "n", true);
        REQUIRE(on_view.num_rows() == on_vec.num_rows());
        for (std::int64_t i = 0; i < on_view.num_rows(); ++i) {
            CHECK(bnum(on_view, i, "n") == bnum(on_vec, i, "n"));
            CHECK(bstr(on_view, i, "cat") == bstr(on_vec, i, "cat"));
        }
        CHECK(bnum(on_view, 0, "n") == 30);  // posix first, descending

        // topk(n, 1): the single largest-count group.
        dataframe::DataFrame tk_view =
            make().topk("n", 1).collect().collect().get();
        dataframe::DataFrame tk_vec = dataframe::topk(base, "n", 1, true);
        REQUIRE(tk_view.num_rows() == 1);
        CHECK(bnum(tk_view, 0, "n") == bnum(tk_vec, 0, "n"));
        CHECK(bstr(tk_view, 0, "cat") == "posix");
    }

    TEST_CASE(
        "View - session collect(View) applies each branch's full plan over one "
        "scan") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);  // posix=30, stdio=20
        std::string idx = determine_index_path(gz, "");
        View base = View::from_file(gz, idx);

        auto run = base.session();
        auto a = run.collect(base.group_by({GroupKey::cat()})
                                 .agg({{AggOp::Count, "", "n"}})
                                 .sort_by("n", true));
        auto b = run.collect(
            base.filter(Query::from_string(R"(cat == "POSIX")").value())
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}}));
        run.execute().get();

        REQUIRE(a->num_rows() == 2);
        CHECK(bstr(*a, 0, "cat") == "posix");  // count desc: 30 before 20
        CHECK(bnum(*a, 0, "n") == 30);
        CHECK(bnum(*a, 1, "n") == 20);

        REQUIRE(b->num_rows() == 1);  // per-branch filter is independent
        CHECK(bstr(*b, 0, "cat") == "posix");
        CHECK(bnum(*b, 0, "n") == 30);
    }

    TEST_CASE(
        "View - session fold factory rides the shared scan with a collect") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);  // 50 events
        std::string idx = determine_index_path(gz, "");
        View base = View::from_file(gz, idx);

        auto run = base.session();
        auto cat = run.collect(
            base.group_by({GroupKey::cat()}).agg({{AggOp::Count, "", "n"}}));
        auto total = std::make_shared<std::atomic<std::uint64_t>>(0);
        run.attach_fold_factory(
            [total](dftracer::utils::StringIntern&)
                -> std::unique_ptr<
                    dftracer::utils::trace::views::detail::Fold> {
                return std::make_unique<CountFold>(total);
            },
            []() {});
        run.execute().get();

        // The factory fold counted every event over the same scan the collect
        // aggregated (posix 30 + stdio 20).
        CHECK(total->load() == 50);
        REQUIRE(cat->num_rows() == 2);
    }

    TEST_CASE(
        "View - session join/compare combine two branches over the one scan") {
        namespace cmp = dftracer::utils::trace::comparator;
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);  // posix=30, stdio=20
        std::string idx = determine_index_path(gz, "");
        View base = View::from_file(gz, idx);
        Query posix = Query::from_string(R"(cat == "POSIX")").value();
        auto by_cat = [] { return std::vector<GroupKey>{GroupKey::cat()}; };
        auto count = [] {
            return std::vector<AggSpec>{{AggOp::Count, "", "n"}};
        };

        auto run = base.session();
        auto a = run.collect(base.group_by(by_cat()).agg(count()));
        auto b =
            run.collect(base.filter(posix).group_by(by_cat()).agg(count()));
        auto j = run.join(a, b, JoinType::INNER);  // n_key inferred (1)
        auto c = run.compare(a, b);
        run.execute().get();

        // Reference: collect both branches standalone, combine with the same
        // primitives the session ops wrap. The session must match exactly.
        auto ra = View::from_file(gz, idx)
                      .group_by(by_cat())
                      .agg(count())
                      .collect()
                      .collect()
                      .get();
        auto rb = View::from_file(gz, idx)
                      .filter(posix)
                      .group_by(by_cat())
                      .agg(count())
                      .collect()
                      .collect()
                      .get();
        auto rj = join_batches(ra, rb, 1, JoinType::INNER);
        auto rc = cmp::CompareView::compare_batches(ra, rb, 1);

        // INNER join keeps only cat=posix (b is posix-only).
        REQUIRE(j->num_rows() == rj.num_rows());
        REQUIRE(j->num_rows() == 1);
        CHECK(bstr(*j, 0, "cat") == "posix");
        CHECK(bnum(*j, 0, "l_n") == 30);
        CHECK(bnum(*j, 0, "r_n") == 30);

        // compare is a FULL join with delta_/pct_ appended: both cats present.
        REQUIRE(c->num_rows() == rc.num_rows());
        REQUIRE(c->num_rows() == 2);
        CHECK(bcol(*c, "delta_n") >= 0);
        CHECK(bcol(*c, "pct_n") >= 0);
    }

    TEST_CASE("View - session join rejects a foreign handle") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");
        View base = View::from_file(gz, idx);

        auto run = base.session();
        auto other = base.session();
        auto a = run.collect(
            base.group_by({GroupKey::cat()}).agg({{AggOp::Count, "", "n"}}));
        auto foreign = other.collect(
            base.group_by({GroupKey::cat()}).agg({{AggOp::Count, "", "n"}}));
        CHECK_THROWS_AS(run.join(a, foreign, JoinType::INNER),
                        DFTUtilsException);
    }

    TEST_CASE("View - agg accepts unified F field expressions") {
        namespace fld = dftracer::utils::dataframe::field;
        const auto& F = fld::F;
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        // Named field aggregates lower to the matching AggSpec, so the F form
        // and the AggSpec form produce the same table (same default col names).
        auto by_expr = View::from_file(gz, idx)
                           .group_by({GroupKey::cat()})
                           .agg(F("dur").sum(), F("dur").mean(), F.any.count())
                           .collect()
                           .collect()
                           .get();
        auto by_spec =
            View::from_file(gz, idx)
                .group_by({GroupKey::cat()})
                .agg(
                    {{AggOp::Sum, "dur"}, {AggOp::Mean, "dur"}, {AggOp::Count}})
                .collect()
                .collect()
                .get();
        REQUIRE(by_expr.num_rows() == by_spec.num_rows());
        REQUIRE(bhas(by_expr, "sum_dur"));
        REQUIRE(bhas(by_expr, "mean_dur"));
        REQUIRE(bhas(by_expr, "count"));
        std::map<std::string, std::array<double, 3>> em, sm;
        for (std::int64_t i = 0; i < by_expr.num_rows(); ++i)
            em[bstr(by_expr, i, "cat")] = {bnum(by_expr, i, "sum_dur"),
                                           bnum(by_expr, i, "mean_dur"),
                                           bnum(by_expr, i, "count")};
        for (std::int64_t i = 0; i < by_spec.num_rows(); ++i)
            sm[bstr(by_spec, i, "cat")] = {bnum(by_spec, i, "sum_dur"),
                                           bnum(by_spec, i, "mean_dur"),
                                           bnum(by_spec, i, "count")};
        CHECK(em == sm);
    }

    TEST_CASE("View - F.any wildcard aggregates numeric args (mean)") {
        namespace fld = dftracer::utils::dataframe::field;
        const auto& F = fld::F;
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // Two cats, each with a numeric arg "level"; mean over the wildcard
        // path must match the mean the AggSpec form computes for that field.
        std::string pfw = env.get_dir() + "/args.pfw";
        {
            std::ofstream ofs(pfw);
            for (int i = 0; i < 10; ++i)
                ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":)"
                    << (1000 + i * 10) << R"(,"dur":5,"args":{"level":)" << i
                    << R"(}})" << "\n";
            for (int i = 0; i < 4; ++i)
                ofs << R"({"ph":"X","name":"seek","cat":"STDIO","pid":1,"tid":1,"ts":)"
                    << (5000 + i * 10) << R"(,"dur":5,"args":{"level":)"
                    << (100 + i) << R"(}})" << "\n";
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        auto any = View::from_file(gz, idx)
                       .group_by({GroupKey::cat()})
                       .agg(F.any.mean(), F.any.count())
                       .collect()
                       .collect()
                       .get();
        REQUIRE(
            bhas(any, "level"));  // discovered numeric arg -> per-group mean
        REQUIRE(bhas(any, "count"));
        std::map<std::string, double> level, count;
        for (std::int64_t i = 0; i < any.num_rows(); ++i) {
            level[bstr(any, i, "cat")] = bnum(any, i, "level");
            count[bstr(any, i, "cat")] = bnum(any, i, "count");
        }
        CHECK(count["posix"] == 10);
        CHECK(count["stdio"] == 4);
        CHECK(level["posix"] == doctest::Approx(4.5));    // mean 0..9
        CHECK(level["stdio"] == doctest::Approx(101.5));  // mean 100..103

        // F.any.sum() applies a sum to every numeric arg (one sum_<arg>).
        auto anysum = View::from_file(gz, idx)
                          .group_by({GroupKey::cat()})
                          .agg(F.any.sum())
                          .collect()
                          .collect()
                          .get();
        REQUIRE(bhas(anysum, "sum_level"));
        std::map<std::string, double> slevel;
        for (std::int64_t i = 0; i < anysum.num_rows(); ++i)
            slevel[bstr(anysum, i, "cat")] = bnum(anysum, i, "sum_level");
        CHECK(slevel["posix"] == doctest::Approx(45));   // sum 0..9
        CHECK(slevel["stdio"] == doctest::Approx(406));  // sum 100..103

        // A per-arg percentile collects a per-arg sketch (p90 of posix level
        // 0..9 is ~9; DDSketch is approximate, so allow a small band).
        auto anypct =
            View::from_file(gz, idx)
                .group_by({GroupKey::cat()})
                .agg_numeric_args({AggSpec(AggOp::Pct, "", "p90", "", 0.9)})
                .collect()
                .collect()
                .get();
        REQUIRE(bhas(anypct, "p90_level"));
        std::map<std::string, double> plevel;
        for (std::int64_t i = 0; i < anypct.num_rows(); ++i)
            plevel[bstr(anypct, i, "cat")] = bnum(anypct, i, "p90_level");
        CHECK(plevel["posix"] >= 7.0);
        CHECK(plevel["posix"] <= 9.0);

        // An op with no per-arg support (Hist) is still rejected on the dyn
        // path (only FieldStat/sketch-derivable reductions are allowed).
        CHECK_THROWS_AS(View::from_file(gz, idx)
                            .group_by({GroupKey::cat()})
                            .agg_numeric_args({AggSpec(AggOp::Hist, "level")}),
                        DFTUtilsException);
    }

    TEST_CASE("View - set_union collects distinct field values per group") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz =
            create_mixed_trace(env, 30, 20);  // read/POSIX, fwrite/STDIO
        std::string idx = determine_index_path(gz, "");

        // Per-cat distinct names: each cat has exactly one name here.
        auto t = View::from_file(gz, idx)
                     .group_by({GroupKey::cat()})
                     .agg({{AggOp::Count, "", "n"},
                           {AggOp::SetUnion, "name", "names"}})
                     .collect()
                     .collect()
                     .get();
        CHECK(bhas(t, "names"));
        std::map<std::string, std::string> got;
        for (std::int64_t i = 0; i < t.num_rows(); ++i)
            got[bstr(t, i, "cat")] = bstr(t, i, "names");
        CHECK(got["posix"] == "read");
        CHECK(got["stdio"] == "fwrite");

        // No grouping: one row unioning both names, sorted + SET_SEP-joined.
        auto whole = View::from_file(gz, idx)
                         .agg({{AggOp::SetUnion, "name", "names"}})
                         .collect()
                         .collect()
                         .get();
        REQUIRE(whole.num_rows() == 1);
        const std::string joined = bstr(whole, 0, "names");
        std::set<std::string> vals;
        for (std::size_t s = 0, i = 0; i <= joined.size(); ++i)
            if (i == joined.size() || joined[i] == '\x1e') {
                vals.insert(joined.substr(s, i - s));
                s = i + 1;
            }
        CHECK(vals == std::set<std::string>{"read", "fwrite"});

        // Distributed partial + merge preserves the union.
        std::string p = View::from_file(gz, idx)
                            .agg({{AggOp::SetUnion, "name", "names"}})
                            .aggregate_partial()
                            .get();
        auto merged = View::from_file(gz, idx)
                          .agg({{AggOp::SetUnion, "name", "names"}})
                          .merge_partials_to_table({p});
        REQUIRE(merged.num_rows() == 1);
        CHECK(bstr(merged, 0, "names") == joined);

        // Rollup round-trip: materialize persists the set, a repeat reads it
        // back.
        auto make = [&]() {
            return View::from_file(gz, idx)
                .group_by({GroupKey::cat()})
                .agg({{AggOp::SetUnion, "name", "names"}});
        };
        make().materialize().run().get();
        auto back = make().collect().collect().get();
        std::map<std::string, std::string> rb;
        for (std::int64_t i = 0; i < back.num_rows(); ++i)
            rb[bstr(back, i, "cat")] = bstr(back, i, "names");
        CHECK(rb["posix"] == "read");
        CHECK(rb["stdio"] == "fwrite");
    }

    TEST_CASE("View - export_counters out-of-core spill matches in-memory") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_bulk_counters(env, 400);
        std::string idx = determine_index_path(gz, "");

        auto run = [&](std::uint64_t budget) {
            StringSink s;
            View::from_file(gz, idx)
                .phase(Phase::Counters)
                .group_by({GroupKey::name()})
                .agg_numeric_args()
                .memory_budget(budget)
                .export_counters(s)
                .get();
            auto l = s.lines();
            std::sort(l.begin(), l.end());
            return l;
        };

        auto in_mem = run(0);       // pure in-memory
        auto spilled = run(1);      // budget 1 -> spill after every batch
        CHECK(in_mem.size() == 2);  // cpu, gpu
        CHECK(in_mem == spilled);   // spill + k-way merge == in-memory result

        // auto_spill() picks ~1/3 memory: large budget, so it runs the spill
        // path but only flushes the final run - still identical output.
        StringSink sa;
        View::from_file(gz, idx)
            .phase(Phase::Counters)
            .group_by({GroupKey::name()})
            .agg_numeric_args()
            .auto_spill()
            .export_counters(sa)
            .get();
        auto autos = sa.lines();
        std::sort(autos.begin(), autos.end());
        CHECK(autos == in_mem);
    }

    TEST_CASE("View - Var/Std match closed-form and survive spill") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 200 POSIX events, dur = 10..209 (one cat group). Var/Std are the
        // SAMPLE convention (matches the dataframe engine and pandas): sample
        // variance of 200 consecutive ints is n(n+1)/12 = 200*201/12 = 3350;
        // mean 109.5.
        std::string gz = create_mixed_trace(env, 200, 0);
        std::string idx = determine_index_path(gz, "");

        auto row = [&](std::uint64_t budget) {
            auto t = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Mean, "dur", "m"},
                               {AggOp::Var, "dur", "v"},
                               {AggOp::Std, "dur", "s"},
                               {AggOp::Skew, "dur", "sk"},
                               {AggOp::Kurt, "dur", "ku"}})
                         .memory_budget(budget)
                         .collect()
                         .collect()
                         .get();
            REQUIRE(t.num_rows() == 1);
            return std::vector<double>{bnum(t, 0, "m"), bnum(t, 0, "v"),
                                       bnum(t, 0, "s"), bnum(t, 0, "sk"),
                                       bnum(t, 0, "ku")};
        };

        auto in_mem = row(0);
        CHECK(in_mem[0] == doctest::Approx(109.5));
        CHECK(in_mem[1] == doctest::Approx(3350.0));    // sample variance
        CHECK(in_mem[2] == doctest::Approx(57.87918));  // sqrt(3350)
        // 200 consecutive ints: symmetric (skew 0), platykurtic (excess ~
        // -1.2).
        CHECK(in_mem[3] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(in_mem[4] == doctest::Approx(-1.2).epsilon(0.01));
        CHECK(row(1) == in_mem);  // spill + k-way merge (m3/m4) is identical
    }

    TEST_CASE("View - integer field Min/Max/Sum are exact integer columns") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 200 POSIX events, dur = 10..209 (one cat group).
        std::string gz = create_mixed_trace(env, 200, 0);
        std::string idx = determine_index_path(gz, "");

        auto t = View::from_file(gz, idx)
                     .group_by({GroupKey::cat()})
                     .agg({{AggOp::Min, "dur", "mn"},
                           {AggOp::Max, "dur", "mx"},
                           {AggOp::Sum, "dur", "sm"}})
                     .collect()
                     .collect()
                     .get();
        REQUIRE(t.num_rows() == 1);
        // dur is a non-negative integer field, so the aggregates come back as
        // an exact Uint64 column, not a Float64 rounded through a double.
        CHECK(t.columns[static_cast<std::size_t>(bcol(t, "mn"))].type() ==
              dataframe::TypeId::Uint64);
        CHECK(t.columns[static_cast<std::size_t>(bcol(t, "sm"))].type() ==
              dataframe::TypeId::Uint64);
        CHECK(bnum(t, 0, "mn") == 10);
        CHECK(bnum(t, 0, "mx") == 209);
        CHECK(bnum(t, 0, "sm") == 21900);  // sum(10..209)
    }

    TEST_CASE(
        "View - partial/merge preserves the integer domain of the direct "
        "collect path") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 0);
        std::string idx = determine_index_path(gz, "");

        auto make = [&] {
            return View::from_file(gz, idx)
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Min, "dur", "mn"},
                      {AggOp::Max, "dur", "mx"},
                      {AggOp::Sum, "dur", "sm"}});
        };

        // Direct fold path.
        dataframe::DataFrame direct = make().collect().collect().get();
        // Partial + merge path (serializes the accumulator and back).
        std::string p = make().aggregate_partial().get();
        dataframe::DataFrame merged = make().merge_partials_to_table({p});

        REQUIRE(direct.num_rows() == 1);
        REQUIRE(merged.num_rows() == 1);

        // The round-trip must not demote the integer field to Float64: both
        // paths yield the same column TypeId and the same exact values.
        for (const char* col : {"mn", "mx", "sm"}) {
            const auto dt =
                direct.columns[static_cast<std::size_t>(bcol(direct, col))]
                    .type();
            const auto mt =
                merged.columns[static_cast<std::size_t>(bcol(merged, col))]
                    .type();
            CHECK(dt == dataframe::TypeId::Uint64);
            CHECK(mt == dataframe::TypeId::Uint64);
            CHECK(dt == mt);
            CHECK(bnum(direct, 0, col) == bnum(merged, 0, col));
        }
        CHECK(bnum(merged, 0, "mn") == 10);
        CHECK(bnum(merged, 0, "mx") == 209);
        CHECK(bnum(merged, 0, "sm") == 21900);
    }

    TEST_CASE("View - percentiles survive spill (DDSketch round-trips)") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 200 POSIX events, dur = 10..209.
        std::string gz = create_mixed_trace(env, 200, 0);
        std::string idx = determine_index_path(gz, "");

        auto p99 = [&](std::uint64_t budget) {
            auto t = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Pct, "dur", "p99", "", 0.99}})
                         .memory_budget(budget)
                         .collect()
                         .collect()
                         .get();
            REQUIRE(t.num_rows() == 1);
            return bnum(t, 0, "p99");
        };

        const double in_mem = p99(0);
        // p99 of 10..209 is ~207, within DDSketch's 1% relative error.
        CHECK(in_mem == doctest::Approx(207).epsilon(0.02));
        // Spilling serializes + k-way merges the sketch; result is identical.
        CHECK(p99(1) == doctest::Approx(in_mem));
    }

    TEST_CASE("View - histogram buckets cover every event and survive spill") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 200 POSIX events, dur = 10..209.
        std::string gz = create_mixed_trace(env, 200, 0);
        std::string idx = determine_index_path(gz, "");

        auto hist = [&](std::uint64_t budget) {
            auto t = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Hist, "dur", "h"}})
                         .memory_budget(budget)
                         .collect()
                         .collect()
                         .get();
            REQUIRE(t.num_rows() == 1);
            REQUIRE(bhas(t, "h"));
            return hist_bins(t, 0, "h");
        };

        const auto in_mem = hist(0);
        std::uint64_t total = 0;
        for (const auto& b : in_mem) {
            CHECK(b.lower <= b.upper);
            total += b.count;
        }
        CHECK(total == 200);  // every event lands in exactly one bucket

        // Spilling serializes + k-way merges the sketch; the histogram is
        // identical.
        const auto spilled = hist(1);
        REQUIRE(spilled.size() == in_mem.size());
        std::uint64_t spilled_total = 0;
        for (const auto& b : spilled) spilled_total += b.count;
        CHECK(spilled_total == 200);
    }

    TEST_CASE("View - collect out-of-core spill matches in-memory") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 200 POSIX events at ts 1000,1100,..; 100us buckets -> ~200 groups,
        // enough that a 1-byte budget spills after nearly every batch.
        std::string gz = create_mixed_trace(env, 200, 0);
        std::string idx = determine_index_path(gz, "");

        auto run = [&](std::uint64_t budget) {
            auto t = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .time_bucket(100)
                         .agg({{AggOp::Count, "", "n"},
                               {AggOp::Sum, "dur", "total"}})
                         .memory_budget(budget)
                         .collect()
                         .collect()
                         .get();
            std::vector<std::string> rows;
            for (std::int64_t i = 0; i < t.num_rows(); ++i) {
                std::string s;
                s += bstr(t, i, "time_bucket") + "|";
                s += bstr(t, i, "cat") + "|";
                s += std::to_string(bnum(t, i, "n")) + "|";
                s += std::to_string(bnum(t, i, "total")) + "|";
                rows.push_back(std::move(s));
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };

        auto in_mem = run(0);
        auto spilled = run(1);
        CHECK(in_mem.size() >= 3);
        CHECK(in_mem == spilled);  // spill + k-way merge == in-memory result
    }

    TEST_CASE("View - materialize() persists a rollup a repeat query reads") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        auto canon = [](const dataframe::DataFrame& df) {
            std::vector<std::string> rows;
            for (std::int64_t i = 0; i < df.num_rows(); ++i) {
                std::string s;
                for (const auto& c : df.columns) {
                    if (c.type() == dataframe::TypeId::String) {
                        s += std::string(c.string_at(i)) + "|";
                    } else if (c.type() == dataframe::TypeId::Int64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::int64_t>()[i])) +
                             "|";
                    } else if (c.type() == dataframe::TypeId::Uint64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::uint64_t>()[i])) +
                             "|";
                    } else {
                        s += std::to_string(c.data<double>()[i]) + "|";
                    }
                }
                rows.push_back(std::move(s));
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };
        // A per-event predicate (dur is not a tier key) forces the real scan
        // path where the materialize hook lives, independent of tier state.
        auto make = [&]() {
            return View::from_file(gz, idx)
                .query("dur >= 0")
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "total"}});
        };

        auto expect = canon(make().collect().collect().get());
        REQUIRE(expect.size() >= 1);

        namespace detail = dftracer::utils::trace::views::detail;
        namespace rdb = dftracer::utils::rocksdb;

        // materialize() persists the result into the index's ROLLUP CF as a
        // byproduct of answering it.
        CHECK(canon(make().materialize().collect().collect().get()) == expect);
        {
            auto db = detail::open_rollup_db(
                idx, rdb::RocksDatabase::OpenMode::ReadOnly);
            REQUIRE(db);
            auto it = db->new_iterator(rdb::cf::ROLLUP);
            it->SeekToFirst();
            CHECK(it->Valid());  // the ROLLUP CF holds the materialized view
        }

        // A repeat query - even without materialize() - reads the rollup back.
        CHECK(canon(make().collect().collect().get()) == expect);
    }

    TEST_CASE("View - run() materializes the rollup without a table") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        namespace detail = dftracer::utils::trace::views::detail;
        namespace rdb = dftracer::utils::rocksdb;

        auto canon = [](const dataframe::DataFrame& df) {
            std::vector<std::string> rows;
            for (std::int64_t i = 0; i < df.num_rows(); ++i) {
                std::string s;
                for (const auto& c : df.columns) {
                    if (c.type() == dataframe::TypeId::String) {
                        s += std::string(c.string_at(i)) + "|";
                    } else if (c.type() == dataframe::TypeId::Int64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::int64_t>()[i])) +
                             "|";
                    } else if (c.type() == dataframe::TypeId::Uint64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::uint64_t>()[i])) +
                             "|";
                    } else {
                        s += std::to_string(c.data<double>()[i]) + "|";
                    }
                }
                rows.push_back(std::move(s));
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };
        auto make = [&]() {
            return View::from_file(gz, idx)
                .query("dur >= 0")
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "total"}});
        };

        auto expect =
            canon(make().collect().collect().get());  // fresh, no rollup yet
        REQUIRE(expect.size() >= 1);

        make().run().get();  // build-only terminal: materialize the rollup
        {
            auto db = detail::open_rollup_db(
                idx, rdb::RocksDatabase::OpenMode::ReadOnly);
            REQUIRE(db);
            auto it = db->new_iterator(rdb::cf::ROLLUP);
            it->SeekToFirst();
            CHECK(it->Valid());
        }
        CHECK(canon(make().collect().collect().get()) ==
              expect);  // now reads the rollup
    }

    TEST_CASE("View - a coarser query is served by rolling up a finer view") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        auto canon = [](const dataframe::DataFrame& df) {
            std::vector<std::string> rows;
            for (std::int64_t i = 0; i < df.num_rows(); ++i) {
                std::string s;
                for (const auto& c : df.columns) {
                    if (c.type() == dataframe::TypeId::String) {
                        s += std::string(c.string_at(i)) + "|";
                    } else if (c.type() == dataframe::TypeId::Int64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::int64_t>()[i])) +
                             "|";
                    } else if (c.type() == dataframe::TypeId::Uint64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::uint64_t>()[i])) +
                             "|";
                    } else {
                        s += std::to_string(c.data<double>()[i]) + "|";
                    }
                }
                rows.push_back(std::move(s));
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };
        // Finer and coarser views differ only in grouping (same filter + aggs),
        // so the coarser is the finer rolled up over `name`.
        auto fine = [&]() {
            return View::from_file(gz, idx)
                .query("dur >= 0")
                .group_by({GroupKey::cat(), GroupKey::name()})
                .agg({{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "total"}});
        };
        auto coarse = [&]() {
            return View::from_file(gz, idx)
                .query("dur >= 0")
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "total"}});
        };

        auto expect =
            canon(coarse().collect().collect().get());  // fresh, no rollup
        REQUIRE(expect.size() >= 1);

        fine().run().get();  // materialize only the finer rollup

        // The coarse query is answered by re-aggregating the finer rollup.
        CHECK(canon(coarse().collect().collect().get()) == expect);
    }

    TEST_CASE(
        "View - a coarser time bucket is served by re-bucketing a finer one") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        auto canon = [](const dataframe::DataFrame& df) {
            std::vector<std::string> rows;
            for (std::int64_t i = 0; i < df.num_rows(); ++i) {
                std::string s;
                for (const auto& c : df.columns) {
                    if (c.type() == dataframe::TypeId::String) {
                        s += std::string(c.string_at(i)) + "|";
                    } else if (c.type() == dataframe::TypeId::Int64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::int64_t>()[i])) +
                             "|";
                    } else if (c.type() == dataframe::TypeId::Uint64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::uint64_t>()[i])) +
                             "|";
                    } else {
                        s += std::to_string(c.data<double>()[i]) + "|";
                    }
                }
                rows.push_back(std::move(s));
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };
        // Same grouping and aggs, differing only in time grain. 1000 is a
        // multiple of 100, so the coarse view is the fine one re-bucketed.
        auto at_bucket = [&](std::uint64_t b) {
            return View::from_file(gz, idx)
                .query("dur >= 0")
                .time_bucket(b)
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "total"}});
        };

        // Baselines from fresh scans, before any rollup exists.
        auto expect_coarse = canon(at_bucket(1000).collect().collect().get());
        auto expect_fine = canon(at_bucket(100).collect().collect().get());
        REQUIRE(expect_coarse.size() >= 1);
        REQUIRE(expect_coarse.size() <= expect_fine.size());  // coarser folds

        at_bucket(100).run().get();  // materialize only the finer grain

        // The coarse query re-buckets the finer rollup; the fine query reads it
        // back exactly.
        CHECK(canon(at_bucket(1000).collect().collect().get()) ==
              expect_coarse);
        CHECK(canon(at_bucket(100).collect().collect().get()) == expect_fine);
    }

    TEST_CASE(
        "View - distributed materialize builds the rollup from partials") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // Two shards sharing one aggregation index (as the tier requires), so
        // the cross-shard rollup has a single anchor.
        std::string a = create_mixed_trace(env, 100, 30);
        std::string b = create_mixed_trace(env, 100, 30);
        std::string shared = env.get_dir() + "/shared_idx";

        auto canon = [](const dataframe::DataFrame& df) {
            std::vector<std::string> rows;
            for (std::int64_t i = 0; i < df.num_rows(); ++i) {
                std::string s;
                for (const auto& c : df.columns) {
                    if (c.type() == dataframe::TypeId::String) {
                        s += std::string(c.string_at(i)) + "|";
                    } else if (c.type() == dataframe::TypeId::Int64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::int64_t>()[i])) +
                             "|";
                    } else if (c.type() == dataframe::TypeId::Uint64) {
                        s += std::to_string(static_cast<double>(
                                 c.data<std::uint64_t>()[i])) +
                             "|";
                    } else {
                        s += std::to_string(c.data<double>()[i]) + "|";
                    }
                }
                rows.push_back(std::move(s));
            }
            std::sort(rows.begin(), rows.end());
            return rows;
        };
        auto view = [&](std::vector<ViewFile> files) {
            return View::from_files(std::move(files))
                .group_by({GroupKey::cat()})
                .agg({{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "total"}});
        };
        const std::vector<ViewFile> both{{a, shared}, {b, shared}};

        auto expect = canon(
            view(both).collect().collect().get());  // single-node baseline
        REQUIRE(expect.size() >= 1);

        // Each rank aggregates its shard; the coordinator reduces +
        // materializes.
        std::string pa = view({{a, shared}}).aggregate_partial().get();
        std::string pb = view({{b, shared}}).aggregate_partial().get();
        view(both).materialize_partials({pa, pb}).get();

        // The full query now reads the distributed-built rollup.
        CHECK(canon(view(both).collect().collect().get()) == expect);
    }

    TEST_CASE("View - distributed partials merge like a single aggregation") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // Two shards ("ranks"): user_pct {40,40} and {80,80} -> merged mean 60.
        std::string a = create_counter_file(env, "a", {40, 40});
        std::string b = create_counter_file(env, "b", {80, 80});
        std::string ia = determine_index_path(a, "");
        std::string ib = determine_index_path(b, "");

        auto view = [](std::vector<ViewFile> files) {
            return View::from_files(std::move(files))
                .phase(Phase::Counters)
                .group_by({GroupKey::name()})
                .agg_numeric_args();
        };

        // Rank-local partials, then merge them (the transport is elided here).
        std::string pa = view({{a, ia}}).aggregate_partial().get();
        std::string pb = view({{b, ib}}).aggregate_partial().get();
        StringSink merged;
        view({}).merge_counter_partials({pa, pb}, merged);

        // Compare against aggregating both shards at once.
        StringSink single;
        view({{a, ia}, {b, ib}}).export_counters(single).get();

        auto ml = merged.lines();
        auto sl = single.lines();
        std::sort(ml.begin(), ml.end());
        std::sort(sl.begin(), sl.end());
        CHECK(ml == sl);
        REQUIRE(ml.size() == 1);
        CHECK(ml[0].find(R"("user_pct":60)") != std::string::npos);
    }

    TEST_CASE("View - group_by + time_bucket yields per-interval rows") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // POSIX ts = 1000,1100,..,3900 (span 2900us). 1000us buckets -> 4
        // buckets.
        std::string gz = create_mixed_trace(env, 30, 0);
        std::string idx = determine_index_path(gz, "");

        auto table = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .time_bucket(1000)
                         .agg({{AggOp::Count, "", "n"}})
                         .collect()
                         .collect()
                         .get();

        REQUIRE(table.num_columns() == 3);  // time_bucket, cat, n
        CHECK(table.names[0] == "time_bucket");
        CHECK(table.names[1] == "cat");
        // buckets 1000,2000,3000 (10 each) + 4000 (ts 4000? no, max 3900) ->
        // rows across buckets, all cat POSIX, counts sum to 30.
        double total = 0;
        for (std::int64_t i = 0; i < table.num_rows(); ++i)
            total += bnum(table, i, "n");
        CHECK(total == doctest::Approx(30));
        CHECK(table.num_rows() >= 3);
    }

    TEST_CASE("View - time_bucket origin/min alignment shifts boundaries") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // POSIX ts = 1000,1100,..,3900; trace min ts = 1000. Width 700 does not
        // divide 1000, so absolute vs min alignment land the first bucket
        // differently.
        std::string gz = create_mixed_trace(env, 30, 0);
        std::string idx = determine_index_path(gz, "");

        auto first_bucket = [&](AggregatedView v) {
            auto t = v.agg({{AggOp::Count, "", "n"}}).collect().collect().get();
            std::int64_t lo = std::numeric_limits<std::int64_t>::max();
            for (std::int64_t i = 0; i < t.num_rows(); ++i)
                lo = std::min<std::int64_t>(
                    lo, std::stoll(bstr(t, i, "time_bucket")));
            return lo;
        };

        // Absolute (aligned to 0): floor(1000/700)*700 = 700.
        CHECK(first_bucket(View::from_file(gz, idx)
                               .group_by({GroupKey::cat()})
                               .time_bucket(700)) == 700);
        // Min-aligned: first bucket starts at the trace min ts (1000).
        CHECK(first_bucket(View::from_file(gz, idx)
                               .group_by({GroupKey::cat()})
                               .time_bucket_min(700)) == 1000);
        // Explicit origin 500: floor((1000-500)/700)*700 + 500 = 500.
        CHECK(first_bucket(View::from_file(gz, idx)
                               .group_by({GroupKey::cat()})
                               .time_bucket(700, 500)) == 500);
    }

    TEST_CASE("View - collect with no group_by folds whole set into one row") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        auto table = View::from_file(gz, idx)
                         .agg({{AggOp::Count, "", "n"}})
                         .collect()
                         .collect()
                         .get();

        REQUIRE(table.num_rows() == 1);
        CHECK(bnum(table, 0, "n") == doctest::Approx(50));
    }

    TEST_CASE("View - ArgMax(name, dur) returns the longest event's name") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // POSIX: read(10), write(99), read(20). The longest is "write".
        std::string pfw = env.get_dir() + "/am.pfw";
        {
            std::ofstream ofs(pfw);
            ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":1000,"dur":10,"args":{}})"
                << "\n"
                << R"({"ph":"X","name":"write","cat":"POSIX","pid":1,"tid":1,"ts":1100,"dur":99,"args":{}})"
                << "\n"
                << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,"ts":1200,"dur":20,"args":{}})"
                << "\n";
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        auto table = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Max, "dur", "max_dur"},
                               {AggOp::ArgMax, "name", "longest", "dur"}})
                         .collect()
                         .collect()
                         .get();

        REQUIRE(table.num_rows() == 1);
        CHECK(bhas(table, "longest"));
        CHECK(bstr(table, 0, "longest") == "write");
        CHECK(bnum(table, 0, "max_dur") == doctest::Approx(99));  // max_dur
    }

    TEST_CASE("View - group_by rank resolves pid via PR metadata") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // Two processes, each declared once by a PR metadata record mapping
        // pid -> rank. Grouping by rank harvests that map during the scan and
        // relabels the pid groups to "0"/"1" at the resolver, like host_name.
        std::string pfw = env.get_dir() + "/rank.pfw";
        {
            std::ofstream ofs(pfw);
            ofs << R"({"ph":"M","name":"PR","cat":"dftracer","pid":100,"tid":0,"args":{"name":"rank","value":"0"}})"
                << "\n"
                << R"({"ph":"M","name":"PR","cat":"dftracer","pid":200,"tid":0,"args":{"name":"rank","value":"1"}})"
                << "\n";
            for (int i = 0; i < 3; ++i)
                ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":100,"tid":1,"ts":)"
                    << (1000 + i * 10) << R"(,"dur":5,"args":{}})" << "\n";
            for (int i = 0; i < 2; ++i)
                ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":200,"tid":1,"ts":)"
                    << (2000 + i * 10) << R"(,"dur":5,"args":{}})" << "\n";
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        auto table = View::from_file(gz, idx)
                         .group_by({GroupKey::rank()})
                         .agg({{AggOp::Count, "", "n"}})
                         .collect()
                         .collect()
                         .get();

        REQUIRE(bhas(table, "rank"));
        std::map<std::string, double> n;
        for (std::int64_t i = 0; i < table.num_rows(); ++i)
            n[bstr(table, i, "rank")] = bnum(table, i, "n");
        CHECK(n["0"] == doctest::Approx(3));
        CHECK(n["1"] == doctest::Approx(2));
    }

    TEST_CASE(
        "View - map_batches folds a custom partial over the scan and reduces "
        "it") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // POSIX dur 10..39 (30), STDIO dur 20..39 (20). The custom fold splits
        // small (dur < 25) from big and keeps the big lines aside - the density
        // shape the built-in agg cannot express in one pass.
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        struct Acc {
            std::size_t small = 0;
            double small_sum = 0;
            std::vector<std::string> big;
        };
        auto dur_of = [](std::string_view e) {
            auto p = e.find("\"dur\":");
            return p == std::string_view::npos
                       ? 0.0
                       : std::strtod(std::string(e.substr(p + 6)).c_str(),
                                     nullptr);
        };
        auto res =
            View::from_file(gz, idx)
                .metadata(false)
                .map_batches<Acc>(
                    [&](Acc& a, const std::vector<std::string_view>& evs) {
                        for (auto e : evs) {
                            double d = dur_of(e);
                            if (d < 25) {
                                ++a.small;
                                a.small_sum += d;
                            } else {
                                a.big.emplace_back(e);
                            }
                        }
                    },
                    [](Acc&& x, Acc&& y) {
                        x.small += y.small;
                        x.small_sum += y.small_sum;
                        for (auto& b : y.big) x.big.emplace_back(std::move(b));
                        return std::move(x);
                    },
                    /*num_slots=*/4)
                .get();

        // small: POSIX 10..24 (15) + STDIO 20..24 (5) = 20; sum 255 + 110.
        CHECK(res.value.small == 20);
        CHECK(res.value.small_sum == doctest::Approx(365));
        // big: the other 30 events, none below the threshold.
        CHECK(res.value.big.size() == 30);
        for (const auto& l : res.value.big) CHECK(dur_of(l) >= 25);
        CHECK(res.stats.events_matched == 50);  // one scan over all events
        CHECK(res.stats.truncated == false);
    }

    TEST_CASE("View - map_batches honors the event limit (truncates)") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        struct Cnt {
            std::size_t n = 0;
        };
        auto res = View::from_file(gz, idx)
                       .metadata(false)
                       .map_batches<Cnt>(
                           [](Cnt& c, const std::vector<std::string_view>& e) {
                               c.n += e.size();
                           },
                           [](Cnt&& a, Cnt&& b) {
                               a.n += b.n;
                               return std::move(a);
                           },
                           /*num_slots=*/2, /*limit=*/10)
                       .get();

        CHECK(res.stats.truncated == true);
        CHECK(res.value.n >= 10);  // cap is checked per batch, may overshoot
        CHECK(res.value.n <= 50);
    }

    TEST_CASE(
        "View - fused partition drives heterogeneous branches in one scan") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // POSIX dur 10..39 (30), STDIO dur 20..39 (20).
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        struct Small {
            std::size_t n = 0;
            double sum = 0;
        };

        StringSink big;
        auto run = View::from_file(gz, idx).metadata(false).session();
        // custom fold: small events (dur < 25), reusing the parsed event.
        auto small = run.fold<Small>(
            Query::from_string("dur < 25").value(),
            [](Small& s, const json::JsonValue& jv, std::string_view) {
                ++s.n;
                s.sum += jv["dur"].get<double>(0);
            },
            [](Small&& a, Small&& b) {
                a.n += b.n;
                a.sum += b.sum;
                return std::move(a);
            });
        // raw passthrough: big events (dur >= 25).
        auto big_stats =
            run.export_json(Query::from_string("dur >= 25").value(), big);
        // built-in agg: POSIX count/sum over the same scan.
        auto posix = run.collect(
            Query::from_string(R"(cat == "POSIX")").value(), {GroupKey::cat()},
            {{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "sum_dur"}});

        auto stats = run.execute().get();

        // small: POSIX 10..24 (15) + STDIO 20..24 (5) = 20; sum 255 + 110.
        CHECK(small->n == 20);
        CHECK(small->sum == doctest::Approx(365));
        // big: the other 30 events, all dur >= 25.
        auto lines = big.lines();
        CHECK(lines.size() == 30);
        CHECK(big_stats->events_matched == 30);
        // POSIX branch: 30 events, dur 10..39. cat | n (count) | sum_dur.
        REQUIRE(posix->num_rows() == 1);
        const dataframe::Series& nc = posix->columns[1];
        const dataframe::Series& sc = posix->columns[2];
        CHECK(nc.data<std::int64_t>()[0] == 30);
        const double sd = sc.type() == dataframe::TypeId::Uint64
                              ? static_cast<double>(sc.data<std::uint64_t>()[0])
                          : sc.type() == dataframe::TypeId::Int64
                              ? static_cast<double>(sc.data<std::int64_t>()[0])
                              : sc.data<double>()[0];
        CHECK(sd == doctest::Approx(30 * 24.5));
        // One scan fed all three branches.
        CHECK(stats.events_matched == 50);
        CHECK(stats.truncated == false);
    }

    TEST_CASE("View - ChunkStatsSource matches a full scan (A/B)") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        View base = View::from_file(gz, idx)
                        .group_by({GroupKey::cat()})
                        .agg({{AggOp::Count, "dur", "n"},
                              {AggOp::Sum, "dur", "sum_dur"},
                              {AggOp::Min, "dur", "min_dur"},
                              {AggOp::Max, "dur", "max_dur"}});

        auto slow = base.collect().collect().get();
        ChunkStatsSource source;
        auto fast = base.with_partial_source(&source).collect().collect().get();

        auto row_of = [](const dataframe::DataFrame& b,
                         const std::string& cat) -> std::int64_t {
            for (std::int64_t i = 0; i < b.num_rows(); ++i)
                if (bstr(b, i, "cat") == cat) return i;
            return -1;
        };
        REQUIRE(slow.num_rows() == fast.num_rows());
        for (const char* c : {"posix", "stdio"}) {
            const std::int64_t s = row_of(slow, c);
            const std::int64_t f = row_of(fast, c);
            REQUIRE(s >= 0);
            REQUIRE(f >= 0);
            for (const char* col : {"n", "sum_dur", "min_dur", "max_dur"})
                CHECK(bnum(fast, f, col) ==
                      doctest::Approx(bnum(slow, s, col)));
        }
    }

    TEST_CASE("View - ChunkStatsSource reads stored per-key sketches") {
        using dftracer::utils::trace::indexing::ChunkStatistics;
        using dftracer::utils::utilities::indexer::IndexDatabase;
        using dftracer::utils::utilities::indexer::internal::get_logical_path;

        auto dir = make_unique_test_path("source");
        fs::create_directories(dir);
        const std::string index_path = (dir / "idx").string();
        const std::string file_path = (dir / "trace.pfw.gz").string();
        {
            IndexDatabase db(index_path);
            auto w = db.begin_write();
            w->init_schema();
            int fid =
                w->get_or_create_file_info(get_logical_path(file_path), 10000);
            ChunkStatistics stats;
            for (std::uint64_t dur : {10u, 20u, 30u})
                stats.update_from_event("read", "POSIX", 1, 1, 1000, dur, true);
            stats.min_timestamp_us = 1000;
            stats.max_timestamp_us = 2000;
            w->insert_chunk_statistics(fid, 0, stats);
            w->commit();
        }

        namespace vdetail = dftracer::utils::trace::views::detail;
        vdetail::ViewPlan plan;
        plan.group_by = {GroupKey::cat()};
        plan.agg = {{AggOp::Count, "", "n"},
                    {AggOp::Sum, "dur", "total"},
                    {AggOp::Min, "dur", "mn"},
                    {AggOp::Max, "dur", "mx"}};
        vdetail::ensure_schema(plan);

        ChunkStatsSource source;
        vdetail::PartialRequest req;
        req.files.push_back(ViewFile{file_path, index_path, 0, 0, 0});
        req.schema = plan.schema.get();
        req.group_by = plan.group_by;
        req.agg_field = "dur";

        std::vector<vdetail::AggAccum> got;
        auto res = source.lookup(
            req, [&](vdetail::AggAccum&& a) { got.push_back(std::move(a)); });

        CHECK(res.handled);
        REQUIRE(got.size() == 1);
        REQUIRE(got[0].keys.size() == 1);
        CHECK(got[0].keys[0] == "posix");
        CHECK(got[0].count == 3);
        const int fi = vdetail::schema_field_index(*plan.schema, "dur");
        REQUIRE(fi >= 0);
        CHECK(got[0].fields[fi].n == 3);
        CHECK(got[0].fields[fi].sum == doctest::Approx(60));
        CHECK(got[0].fields[fi].min == doctest::Approx(10));
        CHECK(got[0].fields[fi].max == doctest::Approx(30));
        REQUIRE(res.covered_chunks.size() == 1);
        CHECK(res.covered_chunks[0].second == 0);
    }

    TEST_CASE(
        "View - phase(Events) vs phase(Counters) separate ph=X and ph=C") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_trace_with_counters(env);
        std::string idx = determine_index_path(gz, "");

        // Events: the 10 ph=X reads, no counters.
        StringSink ev;
        View::from_file(gz, idx)
            .phase(Phase::Events)
            .metadata(false)
            .export_json(ev)
            .get();
        CHECK(ev.lines().size() == 10);
        CHECK(count_containing(ev.lines(), R"("ph":"C")") == 0);

        // Counters: the 2 ph=C events only.
        StringSink ct;
        View::from_file(gz, idx)
            .phase(Phase::Counters)
            .metadata(false)
            .export_json(ct)
            .get();
        CHECK(ct.lines().size() == 2);
        CHECK(count_containing(ct.lines(), "cpu") == 2);
    }

    TEST_CASE(
        "View - export_counters emits re-parseable ph=C aggregate trace") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 30 POSIX reads, ts 1000..3900, dur 10..39.
        std::string gz = create_mixed_trace(env, 30, 0);
        std::string idx = determine_index_path(gz, "");

        StringSink sink;
        View::from_file(gz, idx)
            .group_by({GroupKey::cat()})
            .time_bucket(1000)
            .agg(
                {{AggOp::Count, "", "count"}, {AggOp::Sum, "dur", "total_dur"}})
            .export_counters(sink)
            .get();

        auto lines = sink.lines();
        CHECK(lines.size() >= 3);  // one counter event per interval
        // Every emitted line is a ph=2 (COUNTER) event carrying the agg args.
        CHECK(count_containing(lines, R"("ph":2)") == lines.size());
        CHECK(count_containing(lines, R"("name":"posix")") == lines.size());
        CHECK(count_containing(lines, "count") == lines.size());
        CHECK(count_containing(lines, "total_dur") == lines.size());

        // Re-read the emitted counter trace as a View: phase(Counters) sees
        // them.
        std::string out = env.get_dir() + "/agg.pfw.gz";
        dftu_utils_test::write_gz_trace(out, sink.buffer);
        std::string out_idx = determine_index_path(out, "");
        StringSink reread;
        View::from_file(out, out_idx)
            .phase(Phase::Counters)
            .metadata(false)
            .export_json(reread)
            .get();
        CHECK(reread.lines().size() == lines.size());
    }

    TEST_CASE(
        "View - occupancy busy holds its invariants and honors occ_cell") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // "overlap" has two overlapping intervals (exact union 150 < sum
        // 200); "serial" has two disjoint ones (union == sum(dur)).
        std::string pfw = env.get_dir() + "/occ.pfw";
        {
            std::ofstream ofs(pfw);
            auto ev = [&](const char* name, long ts, long dur) {
                ofs << R"({"ph":"X","name":")" << name
                    << R"(","cat":"POSIX","pid":1,"tid":1,"ts":)" << ts
                    << R"(,"dur":)" << dur << R"(,"args":{}})" << "\n";
            };
            ev("overlap", 1000, 100);
            ev("overlap", 1050, 100);
            ev("serial", 1000, 100);
            ev("serial", 2000, 100);
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");
        dataframe::DataFrame b =
            View::from_file(gz, idx)
                .time_range(900, 2200)
                .occ_cell(5)
                .group_by({GroupKey::name()})
                .agg({{AggOp::Count, "", "n"},
                      {AggOp::Sum, "dur", "sum_dur"},
                      {AggOp::Busy, "", "busy"},
                      {AggOp::Concurrency, "", "concurrency"},
                      {AggOp::Utilization, "", "utilization"}})
                .collect()
                .collect()
                .get();

        REQUIRE(bhas(b, "busy_cell_us"));
        auto row_of = [&](const std::string& nm) {
            for (std::int64_t i = 0; i < b.num_rows(); ++i)
                if (bstr(b, i, "name") == nm) return i;
            return static_cast<std::int64_t>(-1);
        };
        const std::int64_t ov = row_of("overlap");
        const std::int64_t se = row_of("serial");
        REQUIRE(ov >= 0);
        REQUIRE(se >= 0);

        CHECK(bnum(b, ov, "busy_cell_us") == 5);
        // overlap: exact union == 150 (< sum(dur) == 200); the old bitmap
        // regression drifted this toward 200 as input grew.
        CHECK(bnum(b, ov, "sum_dur") == 200);
        CHECK(bnum(b, ov, "busy") == 150);
        CHECK(bnum(b, ov, "concurrency") == doctest::Approx(200.0 / 150.0));
        CHECK(bnum(b, ov, "utilization") == doctest::Approx(1.0));
        // serial: disjoint, so busy == sum(dur) and concurrency == 1.
        CHECK(bnum(b, se, "busy") == 200);
        CHECK(bnum(b, se, "concurrency") == doctest::Approx(1.0));

        for (std::int64_t i = 0; i < b.num_rows(); ++i) {
            CHECK(bnum(b, i, "busy") <= bnum(b, i, "sum_dur"));
            CHECK(bnum(b, i, "concurrency") >= 1.0 - 1e-9);
            CHECK(bnum(b, i, "utilization") <= 1.0 + 1e-9);
        }
    }

    // Exact interval union of half-open [ts, ts+dur) events.
    static std::uint64_t exact_union(
        const std::vector<std::pair<long, long>>& iv) {
        std::vector<std::pair<long, long>> spans;
        for (const auto& [ts, dur] : iv) spans.emplace_back(ts, ts + dur);
        std::sort(spans.begin(), spans.end());
        std::uint64_t total = 0;
        long cur_s = 0, cur_e = 0;
        bool open = false;
        for (const auto& [s, e] : spans) {
            if (!open) {
                cur_s = s;
                cur_e = e;
                open = true;
            } else if (s <= cur_e) {
                if (e > cur_e) cur_e = e;
            } else {
                total += static_cast<std::uint64_t>(cur_e - cur_s);
                cur_s = s;
                cur_e = e;
            }
        }
        if (open) total += static_cast<std::uint64_t>(cur_e - cur_s);
        return total;
    }

    static void write_occ_trace(const std::string& pfw,
                                const std::vector<std::pair<long, long>>& iv,
                                const char* name = "e") {
        std::ofstream ofs(pfw);
        for (const auto& [ts, dur] : iv) {
            ofs << R"({"ph":"X","name":")" << name
                << R"(","cat":"POSIX","pid":1,"tid":1,"ts":)" << ts
                << R"(,"dur":)" << dur << R"(,"args":{}})" << "\n";
        }
    }

    TEST_CASE("View - occupancy busy is the exact union, not sum(dur)") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // Heavy overlap: many events over the same window; the old per-bucket
        // coverage bitmap coarsened cells on a wide window and drifted busy
        // toward sum(dur). The exact delta-map sweep must not.
        std::vector<std::pair<long, long>> iv;
        for (int i = 0; i < 50; ++i) iv.emplace_back(1000 + i, 500);
        const std::uint64_t expect_union = exact_union(iv);
        std::uint64_t sum_dur = 0;
        for (const auto& [s, d] : iv) sum_dur += static_cast<std::uint64_t>(d);
        REQUIRE(expect_union < sum_dur);

        std::string pfw = env.get_dir() + "/occ_dense.pfw";
        write_occ_trace(pfw, iv);
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");
        dataframe::DataFrame b = View::from_file(gz, idx)
                                     .time_range(0, 1000000)
                                     .group_by({GroupKey::name()})
                                     .agg({{AggOp::Sum, "dur", "sum_dur"},
                                           {AggOp::Busy, "", "busy"}})
                                     .collect()
                                     .collect()
                                     .get();
        REQUIRE(b.num_rows() == 1);
        CHECK(bnum(b, 0, "sum_dur") == static_cast<double>(sum_dur));
        CHECK(bnum(b, 0, "busy") == static_cast<double>(expect_union));
        CHECK(bnum(b, 0, "busy") < bnum(b, 0, "sum_dur"));
    }

    TEST_CASE("View - occupancy active is the exact peak concurrency") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // 4 events all overlapping at t=1150: peak depth is exactly 4.
        std::vector<std::pair<long, long>> iv = {
            {1000, 300}, {1050, 300}, {1100, 300}, {1149, 300}};
        const std::uint64_t expect_union = exact_union(iv);

        std::string pfw = env.get_dir() + "/occ_peak.pfw";
        write_occ_trace(pfw, iv);
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");
        dataframe::DataFrame b =
            View::from_file(gz, idx)
                .time_range(0, 1000000)
                .group_by({GroupKey::name()})
                .agg({{AggOp::Busy, "", "busy"}, {AggOp::Active, "", "active"}})
                .collect()
                .collect()
                .get();
        REQUIRE(b.num_rows() == 1);
        CHECK(bnum(b, 0, "busy") == static_cast<double>(expect_union));
        CHECK(bnum(b, 0, "active") == 4.0);
    }

    TEST_CASE(
        "View - occupancy on non-overlapping intervals is exact and serial") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::vector<std::pair<long, long>> iv = {
            {1000, 100}, {2000, 100}, {3000, 100}};
        std::uint64_t sum_dur = 300;

        std::string pfw = env.get_dir() + "/occ_serial.pfw";
        write_occ_trace(pfw, iv);
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");
        dataframe::DataFrame b = View::from_file(gz, idx)
                                     .time_range(0, 1000000)
                                     .group_by({GroupKey::name()})
                                     .agg({{AggOp::Sum, "dur", "sum_dur"},
                                           {AggOp::Busy, "", "busy"},
                                           {AggOp::Active, "", "active"}})
                                     .collect()
                                     .collect()
                                     .get();
        REQUIRE(b.num_rows() == 1);
        CHECK(bnum(b, 0, "sum_dur") == static_cast<double>(sum_dur));
        CHECK(bnum(b, 0, "busy") == static_cast<double>(sum_dur));
        CHECK(bnum(b, 0, "active") == 1.0);
    }

    TEST_CASE("View - occ_cell tolerance bounds busy over the exact union") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        // Two intervals with a small gap; a coarse cell can merge them into
        // one covered span, so busy sits in [exact, exact + cell].
        std::vector<std::pair<long, long>> iv = {{1000, 90}, {1100, 90}};
        const std::uint64_t expect_exact = exact_union(iv);

        std::string pfw = env.get_dir() + "/occ_tol.pfw";
        write_occ_trace(pfw, iv);
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        {
            dataframe::DataFrame b0 = View::from_file(gz, idx)
                                          .time_range(0, 1000000)
                                          .group_by({GroupKey::name()})
                                          .agg({{AggOp::Busy, "", "busy"}})
                                          .collect()
                                          .collect()
                                          .get();
            REQUIRE(bhas(b0, "busy_cell_us"));
            CHECK(bnum(b0, 0, "busy_cell_us") == 0);
            CHECK(bnum(b0, 0, "busy") == static_cast<double>(expect_exact));
        }
        {
            const std::uint64_t cell = 50;
            dataframe::DataFrame b1 = View::from_file(gz, idx)
                                          .time_range(0, 1000000)
                                          .occ_cell(cell)
                                          .group_by({GroupKey::name()})
                                          .agg({{AggOp::Busy, "", "busy"}})
                                          .collect()
                                          .collect()
                                          .get();
            REQUIRE(bhas(b1, "busy_cell_us"));
            CHECK(bnum(b1, 0, "busy_cell_us") == static_cast<double>(cell));
            CHECK(bnum(b1, 0, "busy") >= static_cast<double>(expect_exact));
            CHECK(bnum(b1, 0, "busy") <=
                  static_cast<double>(expect_exact + 2 * cell));
        }
    }

    TEST_CASE("View - group_by resolves any field, top-level and nested") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string pfw = env.get_dir() + "/schemaless.pfw";
        {
            std::ofstream ofs(pfw);
            auto ev = [&](const char* type, const char* host, const char* tag0,
                          int v, long ts) {
                ofs << R"({"ph":"X","name":"op","cat":"POSIX","type":")" << type
                    << R"(","pid":1,"tid":1,"ts":)" << ts
                    << R"(,"dur":10,"args":{"meta":{"host":")" << host
                    << R"("},"tags":[")" << tag0 << R"(","y"],"n":{"v":)" << v
                    << R"(}}})" << "\n";
            };
            ev("c_app", "A", "x", 7, 1000);
            ev("c_app", "A", "x", 7, 1100);
            ev("posix", "B", "z", 9, 1200);
            ev("posix", "B", "z", 9, 1300);
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");
        auto counts = [&](const std::string& col, const GroupKey& gk) {
            dataframe::DataFrame b = View::from_file(gz, idx)
                                         .group_by({gk})
                                         .agg({{AggOp::Count, "", "n"}})
                                         .collect()
                                         .collect()
                                         .get();
            std::map<std::string, double> m;
            for (std::int64_t i = 0; i < b.num_rows(); ++i)
                m[bstr(b, i, col)] = bnum(b, i, "n");
            return m;
        };

        // Top-level "type" is not a POD scalar; it resolves to the raw string.
        auto by_type = counts("type", GroupKey::field("type"));
        CHECK(by_type["c_app"] == 2);
        CHECK(by_type["posix"] == 2);

        auto by_host =
            counts("args.meta.host", GroupKey::field("args.meta.host"));
        CHECK(by_host["A"] == 2);
        CHECK(by_host["B"] == 2);

        // Bracket and dot-numeric array index agree.
        for (const char* p : {"args.tags[0]", "args.tags.0"}) {
            auto by_tag = counts(p, GroupKey::field(p));
            CHECK(by_tag["x"] == 2);
            CHECK(by_tag["z"] == 2);
        }

        dataframe::DataFrame b =
            View::from_file(gz, idx)
                .group_by({GroupKey::field("args.meta.host")})
                .agg({{AggOp::Mean, "args.n.v", "mv"}})
                .collect()
                .collect()
                .get();
        std::map<std::string, double> mv;
        for (std::int64_t i = 0; i < b.num_rows(); ++i)
            mv[bstr(b, i, "args.meta.host")] = bnum(b, i, "mv");
        CHECK(mv["A"] == doctest::Approx(7.0));
        CHECK(mv["B"] == doctest::Approx(9.0));
    }

    TEST_CASE(
        "View - a same-named arg never shadows a top-level schema field") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string pfw = env.get_dir() + "/shadow.pfw";
        {
            std::ofstream ofs(pfw);
            // Top-level type "mpi" on every record; four also carry an
            // unrelated args.type parameter (the shadowing trap).
            for (int i = 0; i < 4; ++i)
                ofs << R"({"ph":"X","name":"kv","cat":"MPI","type":"mpi","pid":1,"tid":1,"ts":)"
                    << (1000 + i) << R"(,"dur":5,"args":{"type":)" << (i + 1)
                    << R"(}})" << "\n";
            for (int i = 0; i < 6; ++i)
                ofs << R"({"ph":"X","name":"send","cat":"MPI","type":"mpi","pid":1,"tid":1,"ts":)"
                    << (2000 + i) << R"(,"dur":5,"args":{"count":10}})" << "\n";
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");
        auto counts = [&](const std::string& col, const GroupKey& gk) {
            dataframe::DataFrame b = View::from_file(gz, idx)
                                         .group_by({gk})
                                         .agg({{AggOp::Count, "", "n"}})
                                         .collect()
                                         .collect()
                                         .get();
            std::map<std::string, double> m;
            for (std::int64_t i = 0; i < b.num_rows(); ++i)
                m[bstr(b, i, col)] = bnum(b, i, "n");
            return m;
        };

        // Bare "type" is the top-level field for every record, not the param.
        auto by_type = counts("type", GroupKey::field("type"));
        CHECK(by_type["mpi"] == 10);
        CHECK(by_type.count("1") == 0);

        // The args parameter stays reachable through the explicit paths.
        auto by_args_type = counts("args.type", GroupKey::field("args.type"));
        CHECK(by_args_type["1"] == 1);
        CHECK(by_args_type["4"] == 1);
        auto by_arg = counts("type", GroupKey::of_arg("type"));
        CHECK(by_arg["1"] == 1);
        CHECK(by_arg.count("mpi") == 0);
    }

    TEST_CASE("View - time_metric reads the CM record, no scan") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string pfw = env.get_dir() + "/time_metric.pfw";
        {
            std::ofstream ofs(pfw);
            ofs << R"({"name":"CM","ph":"M","cat":"dftracer","pid":0,"tid":0,)"
                   R"("args":{"name":"time_metric","value":"NS"}})"
                << "\n";
            int ts = 1000;
            for (int i = 0; i < 20; ++i) {
                ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,)"
                       R"("tid":10,"ts":)"
                    << ts << R"(,"dur":5,"args":{}})" << "\n";
                ts += 100;
            }
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        CHECK(View::from_file(gz, idx).time_metric() == trace::TimeMetric::NS);
    }

    TEST_CASE("View - agg engine path matches the GroupMap path") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string pfw = env.get_dir() + "/agg_engine.pfw";
        {
            std::ofstream ofs(pfw);
            const char* names[] = {"read", "write", "open"};
            const char* cats[] = {"POSIX", "STDIO"};
            const int pids[] = {1, 2, 3};
            const int tids[] = {10, 20};
            int ts = 1000;
            for (int i = 0; i < 90; ++i) {
                ofs << R"({"ph":"X","name":")" << names[i % 3] << R"(","cat":")"
                    << cats[i % 2] << R"(","pid":)" << pids[i % 3]
                    << R"(,"tid":)" << tids[i % 2] << R"(,"ts":)" << ts
                    << R"(,"dur":)" << (5 + (i % 17)) << R"(,"args":{}})"
                    << "\n";
                ts += 100;
            }
        }
        std::string gz = pfw + ".gz";
        dftu_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        // Drive the two internal collection paths directly on the built
        // View's plan, bypassing View::collect_frame's routing entirely.
        namespace detail = dftracer::utils::trace::views::detail;
        auto collect_engine = [](const View& v) {
            dftracer::utils::Runtime rt;
            dataframe::DataFrame result;
            rt.run_blocking(
                "agg-engine",
                [&](dftracer::utils::CoroScope&)
                    -> dftracer::utils::coro::CoroTask<void> {
                    result = co_await detail::run_collect_via_engine(v.plan());
                });
            return detail::apply_agg_post_ops(std::move(result), v.plan());
        };
        auto collect_groupmap = [](const View& v) {
            dftracer::utils::Runtime rt;
            dataframe::DataFrame result;
            rt.run_blocking(
                "groupmap",
                [&](dftracer::utils::CoroScope&)
                    -> dftracer::utils::coro::CoroTask<void> {
                    detail::GroupMap m = co_await detail::run_collect(v.plan());
                    result = detail::finalize_collect_batch(m, v.plan());
                });
            return detail::apply_agg_post_ops(std::move(result), v.plan());
        };

        auto check_match = [](const dataframe::DataFrame& a0,
                              const dataframe::DataFrame& b0,
                              const std::string& key) {
            dataframe::DataFrame a = a0.sort_by(key, false);
            dataframe::DataFrame b = b0.sort_by(key, false);
            REQUIRE(a.names.size() == b.names.size());
            for (std::size_t i = 0; i < a.names.size(); ++i) {
                CHECK(a.names[i] == b.names[i]);
                CHECK(a.columns[i].type() == b.columns[i].type());
            }
            REQUIRE(a.num_rows() == b.num_rows());
            for (std::int64_t r = 0; r < a.num_rows(); ++r)
                for (const auto& name : a.names) {
                    const auto c = static_cast<std::size_t>(bcol(a, name));
                    if (a.columns[c].type() == dataframe::TypeId::String)
                        CHECK(bstr(a, r, name) == bstr(b, r, name));
                    else
                        CHECK(bnum(a, r, name) ==
                              doctest::Approx(bnum(b, r, name)));
                }
        };

        auto run_both_file = [&](const std::string& file,
                                 const std::string& file_idx,
                                 const GroupKey& gk, const std::string& key_col,
                                 std::uint64_t mem_budget) {
            auto build = [&] {
                View v = View::from_file(file, file_idx);
                if (mem_budget) v = v.memory_budget(mem_budget);
                return v.group_by({gk}).agg({
                    {AggOp::Count, "", "n"},
                    {AggOp::Sum, "dur", "sum_dur"},
                    {AggOp::Mean, "dur", "mean_dur"},
                    {AggOp::Min, "dur", "min_dur"},
                    {AggOp::Max, "dur", "max_dur"},
                    {AggOp::Var, "dur", "var_dur"},
                    {AggOp::Std, "dur", "std_dur"},
                    {AggOp::Pct, "dur", "p90_dur", "", 0.9},
                    {AggOp::ArgMax, "name", "top_name", "dur"},
                    {AggOp::SetUnion, "cat", "cats"},
                });
            };
            dataframe::DataFrame legacy = collect_groupmap(build());
            dataframe::DataFrame engine = collect_engine(build());
            check_match(legacy, engine, key_col);
        };
        auto run_both = [&](const GroupKey& gk, const std::string& key_col,
                            std::uint64_t mem_budget) {
            run_both_file(gz, idx, gk, key_col, mem_budget);
        };

        SUBCASE("group_by name") { run_both(GroupKey::name(), "name", 0); }
        SUBCASE("group_by pid") { run_both(GroupKey::pid(), "pid", 0); }
        SUBCASE("group_by tid") { run_both(GroupKey::tid(), "tid", 0); }
        SUBCASE("group_by name, forced spill") {
            run_both(GroupKey::name(), "name", 128);
        }
        // io_cat is a computed key: the event name maps to the dfanalyzer I/O
        // category enum (read -> READ, write -> WRITE, open -> METADATA), so
        // both paths must derive and label the same integer categories.
        SUBCASE("group_by io_cat") {
            run_both(GroupKey::io_cat(), "io_cat", 0);
        }
        SUBCASE("group_by io_cat, forced spill") {
            run_both(GroupKey::io_cat(), "io_cat", 128);
        }

        // The engine path must try the no-scan tier/rollup fast path before
        // scanning: materialize a finer (cat, name) rollup via the legacy
        // path, then run the coarser (cat) query through the engine. A hit
        // re-aggregates the persisted rollup (find_subsuming_rollup), never
        // rescanning the trace, and must match a fresh (pre-rollup) scan.
        SUBCASE("engine path is served by a subsuming rollup, not a rescan") {
            auto fine = [&] {
                return View::from_file(gz, idx)
                    .group_by({GroupKey::cat(), GroupKey::name()})
                    .agg({{AggOp::Count, "", "n"},
                          {AggOp::Sum, "dur", "sum_dur"}});
            };
            auto coarse = [&] {
                return View::from_file(gz, idx)
                    .group_by({GroupKey::cat()})
                    .agg({{AggOp::Count, "", "n"},
                          {AggOp::Sum, "dur", "sum_dur"}});
            };

            dataframe::DataFrame expect = collect_groupmap(coarse());
            fine().run().get();  // materialize only the finer rollup
            dataframe::DataFrame engine_served = collect_engine(coarse());
            check_match(expect, engine_served, "cat");
        }

        // Composite (multi-dim) direct-column keys: pairs rows by their
        // composite key text (exact, not Approx) so a differently-ordered
        // group set from the two paths still compares row for row.
        auto check_match_multi = [](const dataframe::DataFrame& a0,
                                    const dataframe::DataFrame& b0,
                                    const std::vector<std::string>& keys) {
            REQUIRE(a0.names.size() == b0.names.size());
            for (std::size_t i = 0; i < a0.names.size(); ++i) {
                CHECK(a0.names[i] == b0.names[i]);
                CHECK(a0.columns[i].type() == b0.columns[i].type());
            }
            REQUIRE(a0.num_rows() == b0.num_rows());
            auto sort_key = [&](const dataframe::DataFrame& df,
                                std::int64_t r) {
                std::string s;
                for (const auto& k : keys) {
                    s += bstr(df, r, k);
                    s += '\x1f';
                }
                return s;
            };
            std::vector<std::pair<std::string, std::int64_t>> ar, br;
            for (std::int64_t r = 0; r < a0.num_rows(); ++r)
                ar.emplace_back(sort_key(a0, r), r);
            for (std::int64_t r = 0; r < b0.num_rows(); ++r)
                br.emplace_back(sort_key(b0, r), r);
            std::sort(ar.begin(), ar.end());
            std::sort(br.begin(), br.end());
            for (std::size_t i = 0; i < ar.size(); ++i)
                for (const auto& name : a0.names) {
                    const auto c = static_cast<std::size_t>(bcol(a0, name));
                    if (a0.columns[c].type() == dataframe::TypeId::String)
                        CHECK(bstr(a0, ar[i].second, name) ==
                              bstr(b0, br[i].second, name));
                    else
                        CHECK(bnum(a0, ar[i].second, name) ==
                              doctest::Approx(bnum(b0, br[i].second, name)));
                }
        };

        auto run_both_multi = [&](std::vector<ViewFile> files,
                                  const std::vector<GroupKey>& gks,
                                  const std::vector<std::string>& key_cols,
                                  std::uint64_t mem_budget) {
            auto build = [&] {
                View v = View::from_files(files);
                if (mem_budget) v = v.memory_budget(mem_budget);
                return v.group_by(gks).agg({
                    {AggOp::Count, "", "n"},
                    {AggOp::Sum, "dur", "sum_dur"},
                    {AggOp::Mean, "dur", "mean_dur"},
                    {AggOp::Var, "dur", "var_dur"},
                    {AggOp::Pct, "dur", "p90_dur", "", 0.9},
                    {AggOp::ArgMax, "name", "top_name", "dur"},
                });
            };
            dataframe::DataFrame legacy = collect_groupmap(build());
            dataframe::DataFrame engine = collect_engine(build());
            check_match_multi(legacy, engine, key_cols);
        };

        SUBCASE("group_by (name, pid)") {
            run_both_multi({{gz, idx}}, {GroupKey::name(), GroupKey::pid()},
                           {"name", "pid"}, 0);
        }
        SUBCASE("group_by (name, pid, tid), forced spill") {
            run_both_multi({{gz, idx}},
                           {GroupKey::name(), GroupKey::pid(), GroupKey::tid()},
                           {"name", "pid", "tid"}, 128);
        }
        SUBCASE("group_by (cat, io_cat)") {
            run_both_multi({{gz, idx}}, {GroupKey::cat(), GroupKey::io_cat()},
                           {"cat", "io_cat"}, 0);
        }
        SUBCASE("group_by (cat, io_cat), forced spill") {
            run_both_multi({{gz, idx}}, {GroupKey::cat(), GroupKey::io_cat()},
                           {"cat", "io_cat"}, 128);
        }

        // Two files, so fhash genuinely varies across groups (not just pid).
        std::string gz2, idx2;
        {
            std::string pfw2 = env.get_dir() + "/agg_engine2.pfw";
            std::ofstream ofs(pfw2);
            const char* names[] = {"read", "write", "open"};
            const int pids[] = {1, 2, 3};
            int ts = 1000;
            for (int i = 0; i < 60; ++i) {
                ofs << R"({"ph":"X","name":")" << names[i % 3]
                    << R"(","cat":"POSIX","pid":)" << pids[i % 3]
                    << R"(,"tid":10,"ts":)" << ts << R"(,"dur":)"
                    << (5 + (i % 11)) << R"(,"args":{}})" << "\n";
                ts += 100;
            }
            ofs.close();
            gz2 = pfw2 + ".gz";
            dftu_utils_test::compress_file_to_gzip(pfw2, gz2);
            fs::remove(pfw2);
            idx2 = determine_index_path(gz2, "");
        }
        SUBCASE("group_by (pid, fhash) across two files") {
            run_both_multi({{gz, idx}, {gz2, idx2}},
                           {GroupKey::pid(), GroupKey::fhash()},
                           {"pid", "fhash"}, 0);
        }
        SUBCASE("group_by (pid, fhash) across two files, forced spill") {
            run_both_multi({{gz, idx}, {gz2, idx2}},
                           {GroupKey::pid(), GroupKey::fhash()},
                           {"pid", "fhash"}, 128);
        }

        // Mixed-case cat values: the GroupMap path lowercases the group key
        // (agg_fold.h's lower_ascii), so "POSIX"/"posix"/"Stdio"/"STDIO" must
        // merge into two groups ("posix", "stdio") in both paths.
        std::string gz3, idx3;
        {
            std::string pfw3 = env.get_dir() + "/agg_engine_cat.pfw";
            std::ofstream ofs(pfw3);
            const char* names[] = {"read", "write", "open"};
            const char* cats[] = {"POSIX", "posix", "Stdio", "STDIO"};
            const int pids[] = {1, 2};
            int ts = 1000;
            for (int i = 0; i < 80; ++i) {
                ofs << R"({"ph":"X","name":")" << names[i % 3] << R"(","cat":")"
                    << cats[i % 4] << R"(","pid":)" << pids[i % 2]
                    << R"(,"tid":10,"ts":)" << ts << R"(,"dur":)"
                    << (5 + (i % 13)) << R"(,"args":{}})" << "\n";
                ts += 100;
            }
            ofs.close();
            gz3 = pfw3 + ".gz";
            dftu_utils_test::compress_file_to_gzip(pfw3, gz3);
            fs::remove(pfw3);
            idx3 = determine_index_path(gz3, "");
        }
        SUBCASE("group_by cat, mixed case merges") {
            run_both_file(gz3, idx3, GroupKey::cat(), "cat", 0);
        }
        SUBCASE("group_by cat, mixed case merges, forced spill") {
            run_both_file(gz3, idx3, GroupKey::cat(), "cat", 128);
        }
        SUBCASE("group_by (cat, pid), mixed case merges") {
            run_both_multi({{gz3, idx3}}, {GroupKey::cat(), GroupKey::pid()},
                           {"cat", "pid"}, 0);
        }
        SUBCASE("group_by (cat, pid), mixed case merges, forced spill") {
            run_both_multi({{gz3, idx3}}, {GroupKey::cat(), GroupKey::pid()},
                           {"cat", "pid"}, 128);
        }

        // Resolved-name keys (FilePath/FileName/HostName): each file declares
        // its own fhash/hhash via FH/HH metadata and every event references it,
        // so the two files genuinely resolve to different names (distinct
        // basenames, so FileName has no cross-file collision to merge).
        std::string gz4, idx4, gz5, idx5;
        {
            std::string pfw4 = env.get_dir() + "/agg_engine_resolved_a.pfw";
            std::ofstream ofs(pfw4);
            ofs << R"({"name":"FH","cat":"dftracer","pid":1,"tid":1,"ph":"M","args":{"name":"/data/dirA/a.h5","value":"FA1"}})"
                << "\n"
                << R"({"name":"HH","cat":"dftracer","pid":1,"tid":1,"ph":"M","args":{"name":"nodeA","value":"HA1"}})"
                << "\n";
            const char* names[] = {"read", "write", "open"};
            const int pids[] = {1, 2};
            int ts = 1000;
            for (int i = 0; i < 30; ++i) {
                ofs << R"({"ph":"X","name":")" << names[i % 3]
                    << R"(","cat":"POSIX","pid":)" << pids[i % 2]
                    << R"(,"tid":10,"ts":)" << ts << R"(,"dur":)"
                    << (5 + (i % 11))
                    << R"(,"args":{"fhash":"FA1","hhash":"HA1"}})" << "\n";
                ts += 100;
            }
            ofs.close();
            gz4 = pfw4 + ".gz";
            dftu_utils_test::compress_file_to_gzip(pfw4, gz4);
            fs::remove(pfw4);
            idx4 = determine_index_path(gz4, "");

            std::string pfw5 = env.get_dir() + "/agg_engine_resolved_b.pfw";
            std::ofstream ofs5(pfw5);
            ofs5
                << R"({"name":"FH","cat":"dftracer","pid":1,"tid":1,"ph":"M","args":{"name":"/data/dirB/b.h5","value":"FB1"}})"
                << "\n"
                << R"({"name":"HH","cat":"dftracer","pid":1,"tid":1,"ph":"M","args":{"name":"nodeB","value":"HB1"}})"
                << "\n";
            ts = 1000;
            for (int i = 0; i < 20; ++i) {
                ofs5 << R"({"ph":"X","name":")" << names[i % 3]
                     << R"(","cat":"POSIX","pid":)" << pids[i % 2]
                     << R"(,"tid":10,"ts":)" << ts << R"(,"dur":)"
                     << (5 + (i % 7))
                     << R"(,"args":{"fhash":"FB1","hhash":"HB1"}})" << "\n";
                ts += 100;
            }
            ofs5.close();
            gz5 = pfw5 + ".gz";
            dftu_utils_test::compress_file_to_gzip(pfw5, gz5);
            fs::remove(pfw5);
            idx5 = determine_index_path(gz5, "");
        }
        SUBCASE("group_by file_path across two files") {
            run_both_multi({{gz4, idx4}, {gz5, idx5}}, {GroupKey::file_path()},
                           {"file_path"}, 0);
        }
        SUBCASE("group_by file_path across two files, forced spill") {
            run_both_multi({{gz4, idx4}, {gz5, idx5}}, {GroupKey::file_path()},
                           {"file_path"}, 128);
        }
        SUBCASE("group_by file_name across two files") {
            run_both_multi({{gz4, idx4}, {gz5, idx5}}, {GroupKey::file_name()},
                           {"file_name"}, 0);
        }
        SUBCASE("group_by host_name across two files") {
            run_both_multi({{gz4, idx4}, {gz5, idx5}}, {GroupKey::host_name()},
                           {"host_name"}, 0);
        }
        SUBCASE("group_by (file_path, pid) across two files") {
            run_both_multi({{gz4, idx4}, {gz5, idx5}},
                           {GroupKey::file_path(), GroupKey::pid()},
                           {"file_path", "pid"}, 0);
        }
        SUBCASE("group_by (file_path, pid) across two files, forced spill") {
            run_both_multi({{gz4, idx4}, {gz5, idx5}},
                           {GroupKey::file_path(), GroupKey::pid()},
                           {"file_path", "pid"}, 128);
        }

        // Rank is a resolved-like key harvested from PR metadata: each pid is
        // declared once (pid -> rank), events group on pid, and both paths must
        // relabel the pid groups to the same rank strings.
        std::string gz6, idx6;
        {
            std::string pfw6 = env.get_dir() + "/agg_engine_rank.pfw";
            std::ofstream ofs(pfw6);
            ofs << R"({"ph":"M","name":"PR","cat":"dftracer","pid":100,"tid":0,"args":{"name":"rank","value":"0"}})"
                << "\n"
                << R"({"ph":"M","name":"PR","cat":"dftracer","pid":200,"tid":0,"args":{"name":"rank","value":"1"}})"
                << "\n"
                << R"({"ph":"M","name":"PR","cat":"dftracer","pid":300,"tid":0,"args":{"name":"rank","value":"2"}})"
                << "\n";
            const char* names[] = {"read", "write", "open"};
            const int pids[] = {100, 200, 300};
            int ts = 1000;
            for (int i = 0; i < 90; ++i) {
                ofs << R"({"ph":"X","name":")" << names[i % 3]
                    << R"(","cat":"POSIX","pid":)" << pids[i % 3]
                    << R"(,"tid":10,"ts":)" << ts << R"(,"dur":)"
                    << (5 + (i % 17)) << R"(,"args":{}})" << "\n";
                ts += 100;
            }
            ofs.close();
            gz6 = pfw6 + ".gz";
            dftu_utils_test::compress_file_to_gzip(pfw6, gz6);
            fs::remove(pfw6);
            idx6 = determine_index_path(gz6, "");
        }
        SUBCASE("group_by rank") {
            run_both_file(gz6, idx6, GroupKey::rank(), "rank", 0);
        }
        SUBCASE("group_by rank, forced spill") {
            run_both_file(gz6, idx6, GroupKey::rank(), "rank", 128);
        }

        // An Arg/Field key derives a group-key column from a flattened arg (or,
        // for Field, a top-level-then-arg lookup); both paths must render the
        // same key text, including "" for the events missing the arg. The
        // read events carry arg x, the write events carry arg y, so grouping by
        // x exercises the present-and-missing split.
        std::string gz7, idx7;
        {
            std::string pfw7 = env.get_dir() + "/agg_engine_arg.pfw";
            std::ofstream ofs(pfw7);
            int ts = 1000;
            for (int i = 0; i < 30; ++i) {
                ofs << R"({"ph":"X","name":"read","cat":"POSIX","pid":1,)"
                       R"("tid":10,"ts":)"
                    << ts << R"(,"dur":)" << (5 + (i % 11))
                    << R"(,"args":{"x":)" << (i % 5) << R"(}})" << "\n";
                ts += 100;
            }
            for (int i = 0; i < 30; ++i) {
                ofs << R"({"ph":"X","name":"write","cat":"STDIO","pid":2,)"
                       R"("tid":20,"ts":)"
                    << ts << R"(,"dur":)" << (5 + (i % 13))
                    << R"(,"args":{"y":)" << (i % 4) << R"(}})" << "\n";
                ts += 100;
            }
            ofs.close();
            gz7 = pfw7 + ".gz";
            dftu_utils_test::compress_file_to_gzip(pfw7, gz7);
            fs::remove(pfw7);
            idx7 = determine_index_path(gz7, "");
        }
        SUBCASE("group_by arg x") {
            run_both_file(gz7, idx7, GroupKey::of_arg("x"), "x", 0);
        }
        SUBCASE("group_by arg x, forced spill") {
            run_both_file(gz7, idx7, GroupKey::of_arg("x"), "x", 128);
        }
        SUBCASE("group_by field name") {
            run_both_file(gz7, idx7, GroupKey::field("name"), "name", 0);
        }
        SUBCASE("group_by field x (arg via field)") {
            run_both_file(gz7, idx7, GroupKey::field("x"), "x", 0);
        }
        SUBCASE("group_by (name, arg x)") {
            run_both_multi({{gz7, idx7}},
                           {GroupKey::name(), GroupKey::of_arg("x")},
                           {"name", "x"}, 0);
        }
        SUBCASE("group_by (name, arg x), forced spill") {
            run_both_multi({{gz7, idx7}},
                           {GroupKey::name(), GroupKey::of_arg("x")},
                           {"name", "x"}, 128);
        }

        // auto_numeric_metrics discovers the numeric args at scan (gz7's read
        // events carry x, write events carry y), so both the bare-mean legacy
        // path and explicit per-arg reductions must emit the same discovered
        // columns (x/y), with 0.0 where an arg never appears in a group.
        // Grouped by name so read has x-present/y-absent and write the reverse.
        auto run_both_dyn =
            [&](const std::function<AggregatedView(View)>& agg_of,
                std::uint64_t mem_budget) {
                auto build = [&] {
                    View v = View::from_file(gz7, idx7);
                    if (mem_budget) v = v.memory_budget(mem_budget);
                    return agg_of(v.group_by({GroupKey::name()}));
                };
                check_match(collect_groupmap(build()), collect_engine(build()),
                            "name");
            };
        SUBCASE("group_by name + auto_numeric_metrics (legacy bare mean)") {
            run_both_dyn([](View v) { return v.agg_numeric_args(); }, 0);
        }
        SUBCASE("group_by name + auto_numeric_metrics, forced spill") {
            run_both_dyn([](View v) { return v.agg_numeric_args(); }, 128);
        }
        SUBCASE("group_by name + explicit numeric_arg_aggs") {
            run_both_dyn(
                [](View v) {
                    return v.agg_numeric_args({
                        AggSpec(AggOp::Sum),
                        AggSpec(AggOp::Mean),
                        AggSpec(AggOp::Min),
                        AggSpec(AggOp::Max),
                        AggSpec(AggOp::SumSq),
                        AggSpec(AggOp::Var),
                        AggSpec(AggOp::Std),
                        AggSpec(AggOp::Skew),
                        AggSpec(AggOp::Kurt),
                        AggSpec(AggOp::Pct, "", "p90", "", 0.9),
                    });
                },
                0);
        }
        SUBCASE("group_by name + explicit numeric_arg_aggs, forced spill") {
            run_both_dyn(
                [](View v) {
                    return v.agg_numeric_args({
                        AggSpec(AggOp::Sum),
                        AggSpec(AggOp::Mean),
                        AggSpec(AggOp::Var),
                        AggSpec(AggOp::Pct, "", "p90", "", 0.9),
                    });
                },
                128);
        }
        // A trace whose events carry no numeric args discovers nothing, so the
        // dyn path collapses to just the group count column in both paths.
        SUBCASE("group_by name + auto_numeric_metrics, no numeric args") {
            auto build = [&] {
                return View::from_file(gz, idx)
                    .group_by({GroupKey::name()})
                    .agg_numeric_args();
            };
            check_match(collect_groupmap(build()), collect_engine(build()),
                        "name");
        }

        // time_bucket is a computed key too: the bucket column goes first
        // (agg_fold.h prepends it before plan.group_by), so pair rows by
        // [time_bucket, ...extra_key_cols] the same way check_match_multi
        // pairs any other composite key. `bucket_of` applies time_scale/
        // time_bucket(_min); group_by (if any) runs first, matching the
        // group_by-then-time_bucket order used elsewhere in this file.
        auto run_both_bucket =
            [&](const std::function<View(View)>& bucket_of,
                const std::vector<GroupKey>& extra_gks,
                const std::vector<std::string>& extra_key_cols,
                std::uint64_t mem_budget) {
                auto build = [&] {
                    View v = View::from_file(gz, idx);
                    if (mem_budget) v = v.memory_budget(mem_budget);
                    View grouped =
                        extra_gks.empty() ? v : v.group_by(extra_gks);
                    return bucket_of(grouped).agg({
                        {AggOp::Count, "", "n"},
                        {AggOp::Sum, "dur", "sum_dur"},
                        {AggOp::Mean, "dur", "mean_dur"},
                    });
                };
                // Warm the on-disk index once so legacy and engine below see
                // the same (already-built) index: bucket_origin_min reads its
                // zone maps and falls back to origin 0 on a first-touch/missing
                // index, so comparing a first-touch run against a second-touch
                // run would compare two different origins, not the same
                // formula.
                build().collect().collect().get();
                dataframe::DataFrame legacy = collect_groupmap(build());
                dataframe::DataFrame engine = collect_engine(build());
                std::vector<std::string> keys = {"time_bucket"};
                keys.insert(keys.end(), extra_key_cols.begin(),
                            extra_key_cols.end());
                check_match_multi(legacy, engine, keys);
            };

        SUBCASE("time_bucket alone") {
            run_both_bucket([](View v) { return v.time_bucket(1000); }, {}, {},
                            0);
        }
        SUBCASE("time_bucket alone, forced spill") {
            run_both_bucket([](View v) { return v.time_bucket(1000); }, {}, {},
                            128);
        }
        SUBCASE("time_bucket + name") {
            run_both_bucket([](View v) { return v.time_bucket(1000); },
                            {GroupKey::name()}, {"name"}, 0);
        }
        SUBCASE("time_bucket + name, forced spill") {
            run_both_bucket([](View v) { return v.time_bucket(1000); },
                            {GroupKey::name()}, {"name"}, 128);
        }
        SUBCASE("time_bucket with an explicit origin") {
            run_both_bucket([](View v) { return v.time_bucket(700, 500); }, {},
                            {}, 0);
        }
        SUBCASE("time_bucket_min (trace-min-aligned origin)") {
            run_both_bucket([](View v) { return v.time_bucket_min(700); }, {},
                            {}, 0);
        }
        SUBCASE("time_bucket with a non-1.0 time_scale") {
            run_both_bucket(
                [](View v) { return v.time_scale(0.01).time_bucket(10); }, {},
                {}, 0);
        }

        auto build_postop_base = [&] {
            return View::from_file(gz, idx)
                .group_by({GroupKey::name()})
                .agg({
                    {AggOp::Count, "", "n"},
                    {AggOp::Sum, "dur", "sum_dur"},
                });
        };
        SUBCASE("group_by name + sort_by") {
            auto build = [&] { return build_postop_base().sort_by("sum_dur"); };
            check_match(collect_groupmap(build()), collect_engine(build()),
                        "name");
        }
        SUBCASE("group_by name + topk") {
            auto build = [&] { return build_postop_base().topk("sum_dur", 2); };
            dataframe::DataFrame legacy = collect_groupmap(build());
            dataframe::DataFrame engine = collect_engine(build());
            REQUIRE(legacy.num_rows() == 2);
            check_match(legacy, engine, "name");
        }
        SUBCASE("group_by name + offset/limit") {
            auto build = [&] {
                return build_postop_base().sort_by("name").offset(1).limit(1);
            };
            dataframe::DataFrame legacy = collect_groupmap(build());
            dataframe::DataFrame engine = collect_engine(build());
            REQUIRE(legacy.num_rows() == 1);
            check_match(legacy, engine, "name");
        }
        SUBCASE("group_by name + select") {
            auto build = [&] {
                return build_postop_base().select({"name", "sum_dur"});
            };
            dataframe::DataFrame legacy = collect_groupmap(build());
            dataframe::DataFrame engine = collect_engine(build());
            REQUIRE(legacy.names.size() == 2);
            check_match(legacy, engine, "name");
        }
    }
}
