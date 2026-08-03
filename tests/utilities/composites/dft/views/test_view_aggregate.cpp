#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/rocksdb/column_families.h>
#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/utilities/composites/dft/views/rollup_store.h>
#include <doctest/doctest.h>

#include "test_view_common.h"

TEST_SUITE("View") {
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
                     .get();
        REQUIRE(t.text_columns.size() == 1);
        CHECK(t.text_columns[0] == "names");
        std::map<std::string, std::string> got;
        for (const auto& r : t.rows) got[r.keys[0]] = r.texts[0];
        CHECK(got["posix"] == "read");
        CHECK(got["stdio"] == "fwrite");

        // No grouping: one row unioning both names, sorted + SET_SEP-joined.
        auto whole = View::from_file(gz, idx)
                         .agg({{AggOp::SetUnion, "name", "names"}})
                         .collect()
                         .get();
        REQUIRE(whole.rows.size() == 1);
        const std::string joined = whole.rows[0].texts[0];
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
        REQUIRE(merged.rows.size() == 1);
        CHECK(merged.rows[0].texts[0] == joined);

        // Rollup round-trip: materialize persists the set, a repeat reads it
        // back.
        auto make = [&]() {
            return View::from_file(gz, idx)
                .group_by({GroupKey::cat()})
                .agg({{AggOp::SetUnion, "name", "names"}});
        };
        make().materialize().run().get();
        auto back = make().collect().get();
        std::map<std::string, std::string> rb;
        for (const auto& r : back.rows) rb[r.keys[0]] = r.texts[0];
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
        // 200 POSIX events, dur = 10..209 (one cat group). Population variance
        // of 200 consecutive ints is (200^2 - 1)/12 = 3333.25; mean 109.5.
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
                         .get();
            REQUIRE(t.rows.size() == 1);
            return t.rows[0].values;  // [mean, var, std, skew, kurt]
        };

        auto in_mem = row(0);
        CHECK(in_mem[0] == doctest::Approx(109.5));
        CHECK(in_mem[1] == doctest::Approx(3333.25));
        CHECK(in_mem[2] == doctest::Approx(57.7343));
        // 200 consecutive ints: symmetric (skew 0), platykurtic (excess ~
        // -1.2).
        CHECK(in_mem[3] == doctest::Approx(0.0).epsilon(1e-6));
        CHECK(in_mem[4] == doctest::Approx(-1.2).epsilon(0.01));
        CHECK(row(1) == in_mem);  // spill + k-way merge (m3/m4) is identical
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
                         .get();
            REQUIRE(t.rows.size() == 1);
            return t.rows[0].values[0];
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
                         .get();
            REQUIRE(t.rows.size() == 1);
            REQUIRE(t.hist_columns.size() == 1);
            REQUIRE(t.rows[0].hists.size() == 1);
            return t.rows[0].hists[0];
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
                         .get();
            std::vector<std::string> rows;
            for (const auto& r : t.rows) {
                std::string s;
                for (const auto& k : r.keys) s += k + "|";
                for (double v : r.values) s += std::to_string(v) + "|";
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

        auto canon = [](const ResultTable& t) {
            std::vector<std::string> rows;
            for (const auto& r : t.rows) {
                std::string s;
                for (const auto& k : r.keys) s += k + "|";
                for (double v : r.values) s += std::to_string(v) + "|";
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

        auto expect = canon(make().collect().get());
        REQUIRE(expect.size() >= 1);

        namespace detail =
            dftracer::utils::utilities::composites::dft::views::detail;
        namespace rdb = dftracer::utils::rocksdb;

        // materialize() persists the result into the index's ROLLUP CF as a
        // byproduct of answering it.
        CHECK(canon(make().materialize().collect().get()) == expect);
        {
            auto db = detail::open_rollup_db(
                idx, rdb::RocksDatabase::OpenMode::ReadOnly);
            REQUIRE(db);
            auto it = db->new_iterator(rdb::cf::ROLLUP);
            it->SeekToFirst();
            CHECK(it->Valid());  // the ROLLUP CF holds the materialized view
        }

        // A repeat query - even without materialize() - reads the rollup back.
        CHECK(canon(make().collect().get()) == expect);
    }

    TEST_CASE("View - run() materializes the rollup without a table") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        namespace detail =
            dftracer::utils::utilities::composites::dft::views::detail;
        namespace rdb = dftracer::utils::rocksdb;

        auto canon = [](const ResultTable& t) {
            std::vector<std::string> rows;
            for (const auto& r : t.rows) {
                std::string s;
                for (const auto& k : r.keys) s += k + "|";
                for (double v : r.values) s += std::to_string(v) + "|";
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

        auto expect = canon(make().collect().get());  // fresh, no rollup yet
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
        CHECK(canon(make().collect().get()) == expect);  // now reads the rollup
    }

    TEST_CASE("View - a coarser query is served by rolling up a finer view") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        auto canon = [](const ResultTable& t) {
            std::vector<std::string> rows;
            for (const auto& r : t.rows) {
                std::string s;
                for (const auto& k : r.keys) s += k + "|";
                for (double v : r.values) s += std::to_string(v) + "|";
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

        auto expect = canon(coarse().collect().get());  // fresh, no rollup
        REQUIRE(expect.size() >= 1);

        fine().run().get();  // materialize only the finer rollup

        // The coarse query is answered by re-aggregating the finer rollup.
        CHECK(canon(coarse().collect().get()) == expect);
    }

    TEST_CASE(
        "View - a coarser time bucket is served by re-bucketing a finer one") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 200, 60);
        std::string idx = determine_index_path(gz, "");

        auto canon = [](const ResultTable& t) {
            std::vector<std::string> rows;
            for (const auto& r : t.rows) {
                std::string s;
                for (const auto& k : r.keys) s += k + "|";
                for (double v : r.values) s += std::to_string(v) + "|";
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
        auto expect_coarse = canon(at_bucket(1000).collect().get());
        auto expect_fine = canon(at_bucket(100).collect().get());
        REQUIRE(expect_coarse.size() >= 1);
        REQUIRE(expect_coarse.size() <= expect_fine.size());  // coarser folds

        at_bucket(100).run().get();  // materialize only the finer grain

        // The coarse query re-buckets the finer rollup; the fine query reads it
        // back exactly.
        CHECK(canon(at_bucket(1000).collect().get()) == expect_coarse);
        CHECK(canon(at_bucket(100).collect().get()) == expect_fine);
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

        auto canon = [](const ResultTable& t) {
            std::vector<std::string> rows;
            for (const auto& r : t.rows) {
                std::string s;
                for (const auto& k : r.keys) s += k + "|";
                for (double v : r.values) s += std::to_string(v) + "|";
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

        auto expect =
            canon(view(both).collect().get());  // single-node baseline
        REQUIRE(expect.size() >= 1);

        // Each rank aggregates its shard; the coordinator reduces +
        // materializes.
        std::string pa = view({{a, shared}}).aggregate_partial().get();
        std::string pb = view({{b, shared}}).aggregate_partial().get();
        view(both).materialize_partials({pa, pb}).get();

        // The full query now reads the distributed-built rollup.
        CHECK(canon(view(both).collect().get()) == expect);
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
                         .get();

        REQUIRE(table.group_columns.size() == 2);
        CHECK(table.group_columns[0] == "time_bucket");
        CHECK(table.group_columns[1] == "cat");
        // buckets 1000,2000,3000 (10 each) + 4000 (ts 4000? no, max 3900) ->
        // rows across buckets, all cat POSIX, counts sum to 30.
        double total = 0;
        for (const auto& r : table.rows) total += r.values[0];
        CHECK(total == doctest::Approx(30));
        CHECK(table.rows.size() >= 3);
    }

    TEST_CASE("View - collect with no group_by folds whole set into one row") {
        TestEnvironment env(200);
        REQUIRE(env.is_valid());
        std::string gz = create_mixed_trace(env, 30, 20);
        std::string idx = determine_index_path(gz, "");

        auto table = View::from_file(gz, idx)
                         .agg({{AggOp::Count, "", "n"}})
                         .collect()
                         .get();

        REQUIRE(table.rows.size() == 1);
        CHECK(table.rows[0].values[0] == doctest::Approx(50));
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
        dft_utils_test::compress_file_to_gzip(pfw, gz);
        fs::remove(pfw);
        std::string idx = determine_index_path(gz, "");

        auto table = View::from_file(gz, idx)
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Max, "dur", "max_dur"},
                               {AggOp::ArgMax, "name", "longest", "dur"}})
                         .collect()
                         .get();

        REQUIRE(table.rows.size() == 1);
        REQUIRE(table.text_columns.size() == 1);
        CHECK(table.text_columns[0] == "longest");
        REQUIRE(table.rows[0].texts.size() == 1);
        CHECK(table.rows[0].texts[0] == "write");
        REQUIRE(table.rows[0].values.size() == 1);
        CHECK(table.rows[0].values[0] == doctest::Approx(99));  // max_dur
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
        auto run = View::from_file(gz, idx).metadata(false).partition(
            /*num_slots=*/4);
        // custom fold: small events (dur < 25), reusing the parsed event.
        auto small = run.fold<Small>(
            Query::from_string("dur < 25").value(),
            [](Small& s, const utilities::common::json::JsonValue& jv,
               std::string_view) {
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
        // POSIX branch: 30 events, dur 10..39.
        REQUIRE(posix->rows.size() == 1);
        CHECK(posix->rows[0].values[0] == doctest::Approx(30));
        CHECK(posix->rows[0].values[1] == doctest::Approx(30 * 24.5));
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

        auto slow = base.collect().get();
        ChunkStatsSource source;
        auto fast = base.with_partial_source(&source).collect().get();

        auto row_of = [](const ResultTable& t,
                         const std::string& cat) -> const ResultRow* {
            for (const auto& r : t.rows)
                if (!r.keys.empty() && r.keys[0] == cat) return &r;
            return nullptr;
        };
        REQUIRE(slow.rows.size() == fast.rows.size());
        for (const char* c : {"posix", "stdio"}) {
            const ResultRow* s = row_of(slow, c);
            const ResultRow* f = row_of(fast, c);
            REQUIRE(s != nullptr);
            REQUIRE(f != nullptr);
            REQUIRE(s->values.size() == f->values.size());
            for (std::size_t i = 0; i < s->values.size(); ++i)
                CHECK(f->values[i] == doctest::Approx(s->values[i]));
        }
    }

    TEST_CASE("View - ChunkStatsSource reads stored per-key sketches") {
        using dftracer::utils::utilities::composites::dft::indexing::
            ChunkStatistics;
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

        namespace vdetail =
            dftracer::utils::utilities::composites::dft::views::detail;
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
        dft_utils_test::write_gz_trace(out, sink.buffer);
        std::string out_idx = determine_index_path(out, "");
        StringSink reread;
        View::from_file(out, out_idx)
            .phase(Phase::Counters)
            .metadata(false)
            .export_json(reread)
            .get();
        CHECK(reread.lines().size() == lines.size());
    }
}
