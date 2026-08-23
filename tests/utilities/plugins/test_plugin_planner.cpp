// Planner level 1: the host feeds the UNION of every plugin's plan_query into
// the scan's index prune. Covers the union-query construction (dedup, bail on a
// no-query or unparseable plugin, single-vs-multi) and the end-to-end property
// that pruning skips files no plugin wants without changing any plugin's
// result.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/host.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/view.h>
#include <doctest/doctest.h>
#include <testing_utilities.h>

#include <atomic>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::plugins::PluginHost;
using dftracer::utils::plugins::detail::plugin_union_prune_query;
using dftracer::utils::trace::internal::determine_index_path;
using View = dftracer::utils::trace::views::View;
using ViewFile = dftracer::utils::trace::views::ViewFile;
using ExportStats = dftracer::utils::trace::views::ExportStats;
using ExportSink = dftracer::utils::trace::views::ExportSink;
namespace coro = dftracer::utils::coro;

namespace {

std::vector<const char*> qs(std::vector<const char*> v) { return v; }

// A plugin that counts the events its own filter keeps. on_batch sees events
// already filtered by plan_query, so its count is invariant to host pruning;
// on_finalize (once, on the merged master slice) publishes the total.
struct CountState {
    const char* query = nullptr;
    std::atomic<std::uint64_t> total{0};
};
struct CountSlice {
    CountState* st = nullptr;
    std::uint64_t n = 0;
};

std::uint32_t count_needs(void*) { return 0; }
const char* count_plan_query(void* self) {
    return static_cast<CountState*>(self)->query;
}
void* count_make_slice(void* self) {
    auto* s = new CountSlice();
    s->st = static_cast<CountState*>(self);
    return s;
}
dftu_task* count_on_batch(void* slice, const dftu_batch* b, const dftu_host*) {
    static_cast<CountSlice*>(slice)->n += b->count;
    return nullptr;
}
void count_merge(void* into, void* other) {
    static_cast<CountSlice*>(into)->n += static_cast<CountSlice*>(other)->n;
}
dftu_task* count_on_finalize(void* slice, const dftu_host*) {
    auto* s = static_cast<CountSlice*>(slice);
    s->st->total.store(s->n);
    return nullptr;
}
void count_destroy_slice(void* slice) {
    delete static_cast<CountSlice*>(slice);
}
void count_destroy(void*) {}

dftu_plugin make_count_plugin(CountState* st) {
    dftu_plugin p{};
    p.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    p.self = st;
    p.needs = count_needs;
    p.plan_query = count_plan_query;
    p.make_slice = count_make_slice;
    p.on_batch = count_on_batch;
    p.merge = count_merge;
    p.on_finalize = count_on_finalize;
    p.destroy_slice = count_destroy_slice;
    p.destroy = count_destroy;
    return p;
}

// A single-category, single-name trace so a cat filter prunes the whole file.
std::string make_homog_trace(dftu_utils_test::TestEnvironment& env,
                             const std::string& tag, const char* name,
                             const char* cat, int n) {
    // Own subdir per trace: the index root is keyed off the parent directory,
    // so same-dir traces would share (and clobber) one index.
    std::string dir = env.get_dir() + "/" + tag;
    fs::create_directories(dir);
    std::string pfw = dir + "/trace.pfw";
    std::ofstream ofs(pfw);
    for (int i = 0; i < n; ++i)
        ofs << R"({"ph":"X","name":")" << name << R"(","cat":")" << cat
            << R"(","pid":1,"tid":1,"ts":)" << (1000 + i * 100) << R"(,"dur":)"
            << (10 + i) << R"(,"args":{}})" << "\n";
    ofs.close();
    std::string gz = pfw + ".gz";
    dftu_utils_test::compress_file_to_gzip(pfw, gz);
    fs::remove(pfw);
    return gz;
}

ViewFile index_trace(const std::string& gz) {
    std::string idx = determine_index_path(gz, "");
    struct Sink : ExportSink {
        void write(std::string_view) override {}
    } sink;
    View::from_file(gz, idx).metadata(false).export_json(sink).get();
    return ViewFile{gz, idx};
}

ExportStats run_host(std::vector<dftu_plugin*> plugins,
                     std::vector<ViewFile> files) {
    PluginHost host;
    for (auto* p : plugins) host.inject_plugin(p);
    View view = View::from_files(std::move(files));
    ExportStats stats;
    Runtime rt(4);
    auto task = dftracer::utils::run_coro_scope(
        rt.executor(), [&](CoroScope&) -> coro::CoroTask<void> {
            stats = co_await host.run(view);
            co_return;
        });
    rt.submit(std::move(task), "plugin-planner").wait();
    rt.shutdown();
    return stats;
}

}  // namespace

TEST_SUITE("PluginPlannerUnion") {
    TEST_CASE("no plugins yields no prune") {
        CHECK_FALSE(plugin_union_prune_query({}).has_value());
    }

    TEST_CASE("a single query is the union verbatim") {
        auto u = plugin_union_prune_query(qs({R"(cat == "STDIO")"}));
        REQUIRE(u.has_value());
        CHECK(u->source() == R"(cat == "STDIO")");
    }

    TEST_CASE("distinct queries disjoin into a real OR") {
        auto u = plugin_union_prune_query(
            qs({R"(cat == "STDIO")", R"(cat == "POSIX")"}));
        REQUIRE(u.has_value());
        // The parenthesized union must parse to a disjunction that keeps events
        // either side selects.
        CHECK(u->references("cat"));
        dftracer::utils::query::ValueMap posix;
        posix["cat"] = std::string("POSIX");
        dftracer::utils::query::ValueMap stdio;
        stdio["cat"] = std::string("STDIO");
        dftracer::utils::query::ValueMap other;
        other["cat"] = std::string("OTHER");
        CHECK(u->evaluate(posix));
        CHECK(u->evaluate(stdio));
        CHECK_FALSE(u->evaluate(other));
    }

    TEST_CASE("identical queries collapse (no repeated term)") {
        const char* q = R"(cat == "STDIO")";
        auto u = plugin_union_prune_query(qs({q, q, q}));
        REQUIRE(u.has_value());
        CHECK(u->source() == std::string(q));
    }

    TEST_CASE("a plugin without a plan_query disables the prune") {
        CHECK_FALSE(plugin_union_prune_query(qs({R"(cat == "STDIO")", nullptr}))
                        .has_value());
        CHECK_FALSE(plugin_union_prune_query(qs({R"(cat == "STDIO")", ""}))
                        .has_value());
    }

    TEST_CASE("an unparseable plan_query disables the prune") {
        CHECK_FALSE(
            plugin_union_prune_query(qs({R"(cat == "STDIO")", "cat =="}))
                .has_value());
    }
}

TEST_SUITE("PluginPlannerScan") {
    // Two homogeneous files: STDIO-only and POSIX-only. A cat=="STDIO" union
    // must prune the POSIX file (fewer events scanned) while every plugin's
    // filtered count stays identical to an unpruned run.
    TEST_CASE("union prune skips unwanted files without changing results") {
        dftu_utils_test::TestEnvironment env(0);
        REQUIRE(env.is_valid());
        const int N = 40;
        ViewFile stdio =
            index_trace(make_homog_trace(env, "stdio", "fwrite", "STDIO", N));
        ViewFile posix =
            index_trace(make_homog_trace(env, "posix", "read", "POSIX", N));
        std::vector<ViewFile> files{stdio, posix};

        const char* stdio_q = R"(cat == "STDIO")";

        CountState p_a{stdio_q}, p_b{stdio_q};
        dftu_plugin a = make_count_plugin(&p_a), b = make_count_plugin(&p_b);
        ExportStats pruned = run_host({&a, &b}, files);

        // Baseline: a plugin with no plan_query forces a full scan (no prune).
        CountState q_a{stdio_q}, q_all{nullptr};
        dftu_plugin c = make_count_plugin(&q_a), d = make_count_plugin(&q_all);
        ExportStats full = run_host({&c, &d}, files);

        // Every STDIO-filtering plugin sees exactly the STDIO events either
        // way.
        CHECK(p_a.total.load() == N);
        CHECK(p_b.total.load() == N);
        CHECK(q_a.total.load() == N);
        CHECK(p_a.total.load() == q_a.total.load());
        // The unfiltered plugin sees every event, proving the baseline scanned
        // both files.
        CHECK(q_all.total.load() == 2 * N);

        // Prune evidence: the POSIX file was skipped, so fewer events scanned.
        CHECK(pruned.events_scanned < full.events_scanned);
        CHECK(full.events_scanned == 2 * N);
        CHECK(pruned.events_scanned == N);
    }
}
