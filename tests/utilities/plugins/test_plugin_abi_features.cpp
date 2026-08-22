// In-process coverage for the plugin ABI surface the reflected-utility and
// host-service suites leave untested: the config tree, query compile/match and
// plan_query event routing, the coroutine control ops, the async on_batch
// take_pending/drive path, and the scalar functors.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/config.h>
#include <dftracer/utils/plugins/dftu_generated_utilities.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugin.h>

#include <memory>
// After plugin.h/fold_adapter.h so nanoarrow is already set up for arrow_abi.h.
#include <dftracer/utils/plugins/compose.h>
#include <dftracer/utils/trace/schema.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <dftracer/utils/trace/views/view_scan.h>
#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "testing_utilities.h"

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::ConfigTree;
using dftracer::utils::plugins::PluginFold;
namespace coro = dftracer::utils::coro;
namespace views = dftracer::utils::trace::views;
using dftracer::utils::trace::RecordPhase;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::ScanUnit;

namespace {

coro::CoroTask<void>* as_coro(::dftu_task* t) {
    return reinterpret_cast<coro::CoroTask<void>*>(t);
}

dftu_bytes bytes_of(const std::string& s) {
    return dftu_bytes{s.data(), static_cast<std::uint32_t>(s.size())};
}

struct ConfigSlice {
    std::int64_t threshold = 0;
    double ratio = 0.0;
    std::string label;
    bool enabled = false;
    std::int64_t inner = 0;
    std::int64_t missing = 0;
    std::vector<std::int64_t> buckets;

    explicit ConfigSlice(const dftracer::utils::plugins::Config& c) {
        threshold = c.get_int("threshold");
        ratio = c.get_double("ratio");
        label = std::string(c.get("label"));
        enabled = c.get_bool("enabled");
        inner = c.child("nested").get_int("inner");
        missing = c.get_int("missing", -1);
        const dftu_value* arr = c.find("buckets");
        if (arr && arr->kind == DFTU_VAL_ARRAY)
            for (std::uint32_t i = 0; i < arr->count; ++i)
                buckets.push_back(dftu_as_i64(&arr->as.items[i], 0));
    }
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(ConfigSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct CountState {
    std::uint64_t seen = 0;
    std::vector<std::string> fhashes;
    std::vector<std::string> hhashes;
};
CountState g_count;

struct CountSlice {
    explicit CountSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        g_count.seen += b.size();
        for (const dftracer::utils::plugins::Event& e : b) {
            g_count.fhashes.emplace_back(h.str(e.fhash_id()));
            g_count.hhashes.emplace_back(h.str(e.hhash_id()));
        }
    }
    void merge(CountSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

FoldEvent make_event(StringIntern& intern, const char* fhash,
                     const char* hhash) {
    FoldEvent fe;
    fe.cat_id = intern.get_or_insert("POSIX");
    fe.name_id = intern.get_or_insert("read");
    fe.fhash_id = intern.get_or_insert(fhash);
    fe.hhash_id = intern.get_or_insert(hhash);
    fe.phase = RecordPhase::COMPLETE;
    fe.ts = 1;
    fe.dur = 5;
    fe.has_dur = true;
    return fe;
}

void inc_fn(void* p) { static_cast<std::atomic<int>*>(p)->fetch_add(1); }
void append1_fn(void* p) { static_cast<std::vector<int>*>(p)->push_back(1); }
void append2_fn(void* p) { static_cast<std::vector<int>*>(p)->push_back(2); }

const dftu_ext_coro* coro_ext(dftracer::utils::plugins::Host h) {
    return static_cast<const dftu_ext_coro*>(
        h.raw()->get_extension(h.raw()->h, DFTU_EXT_CORO));
}

dftracer::utils::plugins::Task fanout_all(dftracer::utils::plugins::Host h,
                                          std::atomic<int>* ran, int n) {
    const dftu_ext_coro* c = coro_ext(h);
    std::vector<::dftu_task*> kids;
    for (int i = 0; i < n; ++i)
        kids.push_back(c->spawn(h.raw()->h, inc_fn, ran));
    ::dftu_task* all =
        c->when_all(h.raw()->h, kids.data(), static_cast<std::uint32_t>(n));
    co_await h.await(all);
}

dftracer::utils::plugins::Task fanout_any(dftracer::utils::plugins::Host h,
                                          std::atomic<int>* ran) {
    const dftu_ext_coro* c = coro_ext(h);
    ::dftu_task* a = c->spawn(h.raw()->h, inc_fn, ran);
    ::dftu_task* b = c->spawn(h.raw()->h, inc_fn, ran);
    ::dftu_task* kids[2] = {a, b};
    co_await h.await(c->when_any(h.raw()->h, kids, 2));
}

dftracer::utils::plugins::Task chain(dftracer::utils::plugins::Host h,
                                     std::vector<int>* order) {
    const dftu_ext_coro* c = coro_ext(h);
    ::dftu_task* t = c->spawn(h.raw()->h, append1_fn, order);
    co_await h.await(c->then(h.raw()->h, t, append2_fn, order));
}

dftracer::utils::plugins::Task compose_all(dftracer::utils::plugins::Host h,
                                           std::atomic<int>* ran) {
    const dftu_ext_coro* c = coro_ext(h);
    ::dftu_task* a = c->spawn(h.raw()->h, inc_fn, ran);
    ::dftu_task* b = c->spawn(h.raw()->h, inc_fn, ran);
    co_await (dftracer::utils::plugins::compose(h.raw(), a) &&
              dftracer::utils::plugins::compose(h.raw(), b));
}

dftracer::utils::plugins::Task compose_any(dftracer::utils::plugins::Host h,
                                           std::atomic<int>* ran) {
    const dftu_ext_coro* c = coro_ext(h);
    ::dftu_task* a = c->spawn(h.raw()->h, inc_fn, ran);
    ::dftu_task* b = c->spawn(h.raw()->h, inc_fn, ran);
    co_await (dftracer::utils::plugins::compose(h.raw(), a) ||
              dftracer::utils::plugins::compose(h.raw(), b));
}

dftracer::utils::plugins::Task compose_map(dftracer::utils::plugins::Host h,
                                           std::atomic<int>* ran) {
    const dftu_ext_coro* c = coro_ext(h);
    std::vector<int> items = {0, 1, 2, 3};
    co_await dftracer::utils::plugins::map(
        h.raw(), items, [&](int) { return c->spawn(h.raw()->h, inc_fn, ran); });
}

// dftu_op / dftu_ext_compose: leaf value transforms (int64 -> int64), run
// inline.
const dftu_ext_compose* compose_ext(dftracer::utils::plugins::Host h) {
    return static_cast<const dftu_ext_compose*>(
        h.raw()->get_extension(h.raw()->h, DFTU_EXT_COMPOSE));
}
::dftu_task* op_double(void*, const void* in, void* out, int* rc) {
    *static_cast<std::int64_t*>(out) =
        *static_cast<const std::int64_t*>(in) * 2;
    if (rc) *rc = 0;
    return nullptr;
}
::dftu_task* op_plus10(void*, const void* in, void* out, int* rc) {
    *static_cast<std::int64_t*>(out) =
        *static_cast<const std::int64_t*>(in) + 10;
    if (rc) *rc = 0;
    return nullptr;
}

// (in*2) then (+10): value threaded through the pipe.
dftracer::utils::plugins::Task op_pipe(dftracer::utils::plugins::Host h,
                                       std::int64_t* result) {
    const dftu_ext_compose* c = compose_ext(h);
    void* hh = h.raw()->h;
    ::dftu_op* a = c->make_op(hh, op_double, nullptr, nullptr, DFTU_T_I64, 8,
                              DFTU_T_I64, 8);
    ::dftu_op* b = c->make_op(hh, op_plus10, nullptr, nullptr, DFTU_T_I64, 8,
                              DFTU_T_I64, 8);
    ::dftu_op* pipe = c->then(hh, a, b);
    std::int64_t in = 5, out = 0;
    int rc = -1;
    co_await h.await(c->run(hh, pipe, &in, &out, &rc));
    *result = (rc == 0) ? out : -1;
}

// Typed compose: compile-time type safety. A matching pipe is well-formed; a
// mismatched one is not (this is the value over the raw runtime null check).
template <class A, class B>
concept Pipeable = requires(A a, B b) { a | b; };
using OpII = dftracer::utils::plugins::Op<std::int64_t, std::int64_t>;
using OpDD = dftracer::utils::plugins::Op<double, double>;
static_assert(Pipeable<OpII, OpII>, "i64->i64 | i64->i64 must pipe");
static_assert(!Pipeable<OpII, OpDD>, "i64->i64 | f64->f64 must not pipe");

dftracer::utils::plugins::Task typed_pipe(dftracer::utils::plugins::Host h,
                                          std::int64_t* result) {
    OpII a = dftracer::utils::plugins::make_op<std::int64_t, std::int64_t>(
        h, [](std::int64_t x) { return x * 2; });
    OpII b = dftracer::utils::plugins::make_op<std::int64_t, std::int64_t>(
        h, [](std::int64_t x) { return x + 10; });
    std::int64_t in = 5, out = 0;
    int rc = -1;
    co_await dftracer::utils::plugins::run(a | b, in, out, rc);
    *result = (rc == 0) ? out : -1;
}

// A registered host utility as a compose leaf: pipe fnv1a (bytes -> u64) into
// hex64_format (u64 -> hex16), so `then` chains two host utilities.
dftracer::utils::plugins::Task util_pipe(dftracer::utils::plugins::Host h,
                                         dftu_hex16* result, int* rc) {
    const dftu_ext_compose* c = compose_ext(h);
    void* hh = h.raw()->h;
    ::dftu_op* hash = c->util_op(hh, DFTU_UTIL_FNV1A);
    ::dftu_op* fmt = c->util_op(hh, DFTU_UTIL_HEX64_FORMAT);
    ::dftu_op* pipe = c->then(hh, hash, fmt);
    std::string data = "hello-world";
    dftu_bytes in{data.data(), static_cast<std::uint32_t>(data.size())};
    co_await h.await(c->run(hh, pipe, &in, result, rc));
}

// when_any: both racers compute the same value, so the winner is deterministic.
dftracer::utils::plugins::Task op_any(dftracer::utils::plugins::Host h,
                                      std::int64_t* result) {
    const dftu_ext_compose* c = compose_ext(h);
    void* hh = h.raw()->h;
    ::dftu_op* ops[2] = {c->make_op(hh, op_plus10, nullptr, nullptr, DFTU_T_I64,
                                    8, DFTU_T_I64, 8),
                         c->make_op(hh, op_plus10, nullptr, nullptr, DFTU_T_I64,
                                    8, DFTU_T_I64, 8)};
    ::dftu_op* any = c->when_any(hh, ops, 2);
    std::int64_t in = 5, out = 0;
    int rc = -1;
    co_await h.await(c->run(hh, any, &in, &out, &rc));
    *result = out;
}

// when_all: both leaves fan the same input; output is their values
// concatenated.
dftracer::utils::plugins::Task op_all(dftracer::utils::plugins::Host h,
                                      std::int64_t* out0, std::int64_t* out1) {
    const dftu_ext_compose* c = compose_ext(h);
    void* hh = h.raw()->h;
    ::dftu_op* ops[2] = {c->make_op(hh, op_double, nullptr, nullptr, DFTU_T_I64,
                                    8, DFTU_T_I64, 8),
                         c->make_op(hh, op_plus10, nullptr, nullptr, DFTU_T_I64,
                                    8, DFTU_T_I64, 8)};
    ::dftu_op* all = c->when_all(hh, ops, 2);
    std::int64_t in = 5;
    std::int64_t out[2] = {0, 0};
    int rc = -1;
    co_await h.await(c->run(hh, all, &in, out, &rc));
    *out0 = out[0];
    *out1 = out[1];
}

struct AsyncState {
    std::atomic<int> done{0};
    std::uint64_t batch_count = 0;
};
AsyncState g_async;

void mark_done_fn(void* p) { static_cast<AsyncState*>(p)->done.fetch_add(1); }

struct AsyncSlice {
    explicit AsyncSlice(const dftracer::utils::plugins::Config&) {}
    dftracer::utils::plugins::Task on_batch(const dftu_batch& b,
                                            dftracer::utils::plugins::Host h) {
        g_async.batch_count += b.count;
        const dftu_ext_coro* c = static_cast<const dftu_ext_coro*>(
            h.raw()->get_extension(h.raw()->h, DFTU_EXT_CORO));
        co_await h.await(c->spawn(h.raw()->h, mark_done_fn, &g_async));
    }
    void merge(AsyncSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Owns a PluginFold plus its backing plugin; the fold references the plugin, so
// it is destroyed first.
template <class Slice>
struct FoldFixture {
    StringIntern intern;
    dftu_plugin* plugin;
    std::unique_ptr<PluginFold> fold;
    explicit FoldFixture(const dftu_value* config)
        : plugin(dftracer::utils::plugins::make_plugin<Slice>(config)),
          fold(std::make_unique<PluginFold>(plugin, intern)) {}
    dftu_host& host() { return fold->host(); }
    ~FoldFixture() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

}  // namespace

TEST_CASE("plugin ABI: config tree reads scalars, nested, array, and default") {
    dftu_utils_test::TestEnvironment env(0);
    std::string path = env.get_dir() + "/config.json";
    {
        std::ofstream f(path);
        f << R"({"threshold":42,"ratio":2.5,"label":"posix","enabled":true,)"
          << R"("nested":{"inner":7},"buckets":[10,20,30]})";
    }

    ConfigTree tree = ConfigTree::from_json_file(path);
    dftu_plugin* p =
        dftracer::utils::plugins::make_plugin<ConfigSlice>(tree.root());
    auto* slice = static_cast<ConfigSlice*>(p->make_slice(p->self));
    REQUIRE(slice != nullptr);

    CHECK(slice->threshold == 42);
    CHECK(slice->ratio == doctest::Approx(2.5));
    CHECK(slice->label == "posix");
    CHECK(slice->enabled == true);
    CHECK(slice->inner == 7);
    CHECK(slice->missing == -1);
    REQUIRE(slice->buckets.size() == 3);
    CHECK(slice->buckets[0] == 10);
    CHECK(slice->buckets[2] == 30);

    p->destroy_slice(slice);
    p->destroy(p->self);
}

TEST_CASE("plugin ABI: query_compile then query_matches on known events") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();
    dftu_str posix = host.intern(host.h, "POSIX", 5);
    dftu_str stdio = host.intern(host.h, "STDIO", 5);
    dftu_str read = host.intern(host.h, "read", 4);

    const auto* qx = static_cast<const dftu_ext_query*>(
        host.get_extension(host.h, DFTU_EXT_QUERY));
    REQUIRE(qx != nullptr);

    std::string src = "cat == \"POSIX\"";
    dftu_query* q = qx->query_compile(host.h, src.data(),
                                      static_cast<std::uint32_t>(src.size()));
    REQUIRE(q != nullptr);

    dftu_event match{};
    match.cat = posix;
    match.name = read;
    CHECK(qx->query_matches(host.h, q, &match) == 1);

    dftu_event miss{};
    miss.cat = stdio;
    miss.name = read;
    CHECK(qx->query_matches(host.h, q, &miss) == 0);

    SUBCASE("malformed query source does not compile and match is safe") {
        std::string bad = "cat ==";
        dftu_query* bq = qx->query_compile(
            host.h, bad.data(), static_cast<std::uint32_t>(bad.size()));
        CHECK(bq == nullptr);
        // A null query must yield a defined 0, never a throw across the ABI.
        CHECK(qx->query_matches(host.h, nullptr, &match) == 0);
    }
}

TEST_CASE("plugin ABI: plan_query on fhash delivers only matching events") {
    SUBCASE("fhash predicate keeps every match, drops the rest") {
        g_count = CountState{};
        ConfigTree tree;
        tree.set("query", "fhash == \"fhAAA\"");
        FoldFixture<CountSlice> fx(tree.root());

        std::vector<FoldEvent> evs;
        evs.push_back(make_event(fx.intern, "fhAAA", "hh1"));
        evs.push_back(make_event(fx.intern, "fhBBB", "hh1"));
        evs.push_back(make_event(fx.intern, "fhAAA", "hh2"));
        evs.push_back(make_event(fx.intern, "fhCCC", "hh2"));
        evs.push_back(make_event(fx.intern, "fhAAA", "hh3"));
        ScanUnit unit{};
        FoldBatch batch{std::span<const FoldEvent>(evs), unit};

        fx.fold->step(batch);
        CHECK(fx.fold->take_pending() == nullptr);

        // The regression: a plan_query on fhash must not silently drop matches.
        CHECK(g_count.seen == 3);
        for (const auto& f : g_count.fhashes) CHECK(f == "fhAAA");
    }

    SUBCASE("hhash predicate routes on a non cat/name field too") {
        g_count = CountState{};
        ConfigTree tree;
        tree.set("query", "hhash == \"hhZ\"");
        FoldFixture<CountSlice> fx(tree.root());

        std::vector<FoldEvent> evs;
        evs.push_back(make_event(fx.intern, "fh1", "hhZ"));
        evs.push_back(make_event(fx.intern, "fh2", "hhY"));
        evs.push_back(make_event(fx.intern, "fh3", "hhZ"));
        ScanUnit unit{};
        FoldBatch batch{std::span<const FoldEvent>(evs), unit};

        fx.fold->step(batch);
        CHECK(g_count.seen == 2);
        for (const auto& h : g_count.hhashes) CHECK(h == "hhZ");
    }
}

TEST_CASE("plugin ABI: spawn fan-out joined by when_all runs every child") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::atomic<int> ran{0};
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host,
              fanout_all(dftracer::utils::plugins::Host{&host}, &ran, 4));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(ran.load() == 4);
}

TEST_CASE("plugin ABI: when_any returns after the first child") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::atomic<int> ran{0};
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, fanout_any(dftracer::utils::plugins::Host{&host}, &ran));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(ran.load() >= 1);
}

TEST_CASE("plugin ABI: compose && joins both tasks (when_all)") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::atomic<int> ran{0};
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, compose_all(dftracer::utils::plugins::Host{&host}, &ran));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(ran.load() == 2);
}

TEST_CASE("plugin ABI: compose || races the tasks (when_any)") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::atomic<int> ran{0};
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, compose_any(dftracer::utils::plugins::Host{&host}, &ran));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(ran.load() >= 1);
}

TEST_CASE("plugin ABI: compose map joins one task per item (when_all)") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::atomic<int> ran{0};
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, compose_map(dftracer::utils::plugins::Host{&host}, &ran));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(ran.load() == 4);
}

TEST_CASE("plugin ABI: compose dftu_op then threads a value through the pipe") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::int64_t result = 0;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, op_pipe(dftracer::utils::plugins::Host{&host}, &result));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(result == 20);  // (5*2)+10
}

TEST_CASE("plugin ABI: compose dftu_op when_all concatenates outputs") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::int64_t a = 0, b = 0;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, op_all(dftracer::utils::plugins::Host{&host}, &a, &b));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(a == 10);  // 5*2
    CHECK(b == 15);  // 5+10
}

TEST_CASE("plugin ABI: compose dftu_op when_any yields the winner's value") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::int64_t result = 0;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, op_any(dftracer::utils::plugins::Host{&host}, &result));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(result == 15);  // both racers compute 5+10
}

TEST_CASE("plugin ABI: typed compose Op pipes real values") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::int64_t result = 0;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host,
              typed_pipe(dftracer::utils::plugins::Host{&host}, &result));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    CHECK(result == 20);  // (5*2)+10, fully typed
}

TEST_CASE("plugin ABI: compose util_op pipes two host utilities") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    // Expected: fnv1a("hello-world") formatted as 16 hex digits.
    std::string data = "hello-world";
    dftu_bytes bin{data.data(), static_cast<std::uint32_t>(data.size())};
    std::uint64_t hash = 0;
    REQUIRE(dftu_util_fnv1a(&host, &bin, &hash) == 0);
    dftu_hex16 expected{};
    REQUIRE(dftu_util_hex64_format(&host, &hash, &expected) == 0);

    Runtime rt(1);
    dftu_hex16 got{};
    int rc = -1;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host,
              util_pipe(dftracer::utils::plugins::Host{&host}, &got, &rc));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();

    CHECK(rc == 0);
    CHECK(std::string(got.c, 16) == std::string(expected.c, 16));
}

TEST_CASE("plugin ABI: compose dftu_op then rejects a type mismatch") {
    FoldFixture<CountSlice> fx(nullptr);
    dftracer::utils::plugins::Host h{&fx.host()};
    const dftu_ext_compose* c = compose_ext(h);
    void* hh = h.raw()->h;
    ::dftu_op* a = c->make_op(hh, op_double, nullptr, nullptr, DFTU_T_I64, 8,
                              DFTU_T_I64, 8);
    // Same byte size (8) but a different type must not pipe.
    ::dftu_op* f = c->make_op(hh, op_plus10, nullptr, nullptr, DFTU_T_F64, 8,
                              DFTU_T_F64, 8);
    CHECK(c->then(hh, a, f) == nullptr);
    // Matching type pipes fine.
    ::dftu_op* i = c->make_op(hh, op_plus10, nullptr, nullptr, DFTU_T_I64, 8,
                              DFTU_T_I64, 8);
    CHECK(c->then(hh, a, i) != nullptr);
}

TEST_CASE("plugin ABI: then chains a continuation after its task") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    Runtime rt(1);
    std::vector<int> order;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          ::dftu_task* d = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, chain(dftracer::utils::plugins::Host{&host}, &order));
          co_await *as_coro(d);
      }).wait();
    rt.shutdown();
    REQUIRE(order.size() == 2);
    CHECK(order[0] == 1);
    CHECK(order[1] == 2);
}

TEST_CASE("plugin ABI: async on_batch drives to completion via take_pending") {
    g_async.done.store(0);
    g_async.batch_count = 0;
    FoldFixture<AsyncSlice> fx(nullptr);

    std::vector<FoldEvent> evs;
    evs.push_back(make_event(fx.intern, "fh1", "hh1"));
    evs.push_back(make_event(fx.intern, "fh2", "hh2"));
    ScanUnit unit{};
    FoldBatch batch{std::span<const FoldEvent>(evs), unit};

    Runtime rt(1);
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          fx.fold->step(batch);
          ::dftu_task* t = fx.fold->take_pending();
          REQUIRE(t != nullptr);
          co_await *as_coro(t);
      }).wait();
    rt.shutdown();

    CHECK(g_async.batch_count == 2);
    CHECK(g_async.done.load() == 1);
}

TEST_CASE("plugin ABI: scalar functors fnv1a and hex64 round-trip") {
    FoldFixture<CountSlice> fx(nullptr);
    dftu_host& host = fx.host();

    std::string posix = "POSIX";
    dftu_bytes in = bytes_of(posix);
    std::uint64_t h = 0;
    CHECK(dftu_util_fnv1a(&host, &in, &h) == 0);
    CHECK(h == 5527563327133061832ULL);

    std::string empty;
    dftu_bytes ein = bytes_of(empty);
    std::uint64_t he = 0;
    CHECK(dftu_util_fnv1a(&host, &ein, &he) == 0);
    CHECK(he == 14695981039346656037ULL);  // FNV-1a offset basis

    std::uint64_t v = 0x0123456789abcdefULL;
    dftu_hex16 hx{};
    CHECK(dftu_util_hex64_format(&host, &v, &hx) == 0);
    std::string hex(hx.c, 16);
    CHECK(hex == "0123456789abcdef");

    dftu_bytes hb = bytes_of(hex);
    std::uint64_t back = 0;
    CHECK(dftu_util_hex64_parse(&host, &hb, &back) == 0);
    CHECK(back == v);

    std::string bad = "nothexnothex1234";
    dftu_bytes badb = bytes_of(bad);
    std::uint64_t bo = 123;
    CHECK(dftu_util_hex64_parse(&host, &badb, &bo) == -1);

    std::string short_hex = "abc";
    dftu_bytes sb = bytes_of(short_hex);
    CHECK(dftu_util_hex64_parse(&host, &sb, &bo) == -1);
}
