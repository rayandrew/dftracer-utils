// Drives the C consumers (compose_abi_c_consumer.c, query_abi_c_consumer.c):
// the host is built here in C++, but every ABI call happens in a C translation
// unit through the raw vtables, so this proves the plugin C ABIs work
// end-to-end from real C, not just from C++.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugin.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "testing_utilities.h"

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::PluginFold;
namespace coro = dftracer::utils::coro;

extern "C" {
::dftu_task* dftu_test_compose_pipe(const ::dftu_host*, const std::int64_t*,
                                    std::int64_t*, int*);
::dftu_task* dftu_test_compose_all(const ::dftu_host*, const std::int64_t*,
                                   std::int64_t*, int*);
int dftu_test_compose_typecheck(const ::dftu_host*);
int dftu_test_query_match(const ::dftu_host*, const char*, std::uint32_t,
                          const ::dftu_event*);
int dftu_test_query_null_is_safe(const ::dftu_host*, const ::dftu_event*);
int dftu_test_sketch(const ::dftu_host*, double*, std::uint64_t*);
int dftu_test_sketch_merge(const ::dftu_host*, std::uint64_t*);
int dftu_test_ops_fnv1a(const ::dftu_host*, const char*, std::uint32_t,
                        std::uint64_t*);
int dftu_test_ops_find(const ::dftu_host*);
int dftu_test_trace_roundtrip(const ::dftu_host*, const char*,
                              const ::dftu_event*, std::uint32_t, int*);
::dftu_task* dftu_test_io_open(const ::dftu_host*, const char*, int*);
::dftu_task* dftu_test_io_write(const ::dftu_host*, int, const void*,
                                std::uint64_t, std::int64_t*);
::dftu_task* dftu_test_io_pread(const ::dftu_host*, int, void*, std::uint64_t,
                                std::uint64_t, std::int64_t*);
::dftu_task* dftu_test_io_close(const ::dftu_host*, int, int*);
::dftu_writer* dftu_test_writer_create(const ::dftu_host*, const char*);
::dftu_task* dftu_test_writer_open(const ::dftu_host*, ::dftu_writer*);
::dftu_task* dftu_test_writer_chunk(const ::dftu_host*, ::dftu_writer*,
                                    const void*, std::uint64_t);
::dftu_task* dftu_test_writer_close(const ::dftu_host*, ::dftu_writer*);
#ifdef DFTU_TEST_HAS_ARROW
std::int64_t dftu_test_arrow_batch(const ::dftu_host*, const ::dftu_batch*);
#endif
}

namespace {

struct TrivialSlice {
    explicit TrivialSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(TrivialSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct HostFixture {
    StringIntern intern;
    dftu_plugin* plugin;
    std::unique_ptr<PluginFold> fold;
    HostFixture()
        : plugin(dftracer::utils::plugins::make_plugin<TrivialSlice>(nullptr)),
          fold(std::make_unique<PluginFold>(plugin, intern)) {}
    ~HostFixture() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
    dftu_host& host() { return fold->host(); }
};

coro::CoroTask<void>* as_coro(::dftu_task* t) {
    return reinterpret_cast<coro::CoroTask<void>*>(t);
}

void drive(::dftu_task* t) {
    Runtime rt(1);
    rt.scope("c-abi", [&](CoroScope&) -> coro::CoroTask<void> {
          co_await *as_coro(t);
      }).wait();
    rt.shutdown();
}

}  // namespace

TEST_CASE("C ABI: dftu_ext_compose pipe threads a value from C") {
    HostFixture fx;
    std::int64_t in = 5, out = 0;
    int rc = -1;
    ::dftu_task* t = dftu_test_compose_pipe(&fx.host(), &in, &out, &rc);
    REQUIRE(t != nullptr);
    drive(t);
    CHECK(rc == 0);
    CHECK(out == 20);  // (5*2)+10
}

TEST_CASE("C ABI: dftu_ext_compose when_all concatenates from C") {
    HostFixture fx;
    std::int64_t in = 5;
    std::int64_t out[2] = {0, 0};
    int rc = -1;
    ::dftu_task* t = dftu_test_compose_all(&fx.host(), &in, out, &rc);
    REQUIRE(t != nullptr);
    drive(t);
    CHECK(out[0] == 10);  // 5*2
    CHECK(out[1] == 15);  // 5+10
}

TEST_CASE("C ABI: dftu_ext_compose type-checks a pipe from C") {
    HostFixture fx;
    CHECK(dftu_test_compose_typecheck(&fx.host()) == 1);
}

TEST_CASE("C ABI: dftu_ext_query compile + match from C") {
    HostFixture fx;
    dftu_host& host = fx.host();
    dftu_str posix = host.intern(host.h, "POSIX", 5);
    dftu_str stdio = host.intern(host.h, "STDIO", 5);
    dftu_str read = host.intern(host.h, "read", 4);

    std::string src = "cat == \"POSIX\"";
    auto len = static_cast<std::uint32_t>(src.size());

    dftu_event hit{};
    hit.cat = posix;
    hit.name = read;
    dftu_event miss{};
    miss.cat = stdio;
    miss.name = read;

    CHECK(dftu_test_query_match(&host, src.data(), len, &hit) == 1);
    CHECK(dftu_test_query_match(&host, src.data(), len, &miss) == 0);

    std::string bad = "cat ==";
    CHECK(dftu_test_query_match(&host, bad.data(),
                                static_cast<std::uint32_t>(bad.size()),
                                &hit) == -1);  // malformed -> compile failure
    CHECK(dftu_test_query_null_is_safe(&host, &hit) ==
          0);                                  // null query is safe
}

TEST_CASE("C ABI: dftu_ext_sketch add/result/merge from C") {
    HostFixture fx;
    double p50 = 0;
    std::uint64_t count = 0;
    REQUIRE(dftu_test_sketch(&fx.host(), &p50, &count) == 0);
    CHECK(count == 100);
    CHECK(p50 == doctest::Approx(50.0).epsilon(0.1));  // median of 1..100

    std::uint64_t merged = 0;
    REQUIRE(dftu_test_sketch_merge(&fx.host(), &merged) == 0);
    CHECK(merged == 100);  // 50 + 50
}

TEST_CASE("C ABI: dftu_ext_ops runs a host utility op from C") {
    HostFixture fx;
    std::uint64_t hash = 0;
    std::string data = "hello-world";
    REQUIRE(dftu_test_ops_fnv1a(&fx.host(), data.data(),
                                static_cast<std::uint32_t>(data.size()),
                                &hash) == 0);
    CHECK(hash != 0);
    // Deterministic: the same input hashes the same.
    std::uint64_t again = 0;
    REQUIRE(dftu_test_ops_fnv1a(&fx.host(), data.data(),
                                static_cast<std::uint32_t>(data.size()),
                                &again) == 0);
    CHECK(hash == again);
    CHECK(dftu_test_ops_find(&fx.host()) == 1);
}

TEST_CASE("C ABI: dftu_ext_trace write then read round-trips from C") {
    dftu_utils_test::TestEnvironment env(0);
    HostFixture fx;
    dftu_host& host = fx.host();
    dftu_str cat = host.intern(host.h, "POSIX", 5);
    dftu_str name = host.intern(host.h, "write", 5);

    const std::uint32_t N = 20;
    std::vector<dftu_event> evs(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        evs[i] = {};
        evs[i].cat = cat;
        evs[i].name = name;
        evs[i].pid = 7;
        evs[i].tid = 9;
        evs[i].ts = 1000000 + i * 100;
        evs[i].dur = 3;
        evs[i].has_dur = 1;
        evs[i].phase = DFTU_PH_COMPLETE;
    }

    std::string path = env.get_dir() + "/c_abi.pfw.gz";
    int read = 0;
    REQUIRE(dftu_test_trace_roundtrip(&host, path.c_str(), evs.data(), N,
                                      &read) == 0);
    CHECK(read == static_cast<int>(N));
}

TEST_CASE("C ABI: dftu_io write/read round-trips from C") {
    dftu_utils_test::TestEnvironment env(0);
    HostFixture fx;
    dftu_host& host = fx.host();
    std::string path = env.get_dir() + "/io_c.bin";
    std::string data = "hello-io";
    char buf[16] = {0};
    int fd = -1;
    std::int64_t wn = 0, rn = 0;
    int crc = -1;

    // C has no coroutines, so the harness sequences the C-created tasks: the fd
    // from open flows into write/pread/close.
    Runtime rt(1);
    rt.scope("io", [&](CoroScope&) -> coro::CoroTask<void> {
          co_await *as_coro(dftu_test_io_open(&host, path.c_str(), &fd));
          if (fd >= 0) {
              co_await *as_coro(dftu_test_io_write(
                  &host, fd, data.data(),
                  static_cast<std::uint64_t>(data.size()), &wn));
              co_await *as_coro(dftu_test_io_pread(
                  &host, fd, buf, static_cast<std::uint64_t>(data.size()), 0,
                  &rn));
              co_await *as_coro(dftu_test_io_close(&host, fd, &crc));
          }
      }).wait();
    rt.shutdown();

    REQUIRE(fd >= 0);
    CHECK(wn == static_cast<std::int64_t>(data.size()));
    CHECK(rn == static_cast<std::int64_t>(data.size()));
    CHECK(std::string(buf, data.size()) == data);
    CHECK(crc == 0);
}

TEST_CASE("C ABI: dftu_ext_writer create/open/chunk/close from C") {
    dftu_utils_test::TestEnvironment env(0);
    HostFixture fx;
    dftu_host& host = fx.host();
    std::string path = env.get_dir() + "/writer_c.out";
    std::string data = "chunk-from-c";

    ::dftu_writer* w = dftu_test_writer_create(&host, path.c_str());
    REQUIRE(w != nullptr);
    Runtime rt(1);
    rt.scope("writer", [&](CoroScope&) -> coro::CoroTask<void> {
          co_await *as_coro(dftu_test_writer_open(&host, w));
          co_await *as_coro(dftu_test_writer_chunk(
              &host, w, data.data(), static_cast<std::uint64_t>(data.size())));
          co_await *as_coro(dftu_test_writer_close(&host, w));
      }).wait();
    rt.shutdown();

    // The sharded writer emits one file per worker: `<path>.shard_0`.
    std::error_code ec;
    auto sz = std::filesystem::file_size(path + ".shard_0", ec);
    CHECK(!ec);
    CHECK(sz > 0);
}

#ifdef DFTU_TEST_HAS_ARROW
TEST_CASE("C ABI: dftu_ext_arrow batch_to_arrow from C") {
    HostFixture fx;
    dftu_host& host = fx.host();
    dftu_str cat = host.intern(host.h, "POSIX", 5);
    dftu_str name = host.intern(host.h, "read", 4);

    const std::uint32_t N = 8;
    std::vector<dftu_event> evs(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        evs[i] = {};
        evs[i].cat = cat;
        evs[i].name = name;
        evs[i].pid = 1;
        evs[i].tid = 2;
        evs[i].ts = i;
        evs[i].dur = 1;
        evs[i].has_dur = 1;
        evs[i].phase = DFTU_PH_COMPLETE;
    }

    dftu_batch b{evs.data(), N};
    CHECK(dftu_test_arrow_batch(&host, &b) == static_cast<std::int64_t>(N));
}
#endif
