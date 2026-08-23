// Exercises the host services a plugin calls synchronously: Arrow batch export,
// Arrow IPC write, and dftracer trace write/read round-trip. Drives them
// through a real PluginFold-wired dftu_host, the way a loaded plugin would.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/abi.h>
// fold_adapter.h transitively pulls nanoarrow (ArrowArray/ArrowSchema); it must
// precede plugin.h so arrow_abi.h no-ops rather than tripping nanoarrow's outer
// ARROW_FLAG_DICTIONARY_ORDERED sentinel.
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/dftu_generated_utilities.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugin.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <doctest/doctest.h>
#include <fcntl.h>
#include <sys/stat.h>

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "testing_utilities.h"

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::PluginFold;
namespace coro = dftracer::utils::coro;

namespace {

struct TrivialSlice {
    explicit TrivialSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(TrivialSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Owns a PluginFold plus its trivial backing plugin, exposing the wired host.
// The fold references the plugin, so it must be destroyed before the plugin.
struct HostFixture {
    StringIntern intern;
    dftu_plugin* plugin =
        dftracer::utils::plugins::make_plugin<TrivialSlice>(nullptr);
    std::unique_ptr<PluginFold> fold =
        std::make_unique<PluginFold>(plugin, intern);
    dftu_host& host() { return fold->host(); }
    dftu_str intern_str(const char* s) {
        return host().intern(host().h, s,
                             static_cast<std::uint32_t>(strlen(s)));
    }
    ~HostFixture() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

std::uint64_t file_size(const std::string& path) {
    struct stat st{};
    return ::stat(path.c_str(), &st) == 0
               ? static_cast<std::uint64_t>(st.st_size)
               : 0;
}

// A plugin coroutine awaiting util_async, driven exactly as the fuse worker
// drives a plugin's on_batch: the co_await proves executor composition.
dftracer::utils::plugins::Task read_file_async(dftracer::utils::plugins::Host h,
                                               const dftu_file_entry* in,
                                               dftu_text* out, int* rc) {
    co_await h.util_async<dftracer::utils::plugins::util::file_reader>(
        *in, *out, *rc);
}

namespace dfti = dftracer::utils::trace::internal;

// Drive a plugin Task to completion on a fresh Runtime, exactly as the fuse
// worker drives a plugin coroutine.
template <class MakeTask>
void drive_task(dftu_host& host, int workers, MakeTask make) {
    Runtime rt(workers);
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          dftu_task* driver =
              dftracer::utils::plugins::detail::drive_coro<int>(&host, make());
          co_await *reinterpret_cast<coro::CoroTask<void>*>(driver);
      }).wait();
    rt.shutdown();
}

// stop_after 0 pulls the whole stream; a positive value breaks early, closing
// the RAII Stream over a generator still suspended at a co_yield.
dftracer::utils::plugins::Task pull_dir(
    dftracer::utils::plugins::Host h,
    const dftu_directory_scanner_utility_input* in, int* entries,
    std::uint64_t* total, int stop_after) {
    auto s =
        h.util_stream<dftracer::utils::plugins::util::directory_scanner>(*in);
    while (auto e = co_await s.next()) {
        (*entries)++;
        if (e->is_regular_file) *total += e->size;
        if (stop_after && *entries >= stop_after) break;
    }
}

dftracer::utils::plugins::Task pull_view_count(
    dftracer::utils::plugins::Host h, const dftu_view_scanner_input* in,
    int* batches, std::uint64_t* scanned) {
    auto s = h.util_stream<dftracer::utils::plugins::util::view_scanner>(*in);
    while (auto b = co_await s.next()) {
        (*batches)++;
        *scanned += b->events_scanned;
    }
}

// Between every pull the plugin co_awaits its own I/O; completing both proves a
// pull-model plugin can interleave async work mid-stream (push cannot).
dftracer::utils::plugins::Task pull_dir_interleave(
    dftracer::utils::plugins::Host h,
    const dftu_directory_scanner_utility_input* in, const char* path,
    int* entries, std::int64_t* bytes_read) {
    auto s =
        h.util_stream<dftracer::utils::plugins::util::directory_scanner>(*in);
    while (auto e = co_await s.next()) {
        (void)e;
        (*entries)++;
        int fd = -1;
        co_await h.await(h.io().open(path, O_RDONLY, 0, &fd));
        if (fd < 0) continue;
        char buf[128];
        std::int64_t n = 0;
        co_await h.await(h.io().read(fd, buf, sizeof(buf), &n));
        if (n > 0) *bytes_read += n;
        int rc = 0;
        co_await h.await(h.io().close(fd, &rc));
    }
}

// Populate a directory with `n` regular files so a scanner streams many items.
void make_files(const std::string& dir, int n) {
    for (int i = 0; i < n; ++i) {
        std::ofstream f(dir + "/f" + std::to_string(i) + ".dat",
                        std::ios::binary);
        f << "file-" << i << "\n";
    }
}

}  // namespace

#ifdef DFTRACER_UTILS_ENABLE_ARROW
TEST_CASE("plugin host: batch_to_arrow builds a released record batch") {
    HostFixture fx;
    dftu_host& host = fx.host();

    const std::uint32_t N = 4;
    std::vector<dftu_event> evs(N);
    dftu_str cat = fx.intern_str("POSIX");
    dftu_str name = fx.intern_str("read");
    for (std::uint32_t i = 0; i < N; ++i) {
        evs[i] = {};
        evs[i].cat = cat;
        evs[i].name = name;
        evs[i].pid = 100 + i;
        evs[i].tid = 200 + i;
        evs[i].ts = 1000 + i * 10;
        evs[i].dur = 5;
        evs[i].has_dur = 1;
        evs[i].phase = DFTU_PH_COMPLETE;
    }
    dftu_batch b{evs.data(), N};

    const auto* arrow = static_cast<const dftu_ext_arrow*>(
        host.get_extension(host.h, DFTU_EXT_ARROW));
    REQUIRE(arrow != nullptr);

    ArrowArray arr{};
    ArrowSchema sch{};
    int rc = arrow->batch_to_arrow(host.h, &b, &arr, &sch);
    REQUIRE(rc == 0);
    REQUIRE(arr.release != nullptr);
    REQUIRE(sch.release != nullptr);

    CHECK(arr.length == static_cast<std::int64_t>(N));  // rows == batch->count
    REQUIRE(arr.n_children == 7);
    CHECK(arr.children[0]->length ==
          static_cast<std::int64_t>(N));                // col 0 len

    const ArrowArray* ts_col = arr.children[4];
    CHECK(ts_col->length == static_cast<std::int64_t>(N));
    const std::uint64_t* ts =
        static_cast<const std::uint64_t*>(ts_col->buffers[1]);
    for (std::uint32_t i = 0; i < N; ++i) CHECK(ts[i] == evs[i].ts);
    MESSAGE("arrow export: rows=" << arr.length << " cols=" << arr.n_children
                                  << " ts0=" << ts[0]);

    arr.release(&arr);
    sch.release(&sch);
    CHECK(arr.release == nullptr);  // no leak: release ran
    CHECK(sch.release == nullptr);
}
#endif

#ifdef DFTRACER_UTILS_ENABLE_ARROW_IPC
TEST_CASE("plugin host: arrow_write_ipc writes a non-empty IPC file") {
    dftu_utils_test::TestEnvironment env(0);
    HostFixture fx;
    dftu_host& host = fx.host();

    std::vector<dftu_event> evs(3);
    dftu_str cat = fx.intern_str("STDIO");
    dftu_str name = fx.intern_str("fwrite");
    for (std::uint32_t i = 0; i < 3; ++i) {
        evs[i] = {};
        evs[i].cat = cat;
        evs[i].name = name;
        evs[i].ts = 500 + i;
        evs[i].phase = DFTU_PH_COMPLETE;
    }
    dftu_batch b{evs.data(), 3};

    const auto* arrow = static_cast<const dftu_ext_arrow*>(
        host.get_extension(host.h, DFTU_EXT_ARROW));
    REQUIRE(arrow != nullptr);

    ArrowArray arr{};
    ArrowSchema sch{};
    REQUIRE(arrow->batch_to_arrow(host.h, &b, &arr, &sch) == 0);

    std::string ipc = env.get_dir() + "/x.arrow";
    int wr = arrow->arrow_write_ipc(host.h, &arr, &sch, ipc.c_str());
    CHECK(wr == 0);
    CHECK(file_size(ipc) > 0);
    MESSAGE("arrow ipc: " << ipc << " size=" << file_size(ipc));

    if (arr.release) arr.release(&arr);
    if (sch.release) sch.release(&sch);

    ArrowArray back{};
    ArrowSchema back_sch{};
    int rd = arrow->arrow_read_ipc(host.h, ipc.c_str(), &back, &back_sch);
    CHECK(rd == 0);
    REQUIRE(back.release != nullptr);
    REQUIRE(back_sch.release != nullptr);
    CHECK(back.length == 3);  // rows survive the write/read round-trip
    REQUIRE(back.n_children == 7);
    const ArrowArray* ts_col = back.children[4];
    const std::uint64_t* ts =
        static_cast<const std::uint64_t*>(ts_col->buffers[1]);
    CHECK(ts[0] == 500);
    CHECK(ts[2] == 502);
    MESSAGE("arrow ipc read: rows=" << back.length << " ts0=" << ts[0]);
    back.release(&back);
    back_sch.release(&back_sch);
}
#endif

TEST_CASE("plugin host: util_run_async matches the sync path on the executor") {
    HostFixture fx;
    dftu_host& host = fx.host();
    std::string data = "async-vs-sync";
    dftu_bytes in{data.data(), static_cast<std::uint32_t>(data.size())};

    std::uint64_t sync_out = 0;
    REQUIRE(dftu_util_fnv1a(&host, &in, &sync_out) == 0);

    const auto* util = static_cast<const dftu_ext_util*>(
        host.get_extension(host.h, DFTU_EXT_UTIL));
    REQUIRE(util != nullptr);

    // One worker: the async call runs cooperatively on this executor via
    // co_await, so it completes without adopting the blocking pool.
    Runtime rt(1);
    int rc = -1;
    std::uint64_t async_out = 0;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          dftu_task* t = util->util_run_async(host.h, DFTU_UTIL_FNV1A, &in,
                                              &async_out, &rc);
          REQUIRE(t != nullptr);
          co_await *reinterpret_cast<coro::CoroTask<void>*>(t);
      }).wait();
    rt.shutdown();

    CHECK(rc == 0);
    CHECK(async_out == sync_out);
    CHECK(async_out != 0);
}

TEST_CASE(
    "plugin host: util_run_stream_async matches run_stream on the executor") {
    // directory_scanner does real filesystem I/O per entry, so completing it on
    // a single-worker executor purely via co_await is structural proof the
    // stream drive is cooperative and never monopolizes the worker.
    dftu_utils_test::TestEnvironment env(50);
    env.create_dft_test_gzip_file(50);
    HostFixture fx;
    dftu_host& host = fx.host();

    std::string dir = env.get_dir();
    dftu_directory_scanner_utility_input in{};
    in.path = dftu_bytes{dir.data(), static_cast<std::uint32_t>(dir.size())};
    in.recursive = 0;
    in.populate_size = 1;

    struct Sink {
        int entries = 0;
        std::uint64_t total_size = 0;
    };
    auto on_item = [](const void* item, void* ud) {
        const dftu_file_entry* e = static_cast<const dftu_file_entry*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->entries++;
        if (e->is_regular_file) s->total_size += e->size;
    };

    const auto* util = static_cast<const dftu_ext_util*>(
        host.get_extension(host.h, DFTU_EXT_UTIL));
    REQUIRE(util != nullptr);

    Sink sync_sink;
    REQUIRE(util->run_stream(host.h, DFTU_UTIL_DIRECTORY_SCANNER, &in, on_item,
                             &sync_sink) == 0);
    REQUIRE(sync_sink.entries >= 1);
    REQUIRE(sync_sink.total_size > 0);

    Runtime rt(1);
    int rc = -1;
    Sink async_sink;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          dftu_task* t =
              util->util_run_stream_async(host.h, DFTU_UTIL_DIRECTORY_SCANNER,
                                          &in, on_item, &async_sink, &rc);
          REQUIRE(t != nullptr);
          co_await *reinterpret_cast<coro::CoroTask<void>*>(t);
      }).wait();
    rt.shutdown();

    CHECK(rc == 0);
    CHECK(async_sink.entries == sync_sink.entries);
    CHECK(async_sink.total_size == sync_sink.total_size);
    MESSAGE("stream async: entries=" << async_sink.entries << " total_size="
                                     << async_sink.total_size);
}

TEST_CASE("plugin host: util_async wrapper composes real I/O on the executor") {
    dftu_utils_test::TestEnvironment env(0);
    HostFixture fx;
    dftu_host& host = fx.host();

    std::string path = env.get_dir() + "/async_reader.txt";
    const std::string body = "alpha\nbeta\ngamma\n";
    {
        std::ofstream f(path, std::ios::binary);
        f << body;
    }

    dftu_file_entry in{};
    in.path = dftu_bytes{path.data(), static_cast<std::uint32_t>(path.size())};
    in.is_regular_file = 1;

    Runtime rt(1);
    int rc = -1;
    dftu_text out{};
    std::string content;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          dftracer::utils::plugins::Host hw{&host};
          dftu_task* driver = dftracer::utils::plugins::detail::drive_coro<int>(
              &host, read_file_async(hw, &in, &out, &rc));
          co_await *reinterpret_cast<coro::CoroTask<void>*>(driver);
          content.assign(out.content.ptr, out.content.len);
      }).wait();
    rt.shutdown();

    CHECK(rc == 0);
    CHECK(content == body);
}

TEST_CASE("plugin host: DDSketch handle add/merge/result round-trips") {
    HostFixture fx;
    dftu_host& host = fx.host();

    const auto* sk = static_cast<const dftu_ext_sketch*>(
        host.get_extension(host.h, DFTU_EXT_SKETCH));
    REQUIRE(sk != nullptr);

    dftu_sketch* a = sk->sketch_create(host.h);
    dftu_sketch* b = sk->sketch_create(host.h);
    REQUIRE(a != nullptr);
    REQUIRE(b != nullptr);

    for (int v = 1; v <= 500; ++v)
        sk->sketch_add(host.h, a, static_cast<double>(v), 1.0);
    for (int v = 501; v <= 1000; ++v)
        sk->sketch_add(host.h, b, static_cast<double>(v), 1.0);
    sk->sketch_merge(host.h, a, b);

    dftu_quantiles q = sk->sketch_result(host.h, a);
    CHECK(q.count == 1000);
    CHECK(q.min == doctest::Approx(1.0));
    CHECK(q.max == doctest::Approx(1000.0));
    CHECK(q.mean == doctest::Approx(500.5).epsilon(0.01));

    // DDSketch's default 1% relative accuracy bounds the returned quantile to
    // [x*(1-a), x*(1+a)]; allow a slightly wider band for bucket rounding.
    const double p50_band = 500.0 * 0.02;
    CHECK(q.p50 >= 500.0 - p50_band);
    CHECK(q.p50 <= 500.0 + p50_band);
    MESSAGE("sketch handle: count=" << q.count << " min=" << q.min
                                    << " max=" << q.max << " p50=" << q.p50
                                    << " p90=" << q.p90 << " p99=" << q.p99);

    sk->sketch_free(host.h, a);
    sk->sketch_free(host.h, b);
}

TEST_CASE("plugin host: trace write then read round-trips events") {
    dftu_utils_test::TestEnvironment env(0);
    HostFixture fx;
    dftu_host& host = fx.host();

    const auto* tr = static_cast<const dftu_ext_trace*>(
        host.get_extension(host.h, DFTU_EXT_TRACE));
    REQUIRE(tr != nullptr);

    std::string trace = env.get_dir() + "/out.pfw.gz";
    dftu_trace_writer* w = tr->trace_open_write(host.h, trace.c_str());
    REQUIRE(w != nullptr);

    const std::uint32_t N = 50;
    std::vector<dftu_event> evs(N);
    dftu_str cat = fx.intern_str("POSIX");
    dftu_str name = fx.intern_str("write");
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
    CHECK(tr->trace_write(host.h, w, evs.data(), N) == 0);
    CHECK(tr->trace_close(host.h, w) == 0);
    REQUIRE(file_size(trace) > 0);

    struct Sink {
        const dftu_host* host;
        std::uint64_t count = 0;
        bool first_ok = false;
    };
    Sink sink{&host};
    auto on_event = [](const void* item, void* ud) {
        const dftu_event* e = static_cast<const dftu_event*>(item);
        Sink* s = static_cast<Sink*>(ud);
        if (s->count == 0) {
            std::uint32_t len = 0;
            const char* nm = s->host->resolve(s->host->h, e->name, &len);
            s->first_ok =
                e->ts == 1000000 && std::string_view(nm, len) == "write";
        }
        s->count++;
    };

    int rr = tr->trace_read(host.h, trace.c_str(), on_event, &sink);
    CHECK(rr == 0);
    CHECK(sink.count == N);
    CHECK(sink.first_ok);
    MESSAGE("trace round-trip: wrote=" << N << " read=" << sink.count);
}

TEST_CASE("plugin host: repeated utility calls reuse one runtime") {
    HostFixture fx;
    dftu_host& host = fx.host();
    std::string data = "reuse-one-runtime";
    dftu_bytes in{data.data(), static_cast<std::uint32_t>(data.size())};

    std::uint64_t expect = 0;
    REQUIRE(dftu_util_fnv1a(&host, &in, &expect) == 0);

    const int N = 2000;
    auto t0 = std::chrono::steady_clock::now();
    int ok = 0;
    for (int i = 0; i < N; ++i) {
        std::uint64_t out = 0;
        if (dftu_util_fnv1a(&host, &in, &out) == 0 && out == expect) ++ok;
    }
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(
                  std::chrono::steady_clock::now() - t0)
                  .count();
    CHECK(ok == N);
    // A fresh thread pool per call (~ms of thread create+join each) would put
    // this well above 2ms/call; sub-millisecond evidences a reused runtime.
    CHECK(static_cast<double>(us) / N < 2000.0);
    MESSAGE("reuse: " << N << " calls in " << us << "us ("
                      << static_cast<double>(us) / N << "us/call)");
}

TEST_CASE(
    "plugin host: utility call from an executor thread does not deadlock") {
    HostFixture fx;
    dftu_host& host = fx.host();
    std::string data = "no-deadlock";
    dftu_bytes in{data.data(), static_cast<std::uint32_t>(data.size())};

    // One worker: if the blocking call adopted the current executor and drove
    // to completion on this same thread, it would self-deadlock.
    Runtime rt(1);
    int rc = -1;
    std::uint64_t got = 0;
    rt.scope("caller", [&](CoroScope&) -> coro::CoroTask<void> {
          std::uint64_t out = 0;
          rc = dftu_util_fnv1a(&host, &in, &out);
          got = out;
          co_return;
      }).wait();
    rt.shutdown();
    CHECK(rc == 0);
    CHECK(got != 0);
}

TEST_CASE(
    "plugin host: pull-model stream matches the push model item for item") {
    dftu_utils_test::TestEnvironment env(50);
    env.create_dft_test_gzip_file(50);
    HostFixture fx;
    dftu_host& host = fx.host();

    std::string dir = env.get_dir();
    dftu_directory_scanner_utility_input in{};
    in.path = dftu_bytes{dir.data(), static_cast<std::uint32_t>(dir.size())};
    in.recursive = 0;
    in.populate_size = 1;

    struct Sink {
        int entries = 0;
        std::uint64_t total_size = 0;
    };
    Sink push;
    auto on_item = [](const void* item, void* ud) {
        const dftu_file_entry* e = static_cast<const dftu_file_entry*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->entries++;
        if (e->is_regular_file) s->total_size += e->size;
    };
    const auto* util = static_cast<const dftu_ext_util*>(
        host.get_extension(host.h, DFTU_EXT_UTIL));
    REQUIRE(util != nullptr);
    REQUIRE(util->run_stream(host.h, DFTU_UTIL_DIRECTORY_SCANNER, &in, on_item,
                             &push) == 0);
    REQUIRE(push.entries >= 1);
    REQUIRE(push.total_size > 0);

    int pull_entries = 0;
    std::uint64_t pull_total = 0;
    drive_task(host, 1, [&] {
        return pull_dir(dftracer::utils::plugins::Host{&host}, &in,
                        &pull_entries, &pull_total, 0);
    });

    CHECK(pull_entries == push.entries);
    CHECK(pull_total == push.total_size);
    MESSAGE("pull vs push: entries=" << pull_entries
                                     << " total_size=" << pull_total);
}

TEST_CASE(
    "plugin host: pull-model stream interleaves plugin I/O between pulls") {
    dftu_utils_test::TestEnvironment env(0);
    std::string dir = env.get_dir();
    make_files(dir, 8);

    std::string probe = dir + "/f0.dat";
    const std::string body = "file-0\n";

    HostFixture fx;
    dftu_host& host = fx.host();
    dftu_directory_scanner_utility_input in{};
    in.path = dftu_bytes{dir.data(), static_cast<std::uint32_t>(dir.size())};
    in.recursive = 0;
    in.populate_size = 1;

    int entries = 0;
    std::int64_t bytes_read = 0;
    drive_task(host, 1, [&] {
        return pull_dir_interleave(dftracer::utils::plugins::Host{&host}, &in,
                                   probe.c_str(), &entries, &bytes_read);
    });

    REQUIRE(entries >= 2);  // multiple pulls, each with an interleaved read
    // Every pull ran one full read of the probe file: the async work and the
    // stream both completed, which the push model cannot express.
    CHECK(bytes_read == static_cast<std::int64_t>(body.size()) * entries);
    MESSAGE("interleave: entries=" << entries << " bytes_read=" << bytes_read);
}

TEST_CASE("plugin host: early close of a mid-stream generator does not leak") {
    // Built under --preset asan this is the leak/UAF guard: the RAII Stream
    // destroys a generator suspended at a co_yield, freeing its owned input and
    // arena before end-of-stream.
    dftu_utils_test::TestEnvironment env(0);
    std::string dir = env.get_dir();
    make_files(dir, 8);

    HostFixture fx;
    dftu_host& host = fx.host();
    dftu_directory_scanner_utility_input in{};
    in.path = dftu_bytes{dir.data(), static_cast<std::uint32_t>(dir.size())};
    in.recursive = 0;
    in.populate_size = 1;

    const int stop_after = 2;
    int pulled = 0;
    std::uint64_t total = 0;
    drive_task(host, 1, [&] {
        return pull_dir(dftracer::utils::plugins::Host{&host}, &in, &pulled,
                        &total, stop_after);
    });

    CHECK(pulled == stop_after);  // closed before the 8-file stream ran out
    MESSAGE("early close: pulled=" << pulled << " then closed");
}

TEST_CASE(
    "plugin host: pull-model stream completes on a single-worker executor") {
    // view_scanner does real io-backend decompression per batch; completing the
    // whole pull loop on one worker purely via co_await is structural proof the
    // drive is cooperative and never monopolizes the worker.
    dftu_utils_test::TestEnvironment env(200);
    std::string gz = env.create_dft_test_gzip_file(200);
    REQUIRE(dftu_utils_test::build_index(gz));
    std::string idx = dfti::determine_index_path(gz, "");

    HostFixture fx;
    dftu_host& host = fx.host();
    dftu_view_scanner_input in{};
    in.file_path = dftu_bytes{gz.data(), static_cast<std::uint32_t>(gz.size())};
    in.index_path =
        dftu_bytes{idx.data(), static_cast<std::uint32_t>(idx.size())};
    in.checkpoint_size = 1024;
    in.event_batch_size = 16;
    in.start_byte = 0;
    in.end_byte = UINT64_MAX;

    struct Sink {
        int batches = 0;
        std::uint64_t scanned = 0;
    };
    Sink push;
    auto on_item = [](const void* item, void* ud) {
        const dftu_view_scanner_batch* b =
            static_cast<const dftu_view_scanner_batch*>(item);
        Sink* s = static_cast<Sink*>(ud);
        s->batches++;
        s->scanned += b->events_scanned;
    };
    const auto* util = static_cast<const dftu_ext_util*>(
        host.get_extension(host.h, DFTU_EXT_UTIL));
    REQUIRE(util != nullptr);
    REQUIRE(util->run_stream(host.h, DFTU_UTIL_VIEW_SCANNER, &in, on_item,
                             &push) == 0);
    REQUIRE(push.scanned > 0);

    int batches = 0;
    std::uint64_t scanned = 0;
    drive_task(host, 1, [&] {
        return pull_view_count(dftracer::utils::plugins::Host{&host}, &in,
                               &batches, &scanned);
    });

    CHECK(scanned == push.scanned);
    CHECK(batches == push.batches);
    MESSAGE("single-worker pull: batches=" << batches
                                           << " scanned=" << scanned);
}
