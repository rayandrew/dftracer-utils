// Exercises the host services a plugin calls synchronously: Arrow batch export,
// Arrow IPC write, the DDSketch handle, and the dftracer trace write/read
// round-trip. Drives them through a real PluginFold-wired dftu_host, the way a
// loaded plugin would.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/abi.h>
// fold_adapter.h transitively pulls nanoarrow (ArrowArray/ArrowSchema); it must
// precede plugin.h so arrow_abi.h no-ops rather than tripping nanoarrow's outer
// ARROW_FLAG_DICTIONARY_ORDERED sentinel.
#include <dftracer/utils/core/runtime.h>
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

namespace dfti = dftracer::utils::trace::internal;

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
