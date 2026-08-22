// Exercises the additive C++-ergonomics layer over the plugin C ABI: the
// Sketch RAII owner, co_await-able Io ops, typed row/key encoders, the
// MonoidValue view, and the owning Arrow helpers.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/dftu_generated_utilities.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugin.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <doctest/doctest.h>
#include <fcntl.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <variant>
#include <vector>

#include "testing_utilities.h"

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::NamedResultRegistry;
using dftracer::utils::plugins::OwnedArrow;
using dftracer::utils::plugins::PluginFold;
using dftracer::utils::plugins::SharedResultRegistry;
namespace views = dftracer::utils::trace::views;
using views::detail::CoverageSet;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::ScanUnit;
namespace coro = dftracer::utils::coro;

namespace {

struct TrivialSlice {
    explicit TrivialSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(TrivialSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct HostFixture {
    StringIntern intern;
    dftu_plugin* plugin =
        dftracer::utils::plugins::make_plugin<TrivialSlice>(nullptr);
    std::unique_ptr<PluginFold> fold =
        std::make_unique<PluginFold>(plugin, intern);
    dftu_host& host() { return fold->host(); }
    dftu_str intern_str(const char* s) {
        return host().intern(host().h, s,
                             static_cast<std::uint32_t>(std::strlen(s)));
    }
    ~HostFixture() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

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

// co_awaits the Io ops directly, and once through Host::await to prove the
// AsyncOp->dftu_task* decay still composes with the old call site.
dftracer::utils::plugins::Task read_probe(dftracer::utils::plugins::Host h,
                                          const char* path,
                                          std::int64_t* bytes) {
    dftracer::utils::plugins::Io io = h.io();
    int fd = -1;
    co_await io.open(path, O_RDONLY, 0, &fd);
    if (fd < 0) co_return;
    char buf[128];
    std::int64_t n = 0;
    co_await io.read(fd, buf, sizeof(buf), &n);
    if (n > 0) *bytes += n;
    int rc = 0;
    co_await h.await(io.close(fd, &rc));
}

}  // namespace

TEST_CASE("plugin cxx: Sketch RAII adds, merges, and frees") {
    HostFixture fx;
    dftracer::utils::plugins::Host h{&fx.host()};

    dftracer::utils::plugins::Sketch a = h.make_sketch();
    dftracer::utils::plugins::Sketch b{h};
    REQUIRE(a);
    REQUIRE(b);

    for (int v = 1; v <= 500; ++v) a.add(static_cast<double>(v));
    for (int v = 501; v <= 1000; ++v) b.add(static_cast<double>(v));
    a.merge(b);

    dftu_quantiles q = a.result();
    CHECK(q.count == 1000);
    CHECK(q.min == doctest::Approx(1.0));
    CHECK(q.max == doctest::Approx(1000.0));
    CHECK(q.mean == doctest::Approx(500.5).epsilon(0.01));

    // Moved-from Sketch is empty; its destructor must not double-free (asan).
    dftracer::utils::plugins::Sketch c = std::move(a);
    CHECK(!a);
    CHECK(c);
    CHECK(c.result().count == 1000);
}

TEST_CASE("plugin cxx: Io ops are co_await-able directly") {
    dftu_utils_test::TestEnvironment env(0);
    std::string path = env.get_dir() + "/probe.txt";
    const std::string body = "alpha\nbeta\ngamma\n";
    {
        std::ofstream f(path, std::ios::binary);
        f << body;
    }

    HostFixture fx;
    std::int64_t bytes = 0;
    drive_task(fx.host(), 1, [&] {
        return read_probe(dftracer::utils::plugins::Host{&fx.host()},
                          path.c_str(), &bytes);
    });
    CHECK(bytes == static_cast<std::int64_t>(body.size()));
}

TEST_CASE("plugin cxx: typed row and key helpers encode without hand casts") {
    using namespace dftracer::utils::plugins;

    dftu_row_val ru = row_u64(2, 7);
    CHECK(ru.comp == 2);
    CHECK(ru.is_f64 == 0);
    CHECK(ru.value.u == 7u);

    dftu_row_val rf = row_f64(1, 3.5);
    CHECK(rf.comp == 1);
    CHECK(rf.is_f64 == 1);
    CHECK(rf.value.f == doctest::Approx(3.5));

    CHECK(key_of(static_cast<std::int32_t>(-5)) == -5);
    CHECK(key_of(static_cast<std::uint32_t>(11)) == 11);
    CHECK(key_of(static_cast<std::int64_t>(1LL << 40)) == (1LL << 40));

    double d = 2.718281828;
    std::int64_t bits = key_of(d);
    double back = 0;
    std::memcpy(&back, &bits, sizeof(back));
    CHECK(back == d);  // exact-bit round-trip

    HostFixture fx;
    dftracer::utils::plugins::Host h{&fx.host()};
    std::int64_t id = key_of(h, std::string_view("POSIX"));
    CHECK(id != DFTU_STR_NONE);
    CHECK(h.str(static_cast<dftu_str>(id)) == "POSIX");
    CHECK(key_of(h, std::string_view("POSIX")) == id);  // interning is stable
}

TEST_CASE("plugin cxx: MonoidValue typed accessors select by kind") {
    dftu_monoid_value mv{};
    mv.kind = DFTU_MONOID_COUNTER;
    mv.as.u64 = 42;
    dftracer::utils::plugins::MonoidValue counter{mv};
    CHECK(counter.kind() == DFTU_MONOID_COUNTER);
    CHECK(counter.as_u64() == 42u);

    dftu_monoid_value mf{};
    mf.kind = DFTU_MONOID_SUM_F64;
    mf.as.f64 = 1.25;
    dftracer::utils::plugins::MonoidValue sum{mf};
    CHECK(sum.kind() == DFTU_MONOID_SUM_F64);
    CHECK(sum.as_f64() == doctest::Approx(1.25));
}

TEST_CASE("plugin cxx: Batch view iterates typed Events") {
    HostFixture fx;
    dftracer::utils::plugins::Host h{&fx.host()};
    dftu_str cat = fx.intern_str("POSIX");
    dftu_str name = fx.intern_str("read");

    const std::uint32_t N = 3;
    std::vector<dftu_event> evs(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        evs[i] = {};
        evs[i].cat = cat;
        evs[i].name = name;
        evs[i].pid = 100 + i;
        evs[i].tid = 200 + i;
        evs[i].ts = 1000 + i;
        evs[i].dur = 10 * (i + 1);
        evs[i].phase = DFTU_PH_COMPLETE;
        evs[i].has_dur = 1;
    }
    dftu_batch raw{evs.data(), N};
    dftracer::utils::plugins::Batch batch{raw};

    CHECK(batch.size() == N);
    CHECK(!batch.empty());

    std::uint32_t i = 0;
    std::uint64_t pid_sum = 0, dur_sum = 0;
    for (const dftracer::utils::plugins::Event& e : batch) {
        CHECK(e.pid() == 100 + i);
        CHECK(e.tid() == 200 + i);
        CHECK(e.ts() == 1000 + i);
        CHECK(e.dur() == 10 * (i + 1));
        CHECK(e.has_dur());
        CHECK(e.phase() == DFTU_PH_COMPLETE);
        CHECK(e.cat_id() == cat);
        CHECK(e.cat(h) == "POSIX");
        CHECK(e.name(h) == "read");
        CHECK(e.arg_count() == 0);
        pid_sum += e.pid();
        dur_sum += e.dur();
        ++i;
    }
    CHECK(i == N);
    CHECK(pid_sum == 100 + 101 + 102);
    CHECK(dur_sum == 10 + 20 + 30);

    // operator[] agrees with iteration.
    CHECK(batch[0].pid() == 100);
    CHECK(batch[2].dur() == 30);
}

TEST_CASE("plugin cxx: Event args expose typed Arg views and dotted lookup") {
    HostFixture fx;
    dftracer::utils::plugins::Host h{&fx.host()};

    dftu_str k_ret = fx.intern_str("args.ret");
    dftu_str k_fd = fx.intern_str("args.fd");
    dftu_str k_path = fx.intern_str("args.path");
    dftu_str v_path = fx.intern_str("/tmp/a");

    dftu_arg args[3];
    args[0] = {};
    args[0].key = k_ret;
    args[0].kind = DFTU_ARG_I64;
    args[0].v.i64 = 42;
    args[1] = {};
    args[1].key = k_fd;
    args[1].kind = DFTU_ARG_F64;
    args[1].v.f64 = 2.5;
    args[2] = {};
    args[2].key = k_path;
    args[2].kind = DFTU_ARG_STR;
    args[2].v.str = v_path;

    dftu_event ev{};
    ev.pid = 7;
    ev.arg_count = 3;
    ev.args = args;
    dftu_batch raw{&ev, 1};
    dftracer::utils::plugins::Batch batch{raw};

    const dftracer::utils::plugins::Event& e = batch[0];
    CHECK(e.arg_count() == 3);
    CHECK(e.args().size() == 3);

    // Range-for over typed Arg views, resolved through the Host.
    std::uint32_t n = 0;
    for (const dftracer::utils::plugins::Arg& a : e.args()) {
        CHECK(a.key(h).substr(0, 5) == "args.");
        ++n;
    }
    CHECK(n == 3);

    // Dotted lookup as one key or as joined components.
    auto ret = e.find_arg(h, "args.ret");
    REQUIRE(ret.has_value());
    CHECK(ret->is_i64());
    CHECK(ret->i64() == 42);

    auto ret2 = e.find_arg(h, "args", "ret");
    REQUIRE(ret2.has_value());
    CHECK(ret2->i64() == 42);

    auto fd = e.find_arg(h, "args", "fd");
    REQUIRE(fd.has_value());
    CHECK(fd->is_f64());
    CHECK(fd->f64() == doctest::Approx(2.5));

    auto path = e.find_arg(h, "args.path");
    REQUIRE(path.has_value());
    CHECK(path->is_str());
    CHECK(path->str(h) == "/tmp/a");

    CHECK(!e.find_arg(h, "args", "missing").has_value());
}

namespace {

// A Slice authored against the ergonomic Batch view: range-for over typed
// Events, no hand indexing of the C dftu_batch.
struct ViewSlice {
    std::uint64_t count = 0;
    std::uint64_t pid_sum = 0;
    std::uint64_t ts_max = 0;
    explicit ViewSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host) {
        for (const dftracer::utils::plugins::Event& e : b) {
            ++count;
            pid_sum += e.pid();
            if (e.ts() > ts_max) ts_max = e.ts();
        }
    }
    void merge(ViewSlice& o) {
        count += o.count;
        pid_sum += o.pid_sum;
        if (o.ts_max > ts_max) ts_max = o.ts_max;
    }
    void finalize(dftracer::utils::plugins::Host) {}
};

}  // namespace

TEST_CASE("plugin cxx: make_plugin dispatches a Batch-view step") {
    HostFixture fx;
    dftu_plugin* p = dftracer::utils::plugins::make_plugin<ViewSlice>(nullptr);
    auto* slice = static_cast<ViewSlice*>(p->make_slice(p->self));
    REQUIRE(slice != nullptr);

    const std::uint32_t N = 4;
    std::vector<dftu_event> evs(N);
    for (std::uint32_t i = 0; i < N; ++i) {
        evs[i] = {};
        evs[i].pid = 10 + i;
        evs[i].ts = 500 + i;
        evs[i].phase = DFTU_PH_COMPLETE;
    }
    dftu_batch b{evs.data(), N};

    dftu_task* t = p->on_batch(slice, &b, &fx.host());
    CHECK(t == nullptr);  // synchronous step returns no task
    CHECK(slice->count == N);
    CHECK(slice->pid_sum == 10 + 11 + 12 + 13);
    CHECK(slice->ts_max == 503);

    p->destroy_slice(slice);
    if (p->destroy) p->destroy(p->self);
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
TEST_CASE("plugin cxx: owning batch_to_arrow releases in its destructor") {
    HostFixture fx;
    dftracer::utils::plugins::Host h{&fx.host()};

    const std::uint32_t N = 4;
    std::vector<dftu_event> evs(N);
    dftu_str cat = fx.intern_str("POSIX");
    dftu_str name = fx.intern_str("read");
    for (std::uint32_t i = 0; i < N; ++i) {
        evs[i] = {};
        evs[i].cat = cat;
        evs[i].name = name;
        evs[i].ts = 1000 + i;
        evs[i].phase = DFTU_PH_COMPLETE;
    }
    dftu_batch b{evs.data(), N};

    dftracer::utils::plugins::OwnedArrow owned = h.batch_to_arrow(b);
    REQUIRE(owned);
    CHECK(owned.array.length == static_cast<std::int64_t>(N));
    // Leaving scope releases both; asan/valgrind would flag a leak otherwise.
}
#endif

namespace {

using dftracer::utils::plugins::Key;
using dftracer::utils::plugins::Map;
using dftracer::utils::plugins::Monoid;

// Feeds two typed Maps per event: a counter keyed on pid via the map[key] += 1
// sugar, and a single-value ARGMAX_I64 keyed on pid that keeps the tid at the
// largest duration. Neither names a DFTU_MONOID_* nor hand-encodes a key.
struct TypedMapSlice {
    explicit TypedMapSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        Map<std::int64_t> hits = h.counter_map("hits", Key<std::int64_t>{});
        Map<std::int64_t> amax =
            h.map("amax", Monoid::ArgMax_I64, Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            const std::int64_t pid = static_cast<std::int64_t>(e.pid());
            hits[pid] += 1;
            amax.add_argby(pid, static_cast<double>(e.dur()),
                           static_cast<std::int64_t>(e.tid()));
        }
    }
    void merge(TypedMapSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct FoldHolder {
    dftu_plugin* plugin;
    std::unique_ptr<PluginFold> fold;
    FoldHolder(dftu_plugin* p, StringIntern& intern, SharedResultRegistry* reg,
               NamedResultRegistry* named)
        : plugin(p),
          fold(std::make_unique<PluginFold>(p, intern, reg, named)) {}
    ~FoldHolder() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

FoldEvent evt(std::uint64_t pid, std::uint64_t dur, std::uint64_t tid) {
    FoldEvent e;
    e.pid = pid;
    e.dur = dur;
    e.has_dur = true;
    e.tid = tid;
    return e;
}

template <class Slice>
void run_slices(NamedResultRegistry& named, SharedResultRegistry& reg,
                StringIntern& intern,
                const std::vector<std::vector<FoldEvent>>& slices) {
    FoldHolder master(dftracer::utils::plugins::make_plugin<Slice>(nullptr),
                      intern, &reg, &named);
    for (const auto& evs : slices) {
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
    }
    Runtime rt(1);
    rt.scope("fin", [&](CoroScope&) -> coro::CoroTask<void> {
          co_await master.fold->finalize(CoverageSet{});
      }).wait();
    rt.shutdown();
}

}  // namespace

TEST_CASE("plugin cxx: typed Map merges counter and argmax across slices") {
    StringIntern intern;
    SharedResultRegistry reg;
    NamedResultRegistry named;

    // Merged counts: pid 1 -> 3, pid 2 -> 2. Argmax tid by duration: pid 1 at
    // dur 30 -> tid 300; pid 2 at dur 50 -> tid 500.
    const std::vector<std::vector<FoldEvent>> slices = {
        {evt(1, 10, 100), evt(1, 30, 300), evt(2, 5, 50)},
        {evt(1, 20, 200), evt(2, 50, 500)}};
    run_slices<TypedMapSlice>(named, reg, intern, slices);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
    auto column = [](const ArrowArray& a, int c) {
        const ArrowArray* ch = a.children[c];
        return static_cast<const std::int64_t*>(ch->buffers[1]) + ch->offset;
    };
    auto value_by_pid = [&](const char* name, std::int64_t pid) {
        auto it = named.results().find(name);
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const ArrowArray& a = std::get<OwnedArrow>(it->second).array;
        REQUIRE(a.n_children == 2);
        const std::int64_t* k = column(a, 0);
        const std::int64_t* v = column(a, 1);
        for (std::int64_t r = 0; r < a.length; ++r)
            if (k[r + a.offset] == pid) return v[r + a.offset];
        FAIL("pid not found");
        return std::int64_t{0};
    };

    CHECK(value_by_pid("hits", 1) == 3);
    CHECK(value_by_pid("hits", 2) == 2);
    CHECK(value_by_pid("amax", 1) == 300);
    CHECK(value_by_pid("amax", 2) == 500);
#endif
}
