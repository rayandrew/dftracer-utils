// Coverage for the declare/resolve lifecycle and the runtime-authority
// capability registry in PluginHost: provider discovery, semver best-provider
// selection, reserved-namespace enforcement, and required-vs-optional handling.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/host.h>
#include <dftracer/utils/plugins/plugin.h>
#include <doctest/doctest.h>

#include <algorithm>
#include <array>
#include <cstring>
#include <vector>

using dftracer::utils::plugins::PluginHost;

namespace {

std::size_t position_of(const std::vector<std::size_t>& order, std::size_t v) {
    auto it = std::find(order.begin(), order.end(), v);
    return static_cast<std::size_t>(it - order.begin());
}

}  // namespace

namespace {

enum { MODE_UNSET = 0, MODE_WIRED = 1, MODE_STANDALONE = 2 };

struct FakeState {
    std::vector<dftu_capability> provides;
    std::vector<dftu_requirement> reqs;
    int mode = MODE_UNSET;
    int provider_count = 0;
    dftu_version best{};
};

uint32_t fake_provides(void* self, dftu_capability* out, uint32_t max) {
    auto* s = static_cast<FakeState*>(self);
    auto n = static_cast<uint32_t>(s->provides.size());
    for (uint32_t i = 0; i < n && i < max; ++i) out[i] = s->provides[i];
    return n;
}

uint32_t fake_requires(void* self, dftu_requirement* out, uint32_t max) {
    auto* s = static_cast<FakeState*>(self);
    auto n = static_cast<uint32_t>(s->reqs.size());
    for (uint32_t i = 0; i < n && i < max; ++i) out[i] = s->reqs[i];
    return n;
}

void fake_resolve(void* self, const dftu_host* host) {
    auto* s = static_cast<FakeState*>(self);
    if (s->reqs.empty()) return;
    const auto* c = static_cast<const dftu_ext_comms*>(
        host->get_extension(host->h, DFTU_EXT_COMMS));
    if (!c) return;
    const dftu_requirement& req = s->reqs[0];
    s->provider_count = static_cast<int>(c->provider_count(host->h, req.id));
    dftu_version v{};
    s->mode =
        c->provider_best(host->h, &req, &v) == 0 ? MODE_WIRED : MODE_STANDALONE;
    if (s->mode == MODE_WIRED) s->best = v;
}

const dftu_plugin_comms g_comms = {fake_provides, fake_requires, fake_resolve};

const void* fake_get_extension(void*, const char* ext_id) {
    if (ext_id && std::strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_comms;
    return nullptr;
}

dftu_plugin make_fake(FakeState* st) {
    dftu_plugin p{};
    p.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    p.self = st;
    p.get_extension = fake_get_extension;
    return p;
}

dftu_capability cap(const char* id, uint16_t a, uint16_t b, uint16_t c) {
    dftu_capability x{};
    x.id = id;
    x.ver = dftu_version{a, b, c};
    return x;
}

dftu_requirement req(const char* id, dftu_ver_op op, uint16_t a, uint16_t b,
                     uint16_t c, int required) {
    dftu_requirement r{};
    r.id = id;
    r.op = op;
    r.ver = dftu_version{a, b, c};
    r.required = required;
    return r;
}

// A C++ Slice declaring comms through the make_plugin hooks: a static
// provides()/requires_caps() and a static on_resolve() recording what the host
// registry reported. Mirrors the raw-C get_extension path above.
struct CxxResolveRec {
    int provider_count = -1;
    bool wired = false;
    dftu_version best{};
};
CxxResolveRec g_cxx_rec;

struct CxxProvider {
    explicit CxxProvider(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(CxxProvider&) {}
    void finalize(dftracer::utils::plugins::Host) {}
    static std::array<dftu_capability, 1> provides() {
        return {dftracer::utils::plugins::capability("com.example.cxxtag", 2, 1,
                                                     0)};
    }
};

struct CxxConsumer {
    explicit CxxConsumer(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(CxxConsumer&) {}
    void finalize(dftracer::utils::plugins::Host) {}
    static std::array<dftu_requirement, 1> requires_caps() {
        return {dftracer::utils::plugins::requirement(
            "com.example.cxxtag", DFTU_VER_CARET, 2, 0, 0, /*required=*/false)};
    }
    static void on_resolve(dftracer::utils::plugins::Host h) {
        const dftu_requirement r = requires_caps()[0];
        g_cxx_rec.provider_count = static_cast<int>(h.provider_count(r.id));
        if (auto v = h.provider_best(r)) {
            g_cxx_rec.wired = true;
            g_cxx_rec.best = *v;
        }
    }
};

}  // namespace

TEST_SUITE("PluginComms") {
    TEST_CASE("a C++ Slice declares comms through make_plugin") {
        g_cxx_rec = CxxResolveRec{};
        dftu_plugin* provider =
            dftracer::utils::plugins::make_plugin<CxxProvider>(nullptr);
        dftu_plugin* consumer =
            dftracer::utils::plugins::make_plugin<CxxConsumer>(nullptr);
        REQUIRE(provider->get_extension != nullptr);
        REQUIRE(consumer->get_extension != nullptr);

        // The provider publishes its capability through the wired comms table.
        const auto* pc = static_cast<const dftu_plugin_comms*>(
            provider->get_extension(provider->self, DFTU_EXT_COMMS));
        REQUIRE(pc != nullptr);
        REQUIRE(pc->provides != nullptr);
        CHECK(pc->require_caps == nullptr);
        CHECK(pc->resolve == nullptr);
        dftu_capability got{};
        CHECK(pc->provides(provider->self, &got, 1) == 1);
        CHECK(std::strcmp(got.id, "com.example.cxxtag") == 0);
        CHECK(got.ver.major == 2);
        CHECK(got.ver.minor == 1);

        PluginHost host;
        host.inject_plugin(consumer);  // consumer first: resolve reorders
        host.inject_plugin(provider);
        CHECK(host.resolve());

        CHECK(g_cxx_rec.provider_count == 1);
        CHECK(g_cxx_rec.wired);
        CHECK(g_cxx_rec.best.major == 2);
        CHECK(g_cxx_rec.best.minor == 1);

        provider->destroy(provider->self);
        consumer->destroy(consumer->self);
    }

    TEST_CASE("present provider wires the consumer at the highest version") {
        FakeState p12, p15, consumer;
        p12.provides = {cap("com.example.tag", 1, 2, 0)};
        p15.provides = {cap("com.example.tag", 1, 5, 0)};
        consumer.reqs = {
            req("com.example.tag", DFTU_VER_CARET, 1, 0, 0, /*required=*/0)};

        dftu_plugin a = make_fake(&p12), b = make_fake(&p15),
                    c = make_fake(&consumer);
        PluginHost host;
        host.inject_plugin(&a);
        host.inject_plugin(&b);
        host.inject_plugin(&c);

        CHECK(host.resolve());
        CHECK(consumer.mode == MODE_WIRED);
        CHECK(consumer.provider_count == 2);
        CHECK(consumer.best.major == 1);
        CHECK(consumer.best.minor == 5);
        CHECK(consumer.best.patch == 0);
    }

    TEST_CASE("absent provider falls back to standalone") {
        FakeState consumer;
        consumer.reqs = {
            req("com.example.tag", DFTU_VER_CARET, 1, 0, 0, /*required=*/0)};
        dftu_plugin c = make_fake(&consumer);
        PluginHost host;
        host.inject_plugin(&c);

        CHECK(host.resolve());
        CHECK(consumer.mode == MODE_STANDALONE);
        CHECK(consumer.provider_count == 0);
    }

    TEST_CASE("a plugin declaring a dft. capability is rejected") {
        FakeState bad;
        bad.provides = {cap("dftu.cap.events", 1, 0, 0)};
        dftu_plugin p = make_fake(&bad);
        PluginHost host;
        host.inject_plugin(&p);

        CHECK_FALSE(host.resolve());
    }

    TEST_CASE("a plugin providing a blessed host capability id is rejected") {
        FakeState bad;
        bad.provides = {cap(DFTU_CAP_EVENTS, 1, 0, 0)};
        dftu_plugin p = make_fake(&bad);
        PluginHost host;
        host.inject_plugin(&p);

        CHECK_FALSE(host.resolve());  // reserved dft. namespace

        // The blessed ids are usable constants; requiring one is allowed.
        dftu_requirement r =
            req(DFTU_CAP_RESOLVED_EVENTS, DFTU_VER_GE, 1, 0, 0, /*required=*/0);
        CHECK(std::strcmp(r.id, "dftu.cap.resolved_events") == 0);
        CHECK(std::strcmp(DFTU_CAP_TIME_WINDOW, "dftu.cap.time_window") == 0);
    }

    TEST_CASE("an unmet required capability fails resolve") {
        FakeState consumer;
        consumer.reqs = {
            req("com.example.tag", DFTU_VER_GE, 1, 0, 0, /*required=*/1)};
        dftu_plugin c = make_fake(&consumer);
        PluginHost host;
        host.inject_plugin(&c);

        CHECK_FALSE(host.resolve());
    }

    TEST_CASE("an incompatible major hides the provider from the consumer") {
        FakeState p2, consumer;
        p2.provides = {cap("com.example.tag", 2, 0, 0)};
        consumer.reqs = {
            req("com.example.tag", DFTU_VER_CARET, 1, 0, 0, /*required=*/0)};
        dftu_plugin a = make_fake(&p2), c = make_fake(&consumer);
        PluginHost host;
        host.inject_plugin(&a);
        host.inject_plugin(&c);

        CHECK(host.resolve());
        CHECK(consumer.mode == MODE_STANDALONE);
        CHECK(consumer.provider_count == 1);  // any-version count still sees it
    }

    TEST_CASE("resolve orders a provider before a consumer-first requirer") {
        FakeState consumer, producer;
        consumer.reqs = {
            req("com.example.tag", DFTU_VER_GE, 1, 0, 0, /*required=*/0)};
        producer.provides = {cap("com.example.tag", 1, 0, 0)};

        dftu_plugin c = make_fake(&consumer), p = make_fake(&producer);
        PluginHost host;
        host.inject_plugin(&c);  // consumer first in CLI order
        host.inject_plugin(&p);

        CHECK(host.resolve());
        auto order = host.fold_order();
        REQUIRE(order.size() == 2);
        CHECK(position_of(order, 1) < position_of(order, 0));  // producer first
    }

    TEST_CASE("a no-dependency set keeps natural fold order") {
        FakeState a, b, c;
        dftu_plugin pa = make_fake(&a), pb = make_fake(&b), pc = make_fake(&c);
        PluginHost host;
        host.inject_plugin(&pa);
        host.inject_plugin(&pb);
        host.inject_plugin(&pc);

        CHECK(host.resolve());
        CHECK(host.fold_order() == std::vector<std::size_t>{0, 1, 2});
    }

    TEST_CASE("a provide/require cycle falls back to natural order") {
        FakeState x, y;
        x.provides = {cap("com.example.a", 1, 0, 0)};
        x.reqs = {req("com.example.b", DFTU_VER_GE, 1, 0, 0, /*required=*/0)};
        y.provides = {cap("com.example.b", 1, 0, 0)};
        y.reqs = {req("com.example.a", DFTU_VER_GE, 1, 0, 0, /*required=*/0)};

        dftu_plugin px = make_fake(&x), py = make_fake(&y);
        PluginHost host;
        host.inject_plugin(&px);
        host.inject_plugin(&py);

        CHECK(host.resolve());  // cycle does not fail resolve
        CHECK(host.fold_order() == std::vector<std::size_t>{0, 1});
    }
}
