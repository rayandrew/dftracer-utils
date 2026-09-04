// Exercises the additive C++ wrappers layered over the plugin C ABI: the typed
// Handle (merge-handle accumulator), the typed ProductMap/NestedMap factories,
// the Interned STR-key path, the scoped JoinType, the Config array accessors,
// and the typed publish/consume ports. Each keeps the raw Host method it wraps.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugin.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <dftracer/utils/trace/views/view_scan.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <variant>
#include <vector>

namespace views = dftracer::utils::trace::views;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::NamedResultRegistry;
using dftracer::utils::plugins::PluginFold;
using dftracer::utils::plugins::SharedResultRegistry;
using views::detail::CoverageSet;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::FoldPortBus;
using views::detail::ScanUnit;
namespace coro = dftracer::utils::coro;

using dftracer::utils::plugins::Interned;
using dftracer::utils::plugins::interned;
using dftracer::utils::plugins::JoinType;
using dftracer::utils::plugins::Key;
using dftracer::utils::plugins::Monoid;

namespace {

FoldEvent evt(std::uint64_t pid, std::uint64_t dur, std::uint32_t fhash = 0,
              std::uint32_t name = 0) {
    FoldEvent e;
    e.pid = pid;
    e.dur = dur;
    e.has_dur = true;
    e.fhash_id = fhash;
    e.name_id = name;
    return e;
}

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
    rt.scope("fin", [&](dftracer::utils::CoroScope&) -> coro::CoroTask<void> {
          co_await master.fold->finalize(CoverageSet{});
      }).wait();
    rt.shutdown();
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
using dftracer::utils::plugins::OwnedArrow;

const ArrowArray* result_array(NamedResultRegistry& named, const char* name) {
    auto it = named.results().find(name);
    if (it == named.results().end()) return nullptr;
    if (!std::holds_alternative<OwnedArrow>(it->second)) return nullptr;
    return &std::get<OwnedArrow>(it->second).array;
}

const std::int64_t* i64_col(const ArrowArray& a, int c) {
    const ArrowArray* ch = a.children[c];
    return static_cast<const std::int64_t*>(ch->buffers[1]) + ch->offset;
}
const double* f64_col(const ArrowArray& a, int c) {
    const ArrowArray* ch = a.children[c];
    return static_cast<const double*>(ch->buffers[1]) + ch->offset;
}
#endif

// #2 Handle: accumulate a COUNTER and a SUM_F64 through the typed Handle.
#define H_CNT "com.example.wrap@1#cnt"
#define H_SUM "com.example.wrap@1#sum"

struct HandleSlice {
    explicit HandleSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        dftracer::utils::plugins::Handle cnt = h.handle(H_CNT, Monoid::Counter);
        dftracer::utils::plugins::Handle sum = h.handle(H_SUM, Monoid::Sum_F64);
        REQUIRE(cnt);
        REQUIRE(sum);
        cnt.add(static_cast<std::uint64_t>(b.size()));
        for (const dftracer::utils::plugins::Event& e : b)
            sum.add(static_cast<double>(e.dur()));
    }
    void merge(HandleSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// #3 product_map: value is (COUNTER, SUM_F64); component 0 via +=, 1 via
// add_at.
struct ProductSlice {
    explicit ProductSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.product_map("pedges", {Monoid::Counter, Monoid::Sum_F64},
                               Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            const std::int64_t pid = static_cast<std::int64_t>(e.pid());
            m[pid] += 1;
            m.add_at(pid, 1, static_cast<double>(e.dur()));
        }
    }
    void merge(ProductSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// #3 nested_map: outer {pid}, inner {fhash}, inner value (COUNTER, SUM_F64).
struct NestedSlice {
    explicit NestedSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.nested_map("nwide", {Monoid::Counter, Monoid::Sum_F64},
                              Key<std::int64_t>{}, Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            const auto outer = std::tuple{static_cast<std::int64_t>(e.pid())};
            const auto inner =
                std::tuple{static_cast<std::int64_t>(e.fhash_id().raw())};
            m.add(outer, inner, 0, static_cast<std::uint64_t>(1));
            m.add(outer, inner, 1, static_cast<double>(e.dur()));
        }
    }
    void merge(NestedSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// #4 Interned STR key: a (pid, name-id) counter where the name slot is fed an
// already-interned id via interned(), so it is not re-interned.
struct InternedKeySlice {
    explicit InternedKeySlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.counter_map("iname", Key<std::int64_t, Interned>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            if (e.name_id().absent()) continue;
            m[std::tuple{static_cast<std::int64_t>(e.pid()),
                         interned(e.name_id())}] += 1;
        }
    }
    void merge(InternedKeySlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// #5 JoinType: build two typed maps and declare an inner join via the scoped
// enum overload; the host executes it at finalize, emitting "jinner".
struct JoinSlice {
    explicit JoinSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto c = h.counter_map("jl", Key<std::int64_t>{});
        auto d = h.sum_map("jr", Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            const std::int64_t pid = static_cast<std::int64_t>(e.pid());
            c[pid] += 1;
            d.add(pid, static_cast<double>(e.dur()));
        }
        h.map_declare_join("jinner", "jl", "jr", JoinType::Inner);
    }
    void merge(JoinSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

}  // namespace

TEST_SUITE("PluginCxxWrappers") {
    TEST_CASE("Handle accumulates a counter and a sum across worker slices") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<HandleSlice>(nullptr), intern,
            &reg, &named);

        const std::vector<std::vector<FoldEvent>> slices = {
            {evt(1, 10), evt(1, 20)}, {evt(2, 30)}};
        std::uint64_t n = 0;
        double total = 0;
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
            n += evs.size();
            for (const auto& e : evs) total += static_cast<double>(e.dur);
        }
        Runtime rt(1);
        rt.scope("fin",
                 [&](dftracer::utils::CoroScope&) -> coro::CoroTask<void> {
                     co_await master.fold->finalize(CoverageSet{});
                 })
            .wait();
        rt.shutdown();

        dftu_monoid_value cv{};
        REQUIRE(master.fold->handle_result(H_CNT, &cv) == 0);
        CHECK(cv.kind == DFTU_MONOID_COUNTER);
        CHECK(cv.as.u64 == n);

        dftu_monoid_value sv{};
        REQUIRE(master.fold->handle_result(H_SUM, &sv) == 0);
        CHECK(sv.kind == DFTU_MONOID_SUM_F64);
        CHECK(sv.as.f64 == doctest::Approx(total));
    }

    TEST_CASE("Handle::result yields nullopt for an unknown cap") {
        StringIntern intern;
        dftu_plugin* p =
            dftracer::utils::plugins::make_plugin<HandleSlice>(nullptr);
        auto fold = std::make_unique<PluginFold>(p, intern);
        dftracer::utils::plugins::Host h{&fold->host()};
        dftracer::utils::plugins::Handle miss =
            h.handle("com.example.nope@1", Monoid::Counter);
        CHECK_FALSE(miss.result().has_value());
        fold.reset();
        if (p->destroy) p->destroy(p->self);
    }

    TEST_CASE("product_map materializes a counter and a sum component") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        // pid 1: count 3, sum 60; pid 2: count 1, sum 5.
        const std::vector<std::vector<FoldEvent>> slices = {
            {evt(1, 10), evt(1, 20)}, {evt(1, 30), evt(2, 5)}};
        run_slices<ProductSlice>(named, reg, intern, slices);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        const ArrowArray* a = result_array(named, "pedges");
        REQUIRE(a != nullptr);
        REQUIRE(a->n_children == 3);  // key + v0 + v1
        REQUIRE(a->length == 2);
        const std::int64_t* k = i64_col(*a, 0);
        const std::int64_t* v0 = i64_col(*a, 1);
        const double* v1 = f64_col(*a, 2);
        bool saw1 = false, saw2 = false;
        for (std::int64_t r = 0; r < a->length; ++r) {
            const std::int64_t row = r + a->offset;
            if (k[row] == 1) {
                CHECK(v0[row] == 3);
                CHECK(v1[row] == doctest::Approx(60.0));
                saw1 = true;
            } else if (k[row] == 2) {
                CHECK(v0[row] == 1);
                CHECK(v1[row] == doctest::Approx(5.0));
                saw2 = true;
            }
        }
        CHECK(saw1);
        CHECK(saw2);
#endif
    }

    TEST_CASE("nested_map materializes one row per outer key") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        const std::vector<std::vector<FoldEvent>> slices = {
            {evt(1, 10, 100), evt(1, 20, 200)}, {evt(2, 30, 100)}};
        run_slices<NestedSlice>(named, reg, intern, slices);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        const ArrowArray* a = result_array(named, "nwide");
        REQUIRE(a != nullptr);
        // Outer key column + nested value column; one row per distinct pid.
        REQUIRE(a->length == 2);
        const std::int64_t* k = i64_col(*a, 0);
        CHECK(((k[a->offset] == 1 && k[a->offset + 1] == 2) ||
               (k[a->offset] == 2 && k[a->offset + 1] == 1)));
#endif
    }

    TEST_CASE("Interned STR key seeds a name slot without re-interning") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        const std::uint32_t read_id = intern.get_or_insert("read");
        const std::uint32_t write_id = intern.get_or_insert("write");
        const std::vector<std::vector<FoldEvent>> slices = {
            {evt(1, 10, 0, read_id), evt(1, 20, 0, read_id)},
            {evt(1, 5, 0, read_id), evt(2, 7, 0, write_id)}};
        run_slices<InternedKeySlice>(named, reg, intern, slices);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("iname");
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 3);  // pid, name (utf8), count
        REQUIRE(a.length == 2);
        // The name slot resolves to a real utf8 column (not a raw id).
        REQUIRE(owned.schema.children[1]->format != nullptr);
        CHECK(std::string_view(owned.schema.children[1]->format) == "u");

        const ArrowArray* namec = a.children[1];
        const auto* offs =
            static_cast<const std::int32_t*>(namec->buffers[1]) + namec->offset;
        const char* data = static_cast<const char*>(namec->buffers[2]);
        const std::int64_t* cnt = i64_col(a, 2);
        bool saw_read = false, saw_write = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            std::string name(data + offs[r], offs[r + 1] - offs[r]);
            if (name == "read") {
                CHECK(cnt[r + a.offset] == 3);
                saw_read = true;
            } else if (name == "write") {
                CHECK(cnt[r + a.offset] == 1);
                saw_write = true;
            }
        }
        CHECK(saw_read);
        CHECK(saw_write);
#endif
    }

    TEST_CASE("JoinType enumerators mirror the raw dftu_join_type constants") {
        CHECK(static_cast<dftu_join_type>(JoinType::Inner) == DFTU_JOIN_INNER);
        CHECK(static_cast<dftu_join_type>(JoinType::Left) == DFTU_JOIN_LEFT);
        CHECK(static_cast<dftu_join_type>(JoinType::Right) == DFTU_JOIN_RIGHT);
        CHECK(static_cast<dftu_join_type>(JoinType::Full) == DFTU_JOIN_FULL);
    }

    TEST_CASE("map_declare_join(JoinType) emits the joined map at finalize") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        const std::vector<std::vector<FoldEvent>> slices = {
            {evt(1, 10), evt(2, 20)}, {evt(1, 30)}};
        run_slices<JoinSlice>(named, reg, intern, slices);
        CHECK(named.results().find("jinner") != named.results().end());
    }
}

TEST_SUITE("PluginCxxConfigArrays") {
    TEST_CASE("Config array accessors read i64/f64/str arrays") {
        dftu_value ints[3];
        for (int i = 0; i < 3; ++i) {
            ints[i] = {};
            ints[i].kind = DFTU_VAL_I64;
            ints[i].as.i64 = 10 * (i + 1);
        }
        dftu_value dbls[2];
        for (int i = 0; i < 2; ++i) {
            dbls[i] = {};
            dbls[i].kind = DFTU_VAL_F64;
            dbls[i].as.f64 = 1.5 * (i + 1);
        }
        const char* words[2] = {"alpha", "beta"};
        dftu_value strs[2];
        for (int i = 0; i < 2; ++i) {
            strs[i] = {};
            strs[i].kind = DFTU_VAL_STR;
            strs[i].as.str = words[i];
            strs[i].count = static_cast<std::uint32_t>(std::strlen(words[i]));
        }

        dftu_value arr_i{};
        arr_i.kind = DFTU_VAL_ARRAY;
        arr_i.count = 3;
        arr_i.as.items = ints;
        dftu_value arr_f{};
        arr_f.kind = DFTU_VAL_ARRAY;
        arr_f.count = 2;
        arr_f.as.items = dbls;
        dftu_value arr_s{};
        arr_s.kind = DFTU_VAL_ARRAY;
        arr_s.count = 2;
        arr_s.as.items = strs;

        dftu_member members[3] = {
            {"nums", 4, &arr_i}, {"reals", 5, &arr_f}, {"words", 5, &arr_s}};
        dftu_value root{};
        root.kind = DFTU_VAL_OBJECT;
        root.count = 3;
        root.as.members = members;

        dftracer::utils::plugins::Config cfg{&root};
        CHECK(cfg.get_int_array("nums") ==
              std::vector<std::int64_t>{10, 20, 30});
        CHECK(cfg.get_double_array("reals") == std::vector<double>{1.5, 3.0});
        auto ws = cfg.get_string_array("words");
        REQUIRE(ws.size() == 2);
        CHECK(ws[0] == "alpha");
        CHECK(ws[1] == "beta");

        // Absent / wrong-kind keys yield empty vectors and a null array().
        CHECK(cfg.get_int_array("missing").empty());
        CHECK(cfg.array("missing").is_null());
    }
}

namespace {

constexpr const char* PORT_CAP = "com.example.wrap.port";

struct Edge {
    std::uint64_t count;
    double sum;
};

std::uint64_t g_count = 0;
double g_sum = 0;
bool g_present = false;

// #7 typed ports: the producer publishes an Edge; the consumer reads it back
// as a typed value the same batch.
struct PortProducer {
    explicit PortProducer(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        Edge e{b.size(), 0.0};
        for (const dftracer::utils::plugins::Event& ev : b)
            e.sum += static_cast<double>(ev.dur());
        auto port = h.publish_port<Edge>(PORT_CAP);
        port.send(e);
    }
    void merge(PortProducer&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct PortConsumer {
    explicit PortConsumer(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch&,
              dftracer::utils::plugins::Host h) {
        auto in = h.consume_port<Edge>(PORT_CAP);
        std::optional<Edge> v = in.recv();
        g_present = v.has_value();
        g_count = g_present ? v->count : 0;
        g_sum = g_present ? v->sum : 0.0;
    }
    void merge(PortConsumer&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct PortHolder {
    dftu_plugin* plugin;
    std::unique_ptr<PluginFold> fold;
    PortHolder(dftu_plugin* p, StringIntern& intern, FoldPortBus* bus)
        : plugin(p), fold(std::make_unique<PluginFold>(p, intern)) {
        fold->bind_port_bus(bus);
    }
    ~PortHolder() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

}  // namespace

TEST_SUITE("PluginCxxPorts") {
    TEST_CASE("typed publish_port/consume_port round-trip a struct value") {
        StringIntern intern;
        FoldPortBus bus;
        PortHolder prod(
            dftracer::utils::plugins::make_plugin<PortProducer>(nullptr),
            intern, &bus);
        PortHolder cons(
            dftracer::utils::plugins::make_plugin<PortConsumer>(nullptr),
            intern, &bus);

        std::vector<FoldEvent> evs = {evt(1, 10), evt(1, 20), evt(2, 30)};
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};

        g_present = false;
        g_count = 0;
        g_sum = 0;
        bus.clear();
        prod.fold->step(fb);
        cons.fold->step(fb);
        CHECK(g_present);
        CHECK(g_count == 3);
        CHECK(g_sum == doctest::Approx(60.0));

        // Consumer before producer sees nothing (the ordering rule).
        bus.clear();
        g_present = true;
        cons.fold->step(fb);
        CHECK_FALSE(g_present);
    }
}
