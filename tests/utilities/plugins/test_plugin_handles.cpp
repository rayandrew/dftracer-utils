// Named monoid mergeable handles: a producer accumulates a COUNTER and a SKETCH
// across simulated worker slices merged into a master, whose finalize publishes
// into a shared registry a consumer PluginFold then reads by cap id.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/monoid.h>
#include <dftracer/utils/plugins/plugin.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace views = dftracer::utils::trace::views;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::MonoidAccumulator;
using dftracer::utils::plugins::PluginFold;
using dftracer::utils::plugins::SharedResultRegistry;
using views::detail::CoverageSet;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::ScanUnit;
namespace coro = dftracer::utils::coro;

namespace {

#define STATS_CAP "com.example.dur_stats@1.0.0"
#define STATS_SKETCH_CAP "com.example.dur_stats@1.0.0#dur"

// Each batch: sum the durations into a SKETCH handle and count the events into
// a COUNTER handle, both named under the same capability.
struct ProducerSlice {
    explicit ProducerSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        dftu_handle* cnt = h.shared_get(STATS_CAP, DFTU_MONOID_COUNTER);
        dftu_handle* skt = h.shared_get(STATS_SKETCH_CAP, DFTU_MONOID_SKETCH);
        h.handle_add(cnt, static_cast<std::uint64_t>(b.size()));
        for (const dftracer::utils::plugins::Event& e : b)
            h.handle_add(skt, static_cast<double>(e.dur()), 1.0);
    }
    void merge(ProducerSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct ConsumerSlice {
    explicit ConsumerSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch&, dftracer::utils::plugins::Host) {}
    void merge(ConsumerSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

#define M_MIN "com.example.multi@1#min"
#define M_MAX "com.example.multi@1#max"
#define M_AND "com.example.multi@1#and"
#define M_OR "com.example.multi@1#or"
#define M_BITS "com.example.multi@1#bits"
#define M_DISTINCT "com.example.multi@1#distinct"

// Feed each event's duration into one handle of every u64 monoid kind, so a
// merged master reflects the cross-worker reduction of each.
struct MultiSlice {
    explicit MultiSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        dftu_handle* mn = h.shared_get(M_MIN, DFTU_MONOID_MIN_U64);
        dftu_handle* mx = h.shared_get(M_MAX, DFTU_MONOID_MAX_U64);
        dftu_handle* an = h.shared_get(M_AND, DFTU_MONOID_BOOL_AND);
        dftu_handle* orr = h.shared_get(M_OR, DFTU_MONOID_BOOL_OR);
        dftu_handle* bs = h.shared_get(M_BITS, DFTU_MONOID_BITSET_OR);
        dftu_handle* ds = h.shared_get(M_DISTINCT, DFTU_MONOID_DISTINCT);
        for (const dftracer::utils::plugins::Event& e : b) {
            std::uint64_t d = e.dur();
            h.handle_add(mn, d);
            h.handle_add(mx, d);
            h.handle_add(an, d);
            h.handle_add(orr, d);
            h.handle_add(bs, d);
            h.handle_add(ds, d);
        }
    }
    void merge(MultiSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct FoldHolder {
    dftu_plugin* plugin;
    std::unique_ptr<PluginFold> fold;
    FoldHolder(dftu_plugin* p, StringIntern& intern, SharedResultRegistry* reg)
        : plugin(p), fold(std::make_unique<PluginFold>(p, intern, reg)) {}
    ~FoldHolder() {
        fold.reset();
        if (plugin && plugin->destroy) plugin->destroy(plugin->self);
    }
};

std::vector<FoldEvent> make_events(std::uint64_t base, std::uint64_t count) {
    std::vector<FoldEvent> evs(count);
    for (std::uint64_t i = 0; i < count; ++i) {
        evs[i].has_dur = true;
        evs[i].dur = base + i;
    }
    return evs;
}

void finalize_now(PluginFold& f) {
    Runtime rt(1);
    rt.scope("fin", [&](dftracer::utils::CoroScope&) -> coro::CoroTask<void> {
          co_await f.finalize(CoverageSet{});
      }).wait();
    rt.shutdown();
}

}  // namespace

TEST_SUITE("MonoidAccumulator") {
    TEST_CASE("counter/sum/min/max merge and to_value") {
        MonoidAccumulator c(DFTU_MONOID_COUNTER);
        c.add_u64(3);
        MonoidAccumulator c2(DFTU_MONOID_COUNTER);
        c2.add_u64(4);
        c.merge(c2);
        CHECK(c.to_value().as.u64 == 7);

        MonoidAccumulator s(DFTU_MONOID_SUM_F64), s2(DFTU_MONOID_SUM_F64);
        s.add_f64(1.5, 0);
        s2.add_f64(2.25, 0);
        s.merge(s2);
        CHECK(s.to_value().as.f64 == doctest::Approx(3.75));

        MonoidAccumulator mn(DFTU_MONOID_MIN_F64), mn2(DFTU_MONOID_MIN_F64);
        mn.add_f64(5.0, 0);
        mn.add_f64(2.0, 0);
        mn2.add_f64(9.0, 0);
        mn.merge(mn2);
        CHECK(mn.to_value().as.f64 == doctest::Approx(2.0));

        MonoidAccumulator mx(DFTU_MONOID_MAX_F64), mx2(DFTU_MONOID_MAX_F64);
        mx.add_f64(5.0, 0);
        mx2.add_f64(9.0, 0);
        mx2.add_f64(1.0, 0);
        mx.merge(mx2);
        CHECK(mx.to_value().as.f64 == doctest::Approx(9.0));
    }

    TEST_CASE("u64 min/max/bool/bitset/distinct kinds merge and to_value") {
        MonoidAccumulator mn(DFTU_MONOID_MIN_U64), mn2(DFTU_MONOID_MIN_U64);
        mn.add_u64(9);
        mn.add_u64(4);
        mn2.add_u64(7);
        mn.merge(mn2);
        CHECK(mn.to_value().as.u64 == 4);

        MonoidAccumulator mx(DFTU_MONOID_MAX_U64), mx2(DFTU_MONOID_MAX_U64);
        mx.add_u64(9);
        mx2.add_u64(12);
        mx2.add_u64(3);
        mx.merge(mx2);
        CHECK(mx.to_value().as.u64 == 12);

        CHECK(MonoidAccumulator(DFTU_MONOID_BOOL_AND).to_value().as.u64 ==
              1);  // identity
        MonoidAccumulator aa(DFTU_MONOID_BOOL_AND), ab(DFTU_MONOID_BOOL_AND);
        aa.add_u64(1);
        aa.add_u64(5);
        ab.add_u64(0);  // one false flips the merged AND
        aa.merge(ab);
        CHECK(aa.to_value().as.u64 == 0);

        CHECK(MonoidAccumulator(DFTU_MONOID_BOOL_OR).to_value().as.u64 ==
              0);  // identity
        MonoidAccumulator oa(DFTU_MONOID_BOOL_OR), ob(DFTU_MONOID_BOOL_OR);
        oa.add_u64(0);
        ob.add_u64(2);
        oa.merge(ob);
        CHECK(oa.to_value().as.u64 == 1);

        MonoidAccumulator ba(DFTU_MONOID_BITSET_OR), bb(DFTU_MONOID_BITSET_OR);
        ba.add_u64(0x1);
        ba.add_u64(0x2);
        bb.add_u64(0x8);
        ba.merge(bb);
        CHECK(ba.to_value().as.u64 == 0xB);

        // Duplicates across slices count once; the sparse form is exact.
        MonoidAccumulator da(DFTU_MONOID_DISTINCT), db(DFTU_MONOID_DISTINCT);
        da.add_u64(1);
        da.add_u64(2);
        da.add_u64(2);
        db.add_u64(2);
        db.add_u64(3);
        da.merge(db);
        CHECK(da.to_value().as.u64 == 3);
    }
}

TEST_SUITE("PluginHandles") {
    TEST_CASE("cross-worker merge sums the counter and pools the sketch") {
        StringIntern intern;
        SharedResultRegistry reg;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<ProducerSlice>(nullptr),
            intern, &reg);

        // Three simulated worker slices, each fed a distinct batch, then merged
        // into the master the way fuse folds worker slices into the shared
        // fold.
        const std::uint64_t counts[3] = {5, 7, 4};
        std::uint64_t total_events = 0;
        for (int w = 0; w < 3; ++w) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            auto evs = make_events(/*base=*/1 + 10 * w, counts[w]);
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
            total_events += counts[w];
        }
        finalize_now(*master.fold);

        dftu_monoid_value cv{};
        REQUIRE(master.fold->handle_result(STATS_CAP, &cv) == 0);
        CHECK(cv.kind == DFTU_MONOID_COUNTER);
        CHECK(cv.as.u64 == total_events);

        dftu_monoid_value sv{};
        REQUIRE(master.fold->handle_result(STATS_SKETCH_CAP, &sv) == 0);
        CHECK(sv.kind == DFTU_MONOID_SKETCH);
        CHECK(sv.as.quant.count == total_events);
        CHECK(sv.as.quant.min == doctest::Approx(1.0));
        CHECK(sv.as.quant.p50 > 0.0);
        CHECK(sv.as.quant.max >= sv.as.quant.p99);
    }

    TEST_CASE("cross-worker merges every u64 monoid kind") {
        StringIntern intern;
        SharedResultRegistry reg;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<MultiSlice>(nullptr), intern,
            &reg);

        // Crafted per-slice durations; the last slice carries a zero.
        const std::vector<std::vector<std::uint64_t>> slices = {
            {1, 3}, {4, 4}, {8, 0}};
        for (const auto& durs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            std::vector<FoldEvent> evs(durs.size());
            for (std::size_t i = 0; i < durs.size(); ++i) {
                evs[i].has_dur = true;
                evs[i].dur = durs[i];
            }
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

        dftu_monoid_value v{};
        REQUIRE(master.fold->handle_result(M_MIN, &v) == 0);
        CHECK(v.as.u64 == 0);
        REQUIRE(master.fold->handle_result(M_MAX, &v) == 0);
        CHECK(v.as.u64 == 8);
        REQUIRE(master.fold->handle_result(M_AND, &v) == 0);
        CHECK(v.as.u64 == 0);  // the zero event makes the merged AND false
        REQUIRE(master.fold->handle_result(M_OR, &v) == 0);
        CHECK(v.as.u64 == 1);
        REQUIRE(master.fold->handle_result(M_BITS, &v) == 0);
        CHECK(v.as.u64 == 15);  // 1|3|4|8 across disjoint slices
        REQUIRE(master.fold->handle_result(M_DISTINCT, &v) == 0);
        CHECK(v.as.u64 == 5);   // {0,1,3,4,8}; sparse form is exact
    }

    TEST_CASE("a consumer reads the producer's merged value cross-plugin") {
        StringIntern intern;
        SharedResultRegistry reg;
        FoldHolder producer(
            dftracer::utils::plugins::make_plugin<ProducerSlice>(nullptr),
            intern, &reg);
        FoldHolder consumer(
            dftracer::utils::plugins::make_plugin<ConsumerSlice>(nullptr),
            intern, &reg);

        auto slice = producer.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        auto evs = make_events(1, 6);
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        producer.fold->merge(*pf);
        finalize_now(*producer.fold);  // publishes before the consumer reads

        dftu_monoid_value cv{};
        REQUIRE(consumer.fold->handle_result(STATS_CAP, &cv) == 0);
        CHECK(cv.as.u64 == 6);

        // An unknown cap id yields -1.
        dftu_monoid_value miss{};
        CHECK(consumer.fold->handle_result("com.example.nope@1", &miss) == -1);
    }
}
