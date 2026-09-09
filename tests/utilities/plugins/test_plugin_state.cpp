// Tier 2: a plugin registers its own mergeable state type and the host drives
// it like an accumulator without knowing its shape - one instance per worker
// slice, merged at the fan-in, spilled past the memory budget, finalized once.
// The shape here (a per-name map) is deliberately one the engine's agg table
// could do; what is under test is the machinery around it, not the shape.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/fold_adapter.h>

#include <memory>
// After fold_adapter.h so nanoarrow is set up before dataframe/abi.h's
// arrow_abi.
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/trace/schema.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <span>
#include <string>
#include <variant>
#include <vector>

using dftracer::utils::NO_SPILL_BUDGET;
using dftracer::utils::Runtime;
using dftracer::utils::StringIntern;
using dftracer::utils::plugins::NamedResultRegistry;
using dftracer::utils::plugins::PluginFold;
using dftracer::utils::plugins::RegisteredState;
using dftracer::utils::plugins::StateRegistry;
using dftracer::utils::trace::RecordPhase;
namespace coro = dftracer::utils::coro;
namespace views = dftracer::utils::trace::views;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::ScanUnit;

namespace {

constexpr const char* STATE_NAME = "state_test.by_name";
constexpr int GROUPS = 1000;
constexpr int BATCH = 100;
// Small enough that a few dozen distinct names blow past it, so the spill
// engages many times over the batches below.
constexpr std::uint64_t TINY_BUDGET = 4096;
// The plugin's own estimate of one map entry; only its ratio to TINY_BUDGET
// matters.
constexpr std::uint64_t BYTES_PER_ENTRY = 64;

using Totals = std::map<std::string, std::uint64_t>;

// "name\0dur\0..." - a self-contained encoding used both for a spill run and
// for the finalized result, so a spilled state and an in-memory one are read
// back the same way.
std::string encode(const Totals& t) {
    std::string out;
    for (const auto& [name, dur] : t) {
        out.append(name);
        out.push_back('\0');
        out.append(std::to_string(dur));
        out.push_back('\0');
    }
    return out;
}

Totals decode(const char* p, std::size_t n) {
    Totals out;
    std::size_t at = 0;
    while (at < n) {
        const std::string name(p + at);
        at += name.size() + 1;
        if (at > n) break;
        const std::string dur(p + at);
        at += dur.size() + 1;
        out[name] += std::strtoull(dur.c_str(), nullptr, 10);
    }
    return out;
}

/* ---- the plugin's state type, behind the C descriptor ------------------- */

void* state_init(void*) { return new Totals(); }

std::string_view str_at(const dftu_series* col, std::int64_t r) {
    const std::int32_t* off = dftu_series_offsets(col);
    const char* base = static_cast<const char*>(dftu_series_data(col));
    if (!off || !base) return {};
    return {base + off[r], static_cast<std::size_t>(off[r + 1] - off[r])};
}

int state_update(void* state, const dftu_dataframe* df, dftu_error* err) {
    dftu_series* name = dftu_dataframe_column(df, "name");
    dftu_series* dur = dftu_dataframe_column(df, "dur");
    if (!name || !dur) {
        if (name) dftu_series_free(name);
        if (dur) dftu_series_free(dur);
        if (err) err->message = "batch has no name/dur column";
        return -1;
    }
    dftu_series* flat_name = dftu_series_materialize(name);
    dftu_series* flat_dur = dftu_series_materialize(dur);
    const auto* durs =
        static_cast<const std::uint64_t*>(dftu_series_data(flat_dur));
    auto& t = *static_cast<Totals*>(state);
    for (std::int64_t r = 0; r < dftu_series_length(flat_name); ++r)
        t[std::string(str_at(flat_name, r))] += durs[r];
    dftu_series_free(flat_name);
    dftu_series_free(flat_dur);
    dftu_series_free(name);
    dftu_series_free(dur);
    return 0;
}

int state_merge(void* into, void* other, dftu_error*) {
    auto& a = *static_cast<Totals*>(into);
    for (const auto& [k, v] : *static_cast<Totals*>(other)) a[k] += v;
    return 0;
}

std::uint64_t state_bytes(const void* state) {
    return static_cast<const Totals*>(state)->size() * BYTES_PER_ENTRY;
}

void free_buffer(void* data, void*) { delete[] static_cast<char*>(data); }

int state_serialize(const void* state, dftu_bytes* out, dftu_error*) {
    const std::string enc = encode(*static_cast<const Totals*>(state));
    char* buf = new char[enc.size() ? enc.size() : 1];
    std::memcpy(buf, enc.data(), enc.size());
    out->data = buf;
    out->len = enc.size();
    out->free_fn = free_buffer;
    out->ud = nullptr;
    return 0;
}

void* state_deserialize(void*, dftu_bytes in, dftu_error*) {
    return new Totals(decode(static_cast<const char*>(in.data),
                             static_cast<std::size_t>(in.len)));
}

int state_finalize(void* state, dftu_result_value* out, dftu_error*) {
    // Held past the call: the host copies BYTES results, but not until it
    // returns from here.
    static thread_local std::string enc;
    enc = encode(*static_cast<Totals*>(state));
    out->kind = DFTU_RESULT_KIND_BYTES;
    out->u.bytes.data = enc.data();
    out->u.bytes.len = enc.size();
    return 0;
}

void state_destroy(void* state) { delete static_cast<Totals*>(state); }

dftu_state_desc make_desc(bool spillable) {
    dftu_state_desc d{};
    d.name = STATE_NAME;
    d.init = state_init;
    d.update = state_update;
    d.merge = state_merge;
    d.bytes = state_bytes;
    if (spillable) {
        d.serialize = state_serialize;
        d.deserialize = state_deserialize;
    }
    d.finalize = state_finalize;
    d.destroy = state_destroy;
    return d;
}

/* ---- the host-side harness ---------------------------------------------- */

// A plugin that contributes nothing but the registered state, so anything the
// run produces came through the tier-2 path.
dftu_plugin state_only_plugin() {
    dftu_plugin p{};
    p.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    p.make_slice = [](void*) -> void* {
        static int sentinel;
        return &sentinel;
    };
    p.merge = [](void*, void*) {};
    p.on_finalize = [](void*, const dftu_plugin_host*) -> dftu_task* {
        return nullptr;
    };
    p.destroy_slice = [](void*) {};
    p.destroy = [](void*) {};
    return p;
}

FoldEvent evt(StringIntern& intern, const std::string& name,
              std::uint64_t dur) {
    FoldEvent fe;
    fe.cat_id = intern.get_or_insert("POSIX");
    fe.name_id = intern.get_or_insert(name);
    fe.phase = RecordPhase::COMPLETE;
    fe.pid = 1;
    fe.tid = 1;
    fe.ts = dur;
    fe.dur = dur;
    fe.has_dur = true;
    return fe;
}

// GROUPS distinct names, each seen once per slice, so every total depends on
// the cross-slice merge as well as on any spill.
std::vector<std::vector<FoldEvent>> slices(StringIntern& intern, int n_slices) {
    std::vector<std::vector<FoldEvent>> out(static_cast<std::size_t>(n_slices));
    for (int s = 0; s < n_slices; ++s)
        for (int i = 0; i < GROUPS; ++i)
            out[static_cast<std::size_t>(s)].push_back(
                evt(intern, "op_" + std::to_string(i),
                    static_cast<std::uint64_t>((s + 1) * (i + 1))));
    return out;
}

void finalize_now(PluginFold& f) {
    Runtime rt(1);
    rt.scope("fin", [&](dftracer::utils::CoroScope&) -> coro::CoroTask<void> {
          co_await f.finalize(views::detail::CoverageSet{});
      }).wait();
    rt.shutdown();
}

struct RunResult {
    Totals totals;
    std::size_t runs = 0;
    bool refused = false;
};

RunResult run_state(bool spillable, std::uint64_t budget, int n_slices) {
    StateRegistry registry{
        RegisteredState{STATE_NAME, make_desc(spillable), nullptr}};
    dftu_plugin p = state_only_plugin();
    StringIntern intern;
    NamedResultRegistry named;
    PluginFold master(&p, intern, nullptr, &named, "state_test", budget,
                      &registry);
    for (const std::vector<FoldEvent>& evs : slices(intern, n_slices)) {
        auto slice = master.slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit{};
        for (std::size_t at = 0; at < evs.size(); at += BATCH) {
            const std::size_t n = std::min<std::size_t>(BATCH, evs.size() - at);
            pf->step(FoldBatch{
                std::span<const FoldEvent>(evs.data() + at, n), unit, {}});
        }
        master.merge(*pf);
    }

    RunResult out;
    out.runs = master.state_spill_runs(STATE_NAME);
    out.refused = master.state_spill_refused(STATE_NAME);
    finalize_now(master);

    auto it = named.results().find(STATE_NAME);
    REQUIRE(it != named.results().end());
    const auto& bytes = std::get<std::vector<std::byte>>(it->second);
    out.totals =
        decode(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out;
}

Totals expected(int n_slices) {
    Totals want;
    for (int s = 0; s < n_slices; ++s)
        for (int i = 0; i < GROUPS; ++i)
            want["op_" + std::to_string(i)] +=
                static_cast<std::uint64_t>((s + 1) * (i + 1));
    return want;
}

}  // namespace

TEST_CASE("a registered state merges across worker slices") {
    const RunResult one = run_state(true, NO_SPILL_BUDGET, 1);
    const RunResult many = run_state(true, NO_SPILL_BUDGET, 4);

    CHECK(one.runs == 0);
    CHECK(many.runs == 0);
    CHECK(one.totals == expected(1));
    CHECK(many.totals == expected(4));
    // Four slices see four times the work; without the fan-in merge the
    // master would carry only its own (empty) share.
    CHECK(many.totals.size() == static_cast<std::size_t>(GROUPS));
}

TEST_CASE("a serializable state spills past the budget, same result") {
    const RunResult want = run_state(true, NO_SPILL_BUDGET, 4);
    const RunResult got = run_state(true, TINY_BUDGET, 4);

    // The mechanism, not just the answer: a budget that was silently ignored
    // would still produce the right totals.
    CHECK(got.runs > 0);
    CHECK_FALSE(got.refused);
    CHECK(got.totals == want.totals);
}

TEST_CASE("a state with no serialize pair is refused a spill") {
    const RunResult got = run_state(false, TINY_BUDGET, 4);

    CHECK(got.refused);
    CHECK(got.runs == 0);
    // Refused, not dropped: the state stays whole in memory and still
    // finalizes, but the host says out loud that it could not honour the
    // budget.
    CHECK(got.totals == expected(4));
}
