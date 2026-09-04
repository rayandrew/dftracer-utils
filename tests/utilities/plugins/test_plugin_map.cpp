// Mergeable map: a plugin builds a {pid, fhash} -> COUNTER map across simulated
// worker slices merged into a master, whose finalize materializes it to a
// columnar Arrow table in the named-result registry. Asserts the merged keys
// sum across slices and that the Arrow columns carry the same rows.

#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugin.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/trace/aggregators/aggregation_metrics.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <doctest/doctest.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <optional>
#include <set>
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
using dftracer::utils::plugins::OwnedArrow;
using dftracer::utils::plugins::OwnedArrowBatches;
using dftracer::utils::plugins::PluginFold;
using dftracer::utils::plugins::SharedResultRegistry;
using views::detail::CoverageSet;
using views::detail::FoldBatch;
using views::detail::FoldEvent;
using views::detail::ScanUnit;
namespace coro = dftracer::utils::coro;

namespace {

// Each event contributes 1 to the map at key {pid, fhash}: a weighted
// process<->file bipartite graph the host merges and materializes.
struct EdgeSlice {
    explicit EdgeSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.counter_map(
            "edges",
            dftracer::utils::plugins::Key<std::int64_t, std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            if (e.fhash_id().absent()) continue;
            m[std::tuple{static_cast<std::int64_t>(e.pid()),
                         static_cast<std::int64_t>(e.fhash_id().raw())}] += 1;
        }
    }
    void merge(EdgeSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Builds a {pid} -> COUNTER map, then reads it back at finalize through
// FinalizedMap: size(), get(), monoid(), and a full iteration. Results are
// captured into a static struct rather than CHECKed in place, since finalize()
// runs on a Runtime worker thread; the TEST_CASE asserts on the main thread
// after finalize_now() returns.
struct ReadCounterSlice {
    struct Result {
        bool ran = false;
        std::uint64_t size = 0;
        std::optional<std::uint64_t> v1;
        std::optional<std::uint64_t> v2;
        bool missing_key_absent = false;
        bool monoid_matches_get = false;
        std::map<std::int64_t, std::uint64_t> iterated;
    };
    static Result result;

    explicit ReadCounterSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.counter_map("read_counts",
                               dftracer::utils::plugins::Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b)
            m[static_cast<std::int64_t>(e.pid())] += 1;
    }
    void merge(ReadCounterSlice&) {}
    void finalize(dftracer::utils::plugins::Host h) {
        auto m = h.counter_map("read_counts",
                               dftracer::utils::plugins::Key<std::int64_t>{});
        auto fm = m.finalized();
        result.size = fm.size();
        result.v1 = fm.get(1);
        result.v2 = fm.get(2);
        result.missing_key_absent = !fm.get(999).has_value();
        auto mv = fm.monoid(1);
        result.monoid_matches_get = mv.has_value() && result.v1.has_value() &&
                                    mv->as_u64() == *result.v1;
        for (const auto& [key, val] : fm)
            result.iterated[std::get<0>(key)] = val.as_u64();
        result.ran = true;
    }
};
ReadCounterSlice::Result ReadCounterSlice::result;

// Each event contributes (+1, +dur) to a product map at key {pid, fhash}: the
// edge carries both a count and a total duration.
struct WideEdgeSlice {
    explicit WideEdgeSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new_product("wide_edges", {DFTU_T_I64, DFTU_T_I64},
                              {DFTU_MONOID_COUNTER, DFTU_MONOID_SUM_F64});
        for (const dftracer::utils::plugins::Event& e : b) {
            if (e.fhash_id().absent()) continue;
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid()),
                static_cast<std::int64_t>(e.fhash_id().raw())};
            h.map_add_u64_at(m, key, 0, 1);
            h.map_add_f64_at(m, key, 1, static_cast<double>(e.dur()));
        }
    }
    void merge(WideEdgeSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Each event contributes 1 to a map at key {pid, name}, where name is a STR key
// component: the interned event-name id, resolved to a string at materialize.
struct NamedEdgeSlice {
    explicit NamedEdgeSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("named_edges", {DFTU_T_I64, DFTU_T_STR},
                                DFTU_MONOID_COUNTER);
        for (const dftracer::utils::plugins::Event& e : b) {
            if (e.name_id().absent()) continue;
            h.map_add_u64(m,
                          {static_cast<std::int64_t>(e.pid()),
                           static_cast<std::int64_t>(e.name_id().raw())},
                          1);
        }
    }
    void merge(NamedEdgeSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Each file event adds e.fhash (an interned-string id) to a SET_STR value at
// key {pid}: the set of files each process touched, merged as a union.
struct FileSetSlice {
    explicit FileSetSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.map("file_set", dftracer::utils::plugins::Monoid::Set_Str,
                       dftracer::utils::plugins::Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            if (e.fhash_id().absent()) continue;
            m[static_cast<std::int64_t>(e.pid())] +=
                static_cast<std::uint64_t>(e.fhash_id().raw());
        }
    }
    void merge(FileSetSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Each event appends (e.ts, e.name) to a LIST_STR value at key {pid}: the
// ts-ordered sequence of event names each process emitted, merged as concat.
struct EventSeqSlice {
    explicit EventSeqSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto m = h.map("event_seq", dftracer::utils::plugins::Monoid::List_Str,
                       dftracer::utils::plugins::Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            if (e.name_id().absent()) continue;
            m.add_ordered(static_cast<std::int64_t>(e.pid()),
                          static_cast<std::int64_t>(e.ts()),
                          static_cast<std::uint64_t>(e.name_id().raw()));
        }
    }
    void merge(EventSeqSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Each event contributes e.dur to two int64 collections at key {pid}: a
// SET_I64 of distinct durations and a LIST_I64 of durations in ts order.
struct DurSeqSlice {
    explicit DurSeqSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftracer::utils::plugins::Batch& b,
              dftracer::utils::plugins::Host h) {
        auto s = h.map("dur_set", dftracer::utils::plugins::Monoid::Set_I64,
                       dftracer::utils::plugins::Key<std::int64_t>{});
        auto l = h.map("dur_seq", dftracer::utils::plugins::Monoid::List_I64,
                       dftracer::utils::plugins::Key<std::int64_t>{});
        for (const dftracer::utils::plugins::Event& e : b) {
            const std::int64_t pid = static_cast<std::int64_t>(e.pid());
            s[pid] += static_cast<std::uint64_t>(e.dur());
            l.add_ordered(pid, static_cast<std::int64_t>(e.ts()),
                          static_cast<std::uint64_t>(e.dur()));
        }
    }
    void merge(DurSeqSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Two ordered maps (map_set_ordered): asc_i64 keyed by {pid} (I64) and asc_str
// keyed by {name} (STR). Rows must materialize ascending by pid and
// alphabetical by resolved name label, regardless of insertion or interned-id
// order.
struct OrderedSlice {
    explicit OrderedSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* mi = h.map_new("asc_i64", {DFTU_T_I64}, DFTU_MONOID_COUNTER);
        h.map_set_ordered(mi);
        dftu_map* ms = h.map_new("asc_str", {DFTU_T_STR}, DFTU_MONOID_COUNTER);
        h.map_set_ordered(ms);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            h.map_add_u64(mi, {static_cast<std::int64_t>(e.pid)}, 1);
            if (e.name != DFTU_STR_NONE)
                h.map_add_u64(ms, {static_cast<std::int64_t>(e.name)}, 1);
        }
    }
    void merge(OrderedSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Keys each event on {pid as u16, tid as i32}: exercises a narrow unsigned and
// a signed key component (tid may be negative) materialized to exact widths.
struct TypedKeySlice {
    explicit TypedKeySlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("typed", {DFTU_T_U16, DFTU_T_I32}, DFTU_MONOID_COUNTER);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            h.map_add_u64(m,
                          {static_cast<std::int64_t>(e.pid),
                           static_cast<std::int64_t>(e.tid)},
                          1);
        }
    }
    void merge(TypedKeySlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Keys each event on {pid as u32, tid as u64}: both may exceed INT32_MAX, so
// the stored int64 bits must round-trip through the unsigned columns.
struct BigUnsignedSlice {
    explicit BigUnsignedSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("big", {DFTU_T_U32, DFTU_T_U64}, DFTU_MONOID_COUNTER);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            h.map_add_u64(m,
                          {static_cast<std::int64_t>(e.pid),
                           static_cast<std::int64_t>(e.tid)},
                          1);
        }
    }
    void merge(BigUnsignedSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Keys each event on a double whose bit pattern the test stashed in e.ts; the
// slice copies those bits straight into the f64 key slot (exact-bit equality).
struct DoubleKeySlice {
    explicit DoubleKeySlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("dbl", {DFTU_T_F64}, DFTU_MONOID_COUNTER);
        for (std::uint32_t i = 0; i < b.count; ++i)
            h.map_add_u64(m, {static_cast<std::int64_t>(b.events[i].ts)}, 1);
    }
    void merge(DoubleKeySlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Typed min/max value monoids keyed on pid; the per-event payload rides e.dur
// (integer bits) or e.ts (a double's bits) so the test controls each value.
template <dftu_monoid_kind Kind>
struct MinMaxIntSlice {
    explicit MinMaxIntSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("v", {DFTU_T_I64}, Kind);
        for (std::uint32_t i = 0; i < b.count; ++i)
            h.map_add_u64(m, {static_cast<std::int64_t>(b.events[i].pid)},
                          b.events[i].dur);
    }
    void merge(MinMaxIntSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

struct MinF32Slice {
    explicit MinF32Slice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("v", {DFTU_T_I64}, DFTU_MONOID_MIN_F32);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            double d;
            std::uint64_t bits = b.events[i].ts;
            std::memcpy(&d, &bits, sizeof(d));
            h.map_add_f64(m, {static_cast<std::int64_t>(b.events[i].pid)}, d);
        }
    }
    void merge(MinF32Slice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Nested-preserved map: outer key {pid}, inner key {fhash}, value COUNTER. One
// row per pid, whose "value" column is a list<struct<ik0, value>> of per-file
// counts.
struct NestedCountSlice {
    explicit NestedCountSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_nested("nested", {DFTU_T_I64}, {DFTU_T_I64},
                                       {DFTU_MONOID_COUNTER});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_nested_u64(m, {static_cast<std::int64_t>(e.pid)},
                                 {static_cast<std::int64_t>(e.fhash)}, 0, 1);
        }
    }
    void merge(NestedCountSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Nested-preserved product map: outer {pid}, inner {fhash}, inner value a
// product (COUNTER, SUM_F64) so each inner struct carries both v0 and v1.
struct NestedProductSlice {
    explicit NestedProductSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new_nested("nested_wide", {DFTU_T_I64}, {DFTU_T_I64},
                             {DFTU_MONOID_COUNTER, DFTU_MONOID_SUM_F64});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            std::initializer_list<std::int64_t> outer = {
                static_cast<std::int64_t>(e.pid)};
            std::initializer_list<std::int64_t> inner = {
                static_cast<std::int64_t>(e.fhash)};
            h.map_add_nested_u64(m, outer, inner, 0, 1);
            h.map_add_nested_f64(m, outer, inner, 1,
                                 static_cast<double>(e.dur));
        }
    }
    void merge(NestedProductSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Product argby map keyed {pid}: keeps, per process, the tid (i64 payload) and
// the fhash (str payload) at the min and max event duration. `by` is e.dur.
struct ArgBySlice {
    explicit ArgBySlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new_product("argby", {DFTU_T_I64},
                              {DFTU_MONOID_ARGMIN_I64, DFTU_MONOID_ARGMAX_I64,
                               DFTU_MONOID_ARGMIN_STR, DFTU_MONOID_ARGMAX_STR});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid)};
            const double by = static_cast<double>(e.dur);
            h.map_add_argby_at(m, key, 0, by, static_cast<std::int64_t>(e.tid));
            h.map_add_argby_at(m, key, 1, by, static_cast<std::int64_t>(e.tid));
            h.map_add_argby_at(m, key, 2, by,
                               static_cast<std::int64_t>(e.fhash));
            h.map_add_argby_at(m, key, 3, by,
                               static_cast<std::int64_t>(e.fhash));
        }
    }
    void merge(ArgBySlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Product bounded top-k map keyed {pid}, k=3: per process the top/bottom 3 tids
// (i64 payload) and fhash files (str payload) by event duration. `by` is e.dur.
struct TopKSlice {
    explicit TopKSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new_product("topk", {DFTU_T_I64},
                              {DFTU_MONOID_TOPK_I64, DFTU_MONOID_BOTTOMK_I64,
                               DFTU_MONOID_TOPK_STR, DFTU_MONOID_BOTTOMK_STR});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid)};
            const double by = static_cast<double>(e.dur);
            h.map_add_topk_at(m, key, 0, 3, by,
                              static_cast<std::int64_t>(e.tid));
            h.map_add_topk_at(m, key, 1, 3, by,
                              static_cast<std::int64_t>(e.tid));
            h.map_add_topk_at(m, key, 2, 3, by,
                              static_cast<std::int64_t>(e.fhash));
            h.map_add_topk_at(m, key, 3, 3, by,
                              static_cast<std::int64_t>(e.fhash));
        }
    }
    void merge(TopKSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Single-value TOPK_STR map keyed {pid}, k=3: the 3 fhash files at the largest
// e.dur each process touched.
struct TopKFileSlice {
    explicit TopKFileSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("topk_file", {DFTU_T_I64}, DFTU_MONOID_TOPK_STR);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_topk_at(m, {static_cast<std::int64_t>(e.pid)}, 0, 3,
                              static_cast<double>(e.dur),
                              static_cast<std::int64_t>(e.fhash));
        }
    }
    void merge(TopKFileSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// APPROX_TOPK_I64 heavy-hitters map keyed {pid}, capacity K: per process the K
// most frequent int64 values, each value carried in e.dur.
template <std::uint32_t K>
struct ApproxI64Slice {
    explicit ApproxI64Slice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("approx", {DFTU_T_I64}, DFTU_MONOID_APPROX_TOPK_I64);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            h.map_add_approx_topk_at(m, {static_cast<std::int64_t>(e.pid)}, 0,
                                     K, static_cast<std::int64_t>(e.dur));
        }
    }
    void merge(ApproxI64Slice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// APPROX_TOPK_STR heavy-hitters map keyed {pid}, k=3: per process the 3 most
// frequent files (fhash interned string ids).
struct ApproxStrSlice {
    explicit ApproxStrSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("approx_file", {DFTU_T_I64}, DFTU_MONOID_APPROX_TOPK_STR);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_approx_topk_at(m, {static_cast<std::int64_t>(e.pid)}, 0,
                                     3, static_cast<std::int64_t>(e.fhash));
        }
    }
    void merge(ApproxStrSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// SAMPLE_I64 bottom-k-by-hash map keyed {pid}, sample size K: per process a
// deterministic sample of up to K distinct int64 items, each carried in e.dur.
template <std::uint32_t K>
struct SampleI64Slice {
    explicit SampleI64Slice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("sample", {DFTU_T_I64}, DFTU_MONOID_SAMPLE_I64);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            h.map_add_sample_at(m, {static_cast<std::int64_t>(e.pid)}, 0, K,
                                static_cast<std::int64_t>(e.dur));
        }
    }
    void merge(SampleI64Slice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// SAMPLE_STR bottom-k-by-hash map keyed {pid}, k=3: per process a deterministic
// sample of up to 3 distinct files (fhash interned string ids).
struct SampleStrSlice {
    explicit SampleStrSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("sample_file", {DFTU_T_I64}, DFTU_MONOID_SAMPLE_STR);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_sample_at(m, {static_cast<std::int64_t>(e.pid)}, 0, 3,
                                static_cast<std::int64_t>(e.fhash));
        }
    }
    void merge(SampleStrSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Product moment map keyed {pid}: mean/variance/stddev of e.dur per process.
struct MomentSlice {
    explicit MomentSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_product(
            "moments", {DFTU_T_I64},
            {DFTU_MONOID_MEAN, DFTU_MONOID_VARIANCE, DFTU_MONOID_STDDEV});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid)};
            const double d = static_cast<double>(e.dur);
            h.map_add_f64_at(m, key, 0, d);
            h.map_add_f64_at(m, key, 1, d);
            h.map_add_f64_at(m, key, 2, d);
        }
    }
    void merge(MomentSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Single-value SKETCH (quantile) map keyed {pid}: a DDSketch of e.dur asking
// for p50/p90/p99. Materializes to count + one f64 column per quantile.
struct QuantileSlice {
    explicit QuantileSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_sketch("q", {DFTU_T_I64}, {0.5, 0.9, 0.99});
        for (std::uint32_t i = 0; i < b.count; ++i)
            h.map_add_f64(m, {static_cast<std::int64_t>(b.events[i].pid)},
                          static_cast<double>(b.events[i].dur));
    }
    void merge(QuantileSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Fused map keyed {pid}: one lookup per event feeds a COUNTER and a SUM_F64,
// which materialize as two SEPARATE named tables "n" and "tot".
struct FusedSlice {
    explicit FusedSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new_fused("fused", {DFTU_T_I64}, {"n", "tot"},
                            {DFTU_MONOID_COUNTER, DFTU_MONOID_SUM_F64});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            dftu_row_val vals[2];
            vals[0].comp = 0;
            vals[0].is_f64 = 0;
            vals[0].value.u = 1;
            vals[1].comp = 1;
            vals[1].is_f64 = 1;
            vals[1].value.f = static_cast<double>(b.events[i].dur);
            h.map_add_row(m, {static_cast<std::int64_t>(b.events[i].pid)},
                          {vals[0], vals[1]});
        }
    }
    void merge(FusedSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Raw-ABI path used as the oracle for the Host::map_add_xy_at wrapper.
void map_add_xy_at_raw(dftracer::utils::plugins::Host h, dftu_map* m,
                       std::initializer_list<std::int64_t> key,
                       std::uint32_t comp, double x, double y) {
    const dftu_host* raw = h.raw();
    const auto* e = static_cast<const dftu_ext_map*>(
        raw->get_extension(raw->h, DFTU_EXT_MAP));
    if (e && e->map_add_xy_at)
        e->map_add_xy_at(raw->h, m, key.begin(), comp, x, y);
}

// Product higher-moment map keyed {pid}: population skewness and excess
// kurtosis of e.dur per process.
struct HighMomentSlice {
    explicit HighMomentSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new_product("hmoments", {DFTU_T_I64},
                              {DFTU_MONOID_SKEWNESS, DFTU_MONOID_KURTOSIS});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid)};
            const double d = static_cast<double>(e.dur);
            h.map_add_f64_at(m, key, 0, d);
            h.map_add_f64_at(m, key, 1, d);
        }
    }
    void merge(HighMomentSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Product co-moment map keyed {pid}: all six two-variable statistics of (x, y),
// with x = the low bits of e.dur and y = the low bits of e.ts (see xy_event).
struct RegrSlice {
    explicit RegrSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_product(
            "regr", {DFTU_T_I64},
            {DFTU_MONOID_CORR, DFTU_MONOID_COVAR_POP, DFTU_MONOID_COVAR_SAMP,
             DFTU_MONOID_REGR_SLOPE, DFTU_MONOID_REGR_INTERCEPT,
             DFTU_MONOID_REGR_R2});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            double x, y;
            std::memcpy(&x, &e.dur, sizeof(x));
            std::memcpy(&y, &e.ts, sizeof(y));
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid)};
            for (std::uint32_t c = 0; c < 6; ++c)
                map_add_xy_at_raw(h, m, key, c, x, y);
        }
    }
    void merge(RegrSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Same product co-moment map fed through the Host::map_add_xy_at wrapper.
struct RegrWrapperSlice {
    explicit RegrWrapperSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_product(
            "regr", {DFTU_T_I64},
            {DFTU_MONOID_CORR, DFTU_MONOID_COVAR_POP, DFTU_MONOID_COVAR_SAMP,
             DFTU_MONOID_REGR_SLOPE, DFTU_MONOID_REGR_INTERCEPT,
             DFTU_MONOID_REGR_R2});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            double x, y;
            std::memcpy(&x, &e.dur, sizeof(x));
            std::memcpy(&y, &e.ts, sizeof(y));
            std::initializer_list<std::int64_t> key = {
                static_cast<std::int64_t>(e.pid)};
            for (std::uint32_t c = 0; c < 6; ++c)
                h.map_add_xy_at(m, key, c, x, y);
        }
    }
    void merge(RegrWrapperSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Single-value CORR map keyed {pid}: feeds map_add_xy_at through the raw ABI.
struct CorrSlice {
    explicit CorrSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("corr", {DFTU_T_I64}, DFTU_MONOID_CORR);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            double x, y;
            std::memcpy(&x, &e.dur, sizeof(x));
            std::memcpy(&y, &e.ts, sizeof(y));
            map_add_xy_at_raw(h, m, {static_cast<std::int64_t>(e.pid)}, 0, x,
                              y);
        }
    }
    void merge(CorrSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Same CORR map fed through the Host::map_add_xy_at convenience wrapper.
struct CorrWrapperSlice {
    explicit CorrWrapperSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("corr", {DFTU_T_I64}, DFTU_MONOID_CORR);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            double x, y;
            std::memcpy(&x, &e.dur, sizeof(x));
            std::memcpy(&y, &e.ts, sizeof(y));
            h.map_add_xy_at(m, {static_cast<std::int64_t>(e.pid)}, 0, x, y);
        }
    }
    void merge(CorrWrapperSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Single-value ARGMAX_STR map keyed {pid}: the fhash label at max e.dur.
struct ArgMaxFileSlice {
    explicit ArgMaxFileSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("argmax_file", {DFTU_T_I64}, DFTU_MONOID_ARGMAX_STR);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_argby_at(m, {static_cast<std::int64_t>(e.pid)}, 0,
                               static_cast<double>(e.dur),
                               static_cast<std::int64_t>(e.fhash));
        }
    }
    void merge(ArgMaxFileSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Full-row arg-row map (DISTINCT ON) keyed {pid}, is_max by e.dur: keeps the
// WHOLE payload row {fhash (STR), tid (I32), rate (F64)} of the max-duration
// event per process. The float rate rides e.ts's bits (see argrow_event).
struct ArgRowSlice {
    explicit ArgRowSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_argrow("argrow", {DFTU_T_I64}, /*is_max=*/true,
                                       {DFTU_T_STR, DFTU_T_I32, DFTU_T_F64});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            h.map_add_argrow(m, {static_cast<std::int64_t>(e.pid)},
                             static_cast<double>(e.dur),
                             {static_cast<std::int64_t>(e.fhash),
                              static_cast<std::int64_t>(e.tid),
                              static_cast<std::int64_t>(e.ts)});
        }
    }
    void merge(ArgRowSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Single-value VARIANCE map keyed {pid}: sample variance of e.dur.
struct DurVarSlice {
    explicit DurVarSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("dur_var", {DFTU_T_I64}, DFTU_MONOID_VARIANCE);
        for (std::uint32_t i = 0; i < b.count; ++i)
            h.map_add_f64(m, {static_cast<std::int64_t>(b.events[i].pid)},
                          static_cast<double>(b.events[i].dur));
    }
    void merge(DurVarSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// BYTES key component keyed on e.fhash (an interned byte-blob id): rides the
// same interned-id slot as a STR key but materializes to an Arrow binary
// column. COUNTER value.
struct BytesEdgeSlice {
    explicit BytesEdgeSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("byte_edges", {DFTU_T_BYTES}, DFTU_MONOID_COUNTER);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_u64(m, {static_cast<std::int64_t>(e.fhash)}, 1);
        }
    }
    void merge(BytesEdgeSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Two-component key {pid : i64, blob : BYTES}: an int and a byte-blob key
// component side by side.
struct MixedBytesSlice {
    explicit MixedBytesSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new("mixed_bytes", {DFTU_T_I64, DFTU_T_BYTES},
                                DFTU_MONOID_COUNTER);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_u64(m,
                          {static_cast<std::int64_t>(e.pid),
                           static_cast<std::int64_t>(e.fhash)},
                          1);
        }
    }
    void merge(MixedBytesSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Ordered map keyed {blob : BYTES}: rows must materialize sorted by raw bytes
// (byte-lexicographic), independent of interned-id or insertion order.
struct OrderedBytesSlice {
    explicit OrderedBytesSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m =
            h.map_new("asc_bytes", {DFTU_T_BYTES}, DFTU_MONOID_COUNTER);
        h.map_set_ordered(m);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_u64(m, {static_cast<std::int64_t>(e.fhash)}, 1);
        }
    }
    void merge(OrderedBytesSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Nested-preserved map with a BYTES outer key {blob} and an I64 inner key
// {pid}, COUNTER value: one row per blob, inner list<struct<ik0, value>>.
struct NestedBytesSlice {
    explicit NestedBytesSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* m = h.map_new_nested("nested_bytes", {DFTU_T_BYTES},
                                       {DFTU_T_I64}, {DFTU_MONOID_COUNTER});
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            if (e.fhash == DFTU_STR_NONE) continue;
            h.map_add_nested_u64(m, {static_cast<std::int64_t>(e.fhash)},
                                 {static_cast<std::int64_t>(e.pid)}, 0, 1);
        }
    }
    void merge(NestedBytesSlice&) {}
    void finalize(dftracer::utils::plugins::Host) {}
};

// Builds two maps keyed {pid} - "jl_counts" (COUNTER) and "jr_durs" (SUM_F64,
// only for events carrying a duration) - and declares a LEFT and an INNER join
// of them. The host executes both at finalize, emitting "join_left" and
// "join_inner" alongside the two input maps.
struct JoinSlice {
    explicit JoinSlice(const dftracer::utils::plugins::Config&) {}
    void step(const dftu_batch& b, dftracer::utils::plugins::Host h) {
        dftu_map* c = h.map_new("jl_counts", {DFTU_T_I64}, DFTU_MONOID_COUNTER);
        dftu_map* d = h.map_new("jr_durs", {DFTU_T_I64}, DFTU_MONOID_SUM_F64);
        for (std::uint32_t i = 0; i < b.count; ++i) {
            const dftu_event& e = b.events[i];
            const std::int64_t pid = static_cast<std::int64_t>(e.pid);
            h.map_add_u64(c, {pid}, 1);
            if (e.has_dur) h.map_add_f64(d, {pid}, static_cast<double>(e.dur));
        }
        h.map_declare_join("join_left", "jl_counts", "jr_durs", DFTU_JOIN_LEFT);
        h.map_declare_join("join_inner", "jl_counts", "jr_durs",
                           DFTU_JOIN_INNER);
    }
    void merge(JoinSlice&) {}
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

FoldEvent edge(std::uint64_t pid, std::int64_t fhash) {
    FoldEvent e;
    e.pid = pid;
    e.fhash_id = static_cast<std::uint32_t>(fhash);
    return e;
}

FoldEvent edge_dur(std::uint64_t pid, std::int64_t fhash, std::uint64_t dur) {
    FoldEvent e = edge(pid, fhash);
    e.dur = dur;
    e.has_dur = true;
    return e;
}

FoldEvent named_edge(std::uint64_t pid, std::uint32_t name_id) {
    FoldEvent e;
    e.pid = pid;
    e.name_id = name_id;
    return e;
}

FoldEvent seq_event(std::uint64_t pid, std::uint32_t name_id,
                    std::uint64_t ts) {
    FoldEvent e;
    e.pid = pid;
    e.name_id = name_id;
    e.ts = ts;
    return e;
}

FoldEvent dur_event(std::uint64_t pid, std::uint64_t ts, std::uint64_t dur) {
    FoldEvent e;
    e.pid = pid;
    e.ts = ts;
    e.dur = dur;
    e.has_dur = true;
    return e;
}

FoldEvent typed_event(std::uint64_t pid, std::uint64_t tid) {
    FoldEvent e;
    e.pid = pid;
    e.tid = tid;
    return e;
}

FoldEvent dbl_event(double d) {
    FoldEvent e;
    std::uint64_t bits;
    std::memcpy(&bits, &d, sizeof(bits));
    e.ts = bits;
    return e;
}

// pid plus an integer value payload in e.dur (signed values pass their bits).
FoldEvent val_event(std::uint64_t pid, std::uint64_t val) {
    FoldEvent e;
    e.pid = pid;
    e.dur = val;
    return e;
}

FoldEvent fval_event(std::uint64_t pid, double d) {
    FoldEvent e = dbl_event(d);
    e.pid = pid;
    return e;
}

// pid, duration (the argby `by`), fhash (str payload), tid (i64 payload).
FoldEvent argby_event(std::uint64_t pid, std::uint64_t dur, std::uint32_t fhash,
                      std::uint64_t tid) {
    FoldEvent e;
    e.pid = pid;
    e.dur = dur;
    e.has_dur = true;
    e.fhash_id = fhash;
    e.tid = tid;
    return e;
}

// Like argby_event but also carries a per-event f64 `rate` in e.ts's bits, so
// an arg-row payload can keep an independent float column alongside fhash/tid.
FoldEvent argrow_event(std::uint64_t pid, std::uint64_t dur,
                       std::uint32_t fhash, std::uint64_t tid, double rate) {
    FoldEvent e = argby_event(pid, dur, fhash, tid);
    std::uint64_t bits;
    std::memcpy(&bits, &rate, sizeof(bits));
    e.ts = bits;
    return e;
}

// pid plus two f64s: x in e.dur's bits, y in e.ts's bits (RegrSlice/CorrSlice
// decode them symmetrically).
FoldEvent xy_event(std::uint64_t pid, double x, double y) {
    FoldEvent e;
    e.pid = pid;
    std::uint64_t xb, yb;
    std::memcpy(&xb, &x, sizeof(xb));
    std::memcpy(&yb, &y, sizeof(yb));
    e.dur = xb;
    e.has_dur = true;
    e.ts = yb;
    return e;
}

void finalize_now(PluginFold& f) {
    Runtime rt(1);
    rt.scope("fin", [&](dftracer::utils::CoroScope&) -> coro::CoroTask<void> {
          co_await f.finalize(CoverageSet{});
      }).wait();
    rt.shutdown();
}

// Run each slice at the given part_bits (K = 1u << part_bits), merged into a
// master whose maps carry the same part_bits, then materialize into `named`.
template <typename Slice>
void run_partitioned(NamedResultRegistry& named, SharedResultRegistry& reg,
                     StringIntern& intern,
                     const std::vector<std::vector<FoldEvent>>& slices,
                     std::uint32_t part_bits) {
    FoldHolder master(dftracer::utils::plugins::make_plugin<Slice>(nullptr),
                      intern, &reg, &named);
    for (const auto& evs : slices) {
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        pf->set_map_part_bits(part_bits);
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
    }
    finalize_now(*master.fold);
}

// Run each slice as an independent worker with spill forced on at a tiny
// `share` (K=256), merged into a master that also spills, then materialize. The
// spill dir root is `dir`. Returns the total partition-spill count across
// workers and master so a test can prove spilling actually happened.
template <typename Slice>
std::size_t run_spilled(NamedResultRegistry& named, SharedResultRegistry& reg,
                        StringIntern& intern,
                        const std::vector<std::vector<FoldEvent>>& slices,
                        std::size_t share, const std::string& dir) {
    FoldHolder master(dftracer::utils::plugins::make_plugin<Slice>(nullptr),
                      intern, &reg, &named);
    master.fold->set_map_spill(share, dir);
    std::size_t spills = 0;
    for (const auto& evs : slices) {
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        pf->set_map_spill(share, dir);
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
        spills += pf->map_spill_count();
    }
    spills += master.fold->map_spill_count();
    finalize_now(*master.fold);
    return spills;
}

// Run each slice as an independent spilling worker (K=256, tiny `share`) merged
// into a master that spills AND streams: materialize emits a sequence of
// per-partition (or row-chunked) batches. Returns {total streamed batches,
// peak resident partitions} so a test can prove streaming really happened.
template <typename Slice>
std::pair<std::size_t, std::size_t> run_streamed(
    NamedResultRegistry& named, SharedResultRegistry& reg, StringIntern& intern,
    const std::vector<std::vector<FoldEvent>>& slices, std::size_t share,
    const std::string& dir, std::size_t chunk) {
    FoldHolder master(dftracer::utils::plugins::make_plugin<Slice>(nullptr),
                      intern, &reg, &named);
    master.fold->set_map_spill(share, dir);
    master.fold->set_map_stream(true, chunk);
    for (const auto& evs : slices) {
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        pf->set_map_spill(share, dir);
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
    }
    finalize_now(*master.fold);
    return {master.fold->map_stream_batch_count(),
            master.fold->map_stream_max_resident_partitions()};
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
// The materialized batches for `name`: either a single OwnedArrow (one batch)
// or an OwnedArrowBatches sequence (streamed). Concatenating them yields the
// whole result.
struct BatchList {
    std::vector<const ArrowArray*> arrays;
    const ArrowSchema* schema = nullptr;
};

inline BatchList result_batches(NamedResultRegistry& named, const char* name) {
    BatchList bl;
    auto it = named.results().find(name);
    REQUIRE(it != named.results().end());
    if (std::holds_alternative<OwnedArrow>(it->second)) {
        const OwnedArrow& o = std::get<OwnedArrow>(it->second);
        bl.arrays.push_back(&o.array);
        bl.schema = &o.schema;
    } else {
        const auto& ob = std::get<OwnedArrowBatches>(it->second);
        REQUIRE(!ob.batches.empty());
        for (const auto& b : ob.batches) bl.arrays.push_back(&b.array);
        bl.schema = &ob.batches.front().schema;
    }
    return bl;
}

inline const std::int64_t* i64_col(const ArrowArray* a, int c) {
    const ArrowArray* ch = a->children[c];
    return static_cast<const std::int64_t*>(ch->buffers[1]) + ch->offset;
}

// One value of a variable-binary/utf8 child at logical row r, bytes-exact
// (length-based, so an embedded NUL is preserved).
inline std::string bin_at(const ArrowArray* ch, std::int64_t r) {
    const auto* offs =
        static_cast<const std::int32_t*>(ch->buffers[1]) + ch->offset;
    const char* data = static_cast<const char*>(ch->buffers[2]);
    return std::string(data + offs[r], offs[r + 1] - offs[r]);
}
#endif

// A unique, empty spill root under the temp dir; removed on destruction.
struct SpillRoot {
    std::string path;
    SpillRoot() {
        static std::atomic<std::uint64_t> seq{0};
        path = (fs::temp_directory_path() /
                ("dftplugmap_test_" + std::to_string(seq.fetch_add(1)) + "_" +
                 std::to_string(reinterpret_cast<std::uintptr_t>(this))))
                   .string();
        fs::create_directories(path);
    }
    ~SpillRoot() {
        std::error_code ec;
        fs::remove_all(path, ec);
    }
    // Whether any per-run spill subdir survives (should be false
    // post-finalize).
    bool has_run_dirs() const {
        std::error_code ec;
        for (const auto& e : fs::directory_iterator(path, ec))
            if (e.is_directory(ec)) return true;
        return false;
    }
};

}  // namespace

TEST_SUITE("PluginMap") {
    TEST_CASE("merged map sums per-key counts and materializes to Arrow") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<EdgeSlice>(nullptr), intern,
            &reg, &named);

        // Two slices; merged the counts are (1,10)->3 and (2,20)->2.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(1, 10), edge(1, 10), edge(2, 20)},
            {edge(1, 10), edge(2, 20)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("edges");
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const ArrowArray& a = std::get<OwnedArrow>(it->second).array;
        REQUIRE(a.n_children == 3);
        REQUIRE(a.length == 2);

        auto col = [&](int c) {
            const auto* child = a.children[c];
            return static_cast<const std::int64_t*>(child->buffers[1]) +
                   child->offset;
        };
        const std::int64_t* k0 = col(0);
        const std::int64_t* k1 = col(1);
        const std::int64_t* val = col(2);

        bool saw_a = false, saw_b = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            if (k0[row] == 1 && k1[row] == 10) {
                CHECK(val[row] == 3);
                saw_a = true;
            } else if (k0[row] == 2 && k1[row] == 20) {
                CHECK(val[row] == 2);
                saw_b = true;
            }
        }
        CHECK(saw_a);
        CHECK(saw_b);
#endif
    }

    TEST_CASE("product map merges each component and materializes v0,v1") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<WideEdgeSlice>(nullptr),
            intern, &reg, &named);

        // Merged: (1,10) -> count 3, sum 30.0; (2,20) -> count 2, sum 40.0.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge_dur(1, 10, 10), edge_dur(1, 10, 10), edge_dur(2, 20, 20)},
            {edge_dur(1, 10, 10), edge_dur(2, 20, 20)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("wide_edges");
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const ArrowArray& a = std::get<OwnedArrow>(it->second).array;
        REQUIRE(a.n_children == 4);
        REQUIRE(a.length == 2);

        auto i64col = [&](int c) {
            const auto* child = a.children[c];
            return static_cast<const std::int64_t*>(child->buffers[1]) +
                   child->offset;
        };
        auto f64col = [&](int c) {
            const auto* child = a.children[c];
            return static_cast<const double*>(child->buffers[1]) +
                   child->offset;
        };
        const std::int64_t* k0 = i64col(0);
        const std::int64_t* k1 = i64col(1);
        const std::int64_t* v0 = i64col(2);
        const double* v1 = f64col(3);

        bool saw_a = false, saw_b = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            if (k0[row] == 1 && k1[row] == 10) {
                CHECK(v0[row] == 3);
                CHECK(v1[row] == doctest::Approx(30.0));
                saw_a = true;
            } else if (k0[row] == 2 && k1[row] == 20) {
                CHECK(v0[row] == 2);
                CHECK(v1[row] == doctest::Approx(40.0));
                saw_b = true;
            }
        }
        CHECK(saw_a);
        CHECK(saw_b);
#endif
    }

    TEST_CASE("STR key component materializes a resolved string column") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        const std::uint32_t read_id = intern.get_or_insert("read");
        const std::uint32_t write_id = intern.get_or_insert("write");
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<NamedEdgeSlice>(nullptr),
            intern, &reg, &named);

        // Merged the counts are (1,"read")->3 and (2,"write")->2.
        const std::vector<std::vector<FoldEvent>> slices = {
            {named_edge(1, read_id), named_edge(1, read_id),
             named_edge(2, write_id)},
            {named_edge(1, read_id), named_edge(2, write_id)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("named_edges");
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 3);
        REQUIRE(a.length == 2);

        // k1 is an Arrow utf8 column (format "u").
        REQUIRE(owned.schema.children[1]->format != nullptr);
        CHECK(std::string_view(owned.schema.children[1]->format) == "u");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* k1c = a.children[1];
        const ArrowArray* valc = a.children[2];
        const std::int64_t* k0 =
            static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
        const std::int64_t* val =
            static_cast<const std::int64_t*>(valc->buffers[1]) + valc->offset;
        const auto* offs =
            static_cast<const std::int32_t*>(k1c->buffers[1]) + k1c->offset;
        const char* chars = static_cast<const char*>(k1c->buffers[2]);
        auto k1_at = [&](std::int64_t r) {
            return std::string_view(chars + offs[r], offs[r + 1] - offs[r]);
        };

        bool saw_a = false, saw_b = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            if (k0[row] == 1 && k1_at(row) == "read") {
                CHECK(val[row] == 3);
                saw_a = true;
            } else if (k0[row] == 2 && k1_at(row) == "write") {
                CHECK(val[row] == 2);
                saw_b = true;
            }
        }
        CHECK(saw_a);
        CHECK(saw_b);
#endif
    }

    TEST_CASE("SET_STR value merges to a union and materializes list<utf8>") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        const std::uint32_t a_id = intern.get_or_insert("a");
        const std::uint32_t b_id = intern.get_or_insert("b");
        const std::uint32_t c_id = intern.get_or_insert("c");
        const std::uint32_t d_id = intern.get_or_insert("d");
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<FileSetSlice>(nullptr),
            intern, &reg, &named);

        // Merged unions: pid 1 -> {a,b,c} (b overlaps), pid 2 -> {c,d}.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(1, a_id), edge(1, b_id), edge(2, c_id)},
            {edge(1, b_id), edge(1, c_id), edge(2, d_id)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("file_set");
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 2);
        REQUIRE(a.length == 2);

        // value is an Arrow list<utf8> column: outer "+l", item "u".
        REQUIRE(owned.schema.children[1]->format != nullptr);
        CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
        REQUIRE(owned.schema.children[1]->n_children == 1);
        CHECK(std::string_view(owned.schema.children[1]->children[0]->format) ==
              "u");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* setc = a.children[1];
        const std::int64_t* k0 =
            static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
        const auto* loff =
            static_cast<const std::int32_t*>(setc->buffers[1]) + setc->offset;
        const ArrowArray* strc = setc->children[0];
        const auto* soff = static_cast<const std::int32_t*>(strc->buffers[1]);
        const char* chars = static_cast<const char*>(strc->buffers[2]);
        auto row_set = [&](std::int64_t row) {
            std::set<std::string> out;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                out.emplace(chars + soff[j], soff[j + 1] - soff[j]);
            return out;
        };

        bool saw_a = false, saw_b = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            if (k0[row] == 1) {
                CHECK(row_set(row) == std::set<std::string>{"a", "b", "c"});
                saw_a = true;
            } else if (k0[row] == 2) {
                CHECK(row_set(row) == std::set<std::string>{"c", "d"});
                saw_b = true;
            }
        }
        CHECK(saw_a);
        CHECK(saw_b);
#endif
    }

    TEST_CASE(
        "LIST_STR value concats, sorts by order key, and is deterministic") {
        StringIntern intern;
        const std::uint32_t a_id = intern.get_or_insert("a");
        const std::uint32_t b_id = intern.get_or_insert("b");
        const std::uint32_t c_id = intern.get_or_insert("c");
        const std::uint32_t d_id = intern.get_or_insert("d");

        // Names are added out of ts order and across the slice split. Sorted by
        // ts: pid1 -> b(10),c(20),a(30),b(40); pid2 -> d(1),c(5).
        const std::vector<FoldEvent> s0 = {seq_event(1, a_id, 30),
                                           seq_event(1, b_id, 10),
                                           seq_event(2, c_id, 5)};
        const std::vector<FoldEvent> s1 = {seq_event(1, c_id, 20),
                                           seq_event(1, b_id, 40),
                                           seq_event(2, d_id, 1)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<EventSeqSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

            std::map<std::int64_t, std::vector<std::string>> out;
            auto it = named.results().find("event_seq");
            REQUIRE(it != named.results().end());
            const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
            const ArrowArray& a = owned.array;
            REQUIRE(a.n_children == 2);
            CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
            CHECK(std::string_view(
                      owned.schema.children[1]->children[0]->format) == "u");
            const ArrowArray* k0c = a.children[0];
            const ArrowArray* listc = a.children[1];
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* strc = listc->children[0];
            const auto* soff =
                static_cast<const std::int32_t*>(strc->buffers[1]);
            const char* chars = static_cast<const char*>(strc->buffers[2]);
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                std::vector<std::string> seq;
                for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                    seq.emplace_back(chars + soff[j], soff[j + 1] - soff[j]);
                out.emplace(k0[row], std::move(seq));
            }
            return out;
        };

        const auto forward = run({s0, s1});
        const auto reversed = run({s1, s0});

        const std::vector<std::string> exp1 = {"b", "c", "a", "b"};
        const std::vector<std::string> exp2 = {"d", "c"};
        CHECK(forward.at(1) == exp1);
        CHECK(forward.at(2) == exp2);
        // Determinism: merge order does not change the materialized order.
        CHECK(reversed.at(1) == exp1);
        CHECK(reversed.at(2) == exp2);
#endif
    }

    TEST_CASE("SET_I64 sorts, LIST_I64 orders, both materialize list<int64>") {
        StringIntern intern;

        // pid1 durations by ts: 20(ts10), 5(ts20), 20(ts30), 8(ts40); across a
        // slice split. pid2: 3(ts1), 7(ts5). SET is distinct-sorted, LIST is
        // ts-ordered.
        const std::vector<FoldEvent> s0 = {
            dur_event(1, 10, 20), dur_event(1, 30, 20), dur_event(2, 5, 7)};
        const std::vector<FoldEvent> s1 = {
            dur_event(1, 20, 5), dur_event(1, 40, 8), dur_event(2, 1, 3)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto read_list = [&](NamedResultRegistry& named, const char* name) {
            std::map<std::int64_t, std::vector<std::int64_t>> out;
            auto it = named.results().find(name);
            REQUIRE(it != named.results().end());
            const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
            const ArrowArray& a = owned.array;
            REQUIRE(a.n_children == 2);
            CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
            CHECK(std::string_view(
                      owned.schema.children[1]->children[0]->format) == "l");
            const ArrowArray* k0c = a.children[0];
            const ArrowArray* listc = a.children[1];
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* valc = listc->children[0];
            const std::int64_t* vals =
                static_cast<const std::int64_t*>(valc->buffers[1]);
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                std::vector<std::int64_t> v;
                for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                    v.push_back(vals[j]);
                out.emplace(k0[row], std::move(v));
            }
            return out;
        };
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<DurSeqSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            return std::pair{read_list(named, "dur_set"),
                             read_list(named, "dur_seq")};
        };

        const auto forward = run({s0, s1});
        const auto reversed = run({s1, s0});

        const std::vector<std::int64_t> set1 = {5, 8, 20};
        const std::vector<std::int64_t> set2 = {3, 7};
        const std::vector<std::int64_t> seq1 = {20, 5, 20, 8};
        const std::vector<std::int64_t> seq2 = {3, 7};
        CHECK(forward.first.at(1) == set1);
        CHECK(forward.first.at(2) == set2);
        CHECK(forward.second.at(1) == seq1);
        CHECK(forward.second.at(2) == seq2);
        // Determinism: merge order changes neither the set nor the list order.
        CHECK(reversed.first.at(1) == set1);
        CHECK(reversed.first.at(2) == set2);
        CHECK(reversed.second.at(1) == seq1);
        CHECK(reversed.second.at(2) == seq2);
#endif
    }

    TEST_CASE(
        "ordered map materializes rows sorted by key (I64 asc, STR by label)") {
        StringIntern intern;
        // Intern out of alphabetical order so id order != label order: a
        // materialized STR column sorted by id would read cat,apple,banana.
        const std::uint32_t cat_id = intern.get_or_insert("cat");
        const std::uint32_t apple_id = intern.get_or_insert("apple");
        const std::uint32_t banana_id = intern.get_or_insert("banana");

        // Merged counts: pid/name apple->2, banana->1, cat->3 (pids 1,2,3).
        const std::vector<FoldEvent> s0 = {named_edge(1, apple_id),
                                           named_edge(1, apple_id),
                                           named_edge(3, cat_id)};
        const std::vector<FoldEvent> s1 = {named_edge(2, banana_id),
                                           named_edge(3, cat_id),
                                           named_edge(3, cat_id)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<OrderedSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

            std::vector<std::pair<std::int64_t, std::int64_t>> asc_i64;
            std::vector<std::pair<std::string, std::int64_t>> asc_str;

            const auto& ir =
                std::get<OwnedArrow>(named.results().at("asc_i64"));
            const ArrowArray& ia = ir.array;
            const std::int64_t* ik0 =
                static_cast<const std::int64_t*>(ia.children[0]->buffers[1]) +
                ia.children[0]->offset;
            const std::int64_t* iv =
                static_cast<const std::int64_t*>(ia.children[1]->buffers[1]) +
                ia.children[1]->offset;
            for (std::int64_t r = 0; r < ia.length; ++r)
                asc_i64.emplace_back(ik0[r + ia.offset], iv[r + ia.offset]);

            const auto& sr =
                std::get<OwnedArrow>(named.results().at("asc_str"));
            const ArrowArray& sa = sr.array;
            const ArrowArray* k0c = sa.children[0];
            const auto* offs =
                static_cast<const std::int32_t*>(k0c->buffers[1]) + k0c->offset;
            const char* chars = static_cast<const char*>(k0c->buffers[2]);
            const std::int64_t* sv =
                static_cast<const std::int64_t*>(sa.children[1]->buffers[1]) +
                sa.children[1]->offset;
            for (std::int64_t r = 0; r < sa.length; ++r)
                asc_str.emplace_back(
                    std::string(chars + offs[r], offs[r + 1] - offs[r]),
                    sv[r + sa.offset]);
            return std::pair{asc_i64, asc_str};
        };

        const std::vector<std::pair<std::int64_t, std::int64_t>> exp_i64 = {
            {1, 2}, {2, 1}, {3, 3}};
        const std::vector<std::pair<std::string, std::int64_t>> exp_str = {
            {"apple", 2}, {"banana", 1}, {"cat", 3}};

        const auto forward = run({s0, s1});
        CHECK(forward.first == exp_i64);
        CHECK(forward.second == exp_str);
        // Determinism: merge order does not change the materialized row order.
        const auto reversed = run({s1, s0});
        CHECK(reversed.first == exp_i64);
        CHECK(reversed.second == exp_str);
#endif
    }

    TEST_CASE("typed integer key components materialize exact Arrow widths") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<TypedKeySlice>(nullptr),
            intern, &reg, &named);

        // tid -5 rides the int64 key slot as a negative i32; pid 1000 a u16.
        const std::uint64_t neg =
            static_cast<std::uint64_t>(static_cast<std::int64_t>(-5));
        const std::vector<std::vector<FoldEvent>> slices = {
            {typed_event(1000, 7), typed_event(1000, 7),
             typed_event(2000, neg)},
            {typed_event(1000, 7), typed_event(2000, neg)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("typed");
        REQUIRE(it != named.results().end());
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 3);
        REQUIRE(a.length == 2);

        // uint16 is Arrow format "S", int32 "i".
        CHECK(std::string_view(owned.schema.children[0]->format) == "S");
        CHECK(std::string_view(owned.schema.children[1]->format) == "i");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* k1c = a.children[1];
        const ArrowArray* valc = a.children[2];
        const auto* k0 =
            static_cast<const std::uint16_t*>(k0c->buffers[1]) + k0c->offset;
        const auto* k1 =
            static_cast<const std::int32_t*>(k1c->buffers[1]) + k1c->offset;
        const std::int64_t* val =
            static_cast<const std::int64_t*>(valc->buffers[1]) + valc->offset;

        bool saw_a = false, saw_b = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            if (k0[row] == 1000 && k1[row] == 7) {
                CHECK(val[row] == 3);
                saw_a = true;
            } else if (k0[row] == 2000 && k1[row] == -5) {
                CHECK(val[row] == 2);
                saw_b = true;
            }
        }
        CHECK(saw_a);
        CHECK(saw_b);
#endif
    }

    TEST_CASE("unsigned key columns round-trip values above INT32_MAX") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<BigUnsignedSlice>(nullptr),
            intern, &reg, &named);

        const std::uint64_t big_u32 = 4294967295ull;  // UINT32_MAX
        const std::uint64_t big_u64 = 5000000000ull;  // > INT32_MAX
        const std::vector<FoldEvent> evs = {typed_event(big_u32, big_u64),
                                            typed_event(big_u32, big_u64)};
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("big");
        REQUIRE(it != named.results().end());
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 3);
        REQUIRE(a.length == 1);

        // uint32 is Arrow format "I", uint64 "L".
        CHECK(std::string_view(owned.schema.children[0]->format) == "I");
        CHECK(std::string_view(owned.schema.children[1]->format) == "L");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* k1c = a.children[1];
        const auto* k0 =
            static_cast<const std::uint32_t*>(k0c->buffers[1]) + k0c->offset;
        const auto* k1 =
            static_cast<const std::uint64_t*>(k1c->buffers[1]) + k1c->offset;
        CHECK(k0[a.offset] == big_u32);
        CHECK(k1[a.offset] == big_u64);
#endif
    }

    TEST_CASE("f64 key column round-trips exact doubles; equal bits merge") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<DoubleKeySlice>(nullptr),
            intern, &reg, &named);

        // Merged: 1.5 -> 3 (bit-identical duplicates merge), -2.25 -> 1,
        // 3.5 -> 1.
        const std::vector<std::vector<FoldEvent>> slices = {
            {dbl_event(1.5), dbl_event(1.5), dbl_event(-2.25)},
            {dbl_event(1.5), dbl_event(3.5)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("dbl");
        REQUIRE(it != named.results().end());
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 2);
        REQUIRE(a.length == 3);

        // float64 is Arrow format "g".
        CHECK(std::string_view(owned.schema.children[0]->format) == "g");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* valc = a.children[1];
        const double* k0 =
            static_cast<const double*>(k0c->buffers[1]) + k0c->offset;
        const std::int64_t* val =
            static_cast<const std::int64_t*>(valc->buffers[1]) + valc->offset;

        std::map<double, std::int64_t> got;
        for (std::int64_t r = 0; r < a.length; ++r)
            got.emplace(k0[r + a.offset], val[r + a.offset]);
        CHECK(got.at(1.5) == 3);
        CHECK(got.at(-2.25) == 1);
        CHECK(got.at(3.5) == 1);
#endif
    }

    TEST_CASE("typed min/max value monoids materialize at the element width") {
        SUBCASE("MIN_I32 keeps a negative signed min in an int32 column") {
            StringIntern intern;
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(dftracer::utils::plugins::make_plugin<
                                  MinMaxIntSlice<DFTU_MONOID_MIN_I32>>(nullptr),
                              intern, &reg, &named);

            const std::uint64_t neg =
                static_cast<std::uint64_t>(static_cast<std::int64_t>(-5));
            const std::vector<std::vector<FoldEvent>> slices = {
                {val_event(1, 10), val_event(1, neg)},
                {val_event(1, 3), val_event(1, 10)}};
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
            auto it = named.results().find("v");
            REQUIRE(it != named.results().end());
            const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
            const ArrowArray& a = owned.array;
            REQUIRE(a.length == 1);
            CHECK(std::string_view(owned.schema.children[1]->format) == "i");
            const ArrowArray* vc = a.children[1];
            const auto* v =
                static_cast<const std::int32_t*>(vc->buffers[1]) + vc->offset;
            CHECK(v[0] == -5);
#endif
        }

        SUBCASE("MAX_U32 keeps a value above INT32_MAX in a uint32 column") {
            StringIntern intern;
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(dftracer::utils::plugins::make_plugin<
                                  MinMaxIntSlice<DFTU_MONOID_MAX_U32>>(nullptr),
                              intern, &reg, &named);

            const std::uint64_t big = 4000000000ull;  // > INT32_MAX
            const std::vector<std::vector<FoldEvent>> slices = {
                {val_event(1, 100), val_event(1, big)}, {val_event(1, 50)}};
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
            auto it = named.results().find("v");
            REQUIRE(it != named.results().end());
            const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
            const ArrowArray& a = owned.array;
            REQUIRE(a.length == 1);
            CHECK(std::string_view(owned.schema.children[1]->format) == "I");
            const ArrowArray* vc = a.children[1];
            const auto* v =
                static_cast<const std::uint32_t*>(vc->buffers[1]) + vc->offset;
            CHECK(v[0] == 4000000000u);
#endif
        }

        SUBCASE("MIN_U16 materializes a uint16 column") {
            StringIntern intern;
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(dftracer::utils::plugins::make_plugin<
                                  MinMaxIntSlice<DFTU_MONOID_MIN_U16>>(nullptr),
                              intern, &reg, &named);

            const std::vector<std::vector<FoldEvent>> slices = {
                {val_event(1, 50), val_event(1, 30)}, {val_event(1, 10)}};
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
            auto it = named.results().find("v");
            REQUIRE(it != named.results().end());
            const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
            const ArrowArray& a = owned.array;
            REQUIRE(a.length == 1);
            CHECK(std::string_view(owned.schema.children[1]->format) == "S");
            const ArrowArray* vc = a.children[1];
            const auto* v =
                static_cast<const std::uint16_t*>(vc->buffers[1]) + vc->offset;
            CHECK(v[0] == 10);
#endif
        }

        SUBCASE("MIN_F32 materializes a float32 column") {
            StringIntern intern;
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<MinF32Slice>(nullptr),
                intern, &reg, &named);

            const std::vector<std::vector<FoldEvent>> slices = {
                {fval_event(1, 5.0), fval_event(1, 1.5)}, {fval_event(1, 8.0)}};
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
            auto it = named.results().find("v");
            REQUIRE(it != named.results().end());
            const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
            const ArrowArray& a = owned.array;
            REQUIRE(a.length == 1);
            CHECK(std::string_view(owned.schema.children[1]->format) == "f");
            const ArrowArray* vc = a.children[1];
            const auto* v =
                static_cast<const float*>(vc->buffers[1]) + vc->offset;
            CHECK(v[0] == doctest::Approx(1.5f));
#endif
        }
    }

    TEST_CASE(
        "argmin/argmax keep the payload at the extreme, tie-break stable") {
        StringIntern intern;
        const std::uint32_t fa = intern.get_or_insert("fileA");
        const std::uint32_t fb = intern.get_or_insert("fileB");
        const std::uint32_t fc = intern.get_or_insert("fileC");

        // pid1: durs 10/30/5 across the split -> max at dur30 (fb,tid2), min at
        // dur5 (fc,tid3). pid2: two dur20 events (fa/tid10, fb/tid5); equal by
        // keeps the smaller payload id (fa < fb; tid 5 < 10).
        const std::vector<FoldEvent> s0 = {argby_event(1, 10, fa, 1),
                                           argby_event(1, 30, fb, 2),
                                           argby_event(2, 20, fa, 10)};
        const std::vector<FoldEvent> s1 = {argby_event(1, 5, fc, 3),
                                           argby_event(2, 20, fb, 5)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        struct Row {
            std::int64_t argmin_i64, argmax_i64;
            std::string argmin_str, argmax_str;
        };
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<ArgBySlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("argby"));
            const ArrowArray& a = owned.array;
            REQUIRE(a.n_children == 5);
            // v2/v3 are utf8 columns.
            CHECK(std::string_view(owned.schema.children[3]->format) == "u");
            CHECK(std::string_view(owned.schema.children[4]->format) == "u");
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
                a.children[0]->offset;
            const std::int64_t* v0 =
                static_cast<const std::int64_t*>(a.children[1]->buffers[1]) +
                a.children[1]->offset;
            const std::int64_t* v1 =
                static_cast<const std::int64_t*>(a.children[2]->buffers[1]) +
                a.children[2]->offset;
            auto str_at = [&](int c, std::int64_t row) {
                const ArrowArray* ch = a.children[c];
                const auto* offs =
                    static_cast<const std::int32_t*>(ch->buffers[1]) +
                    ch->offset;
                const char* chars = static_cast<const char*>(ch->buffers[2]);
                return std::string(chars + offs[row],
                                   offs[row + 1] - offs[row]);
            };
            std::map<std::int64_t, Row> out;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                out[k0[row]] =
                    Row{v0[row], v1[row], str_at(3, row), str_at(4, row)};
            }
            return out;
        };

        const auto forward = run({s0, s1});
        CHECK(forward.at(1).argmax_i64 == 2);
        CHECK(forward.at(1).argmin_i64 == 3);
        CHECK(forward.at(1).argmax_str == "fileB");
        CHECK(forward.at(1).argmin_str == "fileC");
        CHECK(forward.at(2).argmax_i64 == 5);
        CHECK(forward.at(2).argmin_i64 == 5);
        CHECK(forward.at(2).argmax_str == "fileA");
        CHECK(forward.at(2).argmin_str == "fileA");

        // Merge order-independence: A<-B == B<-A.
        const auto reversed = run({s1, s0});
        for (std::int64_t pid : {std::int64_t{1}, std::int64_t{2}}) {
            CHECK(reversed.at(pid).argmax_i64 == forward.at(pid).argmax_i64);
            CHECK(reversed.at(pid).argmin_i64 == forward.at(pid).argmin_i64);
            CHECK(reversed.at(pid).argmax_str == forward.at(pid).argmax_str);
            CHECK(reversed.at(pid).argmin_str == forward.at(pid).argmin_str);
        }
#endif
    }

    TEST_CASE(
        "arg-row keeps the whole payload row at the extreme, row-consistent") {
        StringIntern intern;
        const std::uint32_t fa = intern.get_or_insert("fileA");
        const std::uint32_t fb = intern.get_or_insert("fileB");
        const std::uint32_t fc = intern.get_or_insert("fileC");

        // pid1: max dur=30 -> whole row {fileB, tid200, 2.5}, NOT the max tid
        //   (300 at dur5) nor the max rate (3.5): proves row-consistency.
        // pid2: two dur20 events tie; the lexicographically smaller payload row
        //   wins ({fileA,..} < {fileB,..}), deterministic under either order.
        const std::vector<FoldEvent> s0 = {argrow_event(1, 10, fa, 100, 1.5),
                                           argrow_event(1, 30, fb, 200, 2.5),
                                           argrow_event(2, 20, fa, 10, 1.0)};
        const std::vector<FoldEvent> s1 = {argrow_event(1, 5, fc, 300, 3.5),
                                           argrow_event(2, 20, fb, 5, 9.0)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        struct Row {
            std::string file;
            std::int32_t tid;
            double rate;
        };
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<ArgRowSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("argrow"));
            const ArrowArray& a = owned.array;
            // k0 (int64) + payload p0 (utf8), p1 (int32), p2 (double).
            REQUIRE(a.n_children == 4);
            CHECK(std::string_view(owned.schema.children[1]->format) == "u");
            CHECK(std::string_view(owned.schema.children[2]->format) == "i");
            CHECK(std::string_view(owned.schema.children[3]->format) == "g");
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
                a.children[0]->offset;
            const std::int32_t* p1 =
                static_cast<const std::int32_t*>(a.children[2]->buffers[1]) +
                a.children[2]->offset;
            const double* p2 =
                static_cast<const double*>(a.children[3]->buffers[1]) +
                a.children[3]->offset;
            std::map<std::int64_t, Row> out;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                out[k0[row]] =
                    Row{bin_at(a.children[1], row), p1[row], p2[row]};
            }
            return out;
        };

        const auto forward = run({s0, s1});
        CHECK(forward.at(1).file == "fileB");
        CHECK(forward.at(1).tid == 200);
        CHECK(forward.at(1).rate == doctest::Approx(2.5));
        CHECK(forward.at(2).file == "fileA");
        CHECK(forward.at(2).tid == 10);
        CHECK(forward.at(2).rate == doctest::Approx(1.0));

        // Tie determinism + merge order-independence: A<-B == B<-A.
        const auto reversed = run({s1, s0});
        for (std::int64_t pid : {std::int64_t{1}, std::int64_t{2}}) {
            CHECK(reversed.at(pid).file == forward.at(pid).file);
            CHECK(reversed.at(pid).tid == forward.at(pid).tid);
            CHECK(reversed.at(pid).rate ==
                  doctest::Approx(forward.at(pid).rate));
        }
#endif
    }

    TEST_CASE(
        "top-k/bottom-k keep the k extreme payloads in by-order, tie stable") {
        StringIntern intern;
        // Intern in order so fa<fb<fc<fd<fe by id; a `by` tie breaks on the
        // smaller payload id, deterministic and merge-order-independent.
        const std::uint32_t fa = intern.get_or_insert("fileA");
        const std::uint32_t fb = intern.get_or_insert("fileB");
        const std::uint32_t fc = intern.get_or_insert("fileC");
        const std::uint32_t fd = intern.get_or_insert("fileD");
        const std::uint32_t fe = intern.get_or_insert("fileE");

        // pid1: 5 distinct durs (>k=3), with a tie at dur 20 (fd/tid4, fe/tid5)
        //   -> bounding drops the two smallest; tie keeps the smaller id first.
        // pid2: 2 events (<k) -> all returned. pid3: exactly 3.
        const std::vector<FoldEvent> s0 = {
            argby_event(1, 10, fa, 1), argby_event(1, 30, fb, 2),
            argby_event(1, 20, fd, 4), argby_event(2, 7, fa, 10),
            argby_event(3, 1, fa, 1),  argby_event(3, 2, fb, 2)};
        const std::vector<FoldEvent> s1 = {
            argby_event(1, 5, fc, 3), argby_event(1, 20, fe, 5),
            argby_event(2, 3, fb, 20), argby_event(3, 3, fc, 3)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        struct Row {
            std::vector<std::int64_t> topk_i64, bottomk_i64;
            std::vector<std::string> topk_str, bottomk_str;
        };
        auto i64_list = [](const ArrowArray& a, int c, std::int64_t row) {
            const ArrowArray* listc = a.children[c];
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* valc = listc->children[0];
            const std::int64_t* vals =
                static_cast<const std::int64_t*>(valc->buffers[1]);
            std::vector<std::int64_t> v;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                v.push_back(vals[j]);
            return v;
        };
        auto str_list = [](const ArrowArray& a, int c, std::int64_t row) {
            const ArrowArray* listc = a.children[c];
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* strc = listc->children[0];
            const auto* soff =
                static_cast<const std::int32_t*>(strc->buffers[1]);
            const char* chars = static_cast<const char*>(strc->buffers[2]);
            std::vector<std::string> v;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                v.emplace_back(chars + soff[j], soff[j + 1] - soff[j]);
            return v;
        };
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<TopKSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("topk"));
            const ArrowArray& a = owned.array;
            REQUIRE(a.n_children == 5);
            // v0/v1 are list<int64> ("+l"/"l"), v2/v3 list<utf8> ("+l"/"u").
            CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
            CHECK(std::string_view(
                      owned.schema.children[1]->children[0]->format) == "l");
            CHECK(std::string_view(owned.schema.children[3]->format) == "+l");
            CHECK(std::string_view(
                      owned.schema.children[3]->children[0]->format) == "u");
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
                a.children[0]->offset;
            std::map<std::int64_t, Row> out;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                out[k0[row]] = Row{i64_list(a, 1, row), i64_list(a, 2, row),
                                   str_list(a, 3, row), str_list(a, 4, row)};
            }
            return out;
        };

        const auto forward = run({s0, s1});
        using I = std::vector<std::int64_t>;
        using S = std::vector<std::string>;
        // pid1: >k with a tie -> exactly 3 kept, by-ordered, tie by payload id.
        CHECK(forward.at(1).topk_i64 == I{2, 4, 5});
        CHECK(forward.at(1).bottomk_i64 == I{3, 1, 4});
        CHECK(forward.at(1).topk_str == S{"fileB", "fileD", "fileE"});
        CHECK(forward.at(1).bottomk_str == S{"fileC", "fileA", "fileD"});
        // pid2: fewer than k -> all returned.
        CHECK(forward.at(2).topk_i64 == I{10, 20});
        CHECK(forward.at(2).topk_str == S{"fileA", "fileB"});
        CHECK(forward.at(2).bottomk_str == S{"fileB", "fileA"});
        // pid3: exactly k.
        CHECK(forward.at(3).topk_str == S{"fileC", "fileB", "fileA"});
        CHECK(forward.at(3).bottomk_str == S{"fileA", "fileB", "fileC"});

        // Merge order-independence: A<-B == B<-A.
        const auto reversed = run({s1, s0});
        for (std::int64_t pid :
             {std::int64_t{1}, std::int64_t{2}, std::int64_t{3}}) {
            CHECK(reversed.at(pid).topk_i64 == forward.at(pid).topk_i64);
            CHECK(reversed.at(pid).bottomk_i64 == forward.at(pid).bottomk_i64);
            CHECK(reversed.at(pid).topk_str == forward.at(pid).topk_str);
            CHECK(reversed.at(pid).bottomk_str == forward.at(pid).bottomk_str);
        }
#endif
    }

#ifdef DFTRACER_UTILS_ENABLE_ARROW
    // Read an APPROX_TOPK_I64 result: key column k0 (int64) and the value
    // column list<struct<value:int64, count:int64>>, into pid -> [(value,
    // count)..].
    inline std::map<std::int64_t,
                    std::vector<std::pair<std::int64_t, std::int64_t>>>
    read_approx_i64(const ArrowArray& a) {
        const std::int64_t* k0 =
            static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
            a.children[0]->offset;
        const ArrowArray* listc = a.children[1];
        const auto* loff =
            static_cast<const std::int32_t*>(listc->buffers[1]) + listc->offset;
        const ArrowArray* st = listc->children[0];
        const std::int64_t* vv =
            static_cast<const std::int64_t*>(st->children[0]->buffers[1]) +
            st->children[0]->offset;
        const std::int64_t* cc =
            static_cast<const std::int64_t*>(st->children[1]->buffers[1]) +
            st->children[1]->offset;
        std::map<std::int64_t,
                 std::vector<std::pair<std::int64_t, std::int64_t>>>
            out;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            std::vector<std::pair<std::int64_t, std::int64_t>> v;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                v.emplace_back(vv[j], cc[j]);
            out.emplace(k0[row], std::move(v));
        }
        return out;
    }

    // Read a SAMPLE_I64 result: key column k0 (int64) and the value column
    // list<int64>, into pid -> [item..] (each list already sorted by item).
    inline std::map<std::int64_t, std::vector<std::int64_t>> read_sample_i64(
        const ArrowArray& a) {
        const std::int64_t* k0 =
            static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
            a.children[0]->offset;
        const ArrowArray* listc = a.children[1];
        const auto* loff =
            static_cast<const std::int32_t*>(listc->buffers[1]) + listc->offset;
        const std::int64_t* vals =
            static_cast<const std::int64_t*>(listc->children[0]->buffers[1]);
        std::map<std::int64_t, std::vector<std::int64_t>> out;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            std::vector<std::int64_t> v;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                v.push_back(vals[j]);
            out.emplace(k0[row], std::move(v));
        }
        return out;
    }
#endif

    TEST_CASE(
        "approx-top-k is exact when the counter capacity exceeds the distinct "
        "count") {
        StringIntern intern;
        // pid1 multiset: 100 x4, 200 x3, 300 x2, 400 x1 (4 distinct <= k=8), so
        // SpaceSaving never evicts and every count is exact.
        std::vector<FoldEvent> evs;
        auto push = [&](std::int64_t v, int n) {
            for (int i = 0; i < n; ++i)
                evs.push_back(val_event(1, static_cast<std::uint64_t>(v)));
        };
        push(100, 4);
        push(200, 3);
        push(300, 2);
        push(400, 1);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<ApproxI64Slice<8>>(nullptr),
            intern, &reg, &named);
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
        finalize_now(*master.fold);

        const OwnedArrow& owned =
            std::get<OwnedArrow>(named.results().at("approx"));
        // Schema: value column is list<struct<value:int64, count:int64>>.
        CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
        const ArrowSchema* stsch = owned.schema.children[1]->children[0];
        CHECK(std::string_view(stsch->format) == "+s");
        REQUIRE(stsch->n_children == 2);
        CHECK(std::string_view(stsch->children[0]->name) == "value");
        CHECK(std::string_view(stsch->children[0]->format) == "l");
        CHECK(std::string_view(stsch->children[1]->name) == "count");
        CHECK(std::string_view(stsch->children[1]->format) == "l");

        const auto got = read_approx_i64(owned.array);
        using P = std::vector<std::pair<std::int64_t, std::int64_t>>;
        // Exact counts, ordered by count descending.
        CHECK(got.at(1) == P{{100, 4}, {200, 3}, {300, 2}, {400, 1}});
#endif
    }

    TEST_CASE(
        "approx-top-k captures the heavy hitters with a bounded over-estimate "
        "when the capacity is smaller than the distinct count") {
        StringIntern intern;
        // k=3, but 9 distinct values. Feed 6 rare singletons first so the heavy
        // hitters must evict them (inflating their counts by the evicted count,
        // bounded by the error). True freqs: 100 x15, 200 x10, 300 x8.
        std::vector<FoldEvent> evs;
        auto push = [&](std::int64_t v, int n) {
            for (int i = 0; i < n; ++i)
                evs.push_back(val_event(1, static_cast<std::uint64_t>(v)));
        };
        for (std::int64_t r = 1; r <= 6; ++r) push(r, 1);  // rare noise
        push(100, 15);
        push(200, 10);
        push(300, 8);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<ApproxI64Slice<3>>(nullptr),
            intern, &reg, &named);
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
        finalize_now(*master.fold);

        const auto got = read_approx_i64(
            std::get<OwnedArrow>(named.results().at("approx")).array);
        const auto& row = got.at(1);
        // Bounded to k monitored counters.
        REQUIRE(row.size() == 3);
        // The top entry is the true most-frequent value.
        CHECK(row.front().first == 100);
        // Every true heavy hitter is present; its reported count over-estimates
        // (>= true) and the error is bounded (<= true + total evicted noise).
        std::map<std::int64_t, std::int64_t> truth = {
            {100, 15}, {200, 10}, {300, 8}};
        std::map<std::int64_t, std::int64_t> reported;
        for (const auto& [v, c] : row) reported[v] = c;
        for (const auto& [v, t] : truth) {
            REQUIRE(reported.count(v) == 1);
            CHECK(reported[v] >= t);
            CHECK(reported[v] <= t + 6);
        }
#endif
    }

    TEST_CASE(
        "approx-top-k merges across slices; exact when capacity >= distinct, "
        "order-independent") {
        StringIntern intern;
        // Same multiset as the exactness case, split across two slices; k=8 >=
        // 4 distinct, so the merged counts are exact regardless of split.
        auto make =
            [&](std::initializer_list<std::pair<std::int64_t, int>> ms) {
                std::vector<FoldEvent> v;
                for (auto [val, n] : ms)
                    for (int i = 0; i < n; ++i)
                        v.push_back(
                            val_event(1, static_cast<std::uint64_t>(val)));
                return v;
            };
        const std::vector<FoldEvent> s0 =
            make({{100, 2}, {200, 2}, {300, 1}, {400, 1}});
        const std::vector<FoldEvent> s1 = make({{100, 2}, {200, 1}, {300, 1}});

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<ApproxI64Slice<8>>(
                    nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            return read_approx_i64(
                std::get<OwnedArrow>(named.results().at("approx")).array);
        };
        const auto forward = run({s0, s1});
        const auto reversed = run({s1, s0});
        using P = std::vector<std::pair<std::int64_t, std::int64_t>>;
        CHECK(forward.at(1) == P{{100, 4}, {200, 3}, {300, 2}, {400, 1}});
        CHECK(forward.at(1) == reversed.at(1));  // merge order-independent
#endif
    }

    TEST_CASE(
        "sample keeps the full distinct set when k >= the distinct count") {
        StringIntern intern;
        // pid1 sees 4 distinct items, each repeated; k=8 > 4, so bottom-k never
        // evicts and the sample is exactly the distinct set (the strongest
        // correctness check). Repeats must not add duplicates.
        std::vector<FoldEvent> evs;
        auto push = [&](std::int64_t v, int n) {
            for (int i = 0; i < n; ++i)
                evs.push_back(val_event(1, static_cast<std::uint64_t>(v)));
        };
        push(10, 3);
        push(20, 1);
        push(30, 4);
        push(40, 2);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<SampleI64Slice<8>>(nullptr),
            intern, &reg, &named);
        auto slice = master.fold->slice();
        auto* pf = static_cast<PluginFold*>(slice.get());
        ScanUnit unit;
        FoldBatch fb{std::span<const FoldEvent>(evs), unit};
        pf->step(fb);
        master.fold->merge(*pf);
        finalize_now(*master.fold);

        const OwnedArrow& owned =
            std::get<OwnedArrow>(named.results().at("sample"));
        // Value column is list<int64> ("+l"/"l").
        CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
        CHECK(std::string_view(owned.schema.children[1]->children[0]->format) ==
              "l");
        const auto got = read_sample_i64(owned.array);
        using I = std::vector<std::int64_t>;
        CHECK(got.at(1) == I{10, 20, 30, 40});  // full distinct set, sorted
#endif
    }

    TEST_CASE(
        "sample is deterministic and order-independent when k < distinct") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            run_partitioned<SampleI64Slice<4>>(named, reg, intern, slices, 0);
            return read_sample_i64(
                std::get<OwnedArrow>(named.results().at("sample")).array);
        };
        // 10 distinct items 1..10; each repeated a few times in a shuffled
        // order, split across two slices. k=4 forces eviction.
        std::vector<FoldEvent> a, b;
        const std::vector<std::int64_t> items = {7, 3, 10, 1, 6, 9, 2, 8, 4, 5};
        for (std::size_t i = 0; i < items.size(); ++i) {
            auto& dst = (i % 2) ? a : b;
            for (int rep = 0; rep < 3; ++rep)  // repeats: distinct semantics
                dst.push_back(
                    val_event(1, static_cast<std::uint64_t>(items[i])));
        }
        const auto forward = run({a, b});
        const auto reversed = run({b, a});

        // Exactly k distinct items, all drawn from the input set.
        REQUIRE(forward.at(1).size() == 4);
        std::set<std::int64_t> input(items.begin(), items.end());
        std::set<std::int64_t> seen;
        for (std::int64_t v : forward.at(1)) {
            CHECK(input.count(v) == 1);
            seen.insert(v);
        }
        CHECK(seen.size() == 4);  // no duplicates despite repeated feeds
        // The whole point of bottom-k over reservoir: same multiset, different
        // order -> identical sample.
        CHECK(reversed.at(1) == forward.at(1));

        // Distinct semantics: feeding each item exactly once yields the same
        // sample as feeding it many times.
        std::vector<FoldEvent> once;
        for (std::int64_t v : items)
            once.push_back(val_event(1, static_cast<std::uint64_t>(v)));
        const auto single = run({once});
        CHECK(single.at(1) == forward.at(1));
#endif
    }

    TEST_CASE("sample merges across slices identically to a single stream") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            run_partitioned<SampleI64Slice<3>>(named, reg, intern, slices, 0);
            return read_sample_i64(
                std::get<OwnedArrow>(named.results().at("sample")).array);
        };
        const std::vector<std::int64_t> items = {5, 1, 9, 3, 7, 2, 8, 4, 6};
        std::vector<FoldEvent> whole, s0, s1;
        for (std::size_t i = 0; i < items.size(); ++i) {
            whole.push_back(val_event(1, static_cast<std::uint64_t>(items[i])));
            ((i % 2) ? s1 : s0)
                .push_back(val_event(1, static_cast<std::uint64_t>(items[i])));
        }
        const auto one = run({whole});
        const auto split = run({s0, s1});
        const auto swapped = run({s1, s0});
        REQUIRE(one.at(1).size() == 3);
        CHECK(split.at(1) == one.at(1));      // merged == single-stream
        CHECK(swapped.at(1) == split.at(1));  // merge(A,B) == merge(B,A)
#endif
    }

    TEST_CASE("spill: SAMPLE_STR map spilled result equals in-memory") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        // 400 pids, each touching >k=3 distinct files (so eviction runs); a
        // tiny share forces many partition spills. Proves the bottom-k sample
        // survives serialize under spill.
        const std::size_t SHARE = 256;
        std::vector<std::uint32_t> ids;
        for (int i = 0; i < 8; ++i)
            ids.push_back(intern.get_or_insert("f" + std::to_string(i)));
        std::vector<FoldEvent> s0, s1;
        for (std::int64_t i = 0; i < 400; ++i) {
            s0.push_back(edge(i, ids[i % 8]));
            s0.push_back(edge(i, ids[(i + 1) % 8]));
            s1.push_back(edge(i, ids[(i + 2) % 8]));
            s1.push_back(edge(i, ids[(i + 3) % 8]));
        }
        const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

        auto extract = [&](bool spill, std::size_t* spills) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            SpillRoot root;
            if (spill)
                *spills = run_spilled<SampleStrSlice>(named, reg, intern,
                                                      slices, SHARE, root.path);
            else
                run_partitioned<SampleStrSlice>(named, reg, intern, slices, 0);
            if (spill) CHECK_FALSE(root.has_run_dirs());
            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("sample_file"));
            const ArrowArray& a = owned.array;
            const ArrowArray* listc = a.children[1];
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* strc = listc->children[0];
            const auto* soff =
                static_cast<const std::int32_t*>(strc->buffers[1]);
            const char* chars = static_cast<const char*>(strc->buffers[2]);
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
                a.children[0]->offset;
            std::map<std::int64_t, std::vector<std::string>> m;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                std::vector<std::string> v;
                for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                    v.emplace_back(chars + soff[j], soff[j + 1] - soff[j]);
                m.emplace(k0[row], std::move(v));
            }
            return m;
        };

        std::size_t spills = 0;
        const auto mem = extract(false, nullptr);
        const auto spilled = extract(true, &spills);
        CHECK(spills > 0);
        CHECK(spilled == mem);  // bottom-k sample preserved through spill
#endif
    }

    TEST_CASE("spill: APPROX_TOPK_STR map spilled result equals in-memory") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        // 400 pids, each touching a skewed file distribution (>k=3 distinct so
        // eviction runs); a tiny share forces many partition spills. Proves the
        // SpaceSaving sketch survives serialize under spill.
        const std::size_t SHARE = 256;
        std::vector<std::uint32_t> ids;
        for (int i = 0; i < 8; ++i)
            ids.push_back(intern.get_or_insert("f" + std::to_string(i)));
        std::vector<FoldEvent> s0, s1;
        for (std::int64_t i = 0; i < 400; ++i) {
            // A dominant file (id 0) plus rotating others, split across slices.
            s0.push_back(edge(i, ids[0]));
            s0.push_back(edge(i, ids[0]));
            s0.push_back(edge(i, ids[1 + (i % 3)]));
            s1.push_back(edge(i, ids[0]));
            s1.push_back(edge(i, ids[4 + (i % 4)]));
        }
        const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

        auto extract = [&](bool spill, std::size_t* spills) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            SpillRoot root;
            if (spill)
                *spills = run_spilled<ApproxStrSlice>(named, reg, intern,
                                                      slices, SHARE, root.path);
            else
                run_partitioned<ApproxStrSlice>(named, reg, intern, slices, 0);
            if (spill) CHECK_FALSE(root.has_run_dirs());
            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("approx_file"));
            const ArrowArray& a = owned.array;
            const ArrowArray* listc = a.children[1];
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* st = listc->children[0];
            const ArrowArray* strc = st->children[0];
            const auto* soff =
                static_cast<const std::int32_t*>(strc->buffers[1]);
            const char* chars = static_cast<const char*>(strc->buffers[2]);
            const std::int64_t* cc =
                static_cast<const std::int64_t*>(st->children[1]->buffers[1]) +
                st->children[1]->offset;
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
                a.children[0]->offset;
            std::map<std::int64_t,
                     std::vector<std::pair<std::string, std::int64_t>>>
                m;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                std::vector<std::pair<std::string, std::int64_t>> v;
                for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                    v.emplace_back(
                        std::string(chars + soff[j], soff[j + 1] - soff[j]),
                        cc[j]);
                m.emplace(k0[row], std::move(v));
            }
            return m;
        };

        std::size_t spills = 0;
        const auto mem = extract(false, nullptr);
        const auto spilled = extract(true, &spills);
        CHECK(spills > 0);
        CHECK(spilled == mem);  // sketch order and counts preserved through
                                // spill
#endif
    }

    TEST_CASE("mean/variance/stddev match hand-computed sample statistics") {
        StringIntern intern;
        // pid1 durs {2,4,4,4,5,5,7,9} split across slices: mean 5, sample
        // variance 32/7, stddev sqrt(32/7). (Population var 4 / std 2 would
        // fail the sample-convention check below.)
        const std::vector<FoldEvent> s0 = {
            dur_event(1, 0, 2), dur_event(1, 0, 4), dur_event(1, 0, 4),
            dur_event(1, 0, 4)};
        const std::vector<FoldEvent> s1 = {
            dur_event(1, 0, 5), dur_event(1, 0, 5), dur_event(1, 0, 7),
            dur_event(1, 0, 9)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<MomentSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            const ArrowArray& a =
                std::get<OwnedArrow>(named.results().at("moments")).array;
            REQUIRE(a.n_children == 4);
            REQUIRE(a.length == 1);
            const std::int64_t row = a.offset;
            auto f64 = [&](int c) {
                return (static_cast<const double*>(a.children[c]->buffers[1]) +
                        a.children[c]->offset)[row];
            };
            return std::array<double, 3>{f64(1), f64(2), f64(3)};
        };

        const auto forward = run({s0, s1});
        CHECK(forward[0] == doctest::Approx(5.0));         // mean
        CHECK(forward[1] == doctest::Approx(32.0 / 7.0));  // sample variance
        CHECK(forward[2] == doctest::Approx(std::sqrt(32.0 / 7.0)));  // stddev
        // Merge-equivalence: two slices merged == the whole set in one slice.
        const auto single =
            run({{s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]}});
        CHECK(single[0] == doctest::Approx(forward[0]));
        CHECK(single[1] == doctest::Approx(forward[1]));
        CHECK(single[2] == doctest::Approx(forward[2]));
#endif
    }

    TEST_CASE(
        "SKETCH map materializes count + a column per requested quantile") {
        StringIntern intern;
        // pid1 durs 1..1000 split across two slices; quantiles are the
        // percentiles of 1..1000 to within DDSketch's ~1% relative error.
        std::vector<FoldEvent> s0, s1;
        for (std::uint64_t d = 1; d <= 500; ++d)
            s0.push_back(dur_event(1, 0, d));
        for (std::uint64_t d = 501; d <= 1000; ++d)
            s1.push_back(dur_event(1, 0, d));

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<QuantileSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            const OwnedArrow& oa =
                std::get<OwnedArrow>(named.results().at("q"));
            const ArrowArray& a = oa.array;
            // key column + count + 3 quantile columns.
            REQUIRE(a.n_children == 5);
            REQUIRE(a.length == 1);
            const std::int64_t row = a.offset;
            const std::int64_t count =
                (static_cast<const std::int64_t*>(a.children[1]->buffers[1]) +
                 a.children[1]->offset)[row];
            auto f64 = [&](int c) {
                return (static_cast<const double*>(a.children[c]->buffers[1]) +
                        a.children[c]->offset)[row];
            };
            // p50, p90, p99 in children 2, 3, 4; assert the schema names too.
            CHECK(std::string(oa.schema.children[1]->name) == "count");
            CHECK(std::string(oa.schema.children[2]->name) == "p50");
            CHECK(std::string(oa.schema.children[3]->name) == "p90");
            CHECK(std::string(oa.schema.children[4]->name) == "p99");
            return std::array<double, 4>{static_cast<double>(count), f64(2),
                                         f64(3), f64(4)};
        };

        const auto forward = run({s0, s1});
        CHECK(forward[0] == doctest::Approx(1000.0));
        CHECK(forward[1] == doctest::Approx(500.0).epsilon(0.02));
        CHECK(forward[2] == doctest::Approx(900.0).epsilon(0.02));
        CHECK(forward[3] == doctest::Approx(990.0).epsilon(0.02));
        // Merge-equivalence: the two slices merged match one combined slice.
        std::vector<FoldEvent> all = s0;
        all.insert(all.end(), s1.begin(), s1.end());
        const auto single = run({all});
        CHECK(single[0] == doctest::Approx(forward[0]));
        CHECK(single[1] == doctest::Approx(forward[1]));
        CHECK(single[2] == doctest::Approx(forward[2]));
        CHECK(single[3] == doctest::Approx(forward[3]));
#endif
    }

    TEST_CASE("fused map splits into one named table per component") {
        StringIntern intern;
        // pid1 durs 1..1000 across two slices; the fused COUNTER/SUM emit as
        // two separate tables "n" and "tot", byte-identical to unfused maps.
        std::vector<FoldEvent> s0, s1;
        for (std::uint64_t d = 1; d <= 500; ++d)
            s0.push_back(dur_event(1, 0, d));
        for (std::uint64_t d = 501; d <= 1000; ++d)
            s1.push_back(dur_event(1, 0, d));

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<FusedSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            // Two distinct named results, each a {k0, value} table with one
            // row.
            REQUIRE(named.results().count("n") == 1);
            REQUIRE(named.results().count("tot") == 1);
            const ArrowArray& na =
                std::get<OwnedArrow>(named.results().at("n")).array;
            const ArrowArray& ta =
                std::get<OwnedArrow>(named.results().at("tot")).array;
            REQUIRE(na.n_children == 2);  // k0, value
            REQUIRE(ta.n_children == 2);
            REQUIRE(na.length == 1);
            REQUIRE(ta.length == 1);
            const std::int64_t count =
                (static_cast<const std::int64_t*>(na.children[1]->buffers[1]) +
                 na.children[1]->offset)[na.offset];
            const double total =
                (static_cast<const double*>(ta.children[1]->buffers[1]) +
                 ta.children[1]->offset)[ta.offset];
            return std::pair<std::int64_t, double>{count, total};
        };

        const auto forward = run({s0, s1});
        CHECK(forward.first == 1000);
        CHECK(forward.second == doctest::Approx(500500.0));  // sum 1..1000
        // Merge-equivalence: two slices merged match one combined slice.
        std::vector<FoldEvent> all = s0;
        all.insert(all.end(), s1.begin(), s1.end());
        const auto single = run({all});
        CHECK(single.first == forward.first);
        CHECK(single.second == doctest::Approx(forward.second));
#endif
    }

    TEST_CASE("skewness/kurtosis match MetricStats population convention") {
        namespace agg = dftracer::utils::trace::aggregators;
        StringIntern intern;
        const std::vector<double> data = {2.0, 4.0, 4.0, 4.0,
                                          5.0, 5.0, 7.0, 9.0};
        const std::vector<FoldEvent> s0 = {
            dur_event(1, 0, 2), dur_event(1, 0, 4), dur_event(1, 0, 4),
            dur_event(1, 0, 4)};
        const std::vector<FoldEvent> s1 = {
            dur_event(1, 0, 5), dur_event(1, 0, 5), dur_event(1, 0, 7),
            dur_event(1, 0, 9)};

        // The aggregator's answer on the same data is the reference.
        agg::MetricStats ref;
        for (double x : data) ref.update(x);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<HighMomentSlice>(nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
                pf->step(fb2);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            const ArrowArray& a =
                std::get<OwnedArrow>(named.results().at("hmoments")).array;
            REQUIRE(a.n_children == 3);
            REQUIRE(a.length == 1);
            const std::int64_t row = a.offset;
            auto f64 = [&](int c) {
                return (static_cast<const double*>(a.children[c]->buffers[1]) +
                        a.children[c]->offset)[row];
            };
            return std::array<double, 2>{f64(1), f64(2)};
        };

        const auto forward = run({s0, s1});
        CHECK(forward[0] == doctest::Approx(ref.get_skewness()));
        CHECK(forward[1] == doctest::Approx(ref.get_kurtosis()));
        // Hand-computed fingerprints (M2=32, M3=42, M4=356, n=8): population
        // skewness sqrt(8)*42/32^1.5, excess kurtosis 8*356/32^2 - 3.
        CHECK(forward[0] == doctest::Approx(0.656221).epsilon(0.0001));
        CHECK(forward[1] == doctest::Approx(-0.21875));
        // Merge-equivalence: two slices == the whole set in one.
        const auto single =
            run({{s0[0], s0[1], s0[2], s0[3], s1[0], s1[1], s1[2], s1[3]}});
        CHECK(single[0] == doctest::Approx(forward[0]));
        CHECK(single[1] == doctest::Approx(forward[1]));
#endif
    }

    TEST_CASE("corr/covar/regr readouts on known two-variable datasets") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        // pid1: y = 3x + 5 (perfect line) split across slices -> slope 3,
        // intercept 5, corr/r2 1, covar_pop 3.75, covar_samp 5.
        // pid2: y = 2x -> corr 1. pid3: y = -x -> corr -1. pid4: a symmetric
        // square (0,0)(0,1)(1,0)(1,1) -> corr 0.
        const std::vector<FoldEvent> s0 = {
            xy_event(1, 1.0, 8.0),  xy_event(1, 2.0, 11.0),
            xy_event(2, 1.0, 2.0),  xy_event(2, 2.0, 4.0),
            xy_event(3, 1.0, -1.0), xy_event(3, 2.0, -2.0),
            xy_event(4, 0.0, 0.0),  xy_event(4, 0.0, 1.0)};
        const std::vector<FoldEvent> s1 = {
            xy_event(1, 3.0, 14.0), xy_event(1, 4.0, 17.0),
            xy_event(2, 3.0, 6.0),  xy_event(2, 4.0, 8.0),
            xy_event(3, 3.0, -3.0), xy_event(3, 4.0, -4.0),
            xy_event(4, 1.0, 0.0),  xy_event(4, 1.0, 1.0)};

        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<RegrSlice>(nullptr), intern,
            &reg, &named);
        for (const auto& evs : {s0, s1}) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb2{std::span<const FoldEvent>(evs), unit};
            pf->step(fb2);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);
        const ArrowArray& a =
            std::get<OwnedArrow>(named.results().at("regr")).array;
        REQUIRE(a.n_children == 7);  // k0 + 6 value columns
        std::map<std::int64_t, std::array<double, 6>> rows;
        auto i64 = [&](int c) {
            return static_cast<const std::int64_t*>(a.children[c]->buffers[1]) +
                   a.children[c]->offset;
        };
        auto f64 = [&](int c) {
            return static_cast<const double*>(a.children[c]->buffers[1]) +
                   a.children[c]->offset;
        };
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            std::array<double, 6> v;
            for (int c = 0; c < 6; ++c) v[c] = f64(c + 1)[row];
            rows[i64(0)[row]] = v;
        }
        REQUIRE(rows.size() == 4);
        // pid1: [corr, covar_pop, covar_samp, slope, intercept, r2].
        CHECK(rows[1][0] == doctest::Approx(1.0));   // corr
        CHECK(rows[1][1] == doctest::Approx(3.75));  // covar_pop
        CHECK(rows[1][2] == doctest::Approx(5.0));   // covar_samp
        CHECK(rows[1][3] == doctest::Approx(3.0));   // regr_slope
        CHECK(rows[1][4] == doctest::Approx(5.0));   // regr_intercept
        CHECK(rows[1][5] == doctest::Approx(1.0));   // regr_r2
        CHECK(rows[2][0] == doctest::Approx(1.0));   // y=2x -> corr 1
        CHECK(rows[2][3] == doctest::Approx(2.0));   // slope 2
        CHECK(rows[3][0] == doctest::Approx(-1.0));  // y=-x -> corr -1
        CHECK(rows[3][3] == doctest::Approx(-1.0));  // slope -1
        CHECK(rows[4][0] == doctest::Approx(0.0));   // uncorrelated
#endif
    }

    TEST_CASE("Host::map_add_xy_at wrapper matches the raw ABI") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        const std::vector<std::vector<FoldEvent>> slices = {
            {xy_event(1, 1.0, 8.0), xy_event(1, 2.0, 11.0),
             xy_event(2, 1.0, 2.0), xy_event(3, 1.0, -1.0)},
            {xy_event(1, 3.0, 14.0), xy_event(1, 4.0, 17.0),
             xy_event(2, 2.0, 4.0), xy_event(3, 2.0, -2.0)}};

        auto rows_of = [&](auto slice_tag, const char* name, int value_n) {
            using Slice = typename decltype(slice_tag)::type;
            StringIntern intern;
            SharedResultRegistry reg;
            NamedResultRegistry named;
            run_partitioned<Slice>(named, reg, intern, slices, 0);
            const ArrowArray& a =
                std::get<OwnedArrow>(named.results().at(name)).array;
            REQUIRE(a.n_children == 1 + value_n);
            std::map<std::int64_t, std::vector<double>> rows;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                std::vector<double> v(value_n);
                for (int c = 0; c < value_n; ++c) {
                    const ArrowArray* ch = a.children[c + 1];
                    v[c] = (static_cast<const double*>(ch->buffers[1]) +
                            ch->offset)[row];
                }
                rows[i64_col(&a, 0)[row]] = v;
            }
            return rows;
        };

        struct RawT {
            using type = RegrSlice;
        };
        struct WrapT {
            using type = RegrWrapperSlice;
        };
        struct RawCorrT {
            using type = CorrSlice;
        };
        struct WrapCorrT {
            using type = CorrWrapperSlice;
        };
        CHECK(rows_of(RawT{}, "regr", 6) == rows_of(WrapT{}, "regr", 6));
        CHECK(rows_of(RawCorrT{}, "corr", 1) ==
              rows_of(WrapCorrT{}, "corr", 1));
#endif
    }

    TEST_CASE("map_new accepts BYTES keys and rejects scalar-less values") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder h(dftracer::utils::plugins::make_plugin<EdgeSlice>(nullptr),
                     intern, &reg, &named);

        const dftu_type bytes_key[1] = {DFTU_T_BYTES};
        CHECK(h.fold->map_get("ok_bytes", bytes_key, 1, DFTU_MONOID_COUNTER) !=
              nullptr);
        const dftu_type str_key[1] = {DFTU_T_STR};
        CHECK(h.fold->map_get("ok_str", str_key, 1, DFTU_MONOID_COUNTER) !=
              nullptr);
        // EVENT is a non-key ABI type and must still be rejected.
        const dftu_type bad_key[1] = {DFTU_T_EVENT};
        CHECK(h.fold->map_get("bad_key", bad_key, 1, DFTU_MONOID_COUNTER) ==
              nullptr);
        const dftu_type i64_key[2] = {DFTU_T_I64, DFTU_T_I64};
        CHECK(h.fold->map_get("bad_val", i64_key, 2, DFTU_MONOID_SKETCH) ==
              nullptr);
        CHECK(h.fold->map_get("ok", i64_key, 2, DFTU_MONOID_COUNTER) !=
              nullptr);
    }

    TEST_CASE("nested map: one row per outer key, merged inner list<struct>") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<NestedCountSlice>(nullptr),
            intern, &reg, &named);

        // Two slices; merged: pid1 -> {10:3, 20:1}, pid2 -> {30:2}.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(1, 10), edge(1, 10), edge(1, 20), edge(2, 30)},
            {edge(1, 10), edge(2, 30)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("nested");
        REQUIRE(it != named.results().end());
        REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 2);
        REQUIRE(a.length == 2);

        // Schema: k0 int64, value list<struct<ik0:int64, value:int64>>.
        CHECK(std::string_view(owned.schema.children[0]->format) == "l");
        CHECK(std::string_view(owned.schema.children[1]->format) == "+l");
        REQUIRE(owned.schema.children[1]->n_children == 1);
        const ArrowSchema* stsch = owned.schema.children[1]->children[0];
        CHECK(std::string_view(stsch->format) == "+s");
        REQUIRE(stsch->n_children == 2);
        CHECK(std::string_view(stsch->children[0]->name) == "ik0");
        CHECK(std::string_view(stsch->children[0]->format) == "l");
        CHECK(std::string_view(stsch->children[1]->name) == "value");
        CHECK(std::string_view(stsch->children[1]->format) == "l");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* listc = a.children[1];
        const std::int64_t* k0 =
            static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
        const auto* loff =
            static_cast<const std::int32_t*>(listc->buffers[1]) + listc->offset;
        const ArrowArray* st = listc->children[0];
        const std::int64_t* ik =
            static_cast<const std::int64_t*>(st->children[0]->buffers[1]) +
            st->children[0]->offset;
        const std::int64_t* iv =
            static_cast<const std::int64_t*>(st->children[1]->buffers[1]) +
            st->children[1]->offset;

        std::map<std::int64_t,
                 std::vector<std::pair<std::int64_t, std::int64_t>>>
            got;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            std::vector<std::pair<std::int64_t, std::int64_t>> inner;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                inner.emplace_back(ik[j], iv[j]);
            got.emplace(k0[row], std::move(inner));
        }
        const std::vector<std::pair<std::int64_t, std::int64_t>> exp1 = {
            {10, 3}, {20, 1}};
        const std::vector<std::pair<std::int64_t, std::int64_t>> exp2 = {
            {30, 2}};
        CHECK(got.at(1) == exp1);
        CHECK(got.at(2) == exp2);
#endif
    }

    TEST_CASE("nested product map: inner struct carries both value fields") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<NestedProductSlice>(nullptr),
            intern, &reg, &named);

        // Merged: pid1 file10 -> count 3, sum 30.0; pid1 file20 -> count 1,
        // sum 5.0; pid2 file30 -> count 2, sum 40.0.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge_dur(1, 10, 10), edge_dur(1, 10, 10), edge_dur(1, 20, 5),
             edge_dur(2, 30, 20)},
            {edge_dur(1, 10, 10), edge_dur(2, 30, 20)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("nested_wide");
        REQUIRE(it != named.results().end());
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 2);
        REQUIRE(a.length == 2);

        // Inner struct is <ik0:int64, v0:int64, v1:double>.
        const ArrowSchema* stsch = owned.schema.children[1]->children[0];
        REQUIRE(stsch->n_children == 3);
        CHECK(std::string_view(stsch->children[0]->name) == "ik0");
        CHECK(std::string_view(stsch->children[1]->name) == "v0");
        CHECK(std::string_view(stsch->children[1]->format) == "l");
        CHECK(std::string_view(stsch->children[2]->name) == "v1");
        CHECK(std::string_view(stsch->children[2]->format) == "g");

        const ArrowArray* k0c = a.children[0];
        const ArrowArray* listc = a.children[1];
        const std::int64_t* k0 =
            static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
        const auto* loff =
            static_cast<const std::int32_t*>(listc->buffers[1]) + listc->offset;
        const ArrowArray* st = listc->children[0];
        const std::int64_t* ik =
            static_cast<const std::int64_t*>(st->children[0]->buffers[1]) +
            st->children[0]->offset;
        const std::int64_t* v0 =
            static_cast<const std::int64_t*>(st->children[1]->buffers[1]) +
            st->children[1]->offset;
        const double* v1 =
            static_cast<const double*>(st->children[2]->buffers[1]) +
            st->children[2]->offset;

        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j) {
                if (k0[row] == 1 && ik[j] == 10) {
                    CHECK(v0[j] == 3);
                    CHECK(v1[j] == doctest::Approx(30.0));
                } else if (k0[row] == 1 && ik[j] == 20) {
                    CHECK(v0[j] == 1);
                    CHECK(v1[j] == doctest::Approx(5.0));
                } else if (k0[row] == 2 && ik[j] == 30) {
                    CHECK(v0[j] == 2);
                    CHECK(v1[j] == doctest::Approx(40.0));
                }
            }
        }
#endif
    }

    TEST_CASE("nested map: outer rows and inner lists are sorted, stable") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        // Files added out of key order and across the slice split; both outer
        // pids and inner fhashes must materialize ascending regardless of merge
        // order.
        const std::vector<FoldEvent> s0 = {edge(2, 30), edge(1, 20),
                                           edge(1, 10)};
        const std::vector<FoldEvent> s1 = {edge(1, 10), edge(2, 5)};

        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<NestedCountSlice>(
                    nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);

            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("nested"));
            const ArrowArray& a = owned.array;
            const ArrowArray* k0c = a.children[0];
            const ArrowArray* listc = a.children[1];
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(k0c->buffers[1]) + k0c->offset;
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* st = listc->children[0];
            const std::int64_t* ik =
                static_cast<const std::int64_t*>(st->children[0]->buffers[1]) +
                st->children[0]->offset;
            std::vector<std::int64_t> outer;
            std::vector<std::vector<std::int64_t>> inners;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                outer.push_back(k0[row]);
                std::vector<std::int64_t> keys;
                for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                    keys.push_back(ik[j]);
                inners.push_back(std::move(keys));
            }
            return std::pair{outer, inners};
        };

        const auto forward = run({s0, s1});
        const auto reversed = run({s1, s0});
        const std::vector<std::int64_t> exp_outer = {1, 2};
        const std::vector<std::vector<std::int64_t>> exp_inner = {{10, 20},
                                                                  {5, 30}};
        CHECK(forward.first == exp_outer);
        CHECK(forward.second == exp_inner);
        CHECK(reversed.first == exp_outer);
        CHECK(reversed.second == exp_inner);
#endif
    }

    // Phase 2 partition-invariance: the same input materializes to the same
    // result at K=1 (part_bits=0, today's single-partition path) and K=8
    // (part_bits=3, the partition-aware merge + materialize). Unordered maps
    // compare order-insensitively (their row order legitimately varies with K);
    // ordered and nested maps compare with exact row order.
    TEST_CASE("partition-invariance: K=1 and K=8 agree") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto i64col = [](const ArrowArray& a, int c) {
            const ArrowArray* ch = a.children[c];
            return static_cast<const std::int64_t*>(ch->buffers[1]) +
                   ch->offset;
        };
        auto f64col = [](const ArrowArray& a, int c) {
            const ArrowArray* ch = a.children[c];
            return static_cast<const double*>(ch->buffers[1]) + ch->offset;
        };

        SUBCASE("COUNTER map (unordered, order-insensitive)") {
            StringIntern intern;
            const std::vector<std::vector<FoldEvent>> slices = {
                {edge(1, 10), edge(1, 10), edge(2, 20), edge(3, 30)},
                {edge(1, 10), edge(2, 20), edge(4, 40)}};
            auto extract = [&](std::uint32_t bits) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                run_partitioned<EdgeSlice>(named, reg, intern, slices, bits);
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("edges")).array;
                const std::int64_t* k0 = i64col(a, 0);
                const std::int64_t* k1 = i64col(a, 1);
                const std::int64_t* v = i64col(a, 2);
                std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[{k0[row], k1[row]}] = v[row];
                }
                return m;
            };
            CHECK(extract(0) == extract(3));
        }

        SUBCASE("PRODUCT map (unordered, order-insensitive)") {
            StringIntern intern;
            const std::vector<std::vector<FoldEvent>> slices = {
                {edge_dur(1, 10, 10), edge_dur(1, 10, 10), edge_dur(2, 20, 20),
                 edge_dur(3, 30, 30)},
                {edge_dur(1, 10, 10), edge_dur(2, 20, 20),
                 edge_dur(4, 40, 40)}};
            auto extract = [&](std::uint32_t bits) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                run_partitioned<WideEdgeSlice>(named, reg, intern, slices,
                                               bits);
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("wide_edges"))
                        .array;
                const std::int64_t* k0 = i64col(a, 0);
                const std::int64_t* k1 = i64col(a, 1);
                const std::int64_t* v0 = i64col(a, 2);
                const double* v1 = f64col(a, 3);
                std::map<std::pair<std::int64_t, std::int64_t>,
                         std::pair<std::int64_t, double>>
                    m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[{k0[row], k1[row]}] = {v0[row], v1[row]};
                }
                return m;
            };
            CHECK(extract(0) == extract(3));
        }

        SUBCASE("ORDERED map (exact row order)") {
            StringIntern intern;
            const std::vector<std::vector<FoldEvent>> slices = {
                {edge(3, 0), edge(1, 0), edge(2, 0), edge(1, 0)},
                {edge(2, 0), edge(4, 0), edge(1, 0)}};
            auto extract = [&](std::uint32_t bits) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                run_partitioned<OrderedSlice>(named, reg, intern, slices, bits);
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("asc_i64")).array;
                const std::int64_t* k0 = i64col(a, 0);
                const std::int64_t* v = i64col(a, 1);
                std::vector<std::pair<std::int64_t, std::int64_t>> rows;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    rows.emplace_back(k0[row], v[row]);
                }
                return rows;
            };
            const auto k1 = extract(0);
            const auto k8 = extract(3);
            CHECK(k1 ==
                  k8);  // exact row order (ordered sorts deterministically)
            REQUIRE(k1.size() == 4);
            CHECK(k1[0].first == 1);
            CHECK(k1[3].first == 4);
        }

        SUBCASE("SET collection map (unordered, order-insensitive)") {
            StringIntern intern;
            const std::uint32_t a_id = intern.get_or_insert("a");
            const std::uint32_t b_id = intern.get_or_insert("b");
            const std::uint32_t c_id = intern.get_or_insert("c");
            const std::uint32_t d_id = intern.get_or_insert("d");
            const std::vector<std::vector<FoldEvent>> slices = {
                {edge(1, a_id), edge(1, b_id), edge(2, c_id), edge(3, a_id)},
                {edge(1, b_id), edge(2, d_id), edge(3, c_id)}};
            auto extract = [&](std::uint32_t bits) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                run_partitioned<FileSetSlice>(named, reg, intern, slices, bits);
                const OwnedArrow& owned =
                    std::get<OwnedArrow>(named.results().at("file_set"));
                const ArrowArray& a = owned.array;
                const std::int64_t* k0 = i64col(a, 0);
                const ArrowArray* setc = a.children[1];
                const auto* loff =
                    static_cast<const std::int32_t*>(setc->buffers[1]) +
                    setc->offset;
                const ArrowArray* strc = setc->children[0];
                const auto* soff =
                    static_cast<const std::int32_t*>(strc->buffers[1]);
                const char* chars = static_cast<const char*>(strc->buffers[2]);
                std::map<std::int64_t, std::set<std::string>> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    std::set<std::string> s;
                    for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                        s.emplace(chars + soff[j], soff[j + 1] - soff[j]);
                    m.emplace(k0[row], std::move(s));
                }
                return m;
            };
            CHECK(extract(0) == extract(3));
        }

        SUBCASE("NESTED map (exact order, outer group never split)") {
            StringIntern intern;
            const std::vector<std::vector<FoldEvent>> slices = {
                {edge(1, 10), edge(1, 10), edge(1, 20), edge(2, 30),
                 edge(3, 40)},
                {edge(1, 10), edge(2, 30), edge(4, 50)}};
            auto extract = [&](std::uint32_t bits) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                run_partitioned<NestedCountSlice>(named, reg, intern, slices,
                                                  bits);
                const OwnedArrow& owned =
                    std::get<OwnedArrow>(named.results().at("nested"));
                const ArrowArray& a = owned.array;
                const std::int64_t* k0 = i64col(a, 0);
                const ArrowArray* listc = a.children[1];
                const auto* loff =
                    static_cast<const std::int32_t*>(listc->buffers[1]) +
                    listc->offset;
                const ArrowArray* st = listc->children[0];
                const std::int64_t* ik = static_cast<const std::int64_t*>(
                                             st->children[0]->buffers[1]) +
                                         st->children[0]->offset;
                const std::int64_t* iv = static_cast<const std::int64_t*>(
                                             st->children[1]->buffers[1]) +
                                         st->children[1]->offset;
                std::vector<std::pair<
                    std::int64_t,
                    std::vector<std::pair<std::int64_t, std::int64_t>>>>
                    rows;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    std::vector<std::pair<std::int64_t, std::int64_t>> inner;
                    for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                        inner.emplace_back(ik[j], iv[j]);
                    rows.emplace_back(k0[row], std::move(inner));
                }
                return rows;
            };
            const auto k1 = extract(0);
            const auto k8 = extract(3);
            CHECK(k1 ==
                  k8);  // exact order (nested always sorts by outer prefix)
            // Outer-key partitioning => each distinct outer key is one whole
            // materialized row at K=8; no outer group is split across
            // partitions.
            std::set<std::int64_t> outer;
            for (const auto& row : k8) outer.insert(row.first);
            CHECK(outer.size() == k8.size());
        }
#endif
    }

    TEST_CASE("BYTES key materializes a binary column, bytes-exact incl NUL") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        // blob_b carries an embedded NUL; a NUL-terminated intern/materialize
        // would truncate it to "x", so getting all three bytes back proves the
        // whole path is length-based and byte-safe.
        const std::string blob_a = "ab";
        const std::string blob_b = std::string("x\0y", 3);
        const std::uint32_t id_a =
            intern.get_or_insert(std::string_view(blob_a));
        const std::uint32_t id_b =
            intern.get_or_insert(std::string_view(blob_b));
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<BytesEdgeSlice>(nullptr),
            intern, &reg, &named);

        // Merged across two slices: blob_a -> 3, blob_b -> 2.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(0, id_a), edge(0, id_a), edge(0, id_b)},
            {edge(0, id_a), edge(0, id_b)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("byte_edges");
        REQUIRE(it != named.results().end());
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 2);
        REQUIRE(a.length == 2);
        // k0 is an Arrow binary column (format "z"), not utf8.
        REQUIRE(owned.schema.children[0]->format != nullptr);
        CHECK(std::string_view(owned.schema.children[0]->format) == "z");

        const ArrowArray* k0c = a.children[0];
        const std::int64_t* val = i64_col(&a, 1);
        std::map<std::string, std::int64_t> got;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            got[bin_at(k0c, row)] = val[row];
        }
        CHECK(got.size() == 2);
        CHECK(got.at(blob_a) == 3);
        CHECK(got.at(blob_b) == 2);
        REQUIRE(got.count(blob_b) == 1);
        // The embedded-NUL blob round-tripped at full length (not truncated).
        CHECK(got.find(blob_b)->first.size() == 3);
#endif
    }

    TEST_CASE("BYTES + int64 two-component key materializes both columns") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        const std::string blob_p = std::string("p\0q", 3);
        const std::string blob_r = "rs";
        const std::uint32_t id_p =
            intern.get_or_insert(std::string_view(blob_p));
        const std::uint32_t id_r =
            intern.get_or_insert(std::string_view(blob_r));
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<MixedBytesSlice>(nullptr),
            intern, &reg, &named);

        // Merged: (1, blob_p) -> 3, (2, blob_r) -> 2.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(1, id_p), edge(1, id_p), edge(2, id_r)},
            {edge(1, id_p), edge(2, id_r)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto it = named.results().find("mixed_bytes");
        REQUIRE(it != named.results().end());
        const OwnedArrow& owned = std::get<OwnedArrow>(it->second);
        const ArrowArray& a = owned.array;
        REQUIRE(a.n_children == 3);
        REQUIRE(a.length == 2);
        CHECK(std::string_view(owned.schema.children[0]->format) ==
              "l");  // i64
        CHECK(std::string_view(owned.schema.children[1]->format) ==
              "z");  // bin

        const std::int64_t* k0 = i64_col(&a, 0);
        const ArrowArray* k1c = a.children[1];
        const std::int64_t* val = i64_col(&a, 2);
        bool saw_a = false, saw_b = false;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            if (k0[row] == 1 && bin_at(k1c, row) == blob_p) {
                CHECK(val[row] == 3);
                saw_a = true;
            } else if (k0[row] == 2 && bin_at(k1c, row) == blob_r) {
                CHECK(val[row] == 2);
                saw_b = true;
            }
        }
        CHECK(saw_a);
        CHECK(saw_b);
#endif
    }

    TEST_CASE("ordered BYTES key sorts rows by raw bytes deterministically") {
        StringIntern intern;
        // Intern out of byte order so id order != byte order: sorting by id
        // would not yield the byte-lexicographic sequence below.
        const std::string b_m = "m";
        const std::string b_a = std::string("a\0", 2);  // {'a','\0'} < "aa"
        const std::string b_aa = "aa";
        const std::string b_z = "z";
        const std::uint32_t id_m = intern.get_or_insert(std::string_view(b_m));
        const std::uint32_t id_a = intern.get_or_insert(std::string_view(b_a));
        const std::uint32_t id_aa =
            intern.get_or_insert(std::string_view(b_aa));
        const std::uint32_t id_z = intern.get_or_insert(std::string_view(b_z));

        const std::vector<FoldEvent> s0 = {edge(0, id_m), edge(0, id_z)};
        const std::vector<FoldEvent> s1 = {edge(0, id_a), edge(0, id_aa)};

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        auto run = [&](const std::vector<std::vector<FoldEvent>>& slices) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<OrderedBytesSlice>(
                    nullptr),
                intern, &reg, &named);
            for (const auto& evs : slices) {
                auto slice = master.fold->slice();
                auto* pf = static_cast<PluginFold*>(slice.get());
                ScanUnit unit;
                FoldBatch fb{std::span<const FoldEvent>(evs), unit};
                pf->step(fb);
                master.fold->merge(*pf);
            }
            finalize_now(*master.fold);
            std::vector<std::string> out;
            const ArrowArray& a =
                std::get<OwnedArrow>(named.results().at("asc_bytes")).array;
            const ArrowArray* k0c = a.children[0];
            for (std::int64_t r = 0; r < a.length; ++r)
                out.push_back(bin_at(k0c, r + a.offset));
            return out;
        };
        // Byte order: {'a','\0'} < "aa" < "m" < "z".
        const std::vector<std::string> exp = {b_a, b_aa, b_m, b_z};
        CHECK(run({s0, s1}) == exp);
        CHECK(run({s1, s0}) == exp);  // merge order does not change row order
#endif
    }

    TEST_CASE("nested map with a BYTES outer key materializes per-blob rows") {
        StringIntern intern;
        const std::string blob_x = std::string("x\0", 2);
        const std::string blob_y = "yy";
        const std::uint32_t id_x =
            intern.get_or_insert(std::string_view(blob_x));
        const std::uint32_t id_y =
            intern.get_or_insert(std::string_view(blob_y));
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<NestedBytesSlice>(nullptr),
            intern, &reg, &named);

        // blob_x: pid1->2, pid2->1; blob_y: pid1->1.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(1, id_x), edge(1, id_x), edge(2, id_x)}, {edge(1, id_y)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        const ArrowArray& a =
            std::get<OwnedArrow>(named.results().at("nested_bytes")).array;
        const OwnedArrow& owned =
            std::get<OwnedArrow>(named.results().at("nested_bytes"));
        REQUIRE(a.n_children == 2);
        REQUIRE(a.length == 2);
        CHECK(std::string_view(owned.schema.children[0]->format) == "z");
        const ArrowArray* k0c = a.children[0];
        const ArrowArray* listc = a.children[1];
        const auto* loff =
            static_cast<const std::int32_t*>(listc->buffers[1]) + listc->offset;
        const ArrowArray* st = listc->children[0];
        const std::int64_t* ik =
            static_cast<const std::int64_t*>(st->children[0]->buffers[1]) +
            st->children[0]->offset;
        const std::int64_t* iv =
            static_cast<const std::int64_t*>(st->children[1]->buffers[1]) +
            st->children[1]->offset;
        std::map<std::string, std::map<std::int64_t, std::int64_t>> got;
        for (std::int64_t r = 0; r < a.length; ++r) {
            const std::int64_t row = r + a.offset;
            std::map<std::int64_t, std::int64_t> inner;
            for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                inner[ik[j]] = iv[j];
            got[bin_at(k0c, row)] = std::move(inner);
        }
        REQUIRE(got.count(blob_x) == 1);
        REQUIRE(got.count(blob_y) == 1);
        CHECK(got.at(blob_x) ==
              std::map<std::int64_t, std::int64_t>{{1, 2}, {2, 1}});
        CHECK(got.at(blob_y) == std::map<std::int64_t, std::int64_t>{{1, 1}});
#endif
    }

    TEST_CASE("spill: BYTES-keyed map spilled result equals in-memory") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        // 400 distinct 8-byte LE blobs across the slice split; a tiny share
        // forces many partition spills.
        const std::size_t SHARE = 256;
        std::vector<FoldEvent> s0, s1;
        for (std::int64_t i = 0; i < 400; ++i) {
            char raw[8];
            std::uint64_t v = static_cast<std::uint64_t>(i);
            std::memcpy(raw, &v, sizeof(raw));
            const std::uint32_t id =
                intern.get_or_insert(std::string_view(raw, sizeof(raw)));
            s0.push_back(edge(0, id));
            s0.push_back(edge(0, id));  // blob i seen 3x total
            s1.push_back(edge(0, id));
        }
        const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

        auto extract = [&](bool spill, std::size_t* spills) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            SpillRoot root;
            if (spill)
                *spills = run_spilled<BytesEdgeSlice>(named, reg, intern,
                                                      slices, SHARE, root.path);
            else
                run_partitioned<BytesEdgeSlice>(named, reg, intern, slices, 0);
            if (spill) CHECK_FALSE(root.has_run_dirs());
            const ArrowArray& a =
                std::get<OwnedArrow>(named.results().at("byte_edges")).array;
            const ArrowArray* k0c = a.children[0];
            const std::int64_t* val = i64_col(&a, 1);
            std::map<std::string, std::int64_t> m;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                m[bin_at(k0c, row)] = val[row];
            }
            return m;
        };
        std::size_t spills = 0;
        const auto mem = extract(false, nullptr);
        const auto spilled = extract(true, &spills);
        CHECK(spills > 0);
        CHECK(spilled == mem);
        CHECK(mem.size() == 400);
#endif
    }

    // Phase 3 spill-invariance: with a forced tiny budget (many spills across
    // partitions), a map materializes IDENTICALLY to the in-memory path, for
    // every value shape. Also proves spilling actually happened (runs > 0) and
    // that the temp dir is cleaned up after finalize.
    TEST_CASE("spill: spilled result equals in-memory for every value shape") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        // A tiny share vs ~56 bytes/key forces spilling once a handful of keys
        // land; the inputs below carry hundreds of keys across the slice split.
        const std::size_t SHARE = 256;

        auto i64col = [](const ArrowArray& a, int c) {
            const ArrowArray* ch = a.children[c];
            return static_cast<const std::int64_t*>(ch->buffers[1]) +
                   ch->offset;
        };
        auto f64col = [](const ArrowArray& a, int c) {
            const ArrowArray* ch = a.children[c];
            return static_cast<const double*>(ch->buffers[1]) + ch->offset;
        };

        SUBCASE("COUNTER map") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(edge(i, 1000 + i));
                s0.push_back(edge(i, 1000 + i));  // pid i seen 3x total
                s1.push_back(edge(i, 1000 + i));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<EdgeSlice>(named, reg, intern, slices,
                                                     SHARE, root.path);
                else
                    run_partitioned<EdgeSlice>(named, reg, intern, slices, 0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("edges")).array;
                std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[{i64col(a, 0)[row], i64col(a, 1)[row]}] =
                        i64col(a, 2)[row];
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);
        }

        SUBCASE("PRODUCT map") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(edge_dur(i, 1000 + i, 10));
                s1.push_back(edge_dur(i, 1000 + i, 10));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<WideEdgeSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<WideEdgeSlice>(named, reg, intern, slices,
                                                   0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("wide_edges"))
                        .array;
                std::map<std::pair<std::int64_t, std::int64_t>,
                         std::pair<std::int64_t, double>>
                    m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[{i64col(a, 0)[row], i64col(a, 1)[row]}] = {
                        i64col(a, 2)[row], f64col(a, 3)[row]};
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);
        }

        SUBCASE("ORDERED map (exact row order)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(edge(i, 0));
                s1.push_back(edge((i * 7) % 400, 0));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<OrderedSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<OrderedSlice>(named, reg, intern, slices,
                                                  0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("asc_i64")).array;
                std::vector<std::pair<std::int64_t, std::int64_t>> rows;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    rows.emplace_back(i64col(a, 0)[row], i64col(a, 1)[row]);
                }
                return rows;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);  // exact ascending row order preserved
        }

        SUBCASE("SET collection map (variable-length state)") {
            StringIntern intern;
            std::vector<std::uint32_t> ids;
            for (int i = 0; i < 8; ++i)
                ids.push_back(intern.get_or_insert("f" + std::to_string(i)));
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(edge(i, ids[i % ids.size()]));
                s1.push_back(edge(i, ids[(i + 1) % ids.size()]));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<FileSetSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<FileSetSlice>(named, reg, intern, slices,
                                                  0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const OwnedArrow& owned =
                    std::get<OwnedArrow>(named.results().at("file_set"));
                const ArrowArray& a = owned.array;
                const ArrowArray* setc = a.children[1];
                const auto* loff =
                    static_cast<const std::int32_t*>(setc->buffers[1]) +
                    setc->offset;
                const ArrowArray* strc = setc->children[0];
                const auto* soff =
                    static_cast<const std::int32_t*>(strc->buffers[1]);
                const char* chars = static_cast<const char*>(strc->buffers[2]);
                std::map<std::int64_t, std::set<std::string>> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    std::set<std::string> s;
                    for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                        s.emplace(chars + soff[j], soff[j + 1] - soff[j]);
                    m.emplace(i64col(a, 0)[row], std::move(s));
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);
        }

        SUBCASE("LIST collection map (ordered variable-length state)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(dur_event(i, /*ts=*/2, 20));
                s0.push_back(dur_event(i, /*ts=*/4, 5));
                s1.push_back(dur_event(i, /*ts=*/1, 8));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto read_seq = [&](NamedResultRegistry& named) {
                const OwnedArrow& owned =
                    std::get<OwnedArrow>(named.results().at("dur_seq"));
                const ArrowArray& a = owned.array;
                const ArrowArray* listc = a.children[1];
                const auto* loff =
                    static_cast<const std::int32_t*>(listc->buffers[1]) +
                    listc->offset;
                const ArrowArray* valc = listc->children[0];
                const std::int64_t* vals =
                    static_cast<const std::int64_t*>(valc->buffers[1]);
                std::map<std::int64_t, std::vector<std::int64_t>> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    std::vector<std::int64_t> v;
                    for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                        v.push_back(vals[j]);
                    m.emplace(i64col(a, 0)[row], std::move(v));
                }
                return m;
            };
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<DurSeqSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<DurSeqSlice>(named, reg, intern, slices, 0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                return read_seq(named);
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);  // ts-order preserved through spill
        }

        SUBCASE("NESTED map (outer group never split)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 200; ++i) {
                s0.push_back(edge(i, 10));
                s0.push_back(edge(i, 10));
                s0.push_back(edge(i, 20));
                s1.push_back(edge(i, 10));
                s1.push_back(edge(i, 30));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<NestedCountSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<NestedCountSlice>(named, reg, intern,
                                                      slices, 0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const OwnedArrow& owned =
                    std::get<OwnedArrow>(named.results().at("nested"));
                const ArrowArray& a = owned.array;
                const ArrowArray* listc = a.children[1];
                const auto* loff =
                    static_cast<const std::int32_t*>(listc->buffers[1]) +
                    listc->offset;
                const ArrowArray* st = listc->children[0];
                const std::int64_t* ik = static_cast<const std::int64_t*>(
                                             st->children[0]->buffers[1]) +
                                         st->children[0]->offset;
                const std::int64_t* iv = static_cast<const std::int64_t*>(
                                             st->children[1]->buffers[1]) +
                                         st->children[1]->offset;
                std::vector<std::pair<
                    std::int64_t,
                    std::vector<std::pair<std::int64_t, std::int64_t>>>>
                    rows;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    std::vector<std::pair<std::int64_t, std::int64_t>> inner;
                    for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                        inner.emplace_back(ik[j], iv[j]);
                    rows.emplace_back(i64col(a, 0)[row], std::move(inner));
                }
                return rows;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);
        }

        SUBCASE("ARGMAX_STR map (payload survives serialize under spill)") {
            StringIntern intern;
            std::vector<std::uint32_t> files;
            for (int i = 0; i < 4; ++i)
                files.push_back(intern.get_or_insert("f" + std::to_string(i)));
            // pid i: durations rise with j so the max-dur file is
            // deterministic.
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(argby_event(i, 10, files[0], 0));
                s0.push_back(argby_event(i, 50, files[2], 0));
                s1.push_back(argby_event(i, 30, files[1], 0));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<ArgMaxFileSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<ArgMaxFileSlice>(named, reg, intern, slices,
                                                     0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("argmax_file"))
                        .array;
                const ArrowArray* vc = a.children[1];
                const auto* offs =
                    static_cast<const std::int32_t*>(vc->buffers[1]) +
                    vc->offset;
                const char* chars = static_cast<const char*>(vc->buffers[2]);
                std::map<std::int64_t, std::string> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[i64col(a, 0)[row]] = std::string(
                        chars + offs[row], offs[row + 1] - offs[row]);
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);
            CHECK(mem.at(0) == "f2");  // dur 50 is the max
        }

        SUBCASE("ARGMAX_ROW map (payload row survives serialize under spill)") {
            StringIntern intern;
            std::vector<std::uint32_t> files;
            for (int i = 0; i < 4; ++i)
                files.push_back(intern.get_or_insert("f" + std::to_string(i)));
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(argrow_event(i, 10, files[0], 1, 1.0));
                s0.push_back(argrow_event(i, 50, files[2], 3, 5.0));  // max dur
                s1.push_back(argrow_event(i, 30, files[1], 2, 3.0));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            using RowT = std::tuple<std::string, std::int32_t, double>;
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<ArgRowSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<ArgRowSlice>(named, reg, intern, slices, 0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("argrow")).array;
                const std::int32_t* p1 = static_cast<const std::int32_t*>(
                                             a.children[2]->buffers[1]) +
                                         a.children[2]->offset;
                const double* p2 = f64col(a, 3);
                std::map<std::int64_t, RowT> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[i64col(a, 0)[row]] =
                        RowT{bin_at(a.children[1], row), p1[row], p2[row]};
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            CHECK(spilled == mem);
            // dur 50 is the max -> the whole row {f2, tid3, 5.0} is kept.
            CHECK(std::get<0>(mem.at(0)) == "f2");
            CHECK(std::get<1>(mem.at(0)) == 3);
            CHECK(std::get<2>(mem.at(0)) == doctest::Approx(5.0));
        }

        SUBCASE("VARIANCE map (FieldStat survives serialize under spill)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(dur_event(i, 0, 2));
                s0.push_back(dur_event(i, 0, 4));
                s1.push_back(dur_event(i, 0, 9));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<DurVarSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<DurVarSlice>(named, reg, intern, slices, 0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("dur_var")).array;
                std::map<std::int64_t, double> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[i64col(a, 0)[row]] = f64col(a, 1)[row];
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            REQUIRE(spilled.size() == mem.size());
            for (const auto& [k, v] : mem)
                CHECK(spilled.at(k) == doctest::Approx(v));
            // {2,4,9}: mean 5, sample variance (9+1+16)/2 = 13.
            CHECK(mem.at(0) == doctest::Approx(13.0));
        }

        SUBCASE("CORR map (co-moment survives serialize under spill)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                // y = 2x per pid -> corr 1 for every key.
                s0.push_back(xy_event(i, 1.0, 2.0));
                s0.push_back(xy_event(i, 2.0, 4.0));
                s1.push_back(xy_event(i, 3.0, 6.0));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<CorrSlice>(named, reg, intern, slices,
                                                     SHARE, root.path);
                else
                    run_partitioned<CorrSlice>(named, reg, intern, slices, 0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("corr")).array;
                std::map<std::int64_t, double> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[i64col(a, 0)[row]] = f64col(a, 1)[row];
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            REQUIRE(spilled.size() == mem.size());
            for (const auto& [k, v] : mem)
                CHECK(spilled.at(k) == doctest::Approx(v));
            CHECK(mem.at(0) == doctest::Approx(1.0));  // y=2x -> corr 1
        }

        SUBCASE(
            "SKEWNESS map (FieldStat moment survives serialize under spill)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(dur_event(i, 0, 2));
                s0.push_back(dur_event(i, 0, 4));
                s0.push_back(dur_event(i, 0, 4));
                s0.push_back(dur_event(i, 0, 4));
                s1.push_back(dur_event(i, 0, 9));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};
            auto extract = [&](bool spill, std::size_t* spills) {
                SharedResultRegistry reg;
                NamedResultRegistry named;
                SpillRoot root;
                if (spill)
                    *spills = run_spilled<HighMomentSlice>(
                        named, reg, intern, slices, SHARE, root.path);
                else
                    run_partitioned<HighMomentSlice>(named, reg, intern, slices,
                                                     0);
                if (spill) CHECK_FALSE(root.has_run_dirs());
                const ArrowArray& a =
                    std::get<OwnedArrow>(named.results().at("hmoments")).array;
                std::map<std::int64_t, double> m;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    m[i64col(a, 0)[row]] = f64col(a, 1)[row];  // skewness
                }
                return m;
            };
            std::size_t spills = 0;
            const auto mem = extract(false, nullptr);
            const auto spilled = extract(true, &spills);
            CHECK(spills > 0);
            REQUIRE(spilled.size() == mem.size());
            for (const auto& [k, v] : mem)
                CHECK(spilled.at(k) == doctest::Approx(v));
        }
#endif
    }

    // The spill temp dir is gone once the fold is destroyed without a finalize
    // (the RAII / exceptional-unwind path), not only after finalize.
    TEST_CASE("spill: temp dir removed on fold destruction without finalize") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        SpillRoot root;
        {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            FoldHolder master(
                dftracer::utils::plugins::make_plugin<EdgeSlice>(nullptr),
                intern, &reg, &named);
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            pf->set_map_spill(256, root.path);
            std::vector<FoldEvent> evs;
            for (std::int64_t i = 0; i < 400; ++i) evs.push_back(edge(i, i));
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            CHECK(pf->map_spill_count() > 0);
            CHECK(
                root.has_run_dirs());  // a run dir exists while the fold lives
            // No merge, no finalize: the slice's destructor must still clean
            // up.
        }
        CHECK_FALSE(root.has_run_dirs());
#endif
    }

    // Phase 4 streamed materialization: with spill forced on (tiny budget,
    // K=256) and streaming enabled, a map materializes as MULTIPLE Arrow
    // batches whose concatenation equals the eager in-memory baseline, for
    // every value shape. Also proves streaming actually happened (>1 batch)
    // and that the unordered path never held more than one partition resident.
    TEST_CASE("stream: multi-batch result equals eager for every value shape") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        const std::size_t SHARE = 256;
        const std::size_t CHUNK = 32;  // small so ordered/nested chunk finely

        SUBCASE("COUNTER map (unordered, per-partition streaming)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(edge(i, 1000 + i));
                s0.push_back(edge(i, 1000 + i));
                s1.push_back(edge(i, 1000 + i));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

            SharedResultRegistry r0;
            NamedResultRegistry n0;
            run_partitioned<EdgeSlice>(n0, r0, intern, slices, 0);
            std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> eager;
            {
                const ArrowArray& a =
                    std::get<OwnedArrow>(n0.results().at("edges")).array;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    eager[{i64_col(&a, 0)[row], i64_col(&a, 1)[row]}] =
                        i64_col(&a, 2)[row];
                }
            }

            SharedResultRegistry r1;
            NamedResultRegistry n1;
            SpillRoot root;
            auto [batches, max_parts] = run_streamed<EdgeSlice>(
                n1, r1, intern, slices, SHARE, root.path, CHUNK);
            CHECK(batches > 1);     // streaming really produced many batches
            CHECK(max_parts == 1);  // never reloaded the whole map at once
            CHECK_FALSE(root.has_run_dirs());

            BatchList bl = result_batches(n1, "edges");
            CHECK(bl.arrays.size() > 1);
            std::map<std::pair<std::int64_t, std::int64_t>, std::int64_t> got;
            for (const ArrowArray* a : bl.arrays)
                for (std::int64_t r = 0; r < a->length; ++r) {
                    const std::int64_t row = r + a->offset;
                    got[{i64_col(a, 0)[row], i64_col(a, 1)[row]}] =
                        i64_col(a, 2)[row];
                }
            CHECK(got == eager);
        }

        SUBCASE("ORDERED map (globally sorted across batches)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(edge(i, 0));
                s1.push_back(edge((i * 7) % 400, 0));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

            SharedResultRegistry r0;
            NamedResultRegistry n0;
            run_partitioned<OrderedSlice>(n0, r0, intern, slices, 0);
            std::vector<std::pair<std::int64_t, std::int64_t>> eager;
            {
                const ArrowArray& a =
                    std::get<OwnedArrow>(n0.results().at("asc_i64")).array;
                for (std::int64_t r = 0; r < a.length; ++r) {
                    const std::int64_t row = r + a.offset;
                    eager.emplace_back(i64_col(&a, 0)[row],
                                       i64_col(&a, 1)[row]);
                }
            }

            SharedResultRegistry r1;
            NamedResultRegistry n1;
            SpillRoot root;
            auto [batches, max_parts] = run_streamed<OrderedSlice>(
                n1, r1, intern, slices, SHARE, root.path, CHUNK);
            (void)max_parts;
            CHECK(batches > 1);
            CHECK_FALSE(root.has_run_dirs());

            BatchList bl = result_batches(n1, "asc_i64");
            CHECK(bl.arrays.size() > 1);
            std::vector<std::pair<std::int64_t, std::int64_t>> got;
            for (const ArrowArray* a : bl.arrays)
                for (std::int64_t r = 0; r < a->length; ++r) {
                    const std::int64_t row = r + a->offset;
                    got.emplace_back(i64_col(a, 0)[row], i64_col(a, 1)[row]);
                }
            CHECK(got == eager);  // exact global order preserved across batches
            for (std::size_t i = 1; i < got.size(); ++i)
                CHECK(got[i - 1].first <= got[i].first);
        }

        SUBCASE("SET_I64 and LIST_I64 (unordered collection values)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 400; ++i) {
                s0.push_back(dur_event(i, /*ts=*/2, 20));
                s0.push_back(dur_event(i, /*ts=*/4, 5));
                s1.push_back(dur_event(i, /*ts=*/1, 8));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

            auto read_list = [](BatchList& bl) {
                std::map<std::int64_t, std::vector<std::int64_t>> out;
                for (const ArrowArray* a : bl.arrays) {
                    const ArrowArray* listc = a->children[1];
                    const auto* loff =
                        static_cast<const std::int32_t*>(listc->buffers[1]) +
                        listc->offset;
                    const std::int64_t* vals = static_cast<const std::int64_t*>(
                        listc->children[0]->buffers[1]);
                    for (std::int64_t r = 0; r < a->length; ++r) {
                        const std::int64_t row = r + a->offset;
                        std::vector<std::int64_t> v;
                        for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                            v.push_back(vals[j]);
                        out.emplace(i64_col(a, 0)[row], std::move(v));
                    }
                }
                return out;
            };

            SharedResultRegistry r0;
            NamedResultRegistry n0;
            run_partitioned<DurSeqSlice>(n0, r0, intern, slices, 0);
            BatchList es = result_batches(n0, "dur_set");
            BatchList eq = result_batches(n0, "dur_seq");
            const auto eager_set = read_list(es);
            const auto eager_seq = read_list(eq);

            SharedResultRegistry r1;
            NamedResultRegistry n1;
            SpillRoot root;
            auto [batches, max_parts] = run_streamed<DurSeqSlice>(
                n1, r1, intern, slices, SHARE, root.path, CHUNK);
            CHECK(batches > 1);
            CHECK(max_parts == 1);
            CHECK_FALSE(root.has_run_dirs());
            BatchList ss = result_batches(n1, "dur_set");
            BatchList sq = result_batches(n1, "dur_seq");
            CHECK(read_list(ss) == eager_set);
            CHECK(read_list(sq) == eager_seq);
        }

        SUBCASE(
            "NESTED map (outer groups sorted across batches, never split)") {
            StringIntern intern;
            std::vector<FoldEvent> s0, s1;
            for (std::int64_t i = 0; i < 200; ++i) {
                s0.push_back(edge(i, 10));
                s0.push_back(edge(i, 10));
                s0.push_back(edge(i, 20));
                s1.push_back(edge(i, 10));
                s1.push_back(edge(i, 30));
            }
            const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

            auto read_nested = [](BatchList& bl) {
                std::vector<std::pair<
                    std::int64_t,
                    std::vector<std::pair<std::int64_t, std::int64_t>>>>
                    rows;
                for (const ArrowArray* a : bl.arrays) {
                    const ArrowArray* listc = a->children[1];
                    const auto* loff =
                        static_cast<const std::int32_t*>(listc->buffers[1]) +
                        listc->offset;
                    const ArrowArray* st = listc->children[0];
                    const std::int64_t* ik = static_cast<const std::int64_t*>(
                                                 st->children[0]->buffers[1]) +
                                             st->children[0]->offset;
                    const std::int64_t* iv = static_cast<const std::int64_t*>(
                                                 st->children[1]->buffers[1]) +
                                             st->children[1]->offset;
                    for (std::int64_t r = 0; r < a->length; ++r) {
                        const std::int64_t row = r + a->offset;
                        std::vector<std::pair<std::int64_t, std::int64_t>>
                            inner;
                        for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                            inner.emplace_back(ik[j], iv[j]);
                        rows.emplace_back(i64_col(a, 0)[row], std::move(inner));
                    }
                }
                return rows;
            };

            SharedResultRegistry r0;
            NamedResultRegistry n0;
            run_partitioned<NestedCountSlice>(n0, r0, intern, slices, 0);
            BatchList eb = result_batches(n0, "nested");
            const auto eager = read_nested(eb);

            SharedResultRegistry r1;
            NamedResultRegistry n1;
            SpillRoot root;
            auto [batches, max_parts] = run_streamed<NestedCountSlice>(
                n1, r1, intern, slices, SHARE, root.path, CHUNK);
            (void)max_parts;
            CHECK(batches > 1);
            CHECK_FALSE(root.has_run_dirs());
            BatchList sb = result_batches(n1, "nested");
            CHECK(sb.arrays.size() > 1);
            const auto got = read_nested(sb);
            CHECK(got == eager);  // exact outer order + merged inner lists
            for (std::size_t i = 1; i < got.size(); ++i)
                CHECK(got[i - 1].first < got[i].first);  // no split outer group
        }
#endif
    }

    TEST_CASE("spill: TOPK_STR map spilled result equals in-memory") {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
        StringIntern intern;
        // 400 pids, each touching 4 distinct files at distinct durations (>k=3
        // so bounding runs); a tiny share forces many partition spills. Proves
        // the bounded top-k state survives serialize under spill.
        const std::size_t SHARE = 256;
        std::vector<std::uint32_t> ids;
        for (int i = 0; i < 8; ++i)
            ids.push_back(intern.get_or_insert("f" + std::to_string(i)));
        std::vector<FoldEvent> s0, s1;
        for (std::int64_t i = 0; i < 400; ++i) {
            s0.push_back(argby_event(i, 10, ids[i % 8], 0));
            s0.push_back(argby_event(i, 20, ids[(i + 1) % 8], 0));
            s1.push_back(argby_event(i, 30, ids[(i + 2) % 8], 0));
            s1.push_back(argby_event(i, 5, ids[(i + 3) % 8], 0));
        }
        const std::vector<std::vector<FoldEvent>> slices = {s0, s1};

        auto extract = [&](bool spill, std::size_t* spills) {
            SharedResultRegistry reg;
            NamedResultRegistry named;
            SpillRoot root;
            if (spill)
                *spills = run_spilled<TopKFileSlice>(named, reg, intern, slices,
                                                     SHARE, root.path);
            else
                run_partitioned<TopKFileSlice>(named, reg, intern, slices, 0);
            if (spill) CHECK_FALSE(root.has_run_dirs());
            const OwnedArrow& owned =
                std::get<OwnedArrow>(named.results().at("topk_file"));
            const ArrowArray& a = owned.array;
            const ArrowArray* listc = a.children[1];
            const auto* loff =
                static_cast<const std::int32_t*>(listc->buffers[1]) +
                listc->offset;
            const ArrowArray* strc = listc->children[0];
            const auto* soff =
                static_cast<const std::int32_t*>(strc->buffers[1]);
            const char* chars = static_cast<const char*>(strc->buffers[2]);
            const std::int64_t* k0 =
                static_cast<const std::int64_t*>(a.children[0]->buffers[1]) +
                a.children[0]->offset;
            std::map<std::int64_t, std::vector<std::string>> m;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                std::vector<std::string> v;  // by-order matters: keep it
                for (std::int32_t j = loff[row]; j < loff[row + 1]; ++j)
                    v.emplace_back(chars + soff[j], soff[j + 1] - soff[j]);
                m.emplace(k0[row], std::move(v));
            }
            return m;
        };

        std::size_t spills = 0;
        const auto mem = extract(false, nullptr);
        const auto spilled = extract(true, &spills);
        CHECK(spills > 0);
        CHECK(spilled == mem);  // bounded top-k order preserved through spill
#endif
    }

    TEST_CASE("declared join emits an additive joined result; LEFT null-pads") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<JoinSlice>(nullptr), intern,
            &reg, &named);

        // Merged: counts pid1->2, pid2->1, pid3->1; durs pid1->40, pid2->20,
        // pid3 absent (its event carried no duration).
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge_dur(1, 0, 10), edge_dur(2, 0, 20), edge(3, 0)},
            {edge_dur(1, 0, 30)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

#ifdef DFTRACER_UTILS_ENABLE_ARROW
        // Input maps still materialize on their own (the join is additive).
        REQUIRE(named.results().find("jl_counts") != named.results().end());
        REQUIRE(named.results().find("jr_durs") != named.results().end());

        auto is_null = [](const ArrowArray* ch, std::int64_t row) {
            const auto* v = static_cast<const std::uint8_t*>(ch->buffers[0]);
            if (!v) return false;
            const std::int64_t idx = ch->offset + row;
            return ((v[idx / 8] >> (idx % 8)) & 1) == 0;
        };

        // LEFT: every left pid, right side null on the left-only pid 3.
        {
            auto it = named.results().find("join_left");
            REQUIRE(it != named.results().end());
            REQUIRE(std::holds_alternative<OwnedArrow>(it->second));
            const ArrowArray& a = std::get<OwnedArrow>(it->second).array;
            REQUIRE(a.n_children == 3);
            REQUIRE(a.length == 3);
            const std::int64_t* k0 = i64_col(&a, 0);
            const std::int64_t* cnt = i64_col(&a, 1);
            const auto* durc = a.children[2];
            const double* dur =
                static_cast<const double*>(durc->buffers[1]) + durc->offset;
            std::map<std::int64_t, std::int64_t> counts;
            for (std::int64_t r = 0; r < a.length; ++r) {
                const std::int64_t row = r + a.offset;
                counts.emplace(k0[row], cnt[row]);
                if (k0[row] == 1) {
                    CHECK_FALSE(is_null(durc, row));
                    CHECK(dur[row] == doctest::Approx(40.0));
                } else if (k0[row] == 2) {
                    CHECK_FALSE(is_null(durc, row));
                    CHECK(dur[row] == doctest::Approx(20.0));
                } else if (k0[row] == 3) {
                    CHECK(is_null(durc, row));
                }
            }
            CHECK(counts[1] == 2);
            CHECK(counts[2] == 1);
            CHECK(counts[3] == 1);
        }

        // INNER: only the matched pids 1 and 2.
        {
            auto it = named.results().find("join_inner");
            REQUIRE(it != named.results().end());
            const ArrowArray& a = std::get<OwnedArrow>(it->second).array;
            REQUIRE(a.n_children == 3);
            REQUIRE(a.length == 2);
            const std::int64_t* k0 = i64_col(&a, 0);
            std::set<std::int64_t> keys;
            for (std::int64_t r = 0; r < a.length; ++r)
                keys.insert(k0[r + a.offset]);
            CHECK(keys == std::set<std::int64_t>{1, 2});
        }
#endif
    }

    TEST_CASE("FinalizedMap reads a merged counter map at finalize") {
        StringIntern intern;
        SharedResultRegistry reg;
        NamedResultRegistry named;
        FoldHolder master(
            dftracer::utils::plugins::make_plugin<ReadCounterSlice>(nullptr),
            intern, &reg, &named);

        // Merged: pid 1 -> 3, pid 2 -> 2.
        const std::vector<std::vector<FoldEvent>> slices = {
            {edge(1, 10), edge(1, 10), edge(2, 20)},
            {edge(1, 10), edge(2, 20)}};
        for (const auto& evs : slices) {
            auto slice = master.fold->slice();
            auto* pf = static_cast<PluginFold*>(slice.get());
            ScanUnit unit;
            FoldBatch fb{std::span<const FoldEvent>(evs), unit};
            pf->step(fb);
            master.fold->merge(*pf);
        }
        finalize_now(*master.fold);

        const ReadCounterSlice::Result& r = ReadCounterSlice::result;
        REQUIRE(r.ran);
        CHECK(r.size == 2);
        REQUIRE(r.v1.has_value());
        CHECK(*r.v1 == 3);
        REQUIRE(r.v2.has_value());
        CHECK(*r.v2 == 2);
        CHECK(r.missing_key_absent);
        CHECK(r.monoid_matches_get);
        REQUIRE(r.iterated.size() == 2);
        CHECK(r.iterated.at(1) == 3);
        CHECK(r.iterated.at(2) == 2);
    }
}
