#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/common/platform_compat.h>
#include <dftracer/utils/core/coro/async_semaphore.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/view_scanner_utility.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

namespace dftracer::utils::trace::views::detail {

namespace {

struct ScanPermit {
    coro::CoroSemaphore& sem;
    std::uint64_t bytes;
    ScanPermit(coro::CoroSemaphore& s, std::uint64_t n) : sem(s), bytes(n) {}
    ~ScanPermit() { sem.release(bytes); }
    ScanPermit(const ScanPermit&) = delete;
    ScanPermit& operator=(const ScanPermit&) = delete;
};

std::uint32_t intern_string(dftracer::utils::StringIntern& intern,
                            simdjson::dom::element val) {
    if (!val.is_string()) return dftracer::utils::StringIntern::NO_ID;
    return intern.get_or_insert(val.get_string().value_unsafe());
}

}  // namespace

void FoldPortBus::publish(std::uint64_t key, const void* data,
                          std::uint32_t len) {
    const std::size_t offset = arena_.size();
    const auto* p = static_cast<const std::byte*>(data);
    arena_.insert(arena_.end(), p, p + len);
    index_[key] = Entry{offset, len};
}

const void* FoldPortBus::consume(std::uint64_t key,
                                 std::uint32_t* out_len) const {
    auto it = index_.find(key);
    if (it == index_.end()) {
        if (out_len) *out_len = 0;
        return nullptr;
    }
    if (out_len) *out_len = it->second.len;
    return arena_.data() + it->second.offset;
}

void FoldPortBus::clear() {
    index_.clear();
    arena_.clear();
}

FoldEvent build_fold_event(const DFTracerEvent& scalars,
                           simdjson::dom::element args, bool has_args,
                           dftracer::utils::StringIntern& intern,
                           bool needs_args) {
    FoldEvent ev;
    ev.phase = scalars.phase;
    ev.pid = scalars.pid;
    ev.tid = scalars.tid;
    ev.ts = scalars.ts;
    ev.dur = scalars.dur;
    ev.has_dur = scalars.has_dur;
    if (!scalars.cat.empty()) ev.cat_id = intern.get_or_insert(scalars.cat);
    if (!scalars.name.empty()) ev.name_id = intern.get_or_insert(scalars.name);

    if (!has_args) return ev;

    // fhash/hhash are group dimensions even for folds that want no other args,
    // so they are always harvested; the rest only when a fold asks.
    for (auto field : args.get_object()) {
        std::string_view key = field.key;
        if (key == "fhash") {
            ev.fhash_id = intern_string(intern, field.value);
        } else if (key == "hhash") {
            ev.hhash_id = intern_string(intern, field.value);
        } else if (needs_args) {
            simdjson::dom::element v = field.value;
            const std::uint32_t key_id = intern.get_or_insert(key);
            if (v.is_string()) {
                ev.args.emplace_back(key_id,
                                     intern.get_or_insert(v.get_string()));
            } else if (v.is_int64()) {
                ev.args.emplace_back(key_id, static_cast<std::int64_t>(
                                                 v.get_int64().value_unsafe()));
            } else if (v.is_uint64()) {
                // Exact as int64 when it fits; a value above INT64_MAX falls
                // back to double.
                const std::uint64_t u = v.get_uint64().value_unsafe();
                if (u <= static_cast<std::uint64_t>(
                             std::numeric_limits<std::int64_t>::max()))
                    ev.args.emplace_back(key_id, static_cast<std::int64_t>(u));
                else
                    ev.args.emplace_back(key_id, static_cast<double>(u));
            } else if (v.is_double()) {
                ev.args.emplace_back(key_id, v.get_double().value_unsafe());
            }
        }
    }
    return ev;
}

FoldEvent extract_fold_event(simdjson::dom::element root,
                             dftracer::utils::StringIntern& intern,
                             bool needs_args,
                             const std::vector<std::string>* extra_fields) {
    DFTracerEvent scalars;
    simdjson::dom::element args;
    bool has_args = false;
    if (!DFTracerEvent::parse_scalars(root, scalars, args, has_args))
        return FoldEvent{};
    FoldEvent ev =
        build_fold_event(scalars, args, has_args, intern, needs_args);
    if (extra_fields)
        for (const auto& name : *extra_fields)
            capture_extra_field(ev, root, intern, name);
    return ev;
}

std::vector<std::string> extra_capture_fields(const ViewPlan& plan) {
    // Flat args come from needs_args; only top-level fields the POD does not
    // carry (anything but the six scalars) and nested paths need capture.
    auto is_pod_scalar = [](const std::string& f) {
        return f == "name" || f == "cat" || f == "pid" || f == "tid" ||
               f == "ts" || f == "dur";
    };
    std::vector<std::string> out;
    auto add = [&](const std::string& f) {
        if (f.empty()) return;
        for (const auto& e : out)
            if (e == f) return;
        out.push_back(f);
    };
    for (const auto& gk : plan.group_by)
        if (gk.kind == GroupKey::Kind::Field && !is_pod_scalar(gk.arg))
            add(gk.arg);
    for (const auto& spec : plan.agg) {
        if (is_nested_path(spec.field)) add(spec.field);
        if (spec.op == AggOp::ArgMax && is_nested_path(spec.by)) add(spec.by);
    }
    return out;
}

coro::CoroTask<ExportStats> fuse(const ViewPlan& plan,
                                 const ViewDefinition& vdef,
                                 std::span<Fold* const> folds,
                                 dftracer::utils::StringIntern& intern,
                                 const CoverageSet* covered,
                                 std::uint64_t limit) {
    std::uint64_t skipped = 0;
    auto units = co_await gather_units(plan, vdef, skipped);
    const std::uint64_t cap =
        limit > 0 ? limit : std::numeric_limits<std::uint64_t>::max();
    std::atomic<std::uint64_t> produced{0};

    ExportStats st;
    st.chunks_skipped = skipped;
    st.chunks_scanned = units.size();
    if (units.empty() || folds.empty()) {
        for (auto* f : folds) co_await f->finalize(CoverageSet{});
        co_return st;
    }

    bool any_needs_args = false;
    bool any_wants_raw = false;
    bool any_wants_fold_event = false;
    for (auto* f : folds) {
        any_needs_args |= f->needs_args();
        if (f->wants_raw())
            any_wants_raw = true;
        else
            any_wants_fold_event = true;
    }

    std::vector<std::string> extra_fields = extra_capture_fields(plan);

    // Coarse fan-out: one worker coroutine per runtime slot draining the shared
    // unit queue, so under the elastic runtime live threads grow toward the
    // cap.
    const std::size_t nworkers =
        std::min<std::size_t>(units.size(), available_parallelism());
    // Bound bytes decoded concurrently so a scan cannot OOM the box. Always on:
    // an explicit memory_budget() or a RAM-fraction default. Workers park when
    // full, so a generous budget never throttles.
    coro::CoroSemaphore budget_sem(compute_memory_budget(plan.memory_budget));
    std::atomic<std::size_t> next_unit{0};
    std::vector<std::uint64_t> matched_v(nworkers, 0), scanned_v(nworkers, 0);
    std::vector<CoverageSet> covered_v(nworkers);
    // Per-worker slice of every fold, owned here so a slice outlives the
    // coroutine that filled it; merged into the shared folds after the join.
    std::vector<std::vector<std::unique_ptr<Fold>>> fslice(nworkers);
    for (std::size_t w = 0; w < nworkers; ++w)
        for (auto* f : folds) fslice[w].push_back(f->slice());

    co_await run_coro_scope([&](CoroScope& scope) -> coro::CoroTask<void> {
        for (std::size_t w = 0; w < nworkers; ++w) {
            scope.spawn([&, w](CoroScope&) -> coro::CoroTask<void> {
                auto& slices = fslice[w];
                FoldPortBus bus;
                for (auto& s : slices) s->bind_port_bus(&bus);
                PendingCoverage pending(covered_v[w]);

                for (;;) {
                    if (is_cancelled(plan)) break;
                    if (produced.load(std::memory_order_relaxed) >= cap) break;
                    std::size_t i =
                        next_unit.fetch_add(1, std::memory_order_relaxed);
                    if (i >= units.size()) break;
                    // A materialized aggregate already answered this chunk.
                    if (covered && covered->covers(units[i].file_path,
                                                   units[i].checkpoint_idx))
                        continue;

                    const std::uint64_t reserve =
                        units[i].end_byte > units[i].start_byte
                            ? units[i].end_byte - units[i].start_byte
                            : 0;
                    co_await budget_sem.acquire(reserve);
                    ScanPermit permit{budget_sem, reserve};

                    ViewScannerInput sin =
                        make_scanner_input(units[i], vdef, vdef.query);
                    // A FoldEvent fold gets the parsed stream; a raw fold
                    // parses the lines itself. When both are present, keep
                    // both.
                    sin.fold_intern = any_wants_fold_event ? &intern : nullptr;
                    sin.fold_needs_args = any_needs_args;
                    sin.fold_keep_raw = any_wants_raw && any_wants_fold_event;
                    if (!extra_fields.empty())
                        sin.fold_extra_fields = &extra_fields;
                    ViewScannerUtility scanner;
                    auto gen = scanner(sin);
                    bool complete = true;
                    while (auto b = co_await gen.next()) {
                        if (is_cancelled(plan)) {
                            complete = false;
                            break;
                        }
                        if (produced.load(std::memory_order_relaxed) >= cap) {
                            complete = false;
                            break;
                        }
                        matched_v[w] += b->events_matched;
                        scanned_v[w] += b->events_scanned;
                        if (b->fold_events.empty() && b->events.empty())
                            continue;
                        produced.fetch_add(
                            b->fold_events.size() + b->events.size(),
                            std::memory_order_relaxed);
                        FoldBatch fb{
                            std::span<const FoldEvent>(b->fold_events),
                            units[i],
                            std::span<const std::string_view>(b->events)};
                        bus.clear();
                        for (auto& s : slices) {
                            s->step(fb);
                            // Await an async plugin fold's task before the next
                            // step recycles the batch; null for sync folds.
                            if (auto* t = s->take_pending())
                                co_await *reinterpret_cast<
                                    coro::CoroTask<void>*>(t);
                        }
                    }
                    if (!complete) {
                        for (auto& s : slices) s->drop_unit(units[i]);
                        break;
                    }
                    for (auto& s : slices) s->seal_unit(units[i]);
                    pending.mark_complete(units[i].file_path,
                                          units[i].checkpoint_idx);
                    pending.seal();
                }
                co_return;
            });
        }
        co_return;
    });

    for (std::size_t w = 0; w < nworkers; ++w)
        for (std::size_t k = 0; k < folds.size(); ++k)
            folds[k]->merge(*fslice[w][k]);

    CoverageSet scanned;
    for (auto& c : covered_v) scanned.absorb(std::move(c));

    // A file is whole only if nothing was pruned and every unit of it sealed.
    if (skipped == 0) {
        std::unordered_map<std::string_view,
                           std::pair<std::size_t, std::size_t>>
            per_file;
        for (const auto& u : units) {
            auto& c = per_file[u.file_path];
            ++c.first;
            if (scanned.covers(u.file_path, u.checkpoint_idx)) ++c.second;
        }
        for (const auto& [file, c] : per_file)
            if (c.first == c.second) scanned.add_file(file);
    }

    for (auto* f : folds) co_await f->finalize(scanned);

    for (auto m : matched_v) st.events_matched += m;
    for (auto s : scanned_v) st.events_scanned += s;
    st.truncated = produced.load(std::memory_order_relaxed) >= cap;
    co_return st;
}

}  // namespace dftracer::utils::trace::views::detail
