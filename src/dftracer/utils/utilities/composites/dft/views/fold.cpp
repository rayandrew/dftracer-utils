#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/utilities/composites/dft/views/fold.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scanner_utility.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <utility>

namespace dftracer::utils::utilities::composites::dft::views::detail {

namespace {

std::uint32_t intern_string(dftracer::utils::StringIntern& intern,
                            simdjson::dom::element val) {
    if (!val.is_string()) return dftracer::utils::StringIntern::NO_ID;
    return intern.get_or_insert(val.get_string().value_unsafe());
}

}  // namespace

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
                // Exact as int64 when it fits; only a value above INT64_MAX
                // falls back to double (rare, and no worse than before).
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
                             bool needs_args) {
    DFTracerEvent scalars;
    simdjson::dom::element args;
    bool has_args = false;
    if (!DFTracerEvent::parse_scalars(root, scalars, args, has_args))
        return FoldEvent{};
    return build_fold_event(scalars, args, has_args, intern, needs_args);
}

coro::CoroTask<ExportStats> fuse(const ViewPlan& plan,
                                 const ViewDefinition& vdef,
                                 std::span<Fold* const> folds,
                                 dftracer::utils::StringIntern& intern,
                                 const CoverageSet* covered) {
    std::uint64_t skipped = 0;
    auto units = co_await gather_units(plan, vdef, skipped);

    ExportStats st;
    st.chunks_skipped = skipped;
    st.chunks_scanned = units.size();
    if (units.empty() || folds.empty()) {
        for (auto* f : folds) co_await f->finalize(CoverageSet{});
        co_return st;
    }

    bool any_needs_args = false;
    for (auto* f : folds) any_needs_args |= f->needs_args();

    const std::size_t nworkers = std::min<std::size_t>(units.size(), 16);
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
                PendingCoverage pending(covered_v[w]);

                for (;;) {
                    if (is_cancelled(plan)) break;
                    std::size_t i =
                        next_unit.fetch_add(1, std::memory_order_relaxed);
                    if (i >= units.size()) break;
                    // A materialized aggregate already answered this chunk.
                    if (covered && covered->covers(units[i].file_path,
                                                   units[i].checkpoint_idx))
                        continue;

                    // Fold mode: the scanner parses each event once and hands
                    // back owned FoldEvents, so nothing here re-parses.
                    ViewScannerInput sin =
                        make_scanner_input(units[i], vdef, vdef.query);
                    sin.fold_intern = &intern;
                    sin.fold_needs_args = any_needs_args;
                    ViewScannerUtility scanner;
                    auto gen = scanner.process(sin);
                    bool complete = true;
                    while (auto b = co_await gen.next()) {
                        if (is_cancelled(plan)) {
                            complete = false;
                            break;
                        }
                        matched_v[w] += b->events_matched;
                        scanned_v[w] += b->events_scanned;
                        if (b->fold_events.empty()) continue;
                        FoldBatch fb{std::span<const FoldEvent>(b->fold_events),
                                     units[i]};
                        for (auto& s : slices) s->step(fb);
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
    co_return st;
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
