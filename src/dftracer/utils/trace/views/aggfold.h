#ifndef DFTRACER_UTILS_TRACE_VIEWS_AGGFOLD_H
#define DFTRACER_UTILS_TRACE_VIEWS_AGGFOLD_H

#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/query/query.h>
#include <dftracer/utils/trace/views/agg_fold.h>
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/trace/views/view_spill.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <functional>
#include <memory>
#include <queue>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace dftracer::utils::trace::views::detail {

/// Fields a bootstrap AggFold can evaluate a predicate on straight from the
/// POD: always-present top-level scalars on a phase-filtered data event. A
/// query touching anything else (args, te, fhash/hhash, resolved.*) cannot be
/// filtered here and must take the indexed path.
inline bool fold_query_field_supported(std::string_view f) {
    return f == "cat" || f == "name" || f == "pid" || f == "tid" || f == "ts" ||
           f == "dur";
}

/// True when every field a query references is POD-evaluable, so the raw-gzip
/// bootstrap can apply the predicate itself instead of deferring to the scan.
inline bool query_evaluable_by_fold(const query::Query& q) {
    for (std::string_view f : q.fields())
        if (!fold_query_field_supported(f)) return false;
    return true;
}

/// The aggregation as a fold on the fused scan. A returning fold: it persists
/// nothing, and the caller pulls result() after fuse. Reads each event through
/// PodSource, so its groups and metrics equal the scan-path fold by
/// construction. `intern` must be the table fuse extracted the events with, and
/// `plan.schema` must be built (ensure_schema) before the scan.
///
/// With `plan.memory_budget` set, the group map spills to sorted runs past the
/// budget and result() k-way merges them, so a high-cardinality aggregation
/// stays bounded. The map keys resolved strings, so spill needs no intern.
class AggFold : public Fold {
   public:
    /// `apply_query` makes this fold evaluate `plan.query` per event (the
    /// raw-gzip bootstrap path, where no scanner pre-filtered). Leave false on
    /// the indexed path: the scanner already applied the predicate there, and
    /// the query must be POD-evaluable (query_evaluable_by_fold).
    AggFold(const ViewPlan& plan, const dftracer::utils::StringIntern& intern,
            bool apply_query = false)
        : plan_(&plan),
          intern_(&intern),
          budget_(plan.memory_budget),
          phase_target_(phase_target(plan)),
          apply_query_(apply_query && plan.query.has_value()) {}

    ~AggFold() override { remove_runs(); }

    bool accepts(const ScanShape&) const override { return true; }

    // Only when the query reads something the POD does not already carry
    // without args. The scalars (cat/name/pid/tid/ts/dur), fhash/hhash, derived
    // te, and Count() need no args; size (ret/size_sum), a grouped arg, or any
    // other field does. Marking too few is a silent wrong answer, so err toward
    // true for anything unrecognized.
    bool needs_args() const override {
        if (plan_->auto_numeric_metrics) return true;
        for (const auto& gk : plan_->group_by)
            if (gk.kind == GroupKey::Kind::Arg) return true;
        for (const auto& spec : plan_->agg) {
            if (!arg_free_field(spec.field)) return true;
            if (spec.op == AggOp::ArgMax && !arg_free_field(spec.by))
                return true;
        }
        return false;
    }

    std::unique_ptr<Fold> slice() const override {
        return std::make_unique<AggFold>(*plan_, *intern_, apply_query_);
    }

    void step(const FoldBatch& batch) override {
        for (const auto& ev : batch.events) {
            // Aggregations never fold ph="M" metadata (it carries no aggregable
            // event), and Events/Counters plans fold only their phase. The
            // indexed scanner already applies both (metadata dropped by the
            // aggregation vdef, phase by the query), so this is a no-op there;
            // on the raw-gzip path the fold is fed every phase and this is what
            // keeps metadata and other phases out of the aggregation.
            if (ev.phase == RecordPhase::METADATA) continue;
            if (phase_target_ != RecordPhase::UNKNOWN &&
                ev.phase != phase_target_)
                continue;
            PodSource src(ev, *intern_);
            // On the bootstrap path the fold is fed every event, so apply the
            // user predicate here; the indexed scanner already did this, hence
            // the flag is off there. Only POD-evaluable fields reach this path
            // (query_evaluable_by_fold gates the bootstrap).
            if (apply_query_ && !passes_query(src)) continue;
            fold_event_over(map_, src, *plan_, keybuf_);
        }
        maybe_spill();
    }

    // A returning fold's result is whatever was folded; a cut-short scan yields
    // a partial aggregate, as the scan-path fold does on cancel.
    void seal_unit(const ScanUnit&) override {}
    void drop_unit(const ScanUnit&) override {}

    void merge(Fold& other) override {
        auto& o = static_cast<AggFold&>(other);
        for (auto& [k, a] : o.map_) {
            auto it = map_.find(k);
            if (it == map_.end())
                map_.emplace(k, std::move(a));
            else
                merge_accum(it->second, a, *plan_);
        }
        o.map_.clear();
        // Adopt its runs and the dirs they live in, so this fold cleans them up
        // and the slice's destructor does not.
        for (auto& r : o.runs_) runs_.push_back(std::move(r));
        for (auto& d : o.dirs_) dirs_.push_back(std::move(d));
        o.runs_.clear();
        o.dirs_.clear();
        o.cur_dir_.clear();
        maybe_spill();
    }

    coro::CoroTask<bool> finalize(const CoverageSet&) override {
        co_return true;
    }

    // The folded groups, spill merged in, before name resolution. The caller
    // may merge these with partials from another source before resolving.
    // Materializes the full result; use for_each_sorted_group to stream.
    GroupMap finish_map() {
        if (runs_.empty()) return std::move(map_);
        GroupMap merged;
        for_each_sorted_group([&](const std::string& k, const AggAccum& a) {
            merged.emplace(k, a);
        });
        return merged;
    }

    // Emit each folded group once, in ascending key order, without ever holding
    // the whole result: spilled runs are k-way merged and an unspilled map is
    // sorted in place. A persist consumer routes this sorted stream to
    // per-shard SST writers, so a huge aggregation never has to fit in memory.
    void for_each_sorted_group(
        const std::function<void(const std::string&, const AggAccum&)>& cb) {
        if (runs_.empty()) {
            std::vector<const GroupMap::value_type*> ents;
            ents.reserve(map_.size());
            for (const auto& kv : map_) ents.push_back(&kv);
            std::sort(ents.begin(), ents.end(),
                      [](auto* a, auto* b) { return a->first < b->first; });
            for (auto* e : ents) cb(e->first, e->second);
            return;
        }
        spill_map();  // fold the in-memory residual in as one more run
        std::vector<std::unique_ptr<RunReader>> readers;
        readers.reserve(runs_.size());
        using Item = std::pair<std::string, std::size_t>;  // (key, reader idx)
        std::priority_queue<Item, std::vector<Item>, std::greater<>> heap;
        for (std::size_t i = 0; i < runs_.size(); ++i) {
            readers.push_back(std::make_unique<RunReader>(runs_[i]));
            if (readers[i]->valid) heap.emplace(readers[i]->key, i);
        }
        while (!heap.empty()) {
            std::string key = heap.top().first;
            AggAccum acc;
            while (!heap.empty() && heap.top().first == key) {
                std::size_t idx = heap.top().second;
                heap.pop();
                merge_accum(acc, readers[idx]->accum, *plan_);
                readers[idx]->advance();
                if (readers[idx]->valid) heap.emplace(readers[idx]->key, idx);
            }
            cb(key, acc);
        }
    }

    dftracer::utils::dataframe::DataFrame result() {
        GroupMap m = finish_map();
        resolve_group_keys(m, *plan_);
        return to_batch(m, *plan_);
    }

   private:
    // Fields the POD serves without capturing args.
    static bool arg_free_field(const std::string& f) {
        return f.empty() || f == "ts" || f == "dur" || f == "te" ||
               f == "cat" || f == "name" || f == "pid" || f == "tid" ||
               f == "fhash" || f == "hhash";
    }

    // Build a ValueMap for the query's referenced fields from the POD and
    // evaluate. Only fold_query_field_supported fields occur here; cat/name go
    // in as strings (needed by like/regex), the rest as numbers. qmap_ is
    // reused across events; keys are views into the query's stable field set.
    bool passes_query(const PodSource& src) {
        qmap_.clear();
        for (std::string_view f : plan_->query->fields()) {
            if (f == "cat" || f == "name") {
                qmap_[f] = src.value(f);
            } else if (auto n = src.number(f)) {
                qmap_[f] = *n;  // dur missing -> unset -> predicate is false
            }
        }
        return plan_->query->evaluate(qmap_);
    }

    void maybe_spill() {
        if (budget_ > 0 && approx_bytes(map_) > budget_) spill_map();
    }

    void spill_map() {
        if (map_.empty()) return;
        if (cur_dir_.empty()) {
            static std::atomic<std::uint64_t> seq{0};
            cur_dir_ = (fs::temp_directory_path() /
                        ("dftaggfold_" + std::to_string(seq.fetch_add(1))))
                           .string();
            fs::create_directories(cur_dir_);
            dirs_.push_back(cur_dir_);
        }
        std::string path = cur_dir_ + "/r" + std::to_string(own_seq_++);
        spill_run(map_, path);  // clears map_
        runs_.push_back(std::move(path));
    }

    void remove_runs() {
        std::error_code ec;
        for (const auto& d : dirs_) fs::remove_all(d, ec);
        dirs_.clear();
        runs_.clear();
        cur_dir_.clear();
    }

    const ViewPlan* plan_;
    const dftracer::utils::StringIntern* intern_;
    std::uint64_t budget_;
    RecordPhase phase_target_;
    bool apply_query_ = false;
    GroupMap map_;
    std::string keybuf_;
    query::ValueMap qmap_;  // reused per-event predicate scratch

    static RecordPhase phase_target(const ViewPlan& plan) {
        if (plan.phase == Phase::Events) return RecordPhase::COMPLETE;
        if (plan.phase == Phase::Counters) return RecordPhase::COUNTER;
        return RecordPhase::UNKNOWN;  // Any: aggregate every phase, no filter
    }
    std::vector<std::string> runs_;   // sorted spill runs, own + adopted
    std::vector<std::string> dirs_;   // dirs to remove (own + adopted)
    std::string cur_dir_;             // this fold's own spill dir
    std::uint64_t own_seq_ = 0;       // names this fold's own runs
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_AGGFOLD_H
