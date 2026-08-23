#ifndef DFTRACER_UTILS_TRACE_VIEWS_AGGREGATION_FOLD_H
#define DFTRACER_UTILS_TRACE_VIEWS_AGGREGATION_FOLD_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/trace/aggregators/aggregation_config.h>
#include <dftracer/utils/trace/aggregators/aggregation_core.h>
#include <dftracer/utils/trace/aggregators/aggregation_intern.h>
#include <dftracer/utils/trace/aggregators/association_tracker.h>
#include <dftracer/utils/trace/event.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <unordered_set>
#include <variant>

namespace dftracer::utils::utilities::indexer {
class IndexBatchSink;
}

namespace dftracer::utils::trace::views::detail {

/// Reads the aggregation core's Accessor concept off an owned FoldEvent,
/// reproducing DFTracerEvent/ArgsValueProxy semantics so a fold-built
/// aggregation tier is byte-identical to the DOM-driven visitor. hhash/fhash
/// are read from the POD's dedicated fields (they are not in `args`); every
/// numeric arg is stored as a double, so arg_uint clamps exactly like
/// ArgsValueProxy::get<uint64>.
class PodAccessor {
   public:
    PodAccessor(const FoldEvent& ev,
                const dftracer::utils::StringIntern& intern)
        : ev_(ev), intern_(intern) {}

    bool is_metadata() const { return ev_.phase == RecordPhase::METADATA; }
    bool is_counter() const { return ev_.phase == RecordPhase::COUNTER; }
    bool is_profile() const { return is_counter() && cat() != "sys"; }
    bool is_system() const { return is_counter() && cat() == "sys"; }
    std::string_view name() const { return resolve(ev_.name_id); }
    std::string_view cat() const { return resolve(ev_.cat_id); }
    std::uint64_t pid() const { return ev_.pid; }
    std::uint64_t tid() const { return ev_.tid; }
    std::uint64_t ts() const { return ev_.ts; }
    std::uint64_t dur() const { return ev_.dur; }

    bool arg_exists(std::string_view k) const {
        if (k == "hhash") return ev_.hhash_id != NO_ID;
        if (k == "fhash") return ev_.fhash_id != NO_ID;
        return find(k) != nullptr;
    }
    bool arg_is_number(std::string_view k) const {
        if (k == "hhash" || k == "fhash") return false;
        const auto* v = find(k);
        return v && (std::holds_alternative<double>(*v) ||
                     std::holds_alternative<std::int64_t>(*v));
    }
    std::uint64_t arg_uint(std::string_view k) const {
        if (k == "hhash" || k == "fhash") return 0;
        const auto* v = find(k);
        if (v) {
            if (const auto* d = std::get_if<double>(v)) return clamp_uint(*d);
            if (const auto* i = std::get_if<std::int64_t>(v))
                return *i < 0 ? 0 : static_cast<std::uint64_t>(*i);
        }
        return 0;
    }
    double arg_double(std::string_view k) const {
        if (k == "hhash" || k == "fhash") return 0.0;
        const auto* v = find(k);
        if (v) {
            if (const auto* d = std::get_if<double>(v)) return *d;
            if (const auto* i = std::get_if<std::int64_t>(v))
                return static_cast<double>(*i);
        }
        return 0.0;
    }
    std::string_view arg_string(std::string_view k) const {
        if (k == "hhash") return resolve(ev_.hhash_id);
        if (k == "fhash") return resolve(ev_.fhash_id);
        const auto* v = find(k);
        if (v)
            if (const auto* id = std::get_if<std::uint32_t>(v))
                return resolve(*id);
        return {};
    }

    template <class Fn>
    void for_each_numeric_arg(Fn&& fn) const {
        for (const auto& [kid, val] : ev_.args) {
            if (const auto* d = std::get_if<double>(&val))
                fn(resolve(kid), clamp_uint(*d), *d);
            else if (const auto* i = std::get_if<std::int64_t>(&val))
                fn(resolve(kid), *i < 0 ? 0 : static_cast<std::uint64_t>(*i),
                   static_cast<double>(*i));
        }
    }

   private:
    static constexpr std::uint32_t NO_ID = dftracer::utils::StringIntern::NO_ID;

    const FoldEvent::ArgValue* find(std::string_view key) const {
        for (const auto& [kid, v] : ev_.args)
            if (resolve(kid) == key) return &v;
        return nullptr;
    }
    std::string_view resolve(std::uint32_t id) const {
        return id == NO_ID ? std::string_view{} : intern_.resolve(id);
    }
    // Reproduce ArgsValueProxy::get<uint64_t> applied to a double.
    static std::uint64_t clamp_uint(double d) {
        if (d >= 0 &&
            d <= static_cast<double>(std::numeric_limits<std::uint64_t>::max()))
            return static_cast<std::uint64_t>(d);
        return 0;
    }

    const FoldEvent& ev_;
    const dftracer::utils::StringIntern& intern_;
};

/// Builds the aggregation index tier (AGGREGATION + SYSTEM_METRICS records) as
/// a fold on the fused scan, so it rides the same single parse as BloomFold /
/// DictFold. `build_intern` is the fused-scan intern that produced the events
/// (resolves the POD's ids); `agg_intern` is the per-index aggregation intern
/// table whose ids the serialized keys embed. write_to_sink emits merge
/// operands + the intern dictionary; the tracker / observed-key / time-bound
/// outputs are returned out-of-band (they are not column-family records).
class AggregationFold : public Fold {
   public:
    AggregationFold(dftracer::utils::StringIntern& build_intern,
                    aggregators::AggInternPtr agg_intern,
                    aggregators::AggregationConfig config,
                    std::uint32_t config_hash)
        : build_intern_(&build_intern),
          agg_intern_(std::move(agg_intern)),
          config_(std::move(config)),
          config_hash_(config_hash) {
        if (config_.track_process_parents || !config_.boundary_events.empty())
            tracker_ = std::make_shared<aggregators::AssociationTracker>();
    }

    bool accepts(const ScanShape&) const override { return true; }
    bool needs_args() const override { return true; }

    std::unique_ptr<Fold> slice() const override {
        return std::make_unique<AggregationFold>(*build_intern_, agg_intern_,
                                                 config_, config_hash_);
    }

    void step(const FoldBatch& batch) override {
        for (const auto& ev : batch.events) {
            PodAccessor acc(ev, *build_intern_);
            // The tracker sees every non-metadata event, fed before the
            // aggregation classifies it (metadata dropped, then tracker, then
            // aggregate).
            if (acc.is_metadata()) continue;
            if (tracker_) tracker_->extract(acc, config_);
            aggregators::aggregate_event(acc, state_, config_, config_hash_,
                                         agg_intern_->intern);
        }
    }

    void seal_unit(const ScanUnit&) override {}
    void drop_unit(const ScanUnit&) override {}
    void merge(Fold& other) override;

    coro::CoroTask<bool> finalize(const CoverageSet&) override {
        co_return false;
    }

    /// Emit the accumulated aggregation + system-metrics merge operands and the
    /// intern dictionary to `sink`. `file_id` is unused (aggregation keys are
    /// file-independent) but kept parallel to the other index folds.
    void write_to_sink(
        dftracer::utils::utilities::indexer::IndexBatchSink& sink, int file_id);

    const std::unordered_set<std::string>& observed_extra_keys() const {
        return state_.observed_extra_keys;
    }
    const std::unordered_set<std::string>& observed_custom_metrics() const {
        return state_.observed_custom_metrics;
    }
    const std::unordered_set<std::string>& observed_system_metrics() const {
        return state_.observed_system_metrics;
    }
    std::uint64_t min_time_bucket() const { return state_.min_time_bucket; }
    std::uint64_t max_time_bucket() const { return state_.max_time_bucket; }
    std::size_t events_processed() const { return state_.events_processed; }

    /// The per-file process/boundary tracker (null when the config tracks
    /// neither). Returned raw (not finalized), matching the visitor's
    /// take_output().local_tracker; the merger finalizes the global tracker.
    std::shared_ptr<aggregators::AssociationTracker> take_tracker() {
        return std::move(tracker_);
    }

   private:
    dftracer::utils::StringIntern* build_intern_;
    aggregators::AggInternPtr agg_intern_;
    aggregators::AggregationConfig config_;
    std::uint32_t config_hash_;
    aggregators::AggState state_;
    std::shared_ptr<aggregators::AssociationTracker> tracker_;
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_AGGREGATION_FOLD_H
