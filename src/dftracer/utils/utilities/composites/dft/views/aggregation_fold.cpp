#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_serialization.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/system_metrics_serialization.h>
#include <dftracer/utils/utilities/composites/dft/views/aggregation_fold.h>
#include <dftracer/utils/utilities/indexer/index_batch_sink.h>

#include <algorithm>
#include <string>
#include <utility>

namespace dftracer::utils::utilities::composites::dft::views::detail {

void AggregationFold::merge(Fold& other) {
    auto& o = static_cast<AggregationFold&>(other);
    auto& os = o.state_;

    for (auto& [k, m] : os.local_buffer) {
        auto it = state_.local_buffer.find(k);
        if (it == state_.local_buffer.end())
            state_.local_buffer.emplace(k, std::move(m));
        else
            it->second.merge_from(m);
    }
    for (auto& [k, m] : os.system_buffer) {
        auto it = state_.system_buffer.find(k);
        if (it == state_.system_buffer.end())
            state_.system_buffer.emplace(k, std::move(m));
        else
            it->second.merge_from(m);
    }
    for (auto& k : os.observed_extra_keys) state_.observed_extra_keys.insert(k);
    for (auto& k : os.observed_custom_metrics)
        state_.observed_custom_metrics.insert(k);
    for (auto& k : os.observed_system_metrics)
        state_.observed_system_metrics.insert(k);
    state_.min_time_bucket =
        std::min(state_.min_time_bucket, os.min_time_bucket);
    state_.max_time_bucket =
        std::max(state_.max_time_bucket, os.max_time_bucket);
    state_.events_processed += os.events_processed;

    if (o.tracker_) {
        if (tracker_)
            tracker_->merge(*o.tracker_);
        else
            tracker_ = std::move(o.tracker_);
    }

    // The last-entry cache points into one buffer; a merge relocates entries.
    state_.last_entry = nullptr;
    state_.last_key = {};
    os.local_buffer.clear();
    os.system_buffer.clear();
    os.last_entry = nullptr;
    os.last_key = {};
}

void AggregationFold::write_to_sink(
    dftracer::utils::utilities::indexer::IndexBatchSink& sink,
    int /*file_id*/) {
    namespace agg = aggregators;
    std::string val_buf;
    std::string sys_val_buf;
    for (auto& [k, m] : state_.local_buffer) {
        agg::serialize_agg_value_into(val_buf, m);
        sink.insert_aggregation_merge(k, val_buf);
    }
    for (auto& [k, m] : state_.system_buffer) {
        agg::serialize_system_value_into(sys_val_buf, m);
        sink.insert_system_metrics_merge(k, sys_val_buf);
    }
    agg::flush_intern_dictionary(sink, *agg_intern_);
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
