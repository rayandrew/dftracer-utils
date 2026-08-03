#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_drain.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/association_tracker.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/event_aggregator.h>

namespace dftracer::utils::utilities::composites::dft::aggregators {

std::vector<std::string> merge_aggregation_folds(
    std::vector<AggFoldOutput>& outputs, EventAggregator* merger) {
    std::vector<std::string> processed_files;
    if (!merger) return processed_files;

    for (auto& out : outputs) {
        for (const auto& k : out.observed_extra_keys)
            merger->add_observed_extra_key(k);
        for (const auto& m : out.observed_custom_metrics)
            merger->add_observed_custom_metric(m);

        // The tracker must be finalized before the merger collects it.
        if (out.tracker) out.tracker->finalize();

        ChunkAggregationOutput chunk;
        chunk.file_path = out.file_path;
        chunk.events_processed = out.events_processed;
        chunk.success = true;
        chunk.local_tracker = std::move(out.tracker);
        chunk.min_time_bucket = out.min_time_bucket;
        chunk.max_time_bucket = out.max_time_bucket;
        processed_files.push_back(chunk.file_path);
        merger->merge_chunk(std::move(chunk));
    }
    return processed_files;
}

}  // namespace dftracer::utils::utilities::composites::dft::aggregators
