#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_AGGREGATORS_AGGREGATION_DRAIN_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_AGGREGATORS_AGGREGATION_DRAIN_H

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::aggregators {

class EventAggregator;
class AssociationTracker;

// The out-of-band outputs an AggregationFold produces per file (the metrics
// themselves are written to the index sink by write_to_sink).
struct AggFoldOutput {
    std::string file_path;
    std::unordered_set<std::string> observed_extra_keys;
    std::unordered_set<std::string> observed_custom_metrics;
    std::shared_ptr<AssociationTracker> tracker;
    std::uint64_t min_time_bucket = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t max_time_bucket = 0;
    std::size_t events_processed = 0;
};

// Merge the per-file AggregationFold out-of-band outputs into `merger` and
// return the processed file paths. Finalizes each tracker before merging.
// No-op when `merger` is null. Synchronous, so safe to call from a coroutine
// without suspending.
std::vector<std::string> merge_aggregation_folds(
    std::vector<AggFoldOutput>& outputs, EventAggregator* merger);

}  // namespace dftracer::utils::utilities::composites::dft::aggregators

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_AGGREGATORS_AGGREGATION_DRAIN_H
