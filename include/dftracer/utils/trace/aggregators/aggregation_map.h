#ifndef DFTRACER_UTILS_TRACE_AGGREGATORS_AGGREGATION_MAP_H
#define DFTRACER_UTILS_TRACE_AGGREGATORS_AGGREGATION_MAP_H

#include <ankerl/unordered_dense.h>
#include <dftracer/utils/trace/aggregators/aggregation_key.h>
#include <dftracer/utils/trace/aggregators/aggregation_metrics.h>

namespace dftracer::utils::trace::aggregators {

/// Segmented (not flat): the aggregation visitor caches a pointer to the last
/// entry across events, so entries must not move on insert.
using AggregationMap =
    ankerl::unordered_dense::segmented_map<AggregationKey, AggregationMetrics,
                                           AggregationKeyHash,
                                           AggregationKeyEqual>;

}  // namespace dftracer::utils::trace::aggregators

#endif  // DFTRACER_UTILS_TRACE_AGGREGATORS_AGGREGATION_MAP_H
