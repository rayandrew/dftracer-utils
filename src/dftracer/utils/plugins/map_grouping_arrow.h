#ifndef DFTRACER_UTILS_PLUGINS_MAP_GROUPING_ARROW_H
#define DFTRACER_UTILS_PLUGINS_MAP_GROUPING_ARROW_H

#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/map_grouping.h>
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>

#include <cstdint>
#include <vector>

// Arrow export of GROUPING SETS into one record batch: all ORIGINAL key columns
// k0.. (each nullable, a dim dropped by a set is null), the value columns, then
// an INT64 grouping_id = the set index. Rows run grouping by grouping, each in
// sorted key order. `original` supplies the unified key/value schema; `intern`
// must be the run's table.
namespace dftracer::utils::plugins {

utilities::common::arrow::ArrowExportResult materialize_grouping_sets(
    const std::vector<MapAccum>& groupings,
    const std::vector<std::vector<std::uint32_t>>& keep_sets,
    const MapAccum& original, dftracer::utils::StringIntern& intern);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_ENABLE_ARROW
#endif  // DFTRACER_UTILS_PLUGINS_MAP_GROUPING_ARROW_H
