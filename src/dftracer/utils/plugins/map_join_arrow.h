#ifndef DFTRACER_UTILS_PLUGINS_MAP_JOIN_ARROW_H
#define DFTRACER_UTILS_PLUGINS_MAP_JOIN_ARROW_H

#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/map_join.h>
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>

// Arrow export of a JoinedMap into one record batch. Columns are key cols
// k0.., then LEFT value cols (prefix "l_"), then RIGHT value cols (prefix
// "r_"); an OUTER row's absent side is null. `intern` must be the run's table.
namespace dftracer::utils::plugins {

utilities::common::arrow::ArrowExportResult materialize_joined(
    const JoinedMap& j, dftracer::utils::StringIntern& intern);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_ENABLE_ARROW
#endif  // DFTRACER_UTILS_PLUGINS_MAP_JOIN_ARROW_H
