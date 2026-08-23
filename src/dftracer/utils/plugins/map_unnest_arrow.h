#ifndef DFTRACER_UTILS_PLUGINS_MAP_UNNEST_ARROW_H
#define DFTRACER_UTILS_PLUGINS_MAP_UNNEST_ARROW_H

#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/plugins/map_unnest.h>
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>

// Arrow EXPORT half of the native unnest: an ExplodedRows into one record batch
// [key cols k0.., kept value cols, the exploded element column]. `intern` must
// be the run's intern table (STR element/key ids are resolved at emit).
namespace dftracer::utils::plugins {

utilities::common::arrow::ArrowExportResult materialize_exploded(
    const ExplodedRows& e, dftracer::utils::StringIntern& intern);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_ENABLE_ARROW
#endif  // DFTRACER_UTILS_PLUGINS_MAP_UNNEST_ARROW_H
