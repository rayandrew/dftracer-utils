#ifndef DFTRACER_UTILS_UTILITIES_COMMON_ARROW_GAPFILL_H
#define DFTRACER_UTILS_UTILITIES_COMMON_ARROW_GAPFILL_H

#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/utilities/common/arrow/arrow_export.h>

#include <cstdint>

namespace dftracer::utils::utilities::common::arrow {

enum class GapFillMode { NONE, LOCF, LINEAR };

/// Materialize a regular time grid over one batch: one row per grid point per
/// partition, keeping a real row that lands on a grid point and generating one
/// where none does. `time_col` must be an integer column; `bucket_width` > 0.
/// Each partition's grid spans its real rows floor-aligned to bucket_width, or
/// the floor-aligned [range_start, range_end] when `has_range`. On a generated
/// row the partition and time columns carry their values, `value_cols` fill per
/// `mode`, every other column is null: NONE -> null, LOCF -> last real value
/// forward (leading gap null), LINEAR -> interpolation as DOUBLE with no
/// extrapolation (and so emits `value_cols` as DOUBLE). Duplicate real times in
/// one bucket keep the first. Throws DFTUtilsException on a bad index/type, a
/// non-positive width, a LINEAR fill over a non-numeric column, or a grid
/// exceeding 10,000,000 points.
ArrowExportResult gap_fill(const ArrowSchema* s, const ArrowArray* a,
                           const std::uint32_t* partition_cols,
                           std::uint32_t n_part, std::uint32_t time_col,
                           std::int64_t bucket_width,
                           const std::uint32_t* value_cols,
                           std::uint32_t n_value, GapFillMode mode,
                           bool has_range, std::int64_t range_start,
                           std::int64_t range_end);

/// Same contract as above for a FLOAT/DOUBLE `time_col` with a double
/// `bucket_width` (> 0); the grid is start, start+width, ... with each point
/// floor-aligned as floor(t / width) * width.
ArrowExportResult gap_fill(const ArrowSchema* s, const ArrowArray* a,
                           const std::uint32_t* partition_cols,
                           std::uint32_t n_part, std::uint32_t time_col,
                           double bucket_width, const std::uint32_t* value_cols,
                           std::uint32_t n_value, GapFillMode mode,
                           bool has_range, double range_start,
                           double range_end);

}  // namespace dftracer::utils::utilities::common::arrow

#endif  // DFTRACER_UTILS_ENABLE_ARROW
#endif  // DFTRACER_UTILS_UTILITIES_COMMON_ARROW_GAPFILL_H
