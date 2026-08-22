#ifndef DFTRACER_UTILS_UTILITIES_COMMON_ARROW_EXPLODE_H
#define DFTRACER_UTILS_UTILITIES_COMMON_ARROW_EXPLODE_H

#include <dftracer/utils/core/common/config.h>
#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/utilities/common/arrow/arrow_export.h>

#include <cstddef>

namespace dftracer::utils::utilities::common::arrow {

/// UNNEST / EXPLODE one list-typed column of a materialized record batch into
/// one row per list element, repeating every other column; the caller retains
/// ownership of the batch. `list_col_idx` must select a list<utf8>, list<int64>
/// (any scalar element type), or list<struct<...>>: a list<struct> flattens
/// each inner field into its own scalar column (named by the field), a
/// scalar-element list becomes one column keeping the list's name, other
/// columns (scalar or list/struct-valued) pass through by value, their cell
/// repeated per output row. By default an empty or null list drops the row;
/// keep_empty emits one row with the exploded column(s) null. Output row order
/// is input row order, then element order within each row. Throws
/// DFTUtilsException on a non-list column, an unsupported element or
/// passthrough column type, or an output name collision.
ArrowExportResult explode(ArrowSchema* schema, ArrowArray* array,
                          std::size_t list_col_idx, bool keep_empty);

}  // namespace dftracer::utils::utilities::common::arrow

#endif  // DFTRACER_UTILS_ENABLE_ARROW
#endif  // DFTRACER_UTILS_UTILITIES_COMMON_ARROW_EXPLODE_H
