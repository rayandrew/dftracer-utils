#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_DICTIONARY_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_DICTIONARY_H

#include <dftracer/utils/dataframe/dataframe.h>

namespace dftracer::utils::dataframe {

/// Dictionary-encode a FLAT String/Binary column: deduplicate the values into a
/// dictionary and return a DICTIONARY column of int32 codes over it. The trace
/// fast path (repeated names/paths). Invalid (empty) for other inputs.
Series dictionary_encode(const Series& v);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_DICTIONARY_H
