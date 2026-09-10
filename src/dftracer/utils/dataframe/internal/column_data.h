#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_COLUMN_DATA_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_COLUMN_DATA_H

#include <dftracer/utils/dataframe/buffer.h>
#include <dftracer/utils/dataframe/types.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// Layout behind the opaque dftu_series handle. Private to the dataframe
/// sources; the public headers only forward-declare it.
struct dftu_series {
    dftracer::utils::dataframe::TypeId type =
        dftracer::utils::dataframe::TypeId::Int64;
    dftracer::utils::dataframe::Encoding encoding =
        dftracer::utils::dataframe::Encoding::Flat;
    std::int64_t length = 0;
    std::int64_t null_count = 0;
    /// FLAT/CONSTANT values, the SELECTION/DICTIONARY index buffer (int32), or
    /// the byte data of a variable-width String/Binary column.
    std::shared_ptr<dftracer::utils::dataframe::Buffer> data;
    /// int32 offsets (length+1 entries) for a variable-width String/Binary/List
    /// column; null for fixed-width types and for the 64-bit-offset Large*
    /// variants, which use `offsets64` instead.
    std::shared_ptr<dftracer::utils::dataframe::Buffer> offsets;
    /// int64 offsets (length+1 entries) for LargeString/LargeBinary/LargeList,
    /// whose 64-bit-offset layout is a distinct physical type from String/
    /// Binary/List's 32-bit offsets; null otherwise. Not exposed through the
    /// public dftu_series_offsets ABI, which is int32-only.
    std::shared_ptr<dftracer::utils::dataframe::Buffer> offsets64;
    /// Arrow-layout validity bitmap (1 = valid); null when there are no nulls.
    std::shared_ptr<dftracer::utils::dataframe::Buffer> validity;
    /// Base for SELECTION/DICTIONARY, or the flattened values of a LIST column;
    /// null otherwise.
    std::shared_ptr<dftu_series> child;
    /// Field columns of a STRUCT (all length == this->length); empty otherwise.
    std::vector<std::shared_ptr<dftu_series>> children;
    /// Field names of a STRUCT, aligned to `children`.
    std::vector<std::string> field_names;
    /// Timestamp/Time32/Time64/Duration only.
    dftracer::utils::dataframe::TimeUnit time_unit =
        dftracer::utils::dataframe::TimeUnit::Micro;
    /// Timestamp only; empty means no timezone.
    std::string timezone;
    /// Decimal128/Decimal256 only.
    std::int32_t decimal_precision = 0;
    std::int32_t decimal_scale = 0;
    /// FixedSizeBinary (byte width) or FixedSizeList (element count) only.
    std::int32_t fixed_size = 0;
};

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_COLUMN_DATA_H
