#ifndef DFTRACER_UTILS_DATAFRAME_INTERNAL_COLUMN_READ_H
#define DFTRACER_UTILS_DATAFRAME_INTERNAL_COLUMN_READ_H

#include <dftracer/utils/dataframe/dataframe.h>

#include <cstdint>

// Read one FLAT numeric cell widened to a common type, dispatching on the
// column's TypeId. Shared by every kernel that needs a domain-agnostic value
// (stats, aggregation, hashing, ...) so the per-type switch lives in one place.
namespace dftracer::utils::dataframe {

/// Cell `i` widened to int64 (Bool -> 0/1). 0 for non-integer columns.
inline std::int64_t read_i64(const Series& c, std::int64_t i) {
    switch (c.type()) {
        case TypeId::Bool: {
            const std::uint8_t* b = c.data<std::uint8_t>();
            return (b[i >> 3] >> (i & 7)) & 1;
        }
        case TypeId::Int8:
            return c.data<std::int8_t>()[i];
        case TypeId::Int16:
            return c.data<std::int16_t>()[i];
        case TypeId::Int32:
            return c.data<std::int32_t>()[i];
        case TypeId::Int64:
            return c.data<std::int64_t>()[i];
        default:
            return 0;
    }
}

/// Cell `i` widened to uint64: unsigned columns keep their value; signed/Bool
/// are read as int64 then reinterpreted (so hashing sees a stable bit pattern).
/// 0 for non-integer columns.
inline std::uint64_t read_u64(const Series& c, std::int64_t i) {
    switch (c.type()) {
        case TypeId::Uint8:
            return c.data<std::uint8_t>()[i];
        case TypeId::Uint16:
            return c.data<std::uint16_t>()[i];
        case TypeId::Uint32:
            return c.data<std::uint32_t>()[i];
        case TypeId::Uint64:
            return c.data<std::uint64_t>()[i];
        default:
            return static_cast<std::uint64_t>(read_i64(c, i));
    }
}

/// Cell `i` widened to double (Bool -> 0/1). 0 for non-numeric columns.
inline double read_f64(const Series& c, std::int64_t i) {
    switch (c.type()) {
        case TypeId::Float32:
            return static_cast<double>(c.data<float>()[i]);
        case TypeId::Float64:
            return c.data<double>()[i];
        case TypeId::Uint8:
        case TypeId::Uint16:
        case TypeId::Uint32:
        case TypeId::Uint64:
            return static_cast<double>(read_u64(c, i));
        default:
            return static_cast<double>(read_i64(c, i));
    }
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_INTERNAL_COLUMN_READ_H
