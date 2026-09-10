#ifndef DFTRACER_UTILS_DATAFRAME_TYPES_H
#define DFTRACER_UTILS_DATAFRAME_TYPES_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

enum class TypeId : std::int32_t {
    /// Schema-only marker: a type not knowable without scanning, or an
    /// unset/zero-initialized TypeId. Never the type of a real Series;
    /// dispatch over actual column data must reject it like any other
    /// out-of-range value, not handle it as data.
    Unknown,
    Bool,
    Int8,
    Int16,
    Int32,
    Int64,
    Uint8,
    Uint16,
    Uint32,
    Uint64,
    Float32,
    Float64,
    String,
    Binary,
    List,
    Struct,
    /// 2-byte IEEE 754 half-precision float. No arithmetic kernel operates on
    /// it; it round-trips through Arrow import/export as opaque 2-byte data.
    Float16,
    /// Days since the Unix epoch, stored as Int32 (DataType::physical_type()).
    Date32,
    /// Milliseconds since the Unix epoch, stored as Int64.
    Date64,
    /// Time of day since midnight in `DataType::time_unit` (Second or Milli),
    /// stored as Int32.
    Time32,
    /// Time of day since midnight in `DataType::time_unit` (Micro or Nano),
    /// stored as Int64.
    Time64,
    /// Instant since the Unix epoch in `DataType::time_unit`, with an
    /// optional `DataType::timezone`, stored as Int64.
    Timestamp,
    /// Elapsed time in `DataType::time_unit`, stored as Int64.
    Duration,
    /// 128-bit fixed-point decimal; `DataType::decimal_precision` and
    /// `decimal_scale` give its meaning. Opaque 16-byte data to every kernel.
    Decimal128,
    /// 256-bit fixed-point decimal; see Decimal128. Opaque 32-byte data.
    Decimal256,
    /// Fixed-width opaque byte string; `DataType::fixed_size` is the width in
    /// bytes. No offsets buffer.
    FixedSizeBinary,
    /// String with 64-bit offsets, for payloads too large for String's
    /// 32-bit offsets. Its own physical layout, not String's.
    LargeString,
    /// Binary with 64-bit offsets. Its own physical layout, not Binary's.
    LargeBinary,
    /// List with 64-bit offsets. Its own physical layout, not List's.
    LargeList,
    /// Fixed-count list: `DataType::fixed_size` elements of the single entry
    /// in `DataType::fields` per row. No offsets buffer.
    FixedSizeList,
    /// Sorted key/value pairs per row, stored as List<Struct{key, value}>
    /// (DataType::physical_type()): `DataType::fields` holds one "entries"
    /// field whose type is that Struct.
    Map,
};

struct Field;

/// Second/Milli/Micro/Nano granularity for Timestamp, Time32, Time64, and
/// Duration.
enum class TimeUnit : std::int32_t {
    Second,
    Milli,
    Micro,
    Nano,
};

/// A column's full type. `id` is the tag; `fields` carries the nested
/// structure - exactly one entry for List/LargeList/FixedSizeList (the
/// element) and for Map (the Struct{key,value} entry type), one per field for
/// Struct, and none for a scalar. The remaining members are parameters that
/// are meaningful only for the `id` values named on each: default-valued and
/// ignored otherwise.
struct DataType {
    TypeId id = TypeId::Unknown;
    std::vector<Field> fields;
    /// Timestamp, Time32, Time64, Duration.
    TimeUnit time_unit = TimeUnit::Micro;
    /// Timestamp only; empty means no timezone (a naive timestamp).
    std::string timezone;
    /// Decimal128, Decimal256.
    std::int32_t decimal_precision = 0;
    std::int32_t decimal_scale = 0;
    /// FixedSizeBinary (byte width), FixedSizeList (element count).
    std::int32_t fixed_size = 0;

    bool operator==(const DataType& other) const;
    bool operator!=(const DataType& other) const { return !(*this == other); }
};

/// A named member of a nested type: a Struct field, or a List's element.
struct Field {
    std::string name;
    DataType type;
    bool nullable = true;

    bool operator==(const Field& other) const {
        return name == other.name && nullable == other.nullable &&
               type == other.type;
    }
    bool operator!=(const Field& other) const { return !(*this == other); }
};

inline bool DataType::operator==(const DataType& other) const {
    return id == other.id && fields == other.fields &&
           time_unit == other.time_unit && timezone == other.timezone &&
           decimal_precision == other.decimal_precision &&
           decimal_scale == other.decimal_scale &&
           fixed_size == other.fixed_size;
}

/// A scalar DataType with no nested fields.
constexpr DataType scalar(TypeId id) noexcept {
    DataType dt;
    dt.id = id;
    return dt;
}

/// A List<item_type> DataType, with the element named `item_name`.
inline DataType list_of(DataType item_type, std::string item_name = "item") {
    DataType dt;
    dt.id = TypeId::List;
    dt.fields.push_back(
        Field{std::move(item_name), std::move(item_type), true});
    return dt;
}

/// A LargeList<item_type> DataType, with the element named `item_name`.
inline DataType large_list_of(DataType item_type,
                              std::string item_name = "item") {
    DataType dt;
    dt.id = TypeId::LargeList;
    dt.fields.push_back(
        Field{std::move(item_name), std::move(item_type), true});
    return dt;
}

/// A FixedSizeList<item_type> DataType of `size` elements per row.
inline DataType fixed_size_list_of(DataType item_type, std::int32_t size,
                                   std::string item_name = "item") {
    DataType dt;
    dt.id = TypeId::FixedSizeList;
    dt.fixed_size = size;
    dt.fields.push_back(
        Field{std::move(item_name), std::move(item_type), true});
    return dt;
}

/// A Struct DataType with the given fields, in order.
inline DataType struct_of(std::vector<Field> fields) {
    DataType dt;
    dt.id = TypeId::Struct;
    dt.fields = std::move(fields);
    return dt;
}

/// A Map<key_type, value_type> DataType, physically List<Struct{key, value}>.
/// The single field in `fields` (named "item", matching List's convention) is
/// that Struct type.
inline DataType map_of(DataType key_type, DataType value_type) {
    DataType entry = struct_of({Field{"key", std::move(key_type), false},
                                Field{"value", std::move(value_type), true}});
    DataType dt;
    dt.id = TypeId::Map;
    dt.fields.push_back(Field{"item", std::move(entry), false});
    return dt;
}

/// A Timestamp DataType at `unit` granularity, optionally with `tz`.
inline DataType timestamp(TimeUnit unit, std::string tz = "") {
    DataType dt;
    dt.id = TypeId::Timestamp;
    dt.time_unit = unit;
    dt.timezone = std::move(tz);
    return dt;
}

/// A Time32/Time64 DataType at `unit` granularity (Second/Milli use Time32's
/// Int32 storage, Micro/Nano use Time64's Int64 storage).
inline DataType time_of(TimeUnit unit) {
    DataType dt;
    dt.id = (unit == TimeUnit::Second || unit == TimeUnit::Milli)
                ? TypeId::Time32
                : TypeId::Time64;
    dt.time_unit = unit;
    return dt;
}

/// A Duration DataType at `unit` granularity.
inline DataType duration_of(TimeUnit unit) {
    DataType dt;
    dt.id = TypeId::Duration;
    dt.time_unit = unit;
    return dt;
}

/// A Decimal128 DataType with the given precision and scale.
inline DataType decimal128(std::int32_t precision, std::int32_t scale) {
    DataType dt;
    dt.id = TypeId::Decimal128;
    dt.decimal_precision = precision;
    dt.decimal_scale = scale;
    return dt;
}

/// A Decimal256 DataType with the given precision and scale.
inline DataType decimal256(std::int32_t precision, std::int32_t scale) {
    DataType dt;
    dt.id = TypeId::Decimal256;
    dt.decimal_precision = precision;
    dt.decimal_scale = scale;
    return dt;
}

/// A FixedSizeBinary DataType of `size` bytes per row.
inline DataType fixed_size_binary(std::int32_t size) {
    DataType dt;
    dt.id = TypeId::FixedSizeBinary;
    dt.fixed_size = size;
    return dt;
}

/// The TypeId whose buffer layout `t` shares, for kernel dispatch: a logical
/// type maps to the physical type its data is actually stored as (e.g.
/// Timestamp -> Int64, Map -> List); every other type maps to itself.
constexpr TypeId physical_type(TypeId t) noexcept {
    switch (t) {
        case TypeId::Date32:
        case TypeId::Time32:
            return TypeId::Int32;
        case TypeId::Date64:
        case TypeId::Time64:
        case TypeId::Timestamp:
        case TypeId::Duration:
            return TypeId::Int64;
        case TypeId::Map:
            return TypeId::List;
        case TypeId::Unknown:
        case TypeId::Bool:
        case TypeId::Int8:
        case TypeId::Int16:
        case TypeId::Int32:
        case TypeId::Int64:
        case TypeId::Uint8:
        case TypeId::Uint16:
        case TypeId::Uint32:
        case TypeId::Uint64:
        case TypeId::Float32:
        case TypeId::Float64:
        case TypeId::String:
        case TypeId::Binary:
        case TypeId::List:
        case TypeId::Struct:
        case TypeId::Float16:
        case TypeId::Decimal128:
        case TypeId::Decimal256:
        case TypeId::FixedSizeBinary:
        case TypeId::LargeString:
        case TypeId::LargeBinary:
        case TypeId::LargeList:
        case TypeId::FixedSizeList:
            return t;
    }
    return t;
}

/// FLAT: values contiguous. CONSTANT: one value, logical length N. DICTIONARY:
/// value[i] = base[codes[i]]. SELECTION: value[i] = base[sel[i]], a view over a
/// base with no data movement.
enum class Encoding : std::int32_t {
    Flat,
    Constant,
    Dictionary,
    Selection,
};

/// C++ mirror of the C ABI op-code enums (same values). Kernels cast the raw
/// int32 code from the ABI to these so their dispatch switches are exhaustive
/// and -Wswitch flags a missing case.

/// Mirrors dftu_cmp_op.
enum class CmpOp : std::int32_t {
    Gt = 0,
    Ge = 1,
    Lt = 2,
    Le = 3,
    Eq = 4,
    Ne = 5,
};

/// Mirrors dftu_logical_op.
enum class LogicalOp : std::int32_t {
    And = 0,
    Or = 1,
};

/// Mirrors dftu_rank_method.
enum class RankMethod : std::int32_t {
    Average = 0,
    Min = 1,
    Dense = 2,
    Ordinal = 3,
};

/// Mirrors dftu_rolling_op.
enum class RollingOp : std::int32_t {
    Sum = 0,
    Mean = 1,
    Min = 2,
    Max = 3,
};

/// Mirrors the dftu_series_prim op codes: a unary numeric primitive over a FLAT
/// Int64/Uint64 column, producing an Int64 column.
enum class PrimOp : std::int32_t {
    Ilog2 = 0,
    BitWidth = 1,
    Popcount = 2,
    Clz = 3,
    Ctz = 4,
    Mix64 = 5,
};

/// Unary numeric op codes for the expression engine (elementwise, one column
/// operand). Log/Sqrt/Exp widen to Float64; Abs/Round/Floor/Ceil/Trunc/Sign/
/// Negate keep the input's numeric type; IsNan/IsFinite/IsInfinite yield Bool.
enum class UnaryOp : std::int32_t {
    Abs = 0,
    Round = 1,
    Floor = 2,
    Ceil = 3,
    Log = 4,
    Sqrt = 5,
    Exp = 6,
    Sign = 7,
    Negate = 8,
    Trunc = 9,
    IsNan = 10,
    IsFinite = 11,
    IsInfinite = 12,
};

/// Mirrors dftu_scalar_tag: the domain a dftu_scalar carries.
enum class ScalarTag : std::int32_t {
    I64 = 0,
    U64 = 1,
    F64 = 2,
};

/// Bytes to hold `n` values of `t` in Arrow layout. Bool is bit-packed
/// (ceil(n/8)); other fixed-width types are n * byte_width; String/Binary
/// (variable-width) return 0 - their sizing is offset-driven, not by this.
constexpr std::size_t buffer_bytes(TypeId t, std::int64_t n) noexcept;

/// Byte width of a fixed-width type; 0 for variable-width (String/Binary and
/// every type whose width is not a per-TypeId constant: FixedSizeBinary and
/// FixedSizeList size by DataType::fixed_size, LargeString/LargeBinary/
/// LargeList by their own 64-bit-offset layout).
constexpr std::size_t byte_width(TypeId t) noexcept {
    switch (t) {
        case TypeId::Bool:
        case TypeId::Int8:
        case TypeId::Uint8:
            return 1;
        case TypeId::Int16:
        case TypeId::Uint16:
        case TypeId::Float16:
            return 2;
        case TypeId::Int32:
        case TypeId::Uint32:
        case TypeId::Float32:
        case TypeId::Date32:
        case TypeId::Time32:
            return 4;
        case TypeId::Int64:
        case TypeId::Uint64:
        case TypeId::Float64:
        case TypeId::Date64:
        case TypeId::Time64:
        case TypeId::Timestamp:
        case TypeId::Duration:
            return 8;
        case TypeId::Decimal128:
            return 16;
        case TypeId::Decimal256:
            return 32;
        case TypeId::String:
        case TypeId::Binary:
        case TypeId::List:
        case TypeId::Struct:
        case TypeId::FixedSizeBinary:
        case TypeId::LargeString:
        case TypeId::LargeBinary:
        case TypeId::LargeList:
        case TypeId::FixedSizeList:
        case TypeId::Map:
        case TypeId::Unknown:
            return 0;
    }
    return 0;
}

constexpr std::size_t buffer_bytes(TypeId t, std::int64_t n) noexcept {
    if (t == TypeId::Bool) return static_cast<std::size_t>((n + 7) / 8);
    return static_cast<std::size_t>(n) * byte_width(t);
}

/// The lowercase dtype name (e.g. "int64", "float64"), for diagnostics.
constexpr const char* type_name(TypeId t) noexcept {
    switch (t) {
        case TypeId::Bool:
            return "bool";
        case TypeId::Int8:
            return "int8";
        case TypeId::Int16:
            return "int16";
        case TypeId::Int32:
            return "int32";
        case TypeId::Int64:
            return "int64";
        case TypeId::Uint8:
            return "uint8";
        case TypeId::Uint16:
            return "uint16";
        case TypeId::Uint32:
            return "uint32";
        case TypeId::Uint64:
            return "uint64";
        case TypeId::Float32:
            return "float32";
        case TypeId::Float64:
            return "float64";
        case TypeId::String:
            return "string";
        case TypeId::Binary:
            return "binary";
        case TypeId::List:
            return "list";
        case TypeId::Struct:
            return "struct";
        case TypeId::Float16:
            return "float16";
        case TypeId::Date32:
            return "date32";
        case TypeId::Date64:
            return "date64";
        case TypeId::Time32:
            return "time32";
        case TypeId::Time64:
            return "time64";
        case TypeId::Timestamp:
            return "timestamp";
        case TypeId::Duration:
            return "duration";
        case TypeId::Decimal128:
            return "decimal128";
        case TypeId::Decimal256:
            return "decimal256";
        case TypeId::FixedSizeBinary:
            return "fixed_size_binary";
        case TypeId::LargeString:
            return "large_string";
        case TypeId::LargeBinary:
            return "large_binary";
        case TypeId::LargeList:
            return "large_list";
        case TypeId::FixedSizeList:
            return "fixed_size_list";
        case TypeId::Map:
            return "map";
        case TypeId::Unknown:
            return "unknown";
    }
    return "unknown";
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_TYPES_H
