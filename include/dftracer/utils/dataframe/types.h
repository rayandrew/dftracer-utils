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
};

struct Field;

/// A column's full type. `id` is the tag; `fields` carries the nested
/// structure - exactly one entry for List (the element), one per field for
/// Struct, and none for a scalar.
struct DataType {
    TypeId id = TypeId::Unknown;
    std::vector<Field> fields;

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
    return id == other.id && fields == other.fields;
}

/// A scalar DataType with no nested fields.
constexpr DataType scalar(TypeId id) noexcept { return DataType{id, {}}; }

/// A List<item_type> DataType, with the element named `item_name`.
inline DataType list_of(DataType item_type, std::string item_name = "item") {
    DataType dt;
    dt.id = TypeId::List;
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

/// Byte width of a fixed-width type; 0 for variable-width (String/Binary).
constexpr std::size_t byte_width(TypeId t) noexcept {
    switch (t) {
        case TypeId::Bool:
        case TypeId::Int8:
        case TypeId::Uint8:
            return 1;
        case TypeId::Int16:
        case TypeId::Uint16:
            return 2;
        case TypeId::Int32:
        case TypeId::Uint32:
        case TypeId::Float32:
            return 4;
        case TypeId::Int64:
        case TypeId::Uint64:
        case TypeId::Float64:
            return 8;
        case TypeId::String:
        case TypeId::Binary:
        case TypeId::List:
        case TypeId::Struct:
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
        case TypeId::Unknown:
            return "unknown";
    }
    return "unknown";
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_TYPES_H
