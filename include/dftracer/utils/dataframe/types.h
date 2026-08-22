#ifndef DFTRACER_UTILS_DATAFRAME_TYPES_H
#define DFTRACER_UTILS_DATAFRAME_TYPES_H

#include <cstddef>
#include <cstdint>

namespace dftracer::utils::dataframe {

enum class TypeId : std::int32_t {
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
            return 0;
    }
    return 0;
}

constexpr std::size_t buffer_bytes(TypeId t, std::int64_t n) noexcept {
    if (t == TypeId::Bool) return static_cast<std::size_t>((n + 7) / 8);
    return static_cast<std::size_t>(n) * byte_width(t);
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_TYPES_H
