#ifndef DFTRACER_UTILS_PLUGINS_PLUGIN_TYPES_H
#define DFTRACER_UTILS_PLUGINS_PLUGIN_TYPES_H

#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/query/builder.h>

#include <cstdint>
#include <string_view>
#include <type_traits>

namespace dftracer::utils::plugins {

// Re-export the query builder so a plugin writes `F("dur") > 1000` with only
// this header and no query:: qualifier.
using dftracer::utils::query::Expr;
using dftracer::utils::query::F;
using dftracer::utils::query::Field;
using dftracer::utils::query::resolved;

/// Build a provided capability from an id and semantic version, for a Slice's
/// static `provides()`. `id` must outlive the plugin (a string literal); the
/// `dftu.` prefix is reserved for the host and rejected at resolve.
constexpr dftu_capability capability(const char* id, std::uint16_t major = 0,
                                     std::uint16_t minor = 0,
                                     std::uint16_t patch = 0) noexcept {
    return dftu_capability{id, dftu_version{major, minor, patch}};
}

/// Build a requirement from an id, version op, and version, for a Slice's
/// static `requires_caps()`. `required` true makes an unmet requirement a load
/// error; false falls back gracefully. `id` must outlive the plugin (a string
/// literal).
constexpr dftu_requirement requirement(const char* id,
                                       dftu_ver_op op = DFTU_VER_GE,
                                       std::uint16_t major = 0,
                                       std::uint16_t minor = 0,
                                       std::uint16_t patch = 0,
                                       bool required = false) noexcept {
    return dftu_requirement{id, op, dftu_version{major, minor, patch},
                            required ? 1 : 0};
}

/// Typed read view over a dftu_monoid_value from Host::handle_result. Read the
/// accessor matching kind(): COUNTER-family kinds carry u64, SUM/MIN/MAX and
/// the statistics kinds f64, SKETCH the quantiles.
struct MonoidValue {
    dftu_monoid_value raw{};

    dftu_monoid_kind kind() const noexcept { return raw.kind; }
    std::uint64_t as_u64() const noexcept { return raw.as.u64; }
    double as_f64() const noexcept { return raw.as.f64; }
    dftu_quantiles as_quantiles() const noexcept { return raw.as.quant; }
};

/// Scoped mirror of dftu_monoid_kind; each enumerator is defined by its ABI
/// constant, so static_cast<dftu_monoid_kind> recovers the raw value. See abi.h
/// for each kind's semantics and its required feed (u64/f64/xy/argby/topk/...).
enum class Monoid : std::int32_t {
    Counter = DFTU_MONOID_COUNTER,
    Sum_F64 = DFTU_MONOID_SUM_F64,
    Min_F64 = DFTU_MONOID_MIN_F64,
    Max_F64 = DFTU_MONOID_MAX_F64,
    Sketch = DFTU_MONOID_SKETCH,
    Min_U64 = DFTU_MONOID_MIN_U64,
    Max_U64 = DFTU_MONOID_MAX_U64,
    Bool_And = DFTU_MONOID_BOOL_AND,
    Bool_Or = DFTU_MONOID_BOOL_OR,
    Bitset_Or = DFTU_MONOID_BITSET_OR,
    Distinct = DFTU_MONOID_DISTINCT,
    Set_Str = DFTU_MONOID_SET_STR,
    List_Str = DFTU_MONOID_LIST_STR,
    Set_I64 = DFTU_MONOID_SET_I64,
    List_I64 = DFTU_MONOID_LIST_I64,
    Min_I8 = DFTU_MONOID_MIN_I8,
    Min_I16 = DFTU_MONOID_MIN_I16,
    Min_I32 = DFTU_MONOID_MIN_I32,
    Min_I64 = DFTU_MONOID_MIN_I64,
    Min_U8 = DFTU_MONOID_MIN_U8,
    Min_U16 = DFTU_MONOID_MIN_U16,
    Min_U32 = DFTU_MONOID_MIN_U32,
    Min_F32 = DFTU_MONOID_MIN_F32,
    Max_I8 = DFTU_MONOID_MAX_I8,
    Max_I16 = DFTU_MONOID_MAX_I16,
    Max_I32 = DFTU_MONOID_MAX_I32,
    Max_I64 = DFTU_MONOID_MAX_I64,
    Max_U8 = DFTU_MONOID_MAX_U8,
    Max_U16 = DFTU_MONOID_MAX_U16,
    Max_U32 = DFTU_MONOID_MAX_U32,
    Max_F32 = DFTU_MONOID_MAX_F32,
    ArgMin_I64 = DFTU_MONOID_ARGMIN_I64,
    ArgMax_I64 = DFTU_MONOID_ARGMAX_I64,
    ArgMin_Str = DFTU_MONOID_ARGMIN_STR,
    ArgMax_Str = DFTU_MONOID_ARGMAX_STR,
    Mean = DFTU_MONOID_MEAN,
    Variance = DFTU_MONOID_VARIANCE,
    Stddev = DFTU_MONOID_STDDEV,
    TopK_I64 = DFTU_MONOID_TOPK_I64,
    TopK_Str = DFTU_MONOID_TOPK_STR,
    BottomK_I64 = DFTU_MONOID_BOTTOMK_I64,
    BottomK_Str = DFTU_MONOID_BOTTOMK_STR,
    Approx_TopK_I64 = DFTU_MONOID_APPROX_TOPK_I64,
    Approx_TopK_Str = DFTU_MONOID_APPROX_TOPK_STR,
    Sample_I64 = DFTU_MONOID_SAMPLE_I64,
    Sample_Str = DFTU_MONOID_SAMPLE_STR,
    ArgMin_Row = DFTU_MONOID_ARGMIN_ROW,
    ArgMax_Row = DFTU_MONOID_ARGMAX_ROW,
    Skewness = DFTU_MONOID_SKEWNESS,
    Kurtosis = DFTU_MONOID_KURTOSIS,
    Corr = DFTU_MONOID_CORR,
    Covar_Pop = DFTU_MONOID_COVAR_POP,
    Covar_Samp = DFTU_MONOID_COVAR_SAMP,
    Regr_Slope = DFTU_MONOID_REGR_SLOPE,
    Regr_Intercept = DFTU_MONOID_REGR_INTERCEPT,
    Regr_R2 = DFTU_MONOID_REGR_R2
};

/// Scoped mirror of dftu_join_type for Host::map_declare_join; each enumerator
/// is its ABI constant, so static_cast<dftu_join_type> recovers the raw value.
enum class JoinType : std::int32_t {
    Inner = DFTU_JOIN_INNER,
    Left = DFTU_JOIN_LEFT,
    Right = DFTU_JOIN_RIGHT,
    Full = DFTU_JOIN_FULL
};

/// Scoped mirror of dftu_ver_op for Version::satisfies; each enumerator is its
/// ABI constant, so static_cast<dftu_ver_op> recovers the raw value.
enum class VerOp : std::int32_t {
    Ge = DFTU_VER_GE,
    Gt = DFTU_VER_GT,
    Le = DFTU_VER_LE,
    Lt = DFTU_VER_LT,
    Eq = DFTU_VER_EQ,
    Caret = DFTU_VER_CARET,
    Tilde = DFTU_VER_TILDE
};
static_assert(static_cast<dftu_ver_op>(VerOp::Ge) == DFTU_VER_GE);
static_assert(static_cast<dftu_ver_op>(VerOp::Gt) == DFTU_VER_GT);
static_assert(static_cast<dftu_ver_op>(VerOp::Le) == DFTU_VER_LE);
static_assert(static_cast<dftu_ver_op>(VerOp::Lt) == DFTU_VER_LT);
static_assert(static_cast<dftu_ver_op>(VerOp::Eq) == DFTU_VER_EQ);
static_assert(static_cast<dftu_ver_op>(VerOp::Caret) == DFTU_VER_CARET);
static_assert(static_cast<dftu_ver_op>(VerOp::Tilde) == DFTU_VER_TILDE);

/// Wraps a dftu_version (major.minor.patch). Comparisons and satisfies() defer
/// to the ABI's dftu_version_cmp / dftu_version_satisfies.
class Version {
   public:
    constexpr Version() noexcept = default;
    constexpr explicit Version(dftu_version v) noexcept : v_(v) {}
    constexpr Version(std::uint16_t major, std::uint16_t minor,
                      std::uint16_t patch) noexcept
        : v_{major, minor, patch} {}

    constexpr std::uint16_t major() const noexcept { return v_.major; }
    constexpr std::uint16_t minor() const noexcept { return v_.minor; }
    constexpr std::uint16_t patch() const noexcept { return v_.patch; }
    constexpr dftu_version raw() const noexcept { return v_; }

    friend bool operator==(Version a, Version b) noexcept {
        return dftu_version_cmp(a.v_, b.v_) == 0;
    }
    friend bool operator!=(Version a, Version b) noexcept { return !(a == b); }
    friend bool operator<(Version a, Version b) noexcept {
        return dftu_version_cmp(a.v_, b.v_) < 0;
    }
    friend bool operator<=(Version a, Version b) noexcept {
        return dftu_version_cmp(a.v_, b.v_) <= 0;
    }
    friend bool operator>(Version a, Version b) noexcept {
        return dftu_version_cmp(a.v_, b.v_) > 0;
    }
    friend bool operator>=(Version a, Version b) noexcept {
        return dftu_version_cmp(a.v_, b.v_) >= 0;
    }

    bool satisfies(dftu_ver_op op, Version want) const noexcept {
        return dftu_version_satisfies(v_, op, want.v_) != 0;
    }
    bool satisfies(VerOp op, Version want) const noexcept {
        return satisfies(static_cast<dftu_ver_op>(op), want);
    }

   private:
    dftu_version v_{};
};

/// Opaque strong handle for an interned-string id (the C ABI's dftu_str, which
/// is an id, not bytes). Compare and hash by identity; resolve to bytes with
/// Host::str. absent() is the DFTU_STR_NONE sentinel. raw() exposes the
/// underlying id only to cross the C ABI seam - it is never an integer to a
/// plugin author.
class StrId {
   public:
    StrId() = default;
    constexpr explicit StrId(dftu_str id) noexcept : id_(id) {}
    constexpr dftu_str raw() const noexcept { return id_; }
    constexpr bool absent() const noexcept { return id_ == DFTU_STR_NONE; }
    constexpr bool operator==(StrId o) const noexcept { return id_ == o.id_; }
    constexpr bool operator!=(StrId o) const noexcept { return id_ != o.id_; }

   private:
    dftu_str id_ = DFTU_STR_NONE;
};

/// A pre-interned STR/BYTES key slot: carries an already-interned id (e.g.
/// Event::name_id(), Event::fhash_id()) as a STR key component without
/// re-interning its bytes. Declare the slot's key type as Interned (in a Key
/// tag) and feed it an interned() value.
struct Interned {
    dftu_str id;
};
/// Wrap an already-interned id for use as a STR key slot; see Interned.
inline Interned interned(StrId id) noexcept { return Interned{id.raw()}; }

/// Compile-time key schema tag naming each key component's C++ type in order,
/// e.g. `Key<std::int64_t, std::string_view>{}` for a (pid, name) key.
template <class... Ts>
struct Key {};

/// Map a key/payload component's C++ type to the dftu_type the map ABI expects;
/// any string-like type is an interned STR column.
template <class T>
constexpr dftu_type type_of() {
    using U = std::remove_cv_t<std::remove_reference_t<T>>;
    if constexpr (std::is_same_v<U, Interned>)
        return DFTU_T_STR;
    else if constexpr (std::is_same_v<U, double>)
        return DFTU_T_F64;
    else if constexpr (std::is_same_v<U, float>)
        return DFTU_T_F32;
    else if constexpr (std::is_same_v<U, std::int8_t>)
        return DFTU_T_I8;
    else if constexpr (std::is_same_v<U, std::int16_t>)
        return DFTU_T_I16;
    else if constexpr (std::is_same_v<U, std::int32_t>)
        return DFTU_T_I32;
    else if constexpr (std::is_same_v<U, std::int64_t>)
        return DFTU_T_I64;
    else if constexpr (std::is_same_v<U, std::uint8_t>)
        return DFTU_T_U8;
    else if constexpr (std::is_same_v<U, std::uint16_t>)
        return DFTU_T_U16;
    else if constexpr (std::is_same_v<U, std::uint32_t>)
        return DFTU_T_U32;
    else if constexpr (std::is_same_v<U, std::uint64_t>)
        return DFTU_T_U64;
    else if constexpr (std::is_convertible_v<U, std::string_view>)
        return DFTU_T_STR;
    else
        static_assert(sizeof(U) == 0, "unsupported map key component type");
}

}  // namespace dftracer::utils::plugins

#endif /* DFTRACER_UTILS_PLUGINS_PLUGIN_TYPES_H */
