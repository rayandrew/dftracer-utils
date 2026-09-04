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

/// Scoped mirror of dftu_phase for Event::phase; each enumerator is its ABI
/// constant, so static_cast<dftu_phase> recovers the raw value.
enum class Phase : std::int32_t {
    Unknown = DFTU_PH_UNKNOWN,
    Complete = DFTU_PH_COMPLETE,
    Counter = DFTU_PH_COUNTER,
    Aggregated = DFTU_PH_AGGREGATED,
    Metadata = DFTU_PH_METADATA
};
static_assert(static_cast<dftu_phase>(Phase::Unknown) == DFTU_PH_UNKNOWN);
static_assert(static_cast<dftu_phase>(Phase::Complete) == DFTU_PH_COMPLETE);
static_assert(static_cast<dftu_phase>(Phase::Counter) == DFTU_PH_COUNTER);
static_assert(static_cast<dftu_phase>(Phase::Aggregated) == DFTU_PH_AGGREGATED);
static_assert(static_cast<dftu_phase>(Phase::Metadata) == DFTU_PH_METADATA);

/// Scoped mirror of dftu_log_level for Host::log; each enumerator is its ABI
/// constant, so static_cast<dftu_log_level> recovers the raw value.
enum class LogLevel : std::int32_t {
    Trace = DFTU_LOG_TRACE,
    Debug = DFTU_LOG_DEBUG,
    Info = DFTU_LOG_INFO,
    Warn = DFTU_LOG_WARN,
    Error = DFTU_LOG_ERROR
};
static_assert(static_cast<dftu_log_level>(LogLevel::Trace) == DFTU_LOG_TRACE);
static_assert(static_cast<dftu_log_level>(LogLevel::Debug) == DFTU_LOG_DEBUG);
static_assert(static_cast<dftu_log_level>(LogLevel::Info) == DFTU_LOG_INFO);
static_assert(static_cast<dftu_log_level>(LogLevel::Warn) == DFTU_LOG_WARN);
static_assert(static_cast<dftu_log_level>(LogLevel::Error) == DFTU_LOG_ERROR);

/// Scoped mirror of dftu_arg_kind for Arg::kind; each enumerator is its ABI
/// constant, so static_cast<dftu_arg_kind> recovers the raw value.
enum class ArgKind : std::int32_t {
    F64 = DFTU_ARG_F64,
    I64 = DFTU_ARG_I64,
    Str = DFTU_ARG_STR
};
static_assert(static_cast<dftu_arg_kind>(ArgKind::F64) == DFTU_ARG_F64);
static_assert(static_cast<dftu_arg_kind>(ArgKind::I64) == DFTU_ARG_I64);
static_assert(static_cast<dftu_arg_kind>(ArgKind::Str) == DFTU_ARG_STR);

/// Scoped mirror of dftu_value_kind for ConfigValue::kind; each enumerator is
/// its ABI constant, so static_cast<dftu_value_kind> recovers the raw value.
enum class ValueKind : std::int32_t {
    Null = DFTU_VAL_NULL,
    Bool = DFTU_VAL_BOOL,
    I64 = DFTU_VAL_I64,
    F64 = DFTU_VAL_F64,
    Str = DFTU_VAL_STR,
    Array = DFTU_VAL_ARRAY,
    Object = DFTU_VAL_OBJECT
};
static_assert(static_cast<dftu_value_kind>(ValueKind::Null) == DFTU_VAL_NULL);
static_assert(static_cast<dftu_value_kind>(ValueKind::Bool) == DFTU_VAL_BOOL);
static_assert(static_cast<dftu_value_kind>(ValueKind::I64) == DFTU_VAL_I64);
static_assert(static_cast<dftu_value_kind>(ValueKind::F64) == DFTU_VAL_F64);
static_assert(static_cast<dftu_value_kind>(ValueKind::Str) == DFTU_VAL_STR);
static_assert(static_cast<dftu_value_kind>(ValueKind::Array) == DFTU_VAL_ARRAY);
static_assert(static_cast<dftu_value_kind>(ValueKind::Object) ==
              DFTU_VAL_OBJECT);

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

/// Read view over a dftu_capability (built with capability()): a namespaced id
/// plus semantic version, as returned to an author from a provider listing.
class Capability {
   public:
    Capability() = default;
    constexpr explicit Capability(dftu_capability c) noexcept : c_(c) {}

    std::string_view id() const noexcept { return c_.id ? c_.id : ""; }
    Version version() const noexcept { return Version{c_.ver}; }
    dftu_capability raw() const noexcept { return c_; }

   private:
    dftu_capability c_{};
};

/// Read view over a dftu_requirement (built with requirement()): an id, a
/// version constraint, and whether it is load-blocking.
class Requirement {
   public:
    Requirement() = default;
    constexpr explicit Requirement(dftu_requirement r) noexcept : r_(r) {}

    std::string_view id() const noexcept { return r_.id ? r_.id : ""; }
    VerOp op() const noexcept { return static_cast<VerOp>(r_.op); }
    Version version() const noexcept { return Version{r_.ver}; }
    bool required() const noexcept { return r_.required != 0; }
    dftu_requirement raw() const noexcept { return r_; }

   private:
    dftu_requirement r_{};
};

/// Scoped mirror of dftu_agg_op for AggCol/Host::agg; each enumerator is its
/// DFTU_AGG_* constant, so static_cast<dftu_agg_op> recovers the raw value.
/// See abi.h/dftu_agg_col for each op's semantics and which of
/// value/out/param/by it consumes.
enum class AggOp : std::int32_t {
    Count = DFTU_AGG_COUNT,
    Sum = DFTU_AGG_SUM,
    Min = DFTU_AGG_MIN,
    Max = DFTU_AGG_MAX,
    Mean = DFTU_AGG_MEAN,
    Var = DFTU_AGG_VAR,
    Std = DFTU_AGG_STD,
    Skew = DFTU_AGG_SKEW,
    Kurt = DFTU_AGG_KURT,
    First = DFTU_AGG_FIRST,
    Last = DFTU_AGG_LAST,
    Pct = DFTU_AGG_PCT,
    Hist = DFTU_AGG_HIST,
    Argmax = DFTU_AGG_ARGMAX,
    Sumsq = DFTU_AGG_SUMSQ,
    SetUnion = DFTU_AGG_SET_UNION,
    Busy = DFTU_AGG_BUSY,
    Concurrency = DFTU_AGG_CONCURRENCY,
    Utilization = DFTU_AGG_UTILIZATION,
    Active = DFTU_AGG_ACTIVE,
    CountValid = DFTU_AGG_COUNT_VALID
};
static_assert(static_cast<dftu_agg_op>(AggOp::Count) == DFTU_AGG_COUNT);
static_assert(static_cast<dftu_agg_op>(AggOp::Sum) == DFTU_AGG_SUM);
static_assert(static_cast<dftu_agg_op>(AggOp::Min) == DFTU_AGG_MIN);
static_assert(static_cast<dftu_agg_op>(AggOp::Max) == DFTU_AGG_MAX);
static_assert(static_cast<dftu_agg_op>(AggOp::Mean) == DFTU_AGG_MEAN);
static_assert(static_cast<dftu_agg_op>(AggOp::Var) == DFTU_AGG_VAR);
static_assert(static_cast<dftu_agg_op>(AggOp::Std) == DFTU_AGG_STD);
static_assert(static_cast<dftu_agg_op>(AggOp::Skew) == DFTU_AGG_SKEW);
static_assert(static_cast<dftu_agg_op>(AggOp::Kurt) == DFTU_AGG_KURT);
static_assert(static_cast<dftu_agg_op>(AggOp::First) == DFTU_AGG_FIRST);
static_assert(static_cast<dftu_agg_op>(AggOp::Last) == DFTU_AGG_LAST);
static_assert(static_cast<dftu_agg_op>(AggOp::Pct) == DFTU_AGG_PCT);
static_assert(static_cast<dftu_agg_op>(AggOp::Hist) == DFTU_AGG_HIST);
static_assert(static_cast<dftu_agg_op>(AggOp::Argmax) == DFTU_AGG_ARGMAX);
static_assert(static_cast<dftu_agg_op>(AggOp::Sumsq) == DFTU_AGG_SUMSQ);
static_assert(static_cast<dftu_agg_op>(AggOp::SetUnion) == DFTU_AGG_SET_UNION);
static_assert(static_cast<dftu_agg_op>(AggOp::Busy) == DFTU_AGG_BUSY);
static_assert(static_cast<dftu_agg_op>(AggOp::Concurrency) ==
              DFTU_AGG_CONCURRENCY);
static_assert(static_cast<dftu_agg_op>(AggOp::Utilization) ==
              DFTU_AGG_UTILIZATION);
static_assert(static_cast<dftu_agg_op>(AggOp::Active) == DFTU_AGG_ACTIVE);
static_assert(static_cast<dftu_agg_op>(AggOp::CountValid) ==
              DFTU_AGG_COUNT_VALID);

/// Fluent builder for one dftu_agg_col spec passed to Host::agg. All names
/// (value/out/by) are borrowed const char* held only for the agg_new call, per
/// the SDK's borrowed-name convention; they must outlive that call.
class AggCol {
   public:
    explicit AggCol(AggOp op) noexcept : op_(op) {}

    AggCol& value(const char* v) noexcept {
        value_ = v;
        return *this;
    }
    AggCol& out(const char* o) noexcept {
        out_ = o;
        return *this;
    }
    AggCol& param(double p) noexcept {
        param_ = p;
        return *this;
    }
    AggCol& by(const char* b) noexcept {
        by_ = b;
        return *this;
    }

    dftu_agg_col raw() const noexcept {
        return dftu_agg_col{static_cast<std::int32_t>(op_), value_, out_,
                            param_, by_};
    }

   private:
    AggOp op_;
    const char* value_ = nullptr;
    const char* out_ = nullptr;
    double param_ = 0.0;
    const char* by_ = nullptr;
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
