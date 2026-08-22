#ifndef DFTRACER_UTILS_PLUGINS_REFLECT_H
#define DFTRACER_UTILS_PLUGINS_REFLECT_H

// Uniform index-addressable view of T's public fields as (name, reference),
// fed either by a DFTU_REFLECT specialization or by pfr aggregate reflection.
// Host/generator-only; not part of the plugin SDK.

#include <cstddef>
#include <functional>
#include <pfr.hpp>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace dftracer::utils::plugins::reflect {

template <class MemberPtr>
struct Field {
    const char* name;
    MemberPtr ptr;
};

template <class C, class M>
constexpr Field<M C::*> make_field(const char* name, M C::* ptr) {
    return Field<M C::*>{name, ptr};
}

template <class T>
struct Reflect;

template <class T>
concept HasReflect = requires { Reflect<T>::fields; };

// DFTU_REFLECT(T, f1, ...): reflect a non-aggregate T at namespace scope.
#define DFTU_REFLECT_FIELD(T, f) \
    ::dftracer::utils::plugins::reflect::make_field(#f, &T::f)

#define DFTU_REFLECT_FE_1(T, f) DFTU_REFLECT_FIELD(T, f)
#define DFTU_REFLECT_FE_2(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_1(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_3(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_2(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_4(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_3(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_5(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_4(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_6(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_5(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_7(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_6(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_8(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_7(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_9(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_8(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_10(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_9(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_11(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_10(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_12(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_11(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_13(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_12(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_14(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_13(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_15(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_14(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_16(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_15(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_17(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_16(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_18(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_17(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_19(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_18(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_20(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_19(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_21(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_20(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_22(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_21(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_23(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_22(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_24(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_23(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_25(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_24(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_26(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_25(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_27(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_26(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_28(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_27(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_29(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_28(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_30(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_29(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_31(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_30(T, __VA_ARGS__)
#define DFTU_REFLECT_FE_32(T, f, ...) \
    DFTU_REFLECT_FIELD(T, f), DFTU_REFLECT_FE_31(T, __VA_ARGS__)

#define DFTU_REFLECT_PICK(_1, _2, _3, _4, _5, _6, _7, _8, _9, _10, _11, _12, \
                          _13, _14, _15, _16, _17, _18, _19, _20, _21, _22,  \
                          _23, _24, _25, _26, _27, _28, _29, _30, _31, _32,  \
                          NAME, ...)                                         \
    NAME
#define DFTU_REFLECT_FOR_EACH(T, ...)                               \
    DFTU_REFLECT_PICK(                                              \
        __VA_ARGS__, DFTU_REFLECT_FE_32, DFTU_REFLECT_FE_31,        \
        DFTU_REFLECT_FE_30, DFTU_REFLECT_FE_29, DFTU_REFLECT_FE_28, \
        DFTU_REFLECT_FE_27, DFTU_REFLECT_FE_26, DFTU_REFLECT_FE_25, \
        DFTU_REFLECT_FE_24, DFTU_REFLECT_FE_23, DFTU_REFLECT_FE_22, \
        DFTU_REFLECT_FE_21, DFTU_REFLECT_FE_20, DFTU_REFLECT_FE_19, \
        DFTU_REFLECT_FE_18, DFTU_REFLECT_FE_17, DFTU_REFLECT_FE_16, \
        DFTU_REFLECT_FE_15, DFTU_REFLECT_FE_14, DFTU_REFLECT_FE_13, \
        DFTU_REFLECT_FE_12, DFTU_REFLECT_FE_11, DFTU_REFLECT_FE_10, \
        DFTU_REFLECT_FE_9, DFTU_REFLECT_FE_8, DFTU_REFLECT_FE_7,    \
        DFTU_REFLECT_FE_6, DFTU_REFLECT_FE_5, DFTU_REFLECT_FE_4,    \
        DFTU_REFLECT_FE_3, DFTU_REFLECT_FE_2, DFTU_REFLECT_FE_1)    \
    (T, __VA_ARGS__)

// Define the specialization inside its namespace. gcc rejects defining a class
// through a leading-`::` qualified name (`struct ::a::b::Reflect<T> {...}`);
// reopening the namespace is the portable form. Used at namespace (file) scope.
#define DFTU_REFLECT(T, ...)                                          \
    namespace dftracer::utils::plugins::reflect {                     \
    template <>                                                       \
    struct Reflect<T> {                                               \
        static constexpr auto fields =                                \
            ::std::make_tuple(DFTU_REFLECT_FOR_EACH(T, __VA_ARGS__)); \
    };                                                                \
    }                                                                 \
    static_assert(true, "require trailing semicolon")

template <class T>
constexpr std::size_t nfields() {
    if constexpr (HasReflect<T>) {
        return std::tuple_size_v<
            std::remove_cv_t<decltype(Reflect<T>::fields)>>;
    } else if constexpr (std::is_aggregate_v<T>) {
        return pfr::tuple_size<T>::value;
    } else {
        static_assert(sizeof(T) == 0,
                      "type is neither DFTU_REFLECT'd nor an aggregate; "
                      "DFTU_REFLECT it centrally or keep the field set "
                      "first-party (aggregate)");
        return 0;
    }
}

template <class T, std::size_t I>
constexpr std::string_view field_name() {
    if constexpr (HasReflect<T>) {
        return std::get<I>(Reflect<T>::fields).name;
    } else {
        return pfr::get_name<I, T>();
    }
}

template <std::size_t I, class T>
constexpr decltype(auto) get_field(T& value) {
    if constexpr (HasReflect<std::remove_cv_t<T>>) {
        return value.*(std::get<I>(Reflect<std::remove_cv_t<T>>::fields).ptr);
    } else {
        return pfr::get<I>(value);
    }
}

template <class T, std::size_t I>
using field_type_t =
    std::remove_cvref_t<decltype(get_field<I>(std::declval<T&>()))>;

template <class T, class Fn>
void for_each_named_field(T& value, Fn&& fn) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (fn(field_name<std::remove_cv_t<T>, Is>(), get_field<Is>(value)), ...);
    }(std::make_index_sequence<nfields<std::remove_cv_t<T>>()>{});
}

/// Field routing: every reflected field is one of data / callback / context /
/// projection, decided by type. emit.h and the host marshaller share these so
/// the generated C struct and the marshalling agree field-for-field.

template <class T>
struct is_callback : std::false_type {};
template <class R, class... A>
struct is_callback<std::function<R(A...)>> : std::true_type {
    using ret = R;
    using arg0 = std::remove_cvref_t<std::tuple_element_t<0, std::tuple<A...>>>;
    static constexpr std::size_t arity = sizeof...(A);
};

/// A field the host owns and injects (CoroScope*/Executor*/StringIntern*, or a
/// reference); omitted from the plugin-facing C struct.
template <class F>
constexpr bool is_context_field =
    std::is_pointer_v<F> || std::is_reference_v<F>;

/// An accumulator projected to a POD on output (DDSketch-like or a
/// DistributionStats wrapping one); detected structurally to keep the heavy
/// statistics headers out of this reflection layer.
template <class F>
concept DDSketchLike = requires(const F& f) {
    { f.quantile(0.5) } -> std::convertible_to<double>;
    { f.count() };
};
template <class F>
concept DistributionStatsLike = requires(const F& f) {
    { f.sketch.quantile(0.5) } -> std::convertible_to<double>;
    { f.mean() } -> std::convertible_to<double>;
};
template <class F>
constexpr bool is_projected_field = DDSketchLike<F> || DistributionStatsLike<F>;

template <class F>
constexpr bool is_data_field =
    !is_callback<F>::value && !is_context_field<F> && !is_projected_field<F>;

/// Count of C-struct members a native field expands to: a callback is a
/// function pointer plus its userdata, a context field is injected (none), all
/// else is one.
template <class F>
constexpr std::size_t c_arity() {
    if constexpr (is_callback<F>::value)
        return 2;
    else if constexpr (is_context_field<F>)
        return 0;
    else
        return 1;
}

}  // namespace dftracer::utils::plugins::reflect

#endif  // DFTRACER_UTILS_PLUGINS_REFLECT_H
