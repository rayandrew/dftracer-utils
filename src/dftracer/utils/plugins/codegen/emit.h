#ifndef DFTRACER_UTILS_SRC_PLUGINS_CODEGEN_EMIT_H
#define DFTRACER_UTILS_SRC_PLUGINS_CODEGEN_EMIT_H

// Generator-only emitters that turn a reflected native type into plugin-facing
// C source text; nothing here runs in the host or a plugin.

#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/coro/async_generator.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/reflect.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <ostream>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace dftracer::utils::plugins::codegen {

template <class T>
constexpr std::string_view pretty() {
#if defined(__clang__) || defined(__GNUC__)
    return __PRETTY_FUNCTION__;
#else
#error "reflected codegen needs __PRETTY_FUNCTION__ (clang/gcc)"
#endif
}

template <class T>
std::string_view type_spelling() {
    std::string_view p = pretty<T>();
    std::size_t s = p.find("T = ");
    s = (s == std::string_view::npos) ? 0 : s + 4;
    std::size_t e = p.find_first_of(";]", s);
    std::string_view q =
        p.substr(s, (e == std::string_view::npos) ? p.size() - s : e - s);
    while (!q.empty() && q.back() == ' ') q.remove_suffix(1);
    return q;
}

// Fully-qualified spelling of T (namespaces kept), used as a c_of<> key.
template <class T>
std::string type_full_name() {
    return std::string(type_spelling<T>());
}

template <class T>
std::string type_leaf_name() {
    // std::string spells differently per compiler (clang: std::string; gcc:
    // std::__cxx11::basic_string<...>), which would make generated names like
    // dftu_span_string vs dftu_span_basic_string diverge. Canonicalize.
    if constexpr (std::is_same_v<T, std::string>)
        return "string";
    else if constexpr (std::is_same_v<T, std::string_view>)
        return "string_view";
    else {
        std::string_view q = type_spelling<T>();
        if (std::size_t lt = q.find('<'); lt != std::string_view::npos)
            q = q.substr(0, lt);
        if (std::size_t cc = q.rfind("::"); cc != std::string_view::npos)
            q = q.substr(cc + 2);
        return std::string(q);
    }
}

inline std::string to_snake(std::string_view name) {
    std::string out;
    out.reserve(name.size() + 4);
    for (std::size_t i = 0; i < name.size(); ++i) {
        char c = name[i];
        bool upper = c >= 'A' && c <= 'Z';
        if (upper && !out.empty()) {
            char prev = name[i - 1];
            bool prev_lower_or_digit =
                (prev >= 'a' && prev <= 'z') || (prev >= '0' && prev <= '9');
            bool next_lower =
                i + 1 < name.size() && name[i + 1] >= 'a' && name[i + 1] <= 'z';
            bool prev_upper = prev >= 'A' && prev <= 'Z';
            if (prev_lower_or_digit || (prev_upper && next_lower)) out += '_';
        }
        out += upper ? char(c - 'A' + 'a') : c;
    }
    return out;
}

template <class T>
struct is_std_vector : std::false_type {};
template <class E, class A>
struct is_std_vector<std::vector<E, A>> : std::true_type {
    using element = E;
};

// A filesystem path crosses as dftu_bytes (its string form), like a string.
template <class T>
constexpr bool is_bytes_like =
    std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string> ||
    std::is_same_v<T, fs::path>;

template <class T>
constexpr bool is_scalar_leaf =
    (std::is_arithmetic_v<T> || std::is_enum_v<T>) && !std::is_same_v<T, bool>;

template <class T>
std::string c_scalar_name() {
    if constexpr (std::is_same_v<T, bool>)
        return "uint8_t";
    else if constexpr (std::is_same_v<T, double>)
        return "double";
    else if constexpr (std::is_same_v<T, float>)
        return "float";
    else if constexpr (std::is_enum_v<T>)
        return c_scalar_name<std::underlying_type_t<T>>();
    else if constexpr (std::is_integral_v<T>) {
        // Map by width and signedness so platform aliases (size_t, long)
        // resolve without a per-type case.
        constexpr bool s = std::is_signed_v<T>;
        std::string base = s ? "int" : "uint";
        return base + std::to_string(sizeof(T) * 8) + "_t";
    } else
        static_assert(sizeof(T) == 0,
                      "no C scalar mapping for this field type");
}

template <class T>
std::string c_struct_name() {
    return "dftu_" + to_snake(type_leaf_name<T>());
}

// A native type that crosses the ABI as a generated dftu_ struct (not a scalar,
// bytes span, or a raw ABI POD spelled by its own name).
template <class T>
constexpr bool is_generated_struct =
    !is_bytes_like<T> && !is_scalar_leaf<T> && !std::is_same_v<T, bool> &&
    !std::is_same_v<T, dftu_bytes> && !std::is_same_v<T, dftu_hex16>;

// C spelling of a top-level IN/OUT/item type: a scalar or raw ABI POD by its
// direct C name, a reflected struct by its generated name.
template <class T>
std::string c_toplevel_name() {
    if constexpr (std::is_same_v<T, dftu_hex16>)
        return "dftu_hex16";
    else if constexpr (std::is_same_v<T, dftu_bytes> || is_bytes_like<T>)
        return "dftu_bytes";
    else if constexpr (is_scalar_leaf<T> || std::is_same_v<T, bool>)
        return c_scalar_name<T>();
    else
        return c_struct_name<T>();
}

// OUT shape resolution: AsyncGenerator<E>/vector<E> is a stream of E,
// optional<E> a single value that may be absent, anything else a single value.
template <class T>
struct stream_of : std::false_type {};
template <class E>
struct stream_of<::dftracer::utils::coro::AsyncGenerator<E>> : std::true_type {
    using elem = E;
};
template <class E, class A>
struct stream_of<std::vector<E, A>> : std::true_type {
    using elem = E;
};

template <class T>
struct single_elem {
    using type = T;
};
template <class E>
struct single_elem<std::optional<E>> {
    using type = E;
};

template <class F>
std::string c_member_type() {
    if constexpr (is_bytes_like<F>)
        return "dftu_bytes";
    else if constexpr (is_std_vector<F>::value)
        return "dftu_span_" +
               to_snake(type_leaf_name<typename F::value_type>());
    else if constexpr (is_scalar_leaf<F> || std::is_same_v<F, bool>)
        return c_scalar_name<F>();
    else
        return c_struct_name<F>();
}

template <class T>
std::string c_ret_name() {
    if constexpr (std::is_same_v<T, bool>)
        return "uint8_t";
    else if constexpr (std::is_void_v<T>)
        return "void";
    else
        return c_scalar_name<T>();
}

// Emit the C struct member(s) a native field F expands to: a callback becomes a
// function pointer plus its userdata, a context field is dropped, a projected
// accumulator becomes dftu_quantiles, everything else is one typed member.
template <class F>
void emit_c_members(std::ostream& os, const std::string& fname) {
    if constexpr (reflect::is_callback<F>::value) {
        using Cb = reflect::is_callback<F>;
        static_assert(Cb::arity == 1,
                      "plugin callback fields must take exactly one argument");
        os << "    " << c_ret_name<typename Cb::ret>() << " (*" << fname
           << ")(const " << c_struct_name<typename Cb::arg0>()
           << "*, void*);\n";
        os << "    void* " << fname << "_ud;\n";
    } else if constexpr (reflect::is_context_field<F>) {
        (void)fname;
    } else if constexpr (reflect::is_projected_field<F>) {
        os << "    dftu_quantiles " << fname << ";\n";
    } else {
        os << "    " << c_member_type<F>() << " " << fname << ";\n";
    }
}

template <class T, class Fn>
void for_each_field(Fn&& fn) {
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (fn.template operator()<reflect::field_type_t<T, Is>>(
             std::string(reflect::field_name<T, Is>())),
         ...);
    }(std::make_index_sequence<reflect::nfields<T>()>{});
}

template <class E>
void emit_span_struct(std::ostream& os, std::set<std::string>& visited) {
    std::string name = "dftu_span_" + to_snake(type_leaf_name<E>());
    if (!visited.insert(name).second) return;
    std::string elem;
    if constexpr (is_bytes_like<E>)
        elem = "dftu_bytes";
    else if constexpr (is_scalar_leaf<E> || std::is_same_v<E, bool>)
        elem = c_scalar_name<E>();
    else
        elem = c_struct_name<E>();
    os << "typedef struct {\n";
    os << "    const " << elem << "* ptr;\n";
    os << "    uint32_t len;\n";
    os << "} " << name << ";\n\n";
}

template <class T>
void emit_c_struct(std::ostream& os, std::set<std::string>& visited) {
    std::string name = c_struct_name<T>();
    if (visited.count(name)) return;

    // Dependencies must be emitted before the struct that uses them.
    for_each_field<T>([&]<class F>(const std::string&) {
        if constexpr (reflect::is_callback<F>::value) {
            emit_c_struct<typename reflect::is_callback<F>::arg0>(os, visited);
        } else if constexpr (reflect::is_context_field<F> ||
                             reflect::is_projected_field<F>) {
            // injected or projected: no nested C struct
        } else if constexpr (is_std_vector<F>::value) {
            using E = typename F::value_type;
            if constexpr (!(is_scalar_leaf<E> || std::is_same_v<E, bool> ||
                            is_bytes_like<E>)) {
                emit_c_struct<E>(os, visited);
            }
            emit_span_struct<E>(os, visited);
        } else if constexpr (!(is_scalar_leaf<F> || std::is_same_v<F, bool> ||
                               is_bytes_like<F>)) {
            emit_c_struct<F>(os, visited);
        }
    });

    if (!visited.insert(name).second) return;
    os << "typedef struct {\n";
    for_each_field<T>([&]<class F>(const std::string& fname) {
        emit_c_members<F>(os, fname);
    });
    os << "} " << name << ";\n\n";
}

template <class In, class Out>
void emit_wrapper(std::ostream& enum_os, std::ostream& wrap_os,
                  std::ostream& tag_os, const std::string& tag,
                  const std::string& name, std::size_t ordinal) {
    std::string in_c = c_toplevel_name<In>();
    std::string out_c = c_toplevel_name<Out>();

    enum_os << "    DFTU_UTIL_" << tag << " = " << ordinal << ",\n";

    wrap_os << "static inline int dftu_util_" << name
            << "(const dftu_host* h, const " << in_c << "* in, " << out_c
            << "* out) {\n";
    wrap_os << "    const dftu_ext_util* e = (const dftu_ext_util*)(\n";
    wrap_os
        << "        h->get_extension ? h->get_extension(h->h, DFTU_EXT_UTIL) "
           ": 0);\n";
    wrap_os << "    const dftu_utility* u =\n";
    wrap_os << "        (e && e->find_by_id) ? e->find_by_id(h->h, "
               "(uint32_t)DFTU_UTIL_"
            << tag << ") : 0;\n";
    wrap_os << "    if (!u) return -1;\n";
    wrap_os << "    return u->run(u->self, in, out);\n";
    wrap_os << "}\n\n";

    tag_os << "    struct " << name << " {\n";
    tag_os << "        using in = " << in_c << ";\n";
    tag_os << "        using out = " << out_c << ";\n";
    tag_os << "        static constexpr uint32_t id = DFTU_UTIL_" << tag
           << ";\n";
    tag_os << "    };\n";
}

// Streaming counterpart of emit_wrapper: the wrapper drives run_stream and the
// plugin gets each item via on_item. `Item` is the per-yield output type.
template <class In, class Item>
void emit_stream_wrapper(std::ostream& enum_os, std::ostream& wrap_os,
                         std::ostream& tag_os, const std::string& tag,
                         const std::string& name, std::size_t ordinal) {
    std::string in_c = c_toplevel_name<In>();
    std::string item_c = c_toplevel_name<Item>();

    enum_os << "    DFTU_UTIL_" << tag << " = " << ordinal << ",\n";

    wrap_os << "static inline int dftu_util_" << name
            << "_stream(const dftu_host* h, const " << in_c
            << "* in, dftu_stream_item_fn on_item, void* ud) {\n";
    wrap_os << "    const dftu_ext_util* e = (const dftu_ext_util*)(\n";
    wrap_os
        << "        h->get_extension ? h->get_extension(h->h, DFTU_EXT_UTIL) "
           ": 0);\n";
    wrap_os << "    return (e && e->run_stream) ? e->run_stream(h->h, "
               "(uint32_t)DFTU_UTIL_"
            << tag << ", in, on_item, ud) : -1;\n";
    wrap_os << "}\n\n";

    tag_os << "    struct " << name << " {\n";
    tag_os << "        using in = " << in_c << ";\n";
    tag_os << "        using item = " << item_c << ";\n";
    tag_os << "        static constexpr uint32_t id = DFTU_UTIL_" << tag
           << ";\n";
    tag_os << "    };\n";
}

// Emit the host-only native<->C type binding (c_of<Native>::type) consumed by
// the marshalling in plugin_exports.cpp, only for reflected structs (scalars
// carry a direct binding host-side); deduped by native name via `seen`.
template <class Native>
void emit_marshal_one(std::ostream& os, std::set<std::string>& seen) {
    if constexpr (is_generated_struct<Native>) {
        std::string native = type_full_name<Native>();
        if (seen.insert(native).second)
            os << "    template <> struct c_of<" << native
               << "> { using type = " << c_struct_name<Native>() << "; };\n";
    }
}

// Emit a generated dftu_ struct for a reflected native type; a scalar or raw
// ABI POD has no struct to emit.
template <class T>
void emit_toplevel_struct(std::ostream& os, std::set<std::string>& visited) {
    if constexpr (is_generated_struct<T>) emit_c_struct<T>(os, visited);
}

// One export entry: resolve OUT's shape, emit its C structs, wrapper, tag, and
// (struct-only) marshalling binding.
template <class In, class Out>
void emit_export(std::ostream& structs, std::set<std::string>& visited,
                 std::ostream& enums, std::ostream& wrappers,
                 std::ostream& tags, std::ostream& marshal,
                 std::set<std::string>& marshal_seen, const std::string& tag,
                 const std::string& name, std::size_t ordinal) {
    emit_toplevel_struct<In>(structs, visited);
    emit_marshal_one<In>(marshal, marshal_seen);
    if constexpr (stream_of<Out>::value) {
        using E = typename stream_of<Out>::elem;
        emit_toplevel_struct<E>(structs, visited);
        emit_stream_wrapper<In, E>(enums, wrappers, tags, tag, name, ordinal);
        emit_marshal_one<E>(marshal, marshal_seen);
    } else {
        using E = typename single_elem<Out>::type;
        emit_toplevel_struct<E>(structs, visited);
        emit_wrapper<In, E>(enums, wrappers, tags, tag, name, ordinal);
        emit_marshal_one<E>(marshal, marshal_seen);
    }
}

}  // namespace dftracer::utils::plugins::codegen

#endif  // DFTRACER_UTILS_SRC_PLUGINS_CODEGEN_EMIT_H
