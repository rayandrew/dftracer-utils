// Host side of the reflected utility path: marshals the plugin's C structs to
// the native I/O types and back, and runs the utility on a host Runtime.

#define DFTU_HOST_MARSHAL
#include <dftracer/utils/plugins/plugin_exports.h>
// Must stay after plugin_exports.h: the DFTU_HOST_MARSHAL section names native
// types that plugin_exports.h declares.
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/plugins/dftu_generated_utilities.h>
#include <dftracer/utils/plugins/utility_registry.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dftracer::utils::plugins {

namespace {

namespace reflect = dftracer::utils::plugins::reflect;
namespace abi_map = dftracer::utils::plugins::abi_map;

// A span/string output borrows into this arena, valid until the next reset.
struct Arena {
    std::vector<std::unique_ptr<char[]>> blocks;
    void reset() { blocks.clear(); }
    void* alloc(std::size_t n) {
        blocks.push_back(std::make_unique<char[]>(n ? n : 1));
        return blocks.back().get();
    }
};

template <class T>
struct is_std_vector : std::false_type {};
template <class E, class A>
struct is_std_vector<std::vector<E, A>> : std::true_type {
    using element = E;
};

template <class T>
constexpr bool is_str_like =
    std::is_same_v<T, std::string_view> || std::is_same_v<T, std::string>;

template <class T>
struct is_optional_t : std::false_type {};
template <class U>
struct is_optional_t<std::optional<U>> : std::true_type {};

template <class T>
concept HasNext = requires(T& t) { t.next(); };
template <class T>
concept HasError = requires(const T& t) { t.error(); };

template <class Native, class CStruct>
void marshal_c_to_native(const CStruct& c, Native& n);
template <class Native, class CStruct>
void marshal_native_to_c(const Native& n, CStruct& c, Arena& arena);

template <class CF, class NF>
void field_c_to_native(const CF& c, NF& n) {
    if constexpr (std::is_same_v<NF, dftu_bytes> ||
                  std::is_same_v<NF, dftu_hex16>) {
        n = c;
    } else if constexpr (std::is_arithmetic_v<NF> || std::is_enum_v<NF>) {
        n = static_cast<NF>(c);
    } else if constexpr (std::is_same_v<NF, std::string_view>) {
        n = std::string_view{c.ptr, c.len};
    } else if constexpr (std::is_same_v<NF, std::string>) {
        n.assign(c.ptr, c.len);
    } else if constexpr (std::is_same_v<NF, fs::path>) {
        n = fs::path(std::string(c.ptr, c.len));
    } else if constexpr (is_std_vector<NF>::value) {
        n.clear();
        n.reserve(c.len);
        for (std::uint32_t i = 0; i < c.len; ++i) {
            typename NF::value_type e{};
            field_c_to_native(c.ptr[i], e);
            n.push_back(std::move(e));
        }
    } else {
        marshal_c_to_native(c, n);
    }
}

template <class NF, class CF>
void field_native_to_c(const NF& n, CF& c, Arena& arena) {
    if constexpr (std::is_same_v<NF, dftu_bytes> ||
                  std::is_same_v<NF, dftu_hex16>) {
        c = n;
    } else if constexpr (std::is_arithmetic_v<NF> || std::is_enum_v<NF>) {
        c = static_cast<CF>(n);
    } else if constexpr (is_str_like<NF> || std::is_same_v<NF, fs::path>) {
        std::string s;
        if constexpr (std::is_same_v<NF, fs::path>)
            s = n.string();
        else
            s.assign(std::string_view{n});
        char* p = static_cast<char*>(arena.alloc(s.size()));
        std::memcpy(p, s.data(), s.size());
        c.ptr = p;
        c.len = static_cast<std::uint32_t>(s.size());
    } else if constexpr (is_std_vector<NF>::value) {
        using CE = std::remove_cv_t<std::remove_pointer_t<decltype(c.ptr)>>;
        CE* arr = static_cast<CE*>(arena.alloc(sizeof(CE) * n.size()));
        for (std::size_t i = 0; i < n.size(); ++i) {
            arr[i] = CE{};
            field_native_to_c(n[i], arr[i], arena);
        }
        c.ptr = arr;
        c.len = static_cast<std::uint32_t>(n.size());
    } else {
        marshal_native_to_c(n, c, arena);
    }
}

// A mergeable accumulator crosses the ABI as a dftu_quantiles POD.
template <class F>
void project_sketch(const F& f, dftu_quantiles& q) {
    q = dftu_quantiles{};
    if constexpr (reflect::DistributionStatsLike<F>) {
        const auto& s = f.sketch;
        q.count = s.count();
        q.mean = f.mean();
        if (s.count()) {
            q.min = s.min();
            q.max = s.max();
            q.p50 = s.quantile(0.5);
            q.p90 = s.quantile(0.9);
            q.p95 = s.quantile(0.95);
            q.p99 = s.quantile(0.99);
        }
    } else {
        q.count = f.count();
        if (f.count()) {
            q.min = f.min();
            q.max = f.max();
            q.p50 = f.quantile(0.5);
            q.p90 = f.quantile(0.9);
            q.p95 = f.quantile(0.95);
            q.p99 = f.quantile(0.99);
        }
    }
}

// Wrap a plugin C callback into a native std::function, marshalling the
// argument native->C on each invocation.
template <class F, class CFn>
void wrap_callback(CFn fn, void* ud, F& out) {
    using Cb = reflect::is_callback<F>;
    using Arg = typename Cb::arg0;
    using R = typename Cb::ret;
    using CArg = typename abi_map::c_of<Arg>::type;
    if (!fn) {
        out = nullptr;
        return;
    }
    out = [fn, ud](const Arg& a) -> R {
        Arena tmp;
        CArg ca{};
        marshal_native_to_c(a, ca, tmp);
        return static_cast<R>(fn(&ca, ud));
    };
}

// C-struct member index at which native field I begins (callback spans 2 C
// members, injected context 0, else 1).
template <class Native, std::size_t I>
constexpr std::size_t c_base() {
    std::size_t b = 0;
    [&]<std::size_t... J>(std::index_sequence<J...>) {
        ((b += reflect::c_arity<reflect::field_type_t<Native, J>>()), ...);
    }(std::make_index_sequence<I>{});
    return b;
}

template <class Native, std::size_t I, class CStruct>
void c_to_native_field(const CStruct& c, Native& n) {
    using F = reflect::field_type_t<Native, I>;
    constexpr std::size_t base = c_base<Native, I>();
    if constexpr (reflect::is_callback<F>::value) {
        wrap_callback<F>(pfr::get<base>(c), pfr::get<base + 1>(c),
                         reflect::get_field<I>(n));
    } else if constexpr (reflect::is_context_field<F>) {
        // Host-owned; the utility keeps its default (nullptr/current scope).
    } else if constexpr (reflect::is_projected_field<F>) {
        static_assert(sizeof(F) == 0,
                      "accumulator field cannot be a plugin input");
    } else {
        field_c_to_native(pfr::get<base>(c), reflect::get_field<I>(n));
    }
}

template <class Native, std::size_t I, class CStruct>
void native_to_c_field(const Native& n, CStruct& c, Arena& arena) {
    using F = reflect::field_type_t<Native, I>;
    constexpr std::size_t base = c_base<Native, I>();
    if constexpr (reflect::is_callback<F>::value) {
        static_assert(sizeof(F) == 0,
                      "callback field cannot be a plugin output");
    } else if constexpr (reflect::is_context_field<F>) {
        // Dropped from the plugin-facing struct.
    } else if constexpr (reflect::is_projected_field<F>) {
        project_sketch(reflect::get_field<I>(n), pfr::get<base>(c));
    } else {
        field_native_to_c(reflect::get_field<I>(n), pfr::get<base>(c), arena);
    }
}

template <class Native, class CStruct>
void marshal_c_to_native(const CStruct& c, Native& n) {
    constexpr std::size_t N = reflect::nfields<Native>();
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (c_to_native_field<Native, Is>(c, n), ...);
    }(std::make_index_sequence<N>{});
}

template <class Native, class CStruct>
void marshal_native_to_c(const Native& n, CStruct& c, Arena& arena) {
    constexpr std::size_t N = reflect::nfields<Native>();
    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
        (native_to_c_field<Native, Is>(n, c, arena), ...);
    }(std::make_index_sequence<N>{});
}

template <class T, class In>
concept HasProcess = requires(T t, const In& in) { t.process(in); };

// Invoke an exported invocable uniformly: a Utility subclass via process(),
// a plain op via operator(). Lets both authoring styles cross the plugin ABI.
template <class Util, class In>
auto invoke_util(Util& util, const In& in) {
    if constexpr (HasProcess<Util, In>)
        return util.process(in);
    else
        return util(in);
}

template <class T>
struct is_coro_task : std::false_type {};
template <class R>
struct is_coro_task<coro::CoroTask<R>> : std::true_type {};

// True when the invocable produces a CoroTask (a Utility via process(), or an
// async op via operator()), so it must be driven; a plain functor yields its
// output directly and is called inline.
template <class Util, class In>
inline constexpr bool yields_task_v =
    is_coro_task<std::remove_cvref_t<decltype(invoke_util(
        std::declval<Util&>(), std::declval<const In&>()))>>::value;

// Drive a utility to completion, calling `sink` once per produced item.
// Returns -1 when a Result item is an error.
template <class Util, class In, class Sink>
coro::CoroTask<int> drive_util(const In& in, Sink sink) {
    Util util;
    auto prod = invoke_util(util, in);
    using P = std::remove_cvref_t<decltype(prod)>;
    if constexpr (HasNext<P>) {
        while (auto item = co_await prod.next()) sink(*item);
        co_return 0;
    } else {
        auto out = co_await std::move(prod);
        using O = std::remove_cvref_t<decltype(out)>;
        if constexpr (is_optional_t<O>::value) {
            if (out) sink(*out);
            co_return 0;
        } else if constexpr (is_std_vector<O>::value) {
            for (const auto& item : out) sink(item);
            co_return 0;
        } else if constexpr (HasError<O>) {
            if (!out) co_return -1;
            sink(*out);
            co_return 0;
        } else {
            sink(out);
            co_return 0;
        }
    }
}

// Drive the utility to completion and block. run_blocking offloads when called
// on a worker thread, so parking the caller cannot starve the executor that
// must complete the utility.
template <class Util, class In, class Sink>
int run_util_blocking(const In& in, Sink&& sink) {
    int rc = 0;
    dftracer::utils::default_runtime().run_blocking(
        "dft-plugin-util", [&](CoroScope&) -> coro::CoroTask<void> {
            rc = co_await drive_util<Util>(in, sink);
        });
    return rc;
}

// Top-level IN/OUT routing: a scalar or raw ABI POD crosses as its direct C
// type, a reflected struct via the generated abi_map::c_of binding.
template <class T>
constexpr bool is_abi_direct =
    is_str_like<T> || std::is_same_v<T, fs::path> || std::is_arithmetic_v<T> ||
    std::is_enum_v<T> || std::is_same_v<T, dftu_bytes> ||
    std::is_same_v<T, dftu_hex16>;

template <class N>
struct direct_c {
    using type = N;
};
template <>
struct direct_c<std::string_view> {
    using type = dftu_bytes;
};
template <>
struct direct_c<std::string> {
    using type = dftu_bytes;
};
template <>
struct direct_c<fs::path> {
    using type = dftu_bytes;
};
template <>
struct direct_c<bool> {
    using type = std::uint8_t;
};

template <class N, bool Direct = is_abi_direct<N>>
struct c_repr;
template <class N>
struct c_repr<N, true> {
    using type = typename direct_c<N>::type;
};
template <class N>
struct c_repr<N, false> {
    using type = typename abi_map::c_of<N>::type;
};
template <class N>
using c_repr_t = typename c_repr<N>::type;

template <class N, class C>
N read_top(const C& c) {
    N n{};
    if constexpr (is_abi_direct<N>)
        field_c_to_native(c, n);
    else
        marshal_c_to_native(c, n);
    return n;
}

template <class N, class C>
void write_top(const N& n, C& c, Arena& arena) {
    if constexpr (is_abi_direct<N>)
        field_native_to_c(n, c, arena);
    else
        marshal_native_to_c(n, c, arena);
}

template <class T>
struct stream_of : std::false_type {};
template <class E>
struct stream_of<coro::AsyncGenerator<E>> : std::true_type {
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

template <class T>
constexpr dftu_type tag_of() {
    if constexpr (is_str_like<T> || std::is_same_v<T, fs::path> ||
                  std::is_same_v<T, dftu_hex16>)
        return DFTU_T_BYTES;
    else if constexpr (std::is_same_v<T, double>)
        return DFTU_T_F64;
    else if constexpr ((std::is_integral_v<T> && !std::is_same_v<T, bool>) ||
                       std::is_enum_v<T>)
        return DFTU_T_I64;
    else
        return DFTU_T_TABLE;
}

// Single-value run: drive a Utility to one item, or call a stateless functor;
// a nullopt result yields no item and run returns -1.
template <class Invocable, class In, class Out>
int single_thunk(void* /*self*/, const void* in, void* out) {
    using CIn = c_repr_t<In>;
    using COut = c_repr_t<Out>;
    try {
        In native_in = read_top<In>(*static_cast<const CIn*>(in));
        // Borrowed output fields point into this arena; thread_local so they
        // stay valid until the next run() on this thread. The utility runs on a
        // worker, so capture the arena by an address taken on THIS (caller)
        // thread: a `&`-captured thread_local would re-resolve to the worker's
        // instance, which no thread ever frees (leak).
        thread_local Arena arena;
        arena.reset();
        Arena* const arena_ptr = &arena;
        bool got = false;
        auto emit = [out, arena_ptr, &got](const Out& o) {
            write_top<Out>(o, *static_cast<COut*>(out), *arena_ptr);
            got = true;
        };
        int rc = 0;
        if constexpr (yields_task_v<Invocable, In>) {
            rc = run_util_blocking<Invocable>(native_in, emit);
        } else if constexpr (is_optional_t<
                                 std::invoke_result_t<Invocable, In>>::value) {
            if (auto r = Invocable{}(native_in)) emit(*r);
        } else {
            emit(Invocable{}(native_in));
        }
        return (rc == 0 && got) ? 0 : -1;
    } catch (...) {
        return -1;
    }
}

// Async mirror of single_thunk: co_await the utility on the current executor so
// the caller's coroutine is never blocked. Output borrows into a thread_local
// arena valid until the next run on this thread.
template <class Invocable, class In, class Out>
coro::CoroTask<int> run_async_thunk(const void* in, void* out) {
    using CIn = c_repr_t<In>;
    using COut = c_repr_t<Out>;
    thread_local Arena arena;
    arena.reset();
    try {
        In native_in = read_top<In>(*static_cast<const CIn*>(in));
        bool got = false;
        auto emit = [&](const Out& o) {
            write_top<Out>(o, *static_cast<COut*>(out), arena);
            got = true;
        };
        int rc = 0;
        if constexpr (yields_task_v<Invocable, In>) {
            rc = co_await drive_util<Invocable>(native_in, emit);
        } else if constexpr (is_optional_t<
                                 std::invoke_result_t<Invocable, In>>::value) {
            if (auto r = Invocable{}(native_in)) emit(*r);
        } else {
            emit(Invocable{}(native_in));
        }
        const int rv = (rc == 0 && got) ? 0 : -1;
        co_return rv;
    } catch (...) {
        co_return -1;
    }
}

template <class Item>
auto stream_sink(Arena& arena, dftu_stream_item_fn on_item, void* ud) {
    return [&arena, on_item, ud](const auto& item) {
        arena.reset();
        c_repr_t<Item> citem{};
        write_top<Item>(item, citem, arena);
        on_item(&citem, ud);
    };
}

template <class Util, class In, class Item>
int stream_thunk(const void* in, dftu_stream_item_fn on_item, void* ud) {
    using CIn = c_repr_t<In>;
    if (!on_item) return -1;
    try {
        In native_in = read_top<In>(*static_cast<const CIn*>(in));
        Arena arena;
        return run_util_blocking<Util>(native_in,
                                       stream_sink<Item>(arena, on_item, ud));
    } catch (...) {
        return -1;
    }
}

// Async mirror of stream_thunk: drive the stream on the current executor so the
// caller's worker is released between yielded items. Each item borrows into a
// frame-local arena valid only for its on_item call.
template <class Util, class In, class Item>
coro::CoroTask<int> run_stream_async_thunk(const void* in,
                                           dftu_stream_item_fn on_item,
                                           void* ud) {
    using CIn = c_repr_t<In>;
    if (!on_item) co_return -1;
    try {
        In native_in = read_top<In>(*static_cast<const CIn*>(in));
        Arena arena;
        co_return co_await drive_util<Util>(
            native_in, stream_sink<Item>(arena, on_item, ud));
    } catch (...) {
        co_return -1;
    }
}

// Native-item generator over any exported OUT shape; the pull-model driver
// stores it suspended between pulls. `in` must outlive the generator.
template <class Util, class In, class Item>
coro::AsyncGenerator<Item> util_stream_gen(const In& in) {
    Util util;
    auto prod = invoke_util(util, in);
    using P = std::remove_cvref_t<decltype(prod)>;
    if constexpr (HasNext<P>) {
        while (auto item = co_await prod.next()) co_yield std::move(*item);
    } else {
        auto out = co_await std::move(prod);
        using O = std::remove_cvref_t<decltype(out)>;
        if constexpr (is_optional_t<O>::value) {
            if (out) co_yield std::move(*out);
        } else if constexpr (is_std_vector<O>::value) {
            for (auto& item : out) co_yield std::move(item);
        } else if constexpr (HasError<O>) {
            if (out) co_yield std::move(*out);
        } else {
            co_yield std::move(out);
        }
    }
}

// Concrete pull-model driver. Member order pins in-before-gen construction and
// gen-before-in teardown so the suspended generator never sees a dangling
// input.
template <class Util, class In, class Item>
struct StreamDriverImpl : StreamDriver {
    In in;
    coro::AsyncGenerator<Item> gen;
    Arena arena;
    c_repr_t<Item> citem{};

    explicit StreamDriverImpl(In native)
        : StreamDriver{&next_thunk, &destroy_thunk},
          in(std::move(native)),
          gen(util_stream_gen<Util, In, Item>(in)) {}

    static coro::CoroTask<int> next_thunk(StreamDriver* self,
                                          const void** out_item) {
        auto* d = static_cast<StreamDriverImpl*>(self);
        d->arena.reset();
        try {
            auto item = co_await d->gen.next();
            if (!item) {
                if (out_item) *out_item = nullptr;
                co_return 0;
            }
            write_top<Item>(*item, d->citem, d->arena);
            if (out_item) *out_item = &d->citem;
            co_return 0;
        } catch (...) {
            if (out_item) *out_item = nullptr;
            co_return -1;
        }
    }

    static void destroy_thunk(StreamDriver* self) {
        delete static_cast<StreamDriverImpl*>(self);
    }
};

template <class Util, class In, class Item>
StreamDriver* make_stream_driver(const void* in) {
    using CIn = c_repr_t<In>;
    try {
        return new StreamDriverImpl<Util, In, Item>(
            read_top<In>(*static_cast<const CIn*>(in)));
    } catch (...) {
        return nullptr;
    }
}

struct Entry {
    dftu_utility run{};
    bool has_run = false;
    // Byte sizes of the C in/out structs (c_repr_t), for building a compose
    // leaf that threads the utility's value through the dftu_op engine.
    std::uint32_t in_size = 0;
    std::uint32_t out_size = 0;
    int (*run_stream)(const void*, dftu_stream_item_fn, void*) = nullptr;
    coro::CoroTask<int> (*run_async)(const void*, void*) = nullptr;
    coro::CoroTask<int> (*run_stream_async)(const void*, dftu_stream_item_fn,
                                            void*) = nullptr;
    StreamDriver* (*stream_open)(const void*) = nullptr;
};

template <class Invocable, class In, class Out>
dftu_utility make_run(std::string_view name) {
    dftu_utility u{};
    u.name = name.data();
    u.name_len = static_cast<std::uint32_t>(name.size());
    u.in_tag = tag_of<In>();
    u.out_tag = tag_of<Out>();
    u.run = &single_thunk<Invocable, In, Out>;
    u.self = nullptr;
    return u;
}

template <class Invocable, class In, class OutDecl>
Entry make_entry(std::string_view name) {
    Entry e{};
    if constexpr (stream_of<OutDecl>::value) {
        using Elem = typename stream_of<OutDecl>::elem;
        e.run_stream = &stream_thunk<Invocable, In, Elem>;
        e.run_stream_async = &run_stream_async_thunk<Invocable, In, Elem>;
        e.stream_open = &make_stream_driver<Invocable, In, Elem>;
    } else {
        using Out = typename single_elem<OutDecl>::type;
        e.run = make_run<Invocable, In, Out>(name);
        e.has_run = true;
        e.in_size = static_cast<std::uint32_t>(sizeof(c_repr_t<In>));
        e.out_size = static_cast<std::uint32_t>(sizeof(c_repr_t<Out>));
        e.run_async = &run_async_thunk<Invocable, In, Out>;
        if constexpr (yields_task_v<Invocable, In>) {
            e.run_stream = &stream_thunk<Invocable, In, Out>;
            e.run_stream_async = &run_stream_async_thunk<Invocable, In, Out>;
            e.stream_open = &make_stream_driver<Invocable, In, Out>;
        }
    }
    return e;
}

const std::array<Entry, DFTU_UTIL__COUNT>& registry_table() {
    static const std::array<Entry, DFTU_UTIL__COUNT> table = [] {
        std::array<Entry, DFTU_UTIL__COUNT> t{};
        std::size_t i = 0;
#define DFTU_EXPORT_UTILITY(TAG, name, INVOCABLE, IN, OUT) \
    t[i] = make_entry<INVOCABLE, IN, OUT>(#name);          \
    ++i;
#include <dftracer/utils/plugins/exported_utilities.def>
        return t;
    }();
    return table;
}

}  // namespace

const dftu_utility* registry_find(std::uint32_t id) {
    if (id >= DFTU_UTIL__COUNT) return nullptr;
    const Entry& e = registry_table()[id];
    return e.has_run ? &e.run : nullptr;
}

bool registry_run_sizes(std::uint32_t id, std::uint32_t* in_size,
                        std::uint32_t* out_size) {
    if (id >= DFTU_UTIL__COUNT) return false;
    const Entry& e = registry_table()[id];
    if (!e.has_run) return false;
    if (in_size) *in_size = e.in_size;
    if (out_size) *out_size = e.out_size;
    return true;
}

int registry_run_stream(std::uint32_t id, const void* in,
                        dftu_stream_item_fn on_item, void* ud) {
    if (id >= DFTU_UTIL__COUNT) return -1;
    const Entry& e = registry_table()[id];
    if (!e.run_stream) return -1;
    return e.run_stream(in, on_item, ud);
}

coro::CoroTask<int> registry_run_async(std::uint32_t id, const void* in,
                                       void* out) {
    if (id >= DFTU_UTIL__COUNT) co_return -1;
    const Entry& e = registry_table()[id];
    if (!e.run_async) co_return -1;
    co_return co_await e.run_async(in, out);
}

coro::CoroTask<int> registry_run_stream_async(std::uint32_t id, const void* in,
                                              dftu_stream_item_fn on_item,
                                              void* ud) {
    if (id >= DFTU_UTIL__COUNT) co_return -1;
    const Entry& e = registry_table()[id];
    if (!e.run_stream_async) co_return -1;
    co_return co_await e.run_stream_async(in, on_item, ud);
}

StreamDriver* registry_stream_open(std::uint32_t id, const void* in) {
    if (id >= DFTU_UTIL__COUNT) return nullptr;
    const Entry& e = registry_table()[id];
    if (!e.stream_open) return nullptr;
    return e.stream_open(in);
}

}  // namespace dftracer::utils::plugins
