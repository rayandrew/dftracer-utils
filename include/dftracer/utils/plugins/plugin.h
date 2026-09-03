#ifndef DFTRACER_UTILS_PLUGINS_PLUGIN_H
#define DFTRACER_UTILS_PLUGINS_PLUGIN_H

/* Header-only C++ wrapper over abi.h; derive a Slice with step/merge/finalize
   and expose it with make_plugin<Slice>(). */

#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/arrow_abi.h>
#include <dftracer/utils/plugins/owned_arrow.h>
#include <dftracer/utils/query/builder.h>

#include <array>
#include <concepts>
#include <coroutine>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <iterator>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

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

/// A lazy coroutine that yields, at each co_await, the next dftu_task the host
/// should await; the host drives it to completion (see dftu_host::drive).
class Task {
   public:
    struct promise_type {
        dftu_task* pending = nullptr;
        const dftu_host* host = nullptr;
        std::exception_ptr exc;

        Task get_return_object() {
            return Task{
                std::coroutine_handle<promise_type>::from_promise(*this)};
        }
        std::suspend_always initial_suspend() noexcept { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() { exc = std::current_exception(); }
    };

    explicit Task(std::coroutine_handle<promise_type> h) : h_(h) {}
    Task(Task&& o) noexcept : h_(o.h_) { o.h_ = nullptr; }
    Task& operator=(Task&& o) noexcept {
        if (this != &o) {
            if (h_) h_.destroy();
            h_ = o.h_;
            o.h_ = nullptr;
        }
        return *this;
    }
    Task(const Task&) = delete;
    Task& operator=(const Task&) = delete;
    ~Task() {
        if (h_) h_.destroy();
    }

    /// Give up ownership; the host driver destroys the frame on completion.
    std::coroutine_handle<promise_type> release() {
        auto h = h_;
        h_ = nullptr;
        return h;
    }

   private:
    std::coroutine_handle<promise_type> h_;
};

/// Suspends the Task and hands `task` to the host as the next thing to await.
/// Decays to dftu_task* so it also passes where a raw task is expected.
struct AsyncOp {
    dftu_task* task;
    bool await_ready() const noexcept { return false; }
    void await_suspend(
        std::coroutine_handle<Task::promise_type> h) const noexcept {
        h.promise().pending = task;
    }
    void await_resume() const noexcept {}
    operator dftu_task*() const noexcept { return task; }
};

namespace detail {
/// A stream tag carries its element as `item`; a single-value-as-stream tag as
/// `out`.
template <class Tag, class = void>
struct stream_elem {
    using type = typename Tag::out;
};
template <class Tag>
struct stream_elem<Tag, std::void_t<typename Tag::item>> {
    using type = typename Tag::item;
};
}  // namespace detail

/// RAII pull-model utility stream. co_await next() to pull one element and do
/// the plugin's own async work between pulls; a returned element's
/// dftu_bytes/span fields stay borrowed only until the next next() or
/// destruction. Move-only; the destructor closes the stream if still open.
template <class Tag>
class Stream {
   public:
    using element = typename detail::stream_elem<Tag>::type;

    Stream(const dftu_host* h, const dftu_ext_util* u, dftu_stream* s)
        : h_(h), u_(u), s_(s) {}
    Stream(Stream&& o) noexcept : h_(o.h_), u_(o.u_), s_(o.s_) {
        o.s_ = nullptr;
    }
    Stream& operator=(Stream&& o) noexcept {
        if (this != &o) {
            close();
            h_ = o.h_;
            u_ = o.u_;
            s_ = o.s_;
            o.s_ = nullptr;
        }
        return *this;
    }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
    ~Stream() { close(); }

    struct NextOp {
        const dftu_host* h;
        const dftu_ext_util* u;
        dftu_stream* s;
        const void* item = nullptr;
        int rc = 0;
        bool await_ready() const noexcept {
            return s == nullptr || u == nullptr;
        }
        void await_suspend(
            std::coroutine_handle<Task::promise_type> co) noexcept {
            co.promise().pending = u->util_stream_next(h->h, s, &item, &rc);
        }
        std::optional<element> await_resume() const {
            if (!item) return std::nullopt;
            return std::optional<element>{*static_cast<const element*>(item)};
        }
    };
    NextOp next() { return NextOp{h_, u_, s_}; }

    explicit operator bool() const { return s_ != nullptr; }

   private:
    void close() {
        if (s_ && u_ && u_->util_stream_close) u_->util_stream_close(h_->h, s_);
        s_ = nullptr;
    }
    const dftu_host* h_;
    const dftu_ext_util* u_ = nullptr;
    dftu_stream* s_ = nullptr;
};

/// Thin typed wrapper over the io ext; each call returns an AsyncOp to
/// `co_await`, a no-op if the host does not provide the io group.
class Io {
   public:
    explicit Io(const dftu_host* h)
        : h_(h),
          io_(h && h->get_extension ? static_cast<const dftu_io*>(
                                          h->get_extension(h->h, DFTU_EXT_IO))
                                    : nullptr) {}
    AsyncOp open(const char* path, int flags, int mode, int* out_fd) const {
        return AsyncOp{io_ ? io_->open(h_->h, path, flags, mode, out_fd)
                           : nullptr};
    }
    AsyncOp close(int fd, int* out_rc) const {
        return AsyncOp{io_ ? io_->close(h_->h, fd, out_rc) : nullptr};
    }
    AsyncOp read(int fd, void* buf, std::uint64_t len,
                 std::int64_t* out_n) const {
        return AsyncOp{io_ ? io_->read(h_->h, fd, buf, len, out_n) : nullptr};
    }
    AsyncOp write(int fd, const void* buf, std::uint64_t len,
                  std::int64_t* out_n) const {
        return AsyncOp{io_ ? io_->write(h_->h, fd, buf, len, out_n) : nullptr};
    }
    AsyncOp pread(int fd, void* buf, std::uint64_t len, std::uint64_t off,
                  std::int64_t* out_n) const {
        return AsyncOp{io_ ? io_->pread(h_->h, fd, buf, len, off, out_n)
                           : nullptr};
    }
    AsyncOp pwrite(int fd, const void* buf, std::uint64_t len,
                   std::uint64_t off, std::int64_t* out_n) const {
        return AsyncOp{io_ ? io_->pwrite(h_->h, fd, buf, len, off, out_n)
                           : nullptr};
    }
    AsyncOp fsync(int fd, int* out_rc) const {
        return AsyncOp{io_ ? io_->fsync(h_->h, fd, out_rc) : nullptr};
    }
    AsyncOp ftruncate(int fd, std::uint64_t len, int* out_rc) const {
        return AsyncOp{io_ ? io_->ftruncate(h_->h, fd, len, out_rc) : nullptr};
    }
    AsyncOp fstat(int fd, dftu_stat* out) const {
        return AsyncOp{io_ ? io_->fstat(h_->h, fd, out) : nullptr};
    }
    /// Scatter-gather sequential read; `iov`/`iovcnt` borrowed for the await.
    AsyncOp readv(int fd, const struct iovec* iov, int iovcnt,
                  std::int64_t* out_n) const {
        return AsyncOp{io_ && io_->readv
                           ? io_->readv(h_->h, fd, iov, iovcnt, out_n)
                           : nullptr};
    }
    /// Scatter-gather sequential write; `iov`/`iovcnt` borrowed for the await.
    AsyncOp writev(int fd, const struct iovec* iov, int iovcnt,
                   std::int64_t* out_n) const {
        return AsyncOp{io_ && io_->writev
                           ? io_->writev(h_->h, fd, iov, iovcnt, out_n)
                           : nullptr};
    }
    /// Scatter-gather positional read (seekable fds only).
    AsyncOp preadv(int fd, const struct iovec* iov, int iovcnt,
                   std::uint64_t off, std::int64_t* out_n) const {
        return AsyncOp{io_ && io_->preadv
                           ? io_->preadv(h_->h, fd, iov, iovcnt, off, out_n)
                           : nullptr};
    }
    /// Scatter-gather positional write (seekable fds only).
    AsyncOp pwritev(int fd, const struct iovec* iov, int iovcnt,
                    std::uint64_t off, std::int64_t* out_n) const {
        return AsyncOp{io_ && io_->pwritev
                           ? io_->pwritev(h_->h, fd, iov, iovcnt, off, out_n)
                           : nullptr};
    }
    /// Reposition the file offset; `whence` is SEEK_SET/CUR/END. `out_off`
    /// receives the resulting absolute offset.
    AsyncOp lseek(int fd, std::int64_t off, int whence,
                  std::int64_t* out_off) const {
        return AsyncOp{io_ && io_->lseek
                           ? io_->lseek(h_->h, fd, off, whence, out_off)
                           : nullptr};
    }
    /// Zero-copy transfer of `count` bytes from `in_fd` (a regular file) to
    /// `out_fd` starting at `off`; `out_n` receives the bytes sent.
    AsyncOp sendfile(int out_fd, int in_fd, std::uint64_t off,
                     std::uint64_t count, std::int64_t* out_n) const {
        return AsyncOp{
            io_ && io_->sendfile
                ? io_->sendfile(h_->h, out_fd, in_fd, off, count, out_n)
                : nullptr};
    }
    /// Accept a connection on a listening socket; `addr`/`addrlen` may be null.
    /// `out_fd` receives the client fd.
    AsyncOp accept(int fd, struct sockaddr* addr, socklen_t* addrlen,
                   int* out_fd) const {
        return AsyncOp{io_ && io_->accept
                           ? io_->accept(h_->h, fd, addr, addrlen, out_fd)
                           : nullptr};
    }
    /// Socket receive; `flags` are the recv(2) flags.
    AsyncOp recv(int fd, void* buf, std::uint64_t len, int flags,
                 std::int64_t* out_n) const {
        return AsyncOp{io_ && io_->recv
                           ? io_->recv(h_->h, fd, buf, len, flags, out_n)
                           : nullptr};
    }
    /// Socket send; `flags` are the send(2) flags.
    AsyncOp send(int fd, const void* buf, std::uint64_t len, int flags,
                 std::int64_t* out_n) const {
        return AsyncOp{io_ && io_->send
                           ? io_->send(h_->h, fd, buf, len, flags, out_n)
                           : nullptr};
    }

   private:
    const dftu_host* h_;
    const dftu_io* io_;
};

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

/// A pre-interned STR/BYTES key slot: carries an already-interned dftu_str id
/// (e.g. Event::name_id(), Event::fhash_id()) as a STR key component without
/// re-interning its bytes. Declare the slot's key type as Interned (in a Key
/// tag) and feed it an interned() value.
struct Interned {
    dftu_str id;
};
/// Wrap an already-interned id for use as a STR key slot; see Interned.
inline Interned interned(dftu_str id) noexcept { return Interned{id}; }

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

template <class... KeyTs>
class Map;
template <class Outer, class Inner>
class NestedMap;
class Handle;
class Writer;
template <class T>
class OutPort;
template <class T>
class InPort;

class Host {
   public:
    explicit Host(const dftu_host* h) : h_(h) {}

    std::string_view str(dftu_str id) const {
        if (id == DFTU_STR_NONE) return {};
        std::uint32_t n = 0;
        const char* p = h_->resolve(h_->h, id, &n);
        return p ? std::string_view{p, n} : std::string_view{};
    }
    dftu_str intern(std::string_view s) const {
        return h_->intern(h_->h, s.data(),
                          static_cast<std::uint32_t>(s.size()));
    }
    void log(dftu_log_level level, std::string_view msg) const {
        h_->log(h_->h, static_cast<std::uint8_t>(level), msg.data(),
                static_cast<std::uint32_t>(msg.size()));
    }

    Io io() const { return Io{h_}; }
    AsyncOp await(dftu_task* t) const { return AsyncOp{t}; }
    AsyncOp all(std::initializer_list<dftu_task*> ts) const {
        const dftu_ext_coro* c = ext(DFTU_EXT_CORO, coro_ext_);
        return AsyncOp{c && c->when_all
                           ? c->when_all(h_->h, ts.begin(),
                                         static_cast<std::uint32_t>(ts.size()))
                           : nullptr};
    }
    AsyncOp any(std::initializer_list<dftu_task*> ts) const {
        const dftu_ext_coro* c = ext(DFTU_EXT_CORO, coro_ext_);
        return AsyncOp{c && c->when_any
                           ? c->when_any(h_->h, ts.begin(),
                                         static_cast<std::uint32_t>(ts.size()))
                           : nullptr};
    }

    /// Spawn `fn(arg)` onto a pool worker and get an AsyncOp to co_await (or
    /// feed to all()/any()); `arg` is plugin-owned and must outlive the await.
    AsyncOp spawn(dftu_work_fn fn, void* arg) const {
        const dftu_ext_coro* c = ext(DFTU_EXT_CORO, coro_ext_);
        return AsyncOp{c && c->spawn ? c->spawn(h_->h, fn, arg) : nullptr};
    }
    /// Spawn any callable onto a pool worker; `fn` is borrowed for the await,
    /// so it must outlive it (a coroutine local naturally does).
    template <class Fn>
    AsyncOp spawn(Fn& fn) const {
        return spawn([](void* p) { (*static_cast<Fn*>(p))(); }, &fn);
    }
    /// Sequence: run `fn(arg)` after `t` completes; returns an AsyncOp to
    /// co_await. `arg` is plugin-owned and must outlive the await.
    AsyncOp then(dftu_task* t, dftu_work_fn fn, void* arg) const {
        const dftu_ext_coro* c = ext(DFTU_EXT_CORO, coro_ext_);
        return AsyncOp{c && c->then ? c->then(h_->h, t, fn, arg) : nullptr};
    }
    /// Sequence a callable after `t`; `fn` is borrowed for the await.
    template <class Fn>
    AsyncOp then(dftu_task* t, Fn& fn) const {
        return then(t, [](void* p) { (*static_cast<Fn*>(p))(); }, &fn);
    }

    /// Run a synchronous blocking call while releasing the worker's run-permit,
    /// so a raw block does not starve the elastic pool. Prefer the async task
    /// path for anything with an awaitable form; use this only when there is
    /// none.
    template <typename Fn>
    void run_blocking(Fn fn) const {
        const dftu_ext_coro* c = ext(DFTU_EXT_CORO, coro_ext_);
        if (c && c->run_blocking)
            c->run_blocking(
                h_->h, [](void* p) { (*static_cast<Fn*>(p))(); }, &fn);
        else
            fn();
    }

    /// Valid for the whole scan; null on parse error. Do not free.
    dftu_query* query_compile(std::string_view src) const {
        const dftu_ext_query* e = ext(DFTU_EXT_QUERY, query_ext_);
        return e && e->query_compile
                   ? e->query_compile(h_->h, src.data(),
                                      static_cast<std::uint32_t>(src.size()))
                   : nullptr;
    }

    /// Compile a predicate built with `F`/`Field`, e.g.
    /// `h.query_compile(F("dur") > 1000)`. Rendered to a DSL string and parsed
    /// host-side, so the plugin links nothing from the query library.
    dftu_query* query_compile(const Expr& e) const {
        return query_compile(e.to_string());
    }
    bool query_matches(const dftu_query* q, const dftu_event& e) const {
        const dftu_ext_query* qe = ext(DFTU_EXT_QUERY, query_ext_);
        return q && qe && qe->query_matches &&
               qe->query_matches(h_->h, q, &e) != 0;
    }

    /// Capability-registry queries, meaningful during a comms resolve() (or any
    /// time after): how many plugins provide `cap_id` at any version.
    std::uint32_t provider_count(const char* cap_id) const {
        const dftu_ext_comms* e = ext(DFTU_EXT_COMMS, comms_ext_);
        return e && e->provider_count ? e->provider_count(h_->h, cap_id) : 0;
    }
    /// Highest provider version satisfying `req` into *out_ver; true on a
    /// match, false if none (or the comms group is absent).
    bool provider_best(const dftu_requirement& req,
                       dftu_version* out_ver) const {
        const dftu_ext_comms* e = ext(DFTU_EXT_COMMS, comms_ext_);
        return e && e->provider_best &&
               e->provider_best(h_->h, &req, out_ver) == 0;
    }
    /// Typed form of provider_best; nullopt if no provider satisfies `req`.
    std::optional<dftu_version> provider_best(
        const dftu_requirement& req) const {
        dftu_version v{};
        return provider_best(req, &v) ? std::optional<dftu_version>{v}
                                      : std::nullopt;
    }

    /// Caller owns the handle and must sketch_free it.
    dftu_sketch* sketch_create() const {
        const dftu_ext_sketch* e = ext(DFTU_EXT_SKETCH, sketch_ext_);
        return e && e->sketch_create ? e->sketch_create(h_->h) : nullptr;
    }
    void sketch_add(dftu_sketch* s, double v, double w = 1.0) const {
        const dftu_ext_sketch* e = ext(DFTU_EXT_SKETCH, sketch_ext_);
        if (e && e->sketch_add) e->sketch_add(h_->h, s, v, w);
    }
    void sketch_merge(dftu_sketch* into, const dftu_sketch* other) const {
        const dftu_ext_sketch* e = ext(DFTU_EXT_SKETCH, sketch_ext_);
        if (e && e->sketch_merge) e->sketch_merge(h_->h, into, other);
    }
    dftu_quantiles sketch_result(const dftu_sketch* s) const {
        dftu_quantiles q{};
        const dftu_ext_sketch* e = ext(DFTU_EXT_SKETCH, sketch_ext_);
        return e && e->sketch_result ? e->sketch_result(h_->h, s) : q;
    }
    void sketch_free(dftu_sketch* s) const {
        const dftu_ext_sketch* e = ext(DFTU_EXT_SKETCH, sketch_ext_);
        if (e && e->sketch_free) e->sketch_free(h_->h, s);
    }
    /// RAII sketch bound to this host; frees the handle in its destructor.
    class Sketch make_sketch() const;

    /// Build an Arrow batch from `b`; the caller owns *out/*out_schema and must
    /// call their release. 0 on success.
    int batch_to_arrow(const dftu_batch& b, ArrowArray* out,
                       ArrowSchema* out_schema) const {
        const dftu_ext_arrow* e = ext(DFTU_EXT_ARROW, arrow_ext_);
        return e && e->batch_to_arrow
                   ? e->batch_to_arrow(h_->h, &b, out, out_schema)
                   : -1;
    }
    /// Owning form of batch_to_arrow; an empty result (`!owned`) signals the
    /// host lacks the Arrow group or export failed.
    OwnedArrow batch_to_arrow(const dftu_batch& b) const {
        OwnedArrow owned;
        batch_to_arrow(b, &owned.array, &owned.schema);
        return owned;
    }
    int arrow_write_ipc(ArrowArray* a, ArrowSchema* s, const char* path) const {
        const dftu_ext_arrow* e = ext(DFTU_EXT_ARROW, arrow_ext_);
        return e && e->arrow_write_ipc ? e->arrow_write_ipc(h_->h, a, s, path)
                                       : -1;
    }
    /// Read the first record batch of an IPC file; the caller owns *out and
    /// *out_schema and must call their release. 0 on success.
    int arrow_read_ipc(const char* path, ArrowArray* out,
                       ArrowSchema* out_schema) const {
        const dftu_ext_arrow* e = ext(DFTU_EXT_ARROW, arrow_ext_);
        return e && e->arrow_read_ipc
                   ? e->arrow_read_ipc(h_->h, path, out, out_schema)
                   : -1;
    }
    /// arrow_write_ipc for an already-owning batch.
    int arrow_write_ipc(OwnedArrow& a, const char* path) const {
        return arrow_write_ipc(&a.array, &a.schema, path);
    }
    /// Owning form of arrow_read_ipc; an empty result (`!owned`) signals the
    /// host lacks the Arrow group or the read failed.
    OwnedArrow arrow_read_ipc(const char* path) const {
        OwnedArrow owned;
        arrow_read_ipc(path, &owned.array, &owned.schema);
        return owned;
    }
    dftu_trace_writer* trace_open_write(const char* path) const {
        const dftu_ext_trace* e = ext(DFTU_EXT_TRACE, trace_ext_);
        return e && e->trace_open_write ? e->trace_open_write(h_->h, path)
                                        : nullptr;
    }
    int trace_write(dftu_trace_writer* w, const dftu_event* evs,
                    std::uint32_t n) const {
        const dftu_ext_trace* e = ext(DFTU_EXT_TRACE, trace_ext_);
        return e && e->trace_write ? e->trace_write(h_->h, w, evs, n) : -1;
    }
    int trace_close(dftu_trace_writer* w) const {
        const dftu_ext_trace* e = ext(DFTU_EXT_TRACE, trace_ext_);
        return e && e->trace_close ? e->trace_close(h_->h, w) : -1;
    }
    /// Scan `path` (auto-indexed) and call on_event per event. 0 on success.
    int trace_read(const char* path, dftu_stream_item_fn on_event,
                   void* ud) const {
        const dftu_ext_trace* e = ext(DFTU_EXT_TRACE, trace_ext_);
        return e && e->trace_read ? e->trace_read(h_->h, path, on_event, ud)
                                  : -1;
    }

    /// Get a host-owned parallel Writer for `path` with `num_workers` lanes
    /// (`gzip` != 0 compresses). The Writer is host-owned and scan-lifetime;
    /// open it, write chunks per worker, then close it - all co_await forms. An
    /// empty result (`!writer`) means the host lacks the writer group.
    Writer writer(const char* path, std::uint32_t num_workers,
                  bool gzip = false) const;
    /// Merge worker shard files into `target`; co_await the returned AsyncOp.
    AsyncOp merge_shards(const char* target,
                         std::initializer_list<const char*> shards) const {
        const dftu_ext_writer* e = ext(DFTU_EXT_WRITER, writer_ext_);
        return AsyncOp{
            e && e->merge_shards
                ? e->merge_shards(h_->h, target, shards.begin(),
                                  static_cast<std::uint32_t>(shards.size()))
                : nullptr};
    }

    /// Invoke a host utility by tag; nullopt if absent or run() failed.
    template <class Tag>
    std::optional<typename Tag::out> util(const typename Tag::in& in) const {
        const dftu_ext_util* u = ext(DFTU_EXT_UTIL, util_ext_);
        const dftu_utility* d =
            u && u->find_by_id
                ? u->find_by_id(h_->h, static_cast<std::uint32_t>(Tag::id))
                : nullptr;
        if (!d) return std::nullopt;
        typename Tag::out out{};
        return d->run(d->self, &in, &out) == 0
                   ? std::optional<typename Tag::out>{out}
                   : std::nullopt;
    }

    /// co_await form of util(): runs the utility on the current executor
    /// without blocking the caller, writing *out and the status into out_rc. in
    /// and out are caller-owned and must outlive the await.
    template <class Tag>
    AsyncOp util_async(const typename Tag::in& in, typename Tag::out& out,
                       int& out_rc) const {
        const dftu_ext_util* u = ext(DFTU_EXT_UTIL, util_ext_);
        return AsyncOp{
            u && u->util_run_async
                ? u->util_run_async(h_->h, static_cast<std::uint32_t>(Tag::id),
                                    &in, &out, &out_rc)
                : nullptr};
    }

    /// co_await form of run_stream: drives the utility on the current executor
    /// without blocking the caller, firing on_item per yielded item and writing
    /// the status into out_rc. in, on_item, and ud must outlive the await.
    template <class Tag>
    AsyncOp util_stream_async(const typename Tag::in& in,
                              dftu_stream_item_fn on_item, void* ud,
                              int& out_rc) const {
        const dftu_ext_util* u = ext(DFTU_EXT_UTIL, util_ext_);
        return AsyncOp{u && u->util_run_stream_async
                           ? u->util_run_stream_async(
                                 h_->h, static_cast<std::uint32_t>(Tag::id),
                                 &in, on_item, ud, &out_rc)
                           : nullptr};
    }

    /// Synchronous push form of a streaming utility: block the calling thread,
    /// invoking `fn(const Tag::element&)` once per yielded item. Prefer the
    /// co_await forms (util_stream_async / util_stream) on an executor; this is
    /// the direct wrapper over the ABI's run_stream. Returns 0 on success, -1
    /// on failure or when the util group is absent.
    template <class Tag, class Fn>
    int util_each(const typename Tag::in& in, Fn&& fn) const {
        const dftu_ext_util* u = ext(DFTU_EXT_UTIL, util_ext_);
        if (!u || !u->run_stream) return -1;
        using element = typename detail::stream_elem<Tag>::type;
        auto thunk = [](const void* item, void* ud) {
            (*static_cast<std::decay_t<Fn>*>(ud))(
                *static_cast<const element*>(item));
        };
        return u->run_stream(
            h_->h, static_cast<std::uint32_t>(Tag::id), &in, thunk,
            const_cast<void*>(static_cast<const void*>(std::addressof(fn))));
    }

    /// Open a pull-model stream; `in` is copied at open and need not outlive
    /// the returned Stream. A non-stream tag yields a Stream that ends
    /// immediately.
    template <class Tag>
    Stream<Tag> util_stream(const typename Tag::in& in) const {
        const dftu_ext_util* u = ext(DFTU_EXT_UTIL, util_ext_);
        dftu_stream* s =
            u && u->util_stream_open
                ? u->util_stream_open(h_->h,
                                      static_cast<std::uint32_t>(Tag::id), &in)
                : nullptr;
        return Stream<Tag>{h_, u, s};
    }

    /// Batch-scoped ports: a producer publishes a derived value a later
    /// consumer reads for the same batch. consume returns NULL (and 0 len) when
    /// the producer has not published this batch; the borrow is valid only
    /// until the current on_batch returns.
    std::uint64_t port_key(const char* cap_id) const {
        const dftu_ext_ports* e = ext(DFTU_EXT_PORTS, ports_ext_);
        return e && e->port_key ? e->port_key(h_->h, cap_id) : 0;
    }
    void publish(std::uint64_t key, const void* data, std::uint32_t len) const {
        const dftu_ext_ports* e = ext(DFTU_EXT_PORTS, ports_ext_);
        if (e && e->publish) e->publish(h_->h, key, data, len);
    }
    const void* consume(std::uint64_t key, std::uint32_t* out_len) const {
        const dftu_ext_ports* e = ext(DFTU_EXT_PORTS, ports_ext_);
        return e && e->consume ? e->consume(h_->h, key, out_len) : nullptr;
    }

    /// Cross-worker mergeable handles: get-or-create a named monoid accumulator
    /// during on_batch; the host merges it across all worker slices and a
    /// consumer reads the merged value at finalize via handle_result.
    dftu_handle* shared_get(const char* cap_id, dftu_monoid_kind kind) const {
        const dftu_ext_handles* e = ext(DFTU_EXT_HANDLES, handles_ext_);
        return e && e->shared_get ? e->shared_get(h_->h, cap_id, kind)
                                  : nullptr;
    }
    void handle_add(dftu_handle* hd, std::uint64_t v) const {
        const dftu_ext_handles* e = ext(DFTU_EXT_HANDLES, handles_ext_);
        if (e && e->add_u64) e->add_u64(h_->h, hd, v);
    }
    void handle_add(dftu_handle* hd, double v, double w = 1.0) const {
        const dftu_ext_handles* e = ext(DFTU_EXT_HANDLES, handles_ext_);
        if (e && e->add_f64) e->add_f64(h_->h, hd, v, w);
    }
    /// 0 and fills *out, or -1 if no plugin produced that handle.
    int handle_result(const char* cap_id, dftu_monoid_value* out) const {
        const dftu_ext_handles* e = ext(DFTU_EXT_HANDLES, handles_ext_);
        return e && e->result ? e->result(h_->h, cap_id, out) : -1;
    }
    /// Typed form of handle_result; nullopt if no plugin produced that handle.
    std::optional<MonoidValue> handle_result(const char* cap_id) const {
        MonoidValue v;
        return handle_result(cap_id, &v.raw) == 0
                   ? std::optional<MonoidValue>{v}
                   : std::nullopt;
    }

    /// Named result channel: emit an opaque blob (the host copies `len` bytes)
    /// or a user-schema Arrow array (the host moves it); both surface from
    /// PluginHost::run keyed by name. Best called at on_finalize.
    void emit_result(const char* name, const void* data,
                     std::uint64_t len) const {
        const dftu_ext_result* e = ext(DFTU_EXT_RESULT, result_ext_);
        if (e && e->emit) e->emit(h_->h, name, data, len);
    }
    void emit_result(const char* name, std::string_view data) const {
        emit_result(name, data.data(), data.size());
    }
    /// Moves *a and *s into the host; -1 if the host lacks the result channel.
    int emit_result_arrow(const char* name, ArrowArray* a,
                          ArrowSchema* s) const {
        const dftu_ext_result* e = ext(DFTU_EXT_RESULT, result_ext_);
        return e && e->emit_arrow ? e->emit_arrow(h_->h, name, a, s) : -1;
    }
    /// Hands a deferred plan to the host, which takes ownership and collects it
    /// at the Python edge; -1 if the host lacks the result channel or the
    /// emit_lazyframe slot. The plan must be self-contained (see the
    /// emit_lazyframe contract in abi.h).
    int emit_result_lazyframe(const char* name, dftu_lazyframe* lf) const {
        const dftu_ext_result* e = ext(DFTU_EXT_RESULT, result_ext_);
        return e && e->emit_lazyframe ? e->emit_lazyframe(h_->h, name, lf) : -1;
    }

    /// Mergeable map: get-or-create a named tuple-keyed map whose value is a
    /// scalar monoid; null if a key type or the value monoid is unsupported.
    /// The result surfaces from PluginHost::run by name.
    dftu_map* map_new(const char* name,
                      std::initializer_list<dftu_type> key_types,
                      dftu_monoid_kind value) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        return e && e->map_new
                   ? e->map_new(h_->h, name, key_types.begin(),
                                static_cast<std::uint32_t>(key_types.size()),
                                value)
                   : nullptr;
    }
    void map_add_u64(dftu_map* m, std::initializer_list<std::int64_t> key,
                     std::uint64_t v) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_u64) e->map_add_u64(h_->h, m, key.begin(), v);
    }
    void map_add_f64(dftu_map* m, std::initializer_list<std::int64_t> key,
                     double v) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_f64) e->map_add_f64(h_->h, m, key.begin(), v);
    }

    /// Product-valued map: the value is a tuple of monoids, materialized to one
    /// value column per component (v0..v{n-1}). map_add_*_at target component
    /// `comp`; map_add_u64/map_add_f64 target component 0.
    dftu_map* map_new_product(
        const char* name, std::initializer_list<dftu_type> key_types,
        std::initializer_list<dftu_monoid_kind> values) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        return e && e->map_new_product
                   ? e->map_new_product(
                         h_->h, name, key_types.begin(),
                         static_cast<std::uint32_t>(key_types.size()),
                         values.begin(),
                         static_cast<std::uint32_t>(values.size()))
                   : nullptr;
    }
    void map_add_u64_at(dftu_map* m, std::initializer_list<std::int64_t> key,
                        std::uint32_t comp, std::uint64_t v) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_u64_at)
            e->map_add_u64_at(h_->h, m, key.begin(), comp, v);
    }
    void map_add_f64_at(dftu_map* m, std::initializer_list<std::int64_t> key,
                        std::uint32_t comp, double v) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_f64_at)
            e->map_add_f64_at(h_->h, m, key.begin(), comp, v);
    }
    void map_add_xy_at(dftu_map* m, std::initializer_list<std::int64_t> key,
                       std::uint32_t comp, double x, double y) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_xy_at)
            e->map_add_xy_at(h_->h, m, key.begin(), comp, x, y);
    }
    /// Append (order_key, element id) to LIST_STR value component `comp`.
    void map_add_ordered_at(dftu_map* m,
                            std::initializer_list<std::int64_t> key,
                            std::uint32_t comp, std::int64_t order_key,
                            std::uint64_t element) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_ordered_at)
            e->map_add_ordered_at(h_->h, m, key.begin(), comp, order_key,
                                  element);
    }
    /// Contribute (by, payload) to an ARGMIN/ARGMAX value component `comp`;
    /// equal `by` keeps the smaller payload for determinism.
    void map_add_argby_at(dftu_map* m, std::initializer_list<std::int64_t> key,
                          std::uint32_t comp, double by,
                          std::int64_t payload) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_argby_at)
            e->map_add_argby_at(h_->h, m, key.begin(), comp, by, payload);
    }
    /// Contribute (by, payload) to a bounded TOPK/BOTTOMK value component
    /// `comp`, keeping the k payloads at the k extreme `by` keys; k is passed
    /// on every add (constant per component).
    void map_add_topk_at(dftu_map* m, std::initializer_list<std::int64_t> key,
                         std::uint32_t comp, std::uint32_t k, double by,
                         std::int64_t payload) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_topk_at)
            e->map_add_topk_at(h_->h, m, key.begin(), comp, k, by, payload);
    }
    /// Observe `value` for an APPROX_TOPK heavy-hitters value component `comp`
    /// (SpaceSaving); k is the counter capacity, passed on every add.
    void map_add_approx_topk_at(dftu_map* m,
                                std::initializer_list<std::int64_t> key,
                                std::uint32_t comp, std::uint32_t k,
                                std::int64_t value) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_approx_topk_at)
            e->map_add_approx_topk_at(h_->h, m, key.begin(), comp, k, value);
    }
    /// Observe `item` for a bottom-k-by-hash SAMPLE value component `comp`; k
    /// is the sample size, passed on every add. Samples distinct items.
    void map_add_sample_at(dftu_map* m, std::initializer_list<std::int64_t> key,
                           std::uint32_t comp, std::uint32_t k,
                           std::int64_t item) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_sample_at)
            e->map_add_sample_at(h_->h, m, key.begin(), comp, k, item);
    }
    /// Arg-row (full-row min-by/max-by, DISTINCT ON): get-or-create a map whose
    /// single value keeps the whole payload row at the extreme `by`. Null if
    /// any key or payload type is unsupported.
    dftu_map* map_new_argrow(
        const char* name, std::initializer_list<dftu_type> key_types,
        bool is_max, std::initializer_list<dftu_type> payload_types) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        return e && e->map_new_argrow
                   ? e->map_new_argrow(
                         h_->h, name, key_types.begin(),
                         static_cast<std::uint32_t>(key_types.size()),
                         is_max ? 1 : 0, payload_types.begin(),
                         static_cast<std::uint32_t>(payload_types.size()))
                   : nullptr;
    }
    /// Get-or-create a DDSketch quantile map over `qs` (each in [0,1]), fed via
    /// map_add_f64; materializes to a count column plus one f64 column per
    /// quantile. Null if unsupported or the host predates map_new_sketch.
    dftu_map* map_new_sketch(const char* name,
                             std::initializer_list<dftu_type> key_types,
                             std::initializer_list<double> qs) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        return e && e->map_new_sketch
                   ? e->map_new_sketch(
                         h_->h, name, key_types.begin(),
                         static_cast<std::uint32_t>(key_types.size()),
                         qs.begin(), static_cast<std::uint32_t>(qs.size()))
                   : nullptr;
    }
    /// Get-or-create a fused map whose components each materialize as their own
    /// named table (out_names). Feed a whole row with one lookup via
    /// map_add_row.
    dftu_map* map_new_fused(
        const char* name, std::initializer_list<dftu_type> key_types,
        std::initializer_list<const char*> out_names,
        std::initializer_list<dftu_monoid_kind> values) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        return e && e->map_new_fused
                   ? e->map_new_fused(
                         h_->h, name, key_types.begin(),
                         static_cast<std::uint32_t>(key_types.size()),
                         out_names.begin(), values.begin(),
                         static_cast<std::uint32_t>(values.size()))
                   : nullptr;
    }
    /// Apply a whole row at `key` with a single hash lookup.
    void map_add_row(dftu_map* m, std::initializer_list<std::int64_t> key,
                     std::initializer_list<dftu_row_val> vals) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_row)
            e->map_add_row(h_->h, m, key.begin(), vals.begin(),
                           static_cast<std::uint32_t>(vals.size()));
    }
    /// Contribute (by, payload-row) at `key`; payload slots use the same
    /// encoding as key slots.
    void map_add_argrow(dftu_map* m, std::initializer_list<std::int64_t> key,
                        double by,
                        std::initializer_list<std::int64_t> payload) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_argrow)
            e->map_add_argrow(h_->h, m, key.begin(), by, payload.begin(),
                              static_cast<std::uint32_t>(payload.size()));
    }
    /// Mark `m` so its rows materialize sorted by key; default is unordered.
    void map_set_ordered(dftu_map* m, bool ordered = true) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_set_ordered)
            e->map_set_ordered(h_->h, m, ordered ? 1 : 0);
    }

    /// Nested-preserved map: the value at each outer key is itself a map (inner
    /// keys -> value monoids), materialized to one list<struct> row per outer
    /// key. map_add_nested_* target value component `comp` at (outer, inner).
    dftu_map* map_new_nested(
        const char* name, std::initializer_list<dftu_type> outer_key_types,
        std::initializer_list<dftu_type> inner_key_types,
        std::initializer_list<dftu_monoid_kind> values) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        return e && e->map_new_nested
                   ? e->map_new_nested(
                         h_->h, name, outer_key_types.begin(),
                         static_cast<std::uint32_t>(outer_key_types.size()),
                         inner_key_types.begin(),
                         static_cast<std::uint32_t>(inner_key_types.size()),
                         values.begin(),
                         static_cast<std::uint32_t>(values.size()))
                   : nullptr;
    }
    void map_add_nested_u64(dftu_map* m,
                            std::initializer_list<std::int64_t> outer_key,
                            std::initializer_list<std::int64_t> inner_key,
                            std::uint32_t comp, std::uint64_t v) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_nested_u64)
            e->map_add_nested_u64(h_->h, m, outer_key.begin(),
                                  inner_key.begin(), comp, v);
    }
    void map_add_nested_f64(dftu_map* m,
                            std::initializer_list<std::int64_t> outer_key,
                            std::initializer_list<std::int64_t> inner_key,
                            std::uint32_t comp, double v) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_add_nested_f64)
            e->map_add_nested_f64(h_->h, m, outer_key.begin(),
                                  inner_key.begin(), comp, v);
    }

    /// Declare a join executed at finalize on the merged master maps: joins the
    /// maps named left_name and right_name on their shared key and emits the
    /// result as an additional map named out_name.
    void map_declare_join(const char* out_name, const char* left_name,
                          const char* right_name, dftu_join_type type) const {
        const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
        if (e && e->map_declare_join)
            e->map_declare_join(h_->h, out_name, left_name, right_name, type);
    }
    /// map_declare_join taking the scoped JoinType.
    void map_declare_join(const char* out_name, const char* left_name,
                          const char* right_name, JoinType type) const {
        map_declare_join(out_name, left_name, right_name,
                         static_cast<dftu_join_type>(type));
    }

    /// Get-or-create a host-owned mergeable map as a typed Map: the Key tag
    /// fixes the key schema, `kind` the value monoid. An empty result (`!map`)
    /// means the host lacks the map group or rejected the schema/monoid.
    template <class... KeyTs>
    Map<KeyTs...> map(const char* name, Monoid kind, Key<KeyTs...> = {}) const;
    template <class... KeyTs>
    Map<KeyTs...> counter_map(const char* name, Key<KeyTs...> = {}) const;
    template <class... KeyTs>
    Map<KeyTs...> sum_map(const char* name, Key<KeyTs...> = {}) const;

    /// Get-or-create a host-owned product-valued map as a typed Map: the value
    /// is a tuple of `values` monoids (one value column per component). Feed
    /// component 0 via the `map[key] += n` sugar or .add, other components via
    /// .add_at / .add_xy_at / .add_argby_at / .add_topk_at etc. An empty result
    /// (`!map`) means the host lacks the map group or rejected the schema.
    template <class... KeyTs>
    Map<KeyTs...> product_map(const char* name,
                              std::initializer_list<Monoid> values,
                              Key<KeyTs...> = {}) const;

    /// Get-or-create a host-owned nested-preserved map as a typed NestedMap:
    /// the value at each outer key is itself a map (inner keys -> `values`
    /// monoids). The two Key tags fix the outer and inner key schemas.
    template <class... OuterTs, class... InnerTs>
    NestedMap<Key<OuterTs...>, Key<InnerTs...>> nested_map(
        const char* name, std::initializer_list<Monoid> values,
        Key<OuterTs...> = {}, Key<InnerTs...> = {}) const;

    /// Get-or-create a named cross-worker mergeable monoid accumulator as a
    /// typed Handle (analogous to a single-key Map): the accumulator is
    /// write-only, merged across all worker slices, and read at finalize via
    /// Handle::result. An empty result (`!handle`) means the host lacks the
    /// handles group.
    Handle handle(const char* cap_id, Monoid kind) const;

    /// Typed batch-scoped output port: send() publishes a trivially-copyable T
    /// under `cap_id` for a same-batch consumer. See publish/consume.
    template <class T>
    OutPort<T> publish_port(const char* cap_id) const;
    /// Typed batch-scoped input port: recv() reads the T a producer published
    /// under `cap_id` this batch, or nullopt if none. See publish/consume.
    template <class T>
    InPort<T> consume_port(const char* cap_id) const;

    const dftu_host* raw() const { return h_; }

   private:
    /// Fetch a host-service group once and memoize it in `slot`; a missing
    /// group stays null so every wrapper falls back exactly as a null
    /// fn-pointer did.
    template <class T>
    const T* ext(const char* id, const T*& slot) const {
        if (!slot && h_->get_extension)
            slot = static_cast<const T*>(h_->get_extension(h_->h, id));
        return slot;
    }

    const dftu_host* h_;
    mutable const dftu_ext_coro* coro_ext_ = nullptr;
    mutable const dftu_ext_query* query_ext_ = nullptr;
    mutable const dftu_ext_comms* comms_ext_ = nullptr;
    mutable const dftu_ext_writer* writer_ext_ = nullptr;
    mutable const dftu_ext_util* util_ext_ = nullptr;
    mutable const dftu_ext_sketch* sketch_ext_ = nullptr;
    mutable const dftu_ext_arrow* arrow_ext_ = nullptr;
    mutable const dftu_ext_trace* trace_ext_ = nullptr;
    mutable const dftu_ext_ports* ports_ext_ = nullptr;
    mutable const dftu_ext_handles* handles_ext_ = nullptr;
    mutable const dftu_ext_result* result_ext_ = nullptr;
    mutable const dftu_ext_map* map_ext_ = nullptr;
};

/// Move-only RAII owner of a host sketch handle; frees it in the destructor.
/// Construct from a Host (`Sketch s{host};`) or via Host::make_sketch().
class Sketch {
   public:
    explicit Sketch(Host host) : host_(host), s_(host_.sketch_create()) {}
    Sketch(Sketch&& o) noexcept : host_(o.host_), s_(o.s_) { o.s_ = nullptr; }
    Sketch& operator=(Sketch&& o) noexcept {
        if (this != &o) {
            reset();
            host_ = o.host_;
            s_ = o.s_;
            o.s_ = nullptr;
        }
        return *this;
    }
    Sketch(const Sketch&) = delete;
    Sketch& operator=(const Sketch&) = delete;
    ~Sketch() { reset(); }

    void add(double value, double weight = 1.0) const {
        host_.sketch_add(s_, value, weight);
    }
    void merge(const Sketch& other) const { host_.sketch_merge(s_, other.s_); }
    dftu_quantiles result() const { return host_.sketch_result(s_); }

    /// Borrowed handle; the Sketch retains ownership. Null if unsupported.
    dftu_sketch* raw() const noexcept { return s_; }
    explicit operator bool() const noexcept { return s_ != nullptr; }

   private:
    void reset() {
        if (s_) host_.sketch_free(s_);
        s_ = nullptr;
    }
    Host host_;
    dftu_sketch* s_ = nullptr;
};

inline Sketch Host::make_sketch() const { return Sketch{*this}; }

/// Typed, non-owning view over a named cross-worker mergeable handle (see
/// Host::handle). The handle is a write-only accumulator: contribute during
/// on_batch with add(), and read the cross-worker-merged value at finalize with
/// result(). The host owns the handle, so there is nothing to free. add() with
/// an integral routes to the u64 feed (COUNTER/MIN_U64/MAX_U64/BOOL/BITSET/
/// DISTINCT), with a double to the f64 feed (SUM/MIN_F64/MAX_F64/SKETCH).
class Handle {
   public:
    Handle() = default;

    /// Contribute an integer observation (u64-fed monoids).
    void add(std::uint64_t v) const { host_.handle_add(hd_, v); }
    /// Contribute a floating observation; `w` is the SKETCH sample weight
    /// (ignored by SUM/MIN/MAX).
    void add(double v, double w = 1.0) const { host_.handle_add(hd_, v, w); }

    /// The cross-worker-merged value; nullopt if no plugin produced this
    /// handle. Call at on_finalize.
    std::optional<MonoidValue> result() const {
        return host_.handle_result(cap_.c_str());
    }

    /// Borrowed handle; the host retains ownership. Null if unsupported.
    dftu_handle* raw() const noexcept { return hd_; }
    explicit operator bool() const noexcept { return hd_ != nullptr; }

   private:
    friend class Host;
    Handle(Host host, std::string cap, dftu_handle* hd)
        : host_(host), cap_(std::move(cap)), hd_(hd) {}

    Host host_{nullptr};
    std::string cap_;
    dftu_handle* hd_ = nullptr;
};

inline Handle Host::handle(const char* cap_id, Monoid kind) const {
    dftu_handle* hd = shared_get(cap_id, static_cast<dftu_monoid_kind>(kind));
    return Handle{*this, cap_id ? std::string{cap_id} : std::string{}, hd};
}

/// Typed, non-owning view over a host-owned parallel Writer (see Host::writer).
/// The writer is host-owned and scan-lifetime, so there is nothing to free; the
/// lifecycle is open() once, chunk() per worker lane, then close(), each a
/// co_await-able AsyncOp. Worker lanes write concurrently, so chunks for
/// distinct `worker` indices may be in flight at once.
class Writer {
   public:
    Writer() = default;

    /// Open the underlying output; co_await before writing any chunk.
    AsyncOp open() const {
        return AsyncOp{e_ && e_->writer_open && w_ ? e_->writer_open(h(), w_)
                                                   : nullptr};
    }
    /// Append `len` bytes to lane `worker`; co_await to complete the write.
    AsyncOp chunk(std::uint32_t worker, const void* data,
                  std::uint64_t len) const {
        return AsyncOp{e_ && e_->writer_chunk && w_
                           ? e_->writer_chunk(h(), w_, worker, data, len)
                           : nullptr};
    }
    AsyncOp chunk(std::uint32_t worker, std::string_view data) const {
        return chunk(worker, data.data(), data.size());
    }
    /// Flush and finalize; co_await after all chunks are written.
    AsyncOp close() const {
        return AsyncOp{e_ && e_->writer_close && w_ ? e_->writer_close(h(), w_)
                                                    : nullptr};
    }

    /// Borrowed handle; the host retains ownership. Null if unsupported.
    dftu_writer* raw() const noexcept { return w_; }
    explicit operator bool() const noexcept { return w_ != nullptr; }

   private:
    friend class Host;
    Writer(Host host, const dftu_ext_writer* e, dftu_writer* w)
        : host_(host), e_(e), w_(w) {}
    void* h() const noexcept { return host_.raw()->h; }

    Host host_{nullptr};
    const dftu_ext_writer* e_ = nullptr;
    dftu_writer* w_ = nullptr;
};

inline Writer Host::writer(const char* path, std::uint32_t num_workers,
                           bool gzip) const {
    const dftu_ext_writer* e = ext(DFTU_EXT_WRITER, writer_ext_);
    dftu_writer* w =
        e && e->writer_create
            ? e->writer_create(h_->h, path, num_workers, gzip ? 1 : 0)
            : nullptr;
    return Writer{*this, e, w};
}

/// Typed batch-scoped output port (see Host::publish_port). send() publishes a
/// trivially-copyable T for a same-batch consumer under the resolved port key.
template <class T>
class OutPort {
    static_assert(std::is_trivially_copyable_v<T>,
                  "a typed port carries a trivially-copyable value");

   public:
    OutPort() = default;

    /// Publish `v` for this batch; the host copies sizeof(T) bytes.
    void send(const T& v) const {
        if (key_ != 0) host_.publish(key_, &v, sizeof(T));
    }

    std::uint64_t key() const noexcept { return key_; }
    explicit operator bool() const noexcept { return key_ != 0; }

   private:
    friend class Host;
    OutPort(Host host, std::uint64_t key) : host_(host), key_(key) {}
    Host host_{nullptr};
    std::uint64_t key_ = 0;
};

/// Typed batch-scoped input port (see Host::consume_port). recv() reads the T a
/// producer published this batch under the resolved port key.
template <class T>
class InPort {
    static_assert(std::is_trivially_copyable_v<T>,
                  "a typed port carries a trivially-copyable value");

   public:
    InPort() = default;

    /// The value a producer published this batch, or nullopt if none was
    /// published or its length did not match sizeof(T). The read is a copy, so
    /// the result outlives the borrowed port buffer.
    std::optional<T> recv() const {
        if (key_ == 0) return std::nullopt;
        std::uint32_t len = 0;
        const void* p = host_.consume(key_, &len);
        if (!p || len != sizeof(T)) return std::nullopt;
        T out;
        std::memcpy(&out, p, sizeof(T));
        return out;
    }

    std::uint64_t key() const noexcept { return key_; }
    explicit operator bool() const noexcept { return key_ != 0; }

   private:
    friend class Host;
    InPort(Host host, std::uint64_t key) : host_(host), key_(key) {}
    Host host_{nullptr};
    std::uint64_t key_ = 0;
};

template <class T>
inline OutPort<T> Host::publish_port(const char* cap_id) const {
    return OutPort<T>{*this, port_key(cap_id)};
}
template <class T>
inline InPort<T> Host::consume_port(const char* cap_id) const {
    return InPort<T>{*this, port_key(cap_id)};
}

/// Non-owning typed view over one flattened event arg. The value is one of
/// i64/f64/str selected by kind(); the key and any string value are interned
/// ids, resolvable to bytes through a Host. Borrowed for the on_batch call.
class Arg {
   public:
    Arg() = default;
    explicit Arg(const dftu_arg& a) noexcept : a_(&a) {}

    dftu_str key_id() const noexcept { return a_->key; }
    std::string_view key(const Host& h) const { return h.str(a_->key); }

    dftu_arg_kind kind() const noexcept { return a_->kind; }
    bool is_i64() const noexcept { return a_->kind == DFTU_ARG_I64; }
    bool is_f64() const noexcept { return a_->kind == DFTU_ARG_F64; }
    bool is_str() const noexcept { return a_->kind == DFTU_ARG_STR; }

    /// Read the accessor matching kind(); another slot holds a stale value.
    std::int64_t i64() const noexcept { return a_->v.i64; }
    double f64() const noexcept { return a_->v.f64; }
    dftu_str str_id() const noexcept { return a_->v.str; }
    std::string_view str(const Host& h) const { return h.str(a_->v.str); }

    const dftu_arg& raw() const noexcept { return *a_; }

   private:
    const dftu_arg* a_ = nullptr;
};

/// Non-owning iterable over an event's args, yielding typed Arg views for
/// range-based iteration: `for (const Arg& a : e.args())`. Empty unless the
/// plugin declared DFTU_NEED_ARGS.
class ArgRange {
   public:
    ArgRange(const dftu_arg* first, std::uint32_t n) noexcept
        : first_(first), n_(n) {}

    std::uint32_t size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }
    Arg operator[](std::uint32_t i) const noexcept { return Arg{first_[i]}; }

    class iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Arg;
        using difference_type = std::ptrdiff_t;
        using pointer = const Arg*;
        using reference = const Arg&;

        explicit iterator(const dftu_arg* p) noexcept : p_(p) {}
        const Arg& operator*() const noexcept {
            cur_ = Arg{*p_};
            return cur_;
        }
        const Arg* operator->() const noexcept {
            cur_ = Arg{*p_};
            return &cur_;
        }
        iterator& operator++() noexcept {
            ++p_;
            return *this;
        }
        iterator operator++(int) noexcept {
            iterator t = *this;
            ++p_;
            return t;
        }
        bool operator==(const iterator& o) const noexcept { return p_ == o.p_; }
        bool operator!=(const iterator& o) const noexcept { return p_ != o.p_; }

       private:
        const dftu_arg* p_;
        mutable Arg cur_;
    };

    iterator begin() const noexcept { return iterator{first_}; }
    iterator end() const noexcept { return iterator{first_ + n_}; }

   private:
    const dftu_arg* first_;
    std::uint32_t n_;
};

/// Non-owning typed view over one parsed event in a batch. Zero-copy and cheap
/// to copy; valid only for the on_batch call that delivered the batch, since it
/// borrows the underlying dftu_event. String-valued fields are interned ids;
/// resolve them to bytes through a Host.
class Event {
   public:
    Event() = default;
    explicit Event(const dftu_event& e) noexcept : e_(&e) {}

    std::uint64_t pid() const noexcept { return e_->pid; }
    std::uint64_t tid() const noexcept { return e_->tid; }
    std::uint64_t ts() const noexcept { return e_->ts; }
    std::uint64_t dur() const noexcept { return e_->dur; }
    /// Whether the event carried an explicit duration (a complete-phase event).
    bool has_dur() const noexcept { return e_->has_dur != 0; }
    dftu_phase phase() const noexcept {
        return static_cast<dftu_phase>(e_->phase);
    }

    /// Interned id of a string field; DFTU_STR_NONE when absent.
    dftu_str cat_id() const noexcept { return e_->cat; }
    dftu_str name_id() const noexcept { return e_->name; }
    dftu_str fhash_id() const noexcept { return e_->fhash; }
    dftu_str hhash_id() const noexcept { return e_->hhash; }

    /// Whether a string field was present on the event, so a plugin never
    /// compares an id against the raw DFTU_STR_NONE sentinel.
    bool has_cat() const noexcept { return e_->cat != DFTU_STR_NONE; }
    bool has_name() const noexcept { return e_->name != DFTU_STR_NONE; }
    bool has_fhash() const noexcept { return e_->fhash != DFTU_STR_NONE; }
    bool has_hhash() const noexcept { return e_->hhash != DFTU_STR_NONE; }

    /// Resolve a string field to its bytes via `h`; empty when absent. The
    /// returned view is stable for the whole scan.
    std::string_view cat(const Host& h) const { return h.str(e_->cat); }
    std::string_view name(const Host& h) const { return h.str(e_->name); }
    std::string_view fhash(const Host& h) const { return h.str(e_->fhash); }
    std::string_view hhash(const Host& h) const { return h.str(e_->hhash); }

    /// Args are present only when the plugin declared DFTU_NEED_ARGS; otherwise
    /// arg_count() is 0. The views are borrowed for the current on_batch call.
    std::uint32_t arg_count() const noexcept { return e_->arg_count; }
    ArgRange args() const noexcept { return ArgRange{e_->args, e_->arg_count}; }
    Arg arg(std::uint32_t i) const noexcept { return Arg{e_->args[i]}; }

    /// First arg whose interned key matches `key`, or nullopt if none (and when
    /// args were not requested). Keys are flattened to dotted paths.
    std::optional<Arg> find_arg(dftu_str key) const noexcept {
        for (std::uint32_t i = 0; i < e_->arg_count; ++i)
            if (e_->args[i].key == key) return Arg{e_->args[i]};
        return std::nullopt;
    }

    /// Find an arg by its flattened dotted key, given either as one string
    /// ("args.ret") or as path components joined with '.' ("args", "ret").
    /// Interns the composed key on `h` and matches by id; nullopt if absent.
    template <class First, class... Rest>
    std::optional<Arg> find_arg(const Host& h, First&& first,
                                Rest&&... rest) const {
        std::string key{std::string_view{std::forward<First>(first)}};
        (
            [&](std::string_view p) {
                key.push_back('.');
                key.append(p.data(), p.size());
            }(std::string_view{std::forward<Rest>(rest)}),
            ...);
        return find_arg(h.intern(key));
    }

    const dftu_event& raw() const noexcept { return *e_; }

   private:
    const dftu_event* e_ = nullptr;
};

/// Non-owning view over a dftu_batch that yields typed Events for range-based
/// iteration: `for (const Event& e : batch)`. Lightweight value type; valid
/// only for the on_batch call that delivered the batch.
class Batch {
   public:
    explicit Batch(const dftu_batch& b) noexcept : b_(&b) {}

    std::uint32_t size() const noexcept { return b_->count; }
    bool empty() const noexcept { return b_->count == 0; }
    Event operator[](std::uint32_t i) const noexcept {
        return Event{b_->events[i]};
    }

    /// Forward iterator over the batch. operator* returns a reference to a
    /// cached Event owned by the iterator, so `for (const Event& e : batch)`
    /// binds to a real object rather than a temporary.
    class iterator {
       public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Event;
        using difference_type = std::ptrdiff_t;
        using pointer = const Event*;
        using reference = const Event&;

        explicit iterator(const dftu_event* p) noexcept : p_(p) {}
        const Event& operator*() const noexcept {
            cur_ = Event{*p_};
            return cur_;
        }
        const Event* operator->() const noexcept {
            cur_ = Event{*p_};
            return &cur_;
        }
        iterator& operator++() noexcept {
            ++p_;
            return *this;
        }
        iterator operator++(int) noexcept {
            iterator t = *this;
            ++p_;
            return t;
        }
        bool operator==(const iterator& o) const noexcept { return p_ == o.p_; }
        bool operator!=(const iterator& o) const noexcept { return p_ != o.p_; }

       private:
        const dftu_event* p_;
        mutable Event cur_;
    };

    iterator begin() const noexcept { return iterator{b_->events}; }
    iterator end() const noexcept { return iterator{b_->events + b_->count}; }

    const dftu_batch& raw() const noexcept { return *b_; }

   private:
    const dftu_batch* b_;
};

/// dftu_row_val factory for a u64 (integer-monoid) contribution to component
/// `comp`.
inline dftu_row_val row_u64(std::uint32_t comp, std::uint64_t v) noexcept {
    dftu_row_val r{};
    r.comp = comp;
    r.is_f64 = 0;
    r.value.u = v;
    return r;
}

/// dftu_row_val factory for an f64 (floating-monoid) contribution to component
/// `comp`.
inline dftu_row_val row_f64(std::uint32_t comp, double v) noexcept {
    dftu_row_val r{};
    r.comp = comp;
    r.is_f64 = 1;
    r.value.f = v;
    return r;
}

/// Encode one map key/payload slot into the int64 slot the map ABI expects:
/// integrals pass through, floats are bit-cast (F32 in the low bits). Slot
/// order still has to match the map's key_types; STR/BYTES use the Host
/// overload.
template <class T>
inline std::int64_t key_of(T v) noexcept {
    static_assert(std::is_arithmetic_v<T>,
                  "key_of takes an integral or floating slot value");
    if constexpr (std::is_floating_point_v<T>) {
        std::int64_t bits = 0;
        std::memcpy(&bits, &v, sizeof(v));
        return bits;
    } else {
        return static_cast<std::int64_t>(v);
    }
}

/// Encode a STR/BYTES key slot: interns `s` on `h` and carries the id in the
/// int64 slot.
inline std::int64_t key_of(const Host& h, std::string_view s) {
    return static_cast<std::int64_t>(h.intern(s));
}

/// Encode one typed key/payload slot into the int64 the map ABI expects,
/// dispatching on the C++ type: an Interned STR slot carries its id verbatim
/// (no re-intern), an arithmetic slot passes through key_of, any string-like
/// slot is interned on `h`.
template <class T>
inline std::int64_t encode_slot(const Host& h, const T& v) {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 Interned>)
        return static_cast<std::int64_t>(v.id);
    else if constexpr (std::is_arithmetic_v<T>)
        return key_of(v);
    else
        return key_of(h, std::string_view{v});
}

/// Typed, non-owning view over a host-owned mergeable map (see Host::map). Keys
/// are passed as natural C++ values (integers pass through, floats bit-cast,
/// strings interned on the host), so authors never hand-encode the int64 key
/// list. Contribute with the std::map-style `map[key] += n` sugar or with the
/// .add* methods; the map is a write-only accumulator (no per-key readback),
/// merged across workers and materialized by the host, so merge() stays empty
/// and there is nothing to free. Each _at form targets a value component of a
/// product map; the bare form targets component 0. Whole-row/nested feeds
/// (map_add_row, map_add_argrow, map_add_nested_*) stay on the raw Host methods
/// via raw().
template <class... KeyTs>
class Map {
    static_assert(sizeof...(KeyTs) >= 1, "a Map needs at least one key column");
    static constexpr std::size_t N = sizeof...(KeyTs);

   public:
    Map() = default;

    /// Borrowed handle; the host retains ownership. Null if unsupported.
    dftu_map* raw() const noexcept { return m_; }
    const Host& host() const noexcept { return host_; }
    explicit operator bool() const noexcept { return m_ != nullptr; }

    /// std::map-style write proxy for one key. `hits[key] += n` contributes n
    /// at that key (component 0), and `++hits[key]` contributes 1; the map is a
    /// write-only accumulator, so the proxy has no readback. `+=` routes an
    /// integral n to the integer monoid, a floating n to the floating monoid.
    class Ref {
       public:
        template <class V>
        const Ref& operator+=(V v) const {
            m_.add_encoded(key_.data(), v);
            return *this;
        }
        const Ref& operator++() const { return operator+=(std::uint64_t{1}); }
        void operator++(int) const { operator+=(std::uint64_t{1}); }

       private:
        friend class Map;
        Ref(const Map& m, const std::array<std::int64_t, N>& key)
            : m_(m), key_(key) {}
        Map m_;
        std::array<std::int64_t, N> key_;
    };

    /// Subscript by the whole key tuple: `hits[std::tuple{pid, name}] += 1`.
    Ref operator[](const std::tuple<KeyTs...>& key) const {
        return std::apply(
            [&](const KeyTs&... ks) {
                return Ref{*this,
                           std::array<std::int64_t, N>{encode_key(ks)...}};
            },
            key);
    }
    /// Single-column convenience: `hits[pid] += 1`.
    template <class K = std::tuple_element_t<0, std::tuple<KeyTs...>>,
              std::size_t M = N, class = std::enable_if_t<M == 1>>
    Ref operator[](const K& single) const {
        return Ref{*this, std::array<std::int64_t, N>{encode_key(single)}};
    }

    /// Contribute `v` at `keys`; integral v routes to the integer monoid,
    /// floating v to the floating monoid.
    template <class V>
    void add(KeyTs... keys, V v) const {
        if (!m_ || !e_) return;
        const std::int64_t k[N] = {encode_key(keys)...};
        add_value(k, 0, v, true);
    }
    template <class V>
    void add_at(KeyTs... keys, std::uint32_t comp, V v) const {
        if (!m_ || !e_) return;
        const std::int64_t k[N] = {encode_key(keys)...};
        add_value(k, comp, v, false);
    }

    /// Contribute (x, y) to a CORR/COVAR/REGR component (x independent).
    void add_xy(KeyTs... keys, double x, double y) const {
        add_xy_at(keys..., 0, x, y);
    }
    void add_xy_at(KeyTs... keys, std::uint32_t comp, double x,
                   double y) const {
        if (!m_ || !e_ || !e_->map_add_xy_at) return;
        const std::int64_t k[N] = {encode_key(keys)...};
        e_->map_add_xy_at(h(), m_, k, comp, x, y);
    }

    /// Contribute (by, payload) to an ARGMIN/ARGMAX component; the string_view
    /// overload interns an ARG*_STR payload on the host.
    void add_argby(KeyTs... keys, double by, std::int64_t payload) const {
        add_argby_at(keys..., 0, by, payload);
    }
    void add_argby(KeyTs... keys, double by, std::string_view payload) const {
        add_argby_at(keys..., 0, by, payload);
    }
    void add_argby_at(KeyTs... keys, std::uint32_t comp, double by,
                      std::int64_t payload) const {
        if (!m_ || !e_ || !e_->map_add_argby_at) return;
        const std::int64_t k[N] = {encode_key(keys)...};
        e_->map_add_argby_at(h(), m_, k, comp, by, payload);
    }
    void add_argby_at(KeyTs... keys, std::uint32_t comp, double by,
                      std::string_view payload) const {
        add_argby_at(keys..., comp, by,
                     static_cast<std::int64_t>(host_.intern(payload)));
    }

    /// Contribute (by, payload) to a bounded TOPK/BOTTOMK component; k is
    /// constant per component. The string_view overload interns a *_STR
    /// payload.
    void add_topk(KeyTs... keys, std::uint32_t k, double by,
                  std::int64_t payload) const {
        add_topk_at(keys..., 0, k, by, payload);
    }
    void add_topk(KeyTs... keys, std::uint32_t k, double by,
                  std::string_view payload) const {
        add_topk_at(keys..., 0, k, by, payload);
    }
    void add_topk_at(KeyTs... keys, std::uint32_t comp, std::uint32_t k,
                     double by, std::int64_t payload) const {
        if (!m_ || !e_ || !e_->map_add_topk_at) return;
        const std::int64_t key[N] = {encode_key(keys)...};
        e_->map_add_topk_at(h(), m_, key, comp, k, by, payload);
    }
    void add_topk_at(KeyTs... keys, std::uint32_t comp, std::uint32_t k,
                     double by, std::string_view payload) const {
        add_topk_at(keys..., comp, k, by,
                    static_cast<std::int64_t>(host_.intern(payload)));
    }

    /// Append (order_key, element id) to an ordered-list (LIST_*) component.
    void add_ordered(KeyTs... keys, std::int64_t order_key,
                     std::uint64_t element) const {
        add_ordered_at(keys..., 0, order_key, element);
    }
    void add_ordered_at(KeyTs... keys, std::uint32_t comp,
                        std::int64_t order_key, std::uint64_t element) const {
        if (!m_ || !e_ || !e_->map_add_ordered_at) return;
        const std::int64_t k[N] = {encode_key(keys)...};
        e_->map_add_ordered_at(h(), m_, k, comp, order_key, element);
    }

    /// Observe `value` for an APPROX_TOPK component; k is the counter capacity.
    void add_approx_topk(KeyTs... keys, std::uint32_t k,
                         std::int64_t value) const {
        add_approx_topk_at(keys..., 0, k, value);
    }
    void add_approx_topk_at(KeyTs... keys, std::uint32_t comp, std::uint32_t k,
                            std::int64_t value) const {
        if (!m_ || !e_ || !e_->map_add_approx_topk_at) return;
        const std::int64_t key[N] = {encode_key(keys)...};
        e_->map_add_approx_topk_at(h(), m_, key, comp, k, value);
    }

    /// Observe `item` for a bottom-k-by-hash SAMPLE component; k is the size.
    void add_sample(KeyTs... keys, std::uint32_t k, std::int64_t item) const {
        add_sample_at(keys..., 0, k, item);
    }
    void add_sample_at(KeyTs... keys, std::uint32_t comp, std::uint32_t k,
                       std::int64_t item) const {
        if (!m_ || !e_ || !e_->map_add_sample_at) return;
        const std::int64_t key[N] = {encode_key(keys)...};
        e_->map_add_sample_at(h(), m_, key, comp, k, item);
    }

    /// Materialize this map's rows sorted by key (default is unordered).
    void set_ordered(bool ordered = true) const {
        if (!m_ || !e_ || !e_->map_set_ordered) return;
        e_->map_set_ordered(h(), m_, ordered ? 1 : 0);
    }

   private:
    friend class Host;
    Map(Host host, const dftu_ext_map* e, dftu_map* m)
        : host_(host), e_(e), m_(m) {}

    void* h() const noexcept { return host_.raw()->h; }

    template <class T>
    std::int64_t encode_key(const T& v) const {
        return encode_slot(host_, v);
    }

    template <class V>
    void add_encoded(const std::int64_t* k, V v) const {
        if (!m_ || !e_) return;
        add_value(k, 0, v, true);
    }

    template <class V>
    void add_value(const std::int64_t* k, std::uint32_t comp, V v,
                   bool comp0) const {
        if constexpr (std::is_floating_point_v<V>) {
            if (comp0) {
                if (e_->map_add_f64)
                    e_->map_add_f64(h(), m_, k, static_cast<double>(v));
            } else if (e_->map_add_f64_at) {
                e_->map_add_f64_at(h(), m_, k, comp, static_cast<double>(v));
            }
        } else {
            if (comp0) {
                if (e_->map_add_u64)
                    e_->map_add_u64(h(), m_, k, static_cast<std::uint64_t>(v));
            } else if (e_->map_add_u64_at) {
                e_->map_add_u64_at(h(), m_, k, comp,
                                   static_cast<std::uint64_t>(v));
            }
        }
    }

    Host host_{nullptr};
    const dftu_ext_map* e_ = nullptr;
    dftu_map* m_ = nullptr;
};

template <class... KeyTs>
inline Map<KeyTs...> Host::map(const char* name, Monoid kind,
                               Key<KeyTs...>) const {
    const dftu_type kt[] = {type_of<KeyTs>()...};
    const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
    dftu_map* m = e && e->map_new
                      ? e->map_new(h_->h, name, kt, sizeof...(KeyTs),
                                   static_cast<dftu_monoid_kind>(kind))
                      : nullptr;
    return Map<KeyTs...>{*this, e, m};
}

template <class... KeyTs>
inline Map<KeyTs...> Host::counter_map(const char* name,
                                       Key<KeyTs...> k) const {
    return map(name, Monoid::Counter, k);
}

template <class... KeyTs>
inline Map<KeyTs...> Host::sum_map(const char* name, Key<KeyTs...> k) const {
    return map(name, Monoid::Sum_F64, k);
}

template <class... KeyTs>
inline Map<KeyTs...> Host::product_map(const char* name,
                                       std::initializer_list<Monoid> values,
                                       Key<KeyTs...>) const {
    const dftu_type kt[] = {type_of<KeyTs>()...};
    std::vector<dftu_monoid_kind> vs;
    vs.reserve(values.size());
    for (Monoid m : values) vs.push_back(static_cast<dftu_monoid_kind>(m));
    const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
    dftu_map* m =
        e && e->map_new_product
            ? e->map_new_product(h_->h, name, kt, sizeof...(KeyTs), vs.data(),
                                 static_cast<std::uint32_t>(vs.size()))
            : nullptr;
    return Map<KeyTs...>{*this, e, m};
}

/// Typed, non-owning view over a host-owned nested-preserved map (see
/// Host::nested_map). Keys are passed as natural C++ value tuples (integers
/// pass through, floats bit-cast, strings interned on the host, Interned ids
/// carried verbatim), so authors never hand-encode the int64 key lists. The map
/// is a write-only accumulator merged across workers and materialized by the
/// host; there is nothing to read back or free.
template <class... OuterTs, class... InnerTs>
class NestedMap<Key<OuterTs...>, Key<InnerTs...>> {
    static_assert(sizeof...(OuterTs) >= 1,
                  "a nested map needs at least one outer key column");
    static_assert(sizeof...(InnerTs) >= 1,
                  "a nested map needs at least one inner key column");
    static constexpr std::size_t NO = sizeof...(OuterTs);
    static constexpr std::size_t NI = sizeof...(InnerTs);

   public:
    NestedMap() = default;

    /// Borrowed handle; the host retains ownership. Null if unsupported.
    dftu_map* raw() const noexcept { return m_; }
    const Host& host() const noexcept { return host_; }
    explicit operator bool() const noexcept { return m_ != nullptr; }

    /// Contribute `v` to value component `comp` at (outer, inner); an integral
    /// `v` routes to the integer monoid, a floating `v` to the floating monoid.
    template <class V>
    void add(const std::tuple<OuterTs...>& outer,
             const std::tuple<InnerTs...>& inner, std::uint32_t comp,
             V v) const {
        if (!m_ || !e_) return;
        const std::array<std::int64_t, NO> ok = encode_tuple(outer);
        const std::array<std::int64_t, NI> ik = encode_tuple(inner);
        if constexpr (std::is_floating_point_v<V>) {
            if (e_->map_add_nested_f64)
                e_->map_add_nested_f64(h(), m_, ok.data(), ik.data(), comp,
                                       static_cast<double>(v));
        } else {
            if (e_->map_add_nested_u64)
                e_->map_add_nested_u64(h(), m_, ok.data(), ik.data(), comp,
                                       static_cast<std::uint64_t>(v));
        }
    }
    /// Single-component convenience: contribute to component 0.
    template <class V>
    void add(const std::tuple<OuterTs...>& outer,
             const std::tuple<InnerTs...>& inner, V v) const {
        add(outer, inner, 0u, v);
    }

   private:
    friend class Host;
    NestedMap(Host host, const dftu_ext_map* e, dftu_map* m)
        : host_(host), e_(e), m_(m) {}

    void* h() const noexcept { return host_.raw()->h; }

    template <class... Ts>
    std::array<std::int64_t, sizeof...(Ts)> encode_tuple(
        const std::tuple<Ts...>& t) const {
        return std::apply(
            [&](const Ts&... xs) {
                return std::array<std::int64_t, sizeof...(Ts)>{
                    encode_slot(host_, xs)...};
            },
            t);
    }

    Host host_{nullptr};
    const dftu_ext_map* e_ = nullptr;
    dftu_map* m_ = nullptr;
};

template <class... OuterTs, class... InnerTs>
inline NestedMap<Key<OuterTs...>, Key<InnerTs...>> Host::nested_map(
    const char* name, std::initializer_list<Monoid> values, Key<OuterTs...>,
    Key<InnerTs...>) const {
    const dftu_type okt[] = {type_of<OuterTs>()...};
    const dftu_type ikt[] = {type_of<InnerTs>()...};
    std::vector<dftu_monoid_kind> vs;
    vs.reserve(values.size());
    for (Monoid m : values) vs.push_back(static_cast<dftu_monoid_kind>(m));
    const dftu_ext_map* e = ext(DFTU_EXT_MAP, map_ext_);
    dftu_map* m = e && e->map_new_nested
                      ? e->map_new_nested(h_->h, name, okt, sizeof...(OuterTs),
                                          ikt, sizeof...(InnerTs), vs.data(),
                                          static_cast<std::uint32_t>(vs.size()))
                      : nullptr;
    return NestedMap<Key<OuterTs...>, Key<InnerTs...>>{*this, e, m};
}

/** View over the config tree; a Slice may keep returned string_views since the
   host-owned tree outlives it. */
class Config {
   public:
    Config() = default;
    explicit Config(const dftu_value* root) : root_(root) {}

    const dftu_value* find(std::string_view key) const {
        if (!root_ || root_->kind != DFTU_VAL_OBJECT) return nullptr;
        for (std::uint32_t i = 0; i < root_->count; ++i) {
            const dftu_member& m = root_->as.members[i];
            if (std::string_view{m.key, m.key_len} == key) return m.value;
        }
        return nullptr;
    }

    std::string_view get(std::string_view key,
                         std::string_view dflt = {}) const {
        const dftu_value* v = find(key);
        return (v && v->kind == DFTU_VAL_STR)
                   ? std::string_view{v->as.str, v->count}
                   : dflt;
    }
    std::int64_t get_int(std::string_view key, std::int64_t dflt = 0) const {
        return dftu_as_i64(find(key), dflt);
    }
    double get_double(std::string_view key, double dflt = 0.0) const {
        return dftu_as_f64(find(key), dflt);
    }
    bool get_bool(std::string_view key, bool dflt = false) const {
        return dftu_as_bool(find(key), dflt ? 1 : 0) != 0;
    }
    Config child(std::string_view key) const {
        const dftu_value* v = find(key);
        return Config{(v && v->kind == DFTU_VAL_OBJECT) ? v : nullptr};
    }

    /// The raw ARRAY value at `key`, or null if absent or not an array. Iterate
    /// its elements as `arr->as.items[i]` over `arr->count`.
    const dftu_value* array(std::string_view key) const {
        const dftu_value* v = find(key);
        return (v && v->kind == DFTU_VAL_ARRAY) ? v : nullptr;
    }
    /// The ARRAY at `key` coerced to int64 per element (dftu_as_i64 rules);
    /// empty if the key is absent or not an array.
    std::vector<std::int64_t> get_int_array(std::string_view key) const {
        std::vector<std::int64_t> out;
        if (const dftu_value* a = array(key)) {
            out.reserve(a->count);
            for (std::uint32_t i = 0; i < a->count; ++i)
                out.push_back(dftu_as_i64(&a->as.items[i], 0));
        }
        return out;
    }
    /// The ARRAY at `key` coerced to double per element (dftu_as_f64 rules);
    /// empty if the key is absent or not an array.
    std::vector<double> get_double_array(std::string_view key) const {
        std::vector<double> out;
        if (const dftu_value* a = array(key)) {
            out.reserve(a->count);
            for (std::uint32_t i = 0; i < a->count; ++i)
                out.push_back(dftu_as_f64(&a->as.items[i], 0.0));
        }
        return out;
    }
    /// The ARRAY at `key` as string_views (non-STR elements yield an empty
    /// view); the views borrow the host-owned config tree. Empty if the key is
    /// absent or not an array.
    std::vector<std::string_view> get_string_array(std::string_view key) const {
        std::vector<std::string_view> out;
        if (const dftu_value* a = array(key)) {
            out.reserve(a->count);
            for (std::uint32_t i = 0; i < a->count; ++i) {
                const dftu_value& e = a->as.items[i];
                out.push_back(e.kind == DFTU_VAL_STR
                                  ? std::string_view{e.as.str, e.count}
                                  : std::string_view{});
            }
        }
        return out;
    }

    explicit operator bool() const { return root_ != nullptr; }
    const dftu_value* raw() const { return root_; }

   private:
    const dftu_value* root_ = nullptr;
};

namespace detail {

template <class Slice>
constexpr std::uint32_t slice_needs() {
    if constexpr (requires { Slice::needs; })
        return Slice::needs;
    else
        return 0;
}

template <class Slice>
struct Holder {
    dftu_plugin vt{};
    Config config;
    std::string plan; /**< backs plan_query's const char* */
};

template <class Slice>
Holder<Slice>* holder_of(void* self) {
    return static_cast<Holder<Slice>*>(self);
}

/// A Slice whose async on_batch takes the ergonomic Batch view.
template <class Slice>
concept AsyncBatchView = requires(Slice& s, const Batch& b, Host h) {
    { s.on_batch(b, h) } -> std::same_as<Task>;
};

/// A Slice whose async on_batch takes the raw C dftu_batch (legacy signature).
template <class Slice>
concept AsyncBatch = requires(Slice& s, const dftu_batch& b, Host h) {
    { s.on_batch(b, h) } -> std::same_as<Task>;
};

/// A Slice whose synchronous step takes the ergonomic Batch view.
template <class Slice>
concept BatchViewStep =
    requires(Slice& s, const Batch& b, Host h) { s.step(b, h); };

template <class Slice>
concept AsyncFinalize = requires(Slice& s, Host h) {
    { s.on_finalize(h) } -> std::same_as<Task>;
};

/// Resume the driven coroutine one step; NULL once it has run to completion.
inline dftu_task* step_thunk(void* coro) {
    auto h = std::coroutine_handle<Task::promise_type>::from_address(coro);
    h.resume();
    if (!h.done()) return h.promise().pending;
    if (h.promise().exc) {
        if (const dftu_host* host = h.promise().host) {
            try {
                std::rethrow_exception(h.promise().exc);
            } catch (const std::exception& e) {
                Host{host}.log(DFTU_LOG_ERROR, e.what());
            } catch (...) {
                Host{host}.log(DFTU_LOG_ERROR, "plugin coroutine threw");
            }
        }
    }
    h.destroy();
    return nullptr;
}

template <class Slice, class Coro>
dftu_task* drive_coro(const dftu_host* host, Coro&& coro) {
    Task t = std::forward<Coro>(coro);
    auto h = t.release();
    h.promise().host = host;
    const dftu_ext_coro* c =
        host->get_extension ? static_cast<const dftu_ext_coro*>(
                                  host->get_extension(host->h, DFTU_EXT_CORO))
                            : nullptr;
    return c && c->drive ? c->drive(host->h, &step_thunk, h.address())
                         : nullptr;
}

/// A Slice that publishes capabilities via a static `provides()` returning a
/// range of dftu_capability (build them with capability()).
template <class Slice>
concept DeclaresProvides = requires {
    { std::begin(Slice::provides()) };
    { std::end(Slice::provides()) };
};

/// A Slice that consumes capabilities via a static `requires_caps()` returning
/// a range of dftu_requirement (build them with requirement()). Named
/// requires_caps because `requires` is a C++20 keyword.
template <class Slice>
concept DeclaresRequires = requires {
    { std::begin(Slice::requires_caps()) };
    { std::end(Slice::requires_caps()) };
};

/// A Slice that reacts to the post-declare capability resolution via a static
/// `on_resolve(Host)`; use Host::provider_count / Host::provider_best inside.
template <class Slice>
concept DeclaresResolve = requires(Host h) { Slice::on_resolve(h); };

/// A Slice that declares any comms hook, so make_plugin wires get_extension.
template <class Slice>
concept HasComms = DeclaresProvides<Slice> || DeclaresRequires<Slice> ||
                   DeclaresResolve<Slice>;

template <class Slice>
std::uint32_t comms_provides(void*, dftu_capability* out, std::uint32_t max) {
    std::uint32_t n = 0;
    for (const dftu_capability& c : Slice::provides()) {
        if (out && n < max) out[n] = c;
        ++n;
    }
    return n;
}

template <class Slice>
std::uint32_t comms_requires(void*, dftu_requirement* out, std::uint32_t max) {
    std::uint32_t n = 0;
    for (const dftu_requirement& r : Slice::requires_caps()) {
        if (out && n < max) out[n] = r;
        ++n;
    }
    return n;
}

template <class Slice>
void comms_resolve(void*, const dftu_host* host) {
    try {
        Slice::on_resolve(Host{host});
    } catch (const std::exception& e) {
        Host{host}.log(DFTU_LOG_ERROR, e.what());
    } catch (...) {
        Host{host}.log(DFTU_LOG_ERROR, "plugin resolve threw");
    }
}

template <class Slice>
constexpr auto provides_thunk() {
    if constexpr (DeclaresProvides<Slice>)
        return &comms_provides<Slice>;
    else
        return static_cast<decltype(dftu_plugin_comms::provides)>(nullptr);
}
template <class Slice>
constexpr auto requires_thunk() {
    if constexpr (DeclaresRequires<Slice>)
        return &comms_requires<Slice>;
    else
        return static_cast<decltype(dftu_plugin_comms::require_caps)>(nullptr);
}
template <class Slice>
constexpr auto resolve_thunk() {
    if constexpr (DeclaresResolve<Slice>)
        return &comms_resolve<Slice>;
    else
        return static_cast<decltype(dftu_plugin_comms::resolve)>(nullptr);
}

/// Per-Slice comms table with static storage; only ODR-used when HasComms.
template <class Slice>
inline const dftu_plugin_comms comms_table = {
    provides_thunk<Slice>(), requires_thunk<Slice>(), resolve_thunk<Slice>()};

}  // namespace detail

/** Build a dftu_plugin from a Slice providing Slice(const Config&), merge, and
   either sync step/finalize or a Task-returning on_batch/on_finalize coroutine,
   plus optionally `static constexpr uint32_t needs`. For inter-plugin comms a
   Slice may also declare any of: `static ... provides()` (a range of
   dftu_capability, built with capability()), `static ... requires_caps()` (a
   range of dftu_requirement, built with requirement()), and `static void
   on_resolve(Host)`; when present, make_plugin wires dftu_plugin::get_extension
   to a per-Slice dftu_plugin_comms so the host discovers and resolves them.
   Exceptions must not escape the ABI boundary, so every callback catches. */
template <class Slice>
dftu_plugin* make_plugin(const dftu_value* config) {
    auto* hd = new detail::Holder<Slice>();
    hd->config = Config(config);
    hd->plan = std::string(hd->config.get("query"));

    dftu_plugin& vt = hd->vt;
    vt.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    vt.self = hd;

    vt.needs = [](void*) -> std::uint32_t {
        return detail::slice_needs<Slice>();
    };

    vt.plan_query = [](void* self) -> const char* {
        auto* h = detail::holder_of<Slice>(self);
        return h->plan.empty() ? nullptr : h->plan.c_str();
    };

    vt.make_slice = [](void* self) -> void* {
        try {
            return new Slice(detail::holder_of<Slice>(self)->config);
        } catch (...) {
            return nullptr;
        }
    };

    vt.on_batch = [](void* slice, const dftu_batch* b,
                     const dftu_host* host) -> dftu_task* {
        try {
            Slice* sl = static_cast<Slice*>(slice);
            Host h{host};
            if constexpr (detail::AsyncBatchView<Slice>) {
                return detail::drive_coro<Slice>(host,
                                                 sl->on_batch(Batch{*b}, h));
            } else if constexpr (detail::AsyncBatch<Slice>) {
                return detail::drive_coro<Slice>(host, sl->on_batch(*b, h));
            } else if constexpr (detail::BatchViewStep<Slice>) {
                sl->step(Batch{*b}, h);
            } else {
                sl->step(*b, h);
            }
        } catch (const std::exception& e) {
            Host{host}.log(DFTU_LOG_ERROR, e.what());
        } catch (...) {
            Host{host}.log(DFTU_LOG_ERROR, "plugin step threw");
        }
        return nullptr;
    };

    vt.merge = [](void* into, void* other) {
        try {
            static_cast<Slice*>(into)->merge(*static_cast<Slice*>(other));
        } catch (...) {
        }
    };

    vt.on_finalize = [](void* slice, const dftu_host* host) -> dftu_task* {
        try {
            if constexpr (detail::AsyncFinalize<Slice>) {
                return detail::drive_coro<Slice>(
                    host, static_cast<Slice*>(slice)->on_finalize(Host{host}));
            } else {
                static_cast<Slice*>(slice)->finalize(Host{host});
            }
        } catch (const std::exception& e) {
            Host{host}.log(DFTU_LOG_ERROR, e.what());
        } catch (...) {
            Host{host}.log(DFTU_LOG_ERROR, "plugin finalize threw");
        }
        return nullptr;
    };

    vt.destroy_slice = [](void* slice) { delete static_cast<Slice*>(slice); };
    vt.destroy = [](void* self) { delete detail::holder_of<Slice>(self); };

    if constexpr (detail::HasComms<Slice>) {
        vt.get_extension = [](void*, const char* ext_id) -> const void* {
            if (ext_id && std::strcmp(ext_id, DFTU_EXT_COMMS) == 0)
                return &detail::comms_table<Slice>;
            return nullptr;
        };
    }

    return &vt;
}

}  // namespace dftracer::utils::plugins

#endif /* DFTRACER_UTILS_PLUGINS_PLUGIN_H */
