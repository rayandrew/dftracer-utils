#ifndef DFTRACER_UTILS_PLUGINS_PLUGIN_MAP_H
#define DFTRACER_UTILS_PLUGINS_PLUGIN_MAP_H

#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/arrow_abi.h>
#include <dftracer/utils/plugins/owned_arrow.h>
#include <dftracer/utils/plugins/plugin/async.h>
#include <dftracer/utils/plugins/plugin/types.h>
#include <dftracer/utils/plugins/result_registry.h>

#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dftracer::utils::plugins {

class Writer;
class Agg;
template <class T>
class OutPort;
template <class T>
class InPort;

class Host {
   public:
    explicit Host(const dftu_host* h) : h_(h) {}

    std::string_view str(StrId id) const {
        if (id.absent()) return {};
        std::uint32_t n = 0;
        const char* p = h_->resolve(h_->h, id.raw(), &n);
        return p ? std::string_view{p, n} : std::string_view{};
    }
    StrId intern(std::string_view s) const {
        return StrId{
            h_->intern(h_->h, s.data(), static_cast<std::uint32_t>(s.size()))};
    }
    void log(dftu_log_level level, std::string_view msg) const {
        h_->log(h_->h, static_cast<std::uint8_t>(level), msg.data(),
                static_cast<std::uint32_t>(msg.size()));
    }
    void log(LogLevel level, std::string_view msg) const {
        log(static_cast<dftu_log_level>(level), msg);
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
    /// Compile and wrap as a non-owning Query bound to this host; see
    /// query_compile for lifetime.
    class Query compile(std::string_view src) const;
    /// compile() for a predicate built with `F`/`Field`.
    class Query compile(const Expr& e) const;

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
    /// Version-wrapped form of provider_best; nullopt if no provider satisfies
    /// `req`.
    std::optional<Version> provider_best_v(const dftu_requirement& req) const {
        dftu_version v{};
        return provider_best(req, &v) ? std::optional<Version>{Version{v}}
                                      : std::nullopt;
    }
    /// Version-wrapped provider_best for a wrapped Requirement.
    std::optional<Version> provider_best_v(const Requirement& req) const {
        return provider_best_v(req.raw());
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
    /// Scan `path` (auto-indexed), invoking `fn` once per event as either
    /// `fn(const Event&)` or `fn(const dftu_event&)`. `fn` is borrowed for the
    /// call. 0 on success.
    template <class Fn>
    int trace_read(const char* path, Fn&& fn) const;

    /// RAII trace writer for `path` (a gzip .pfw.gz); empty (`!writer`) if the
    /// host lacks the trace group.
    class TraceWriter trace_writer(const char* path) const;

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
    dftu_agg* agg_new(const char* name, const char* const* key_names,
                      std::uint32_t key_n, const dftu_agg_col* specs,
                      std::uint32_t spec_n) const {
        const dftu_ext_agg* e = ext(DFTU_EXT_AGG, agg_ext_);
        return e && e->agg_new
                   ? e->agg_new(h_->h, name, key_names, key_n, specs, spec_n)
                   : nullptr;
    }
    void agg_accumulate(dftu_agg* a, const dftu_dataframe* df) const {
        const dftu_ext_agg* e = ext(DFTU_EXT_AGG, agg_ext_);
        if (e && e->agg_accumulate) e->agg_accumulate(h_->h, a, df);
    }
    /// Get-or-create a named cross-batch aggregation accumulator grouping by
    /// `key_names` and computing each of `cols`; empty (`!agg`) if the host
    /// lacks the agg group or agg_new rejects the spec (a bad op code or
    /// missing output name). See dftu_ext_agg::agg_new; a name seen before
    /// returns the existing accumulator and ignores `cols`.
    Agg agg(const char* name, std::initializer_list<const char*> key_names,
            std::initializer_list<AggCol> cols) const;

    /// The cross-worker-merged, finalized result of the accumulator named
    /// `name` (any plugin's), as an owned frame. Call at on_finalize; the
    /// producing plugin must be registered first. Empty (`!frame.handle`) when
    /// no plugin produced that name.
    OwnedDataFrame agg_result(const char* name) const;

    /// Look up and run a registered dataframe column op by name on `in` (see
    /// dftu_ext_ops::run); NULL if the host lacks the ops group, `name` is
    /// unregistered, or the call mismatches the op's kind/arity/shape. The
    /// result is newly owned by the caller (free with dftu_series_free).
    dftu_series* run_op(const char* name,
                        std::initializer_list<const dftu_series*> in,
                        const dftu_op_arg* args = nullptr) const {
        const dftu_ext_ops* e = ext(DFTU_EXT_OPS, ops_ext_);
        if (!e || !e->run) return nullptr;
        std::vector<const dftu_series*> vin(in);
        return e->run(h_->h, name, vin.data(),
                      static_cast<std::uint32_t>(vin.size()), args);
    }
    /// run_op for a signature whose return token is FRAME; the result is
    /// newly owned by the caller (free with dftu_dataframe_free).
    dftu_dataframe* run_op_frame(
        const char* name, std::initializer_list<const dftu_dataframe*> in,
        const dftu_op_arg* args = nullptr) const {
        const dftu_ext_ops* e = ext(DFTU_EXT_OPS, ops_ext_);
        if (!e || !e->run_frame) return nullptr;
        std::vector<const dftu_dataframe*> vin(in);
        return e->run_frame(h_->h, name, vin.data(),
                            static_cast<std::uint32_t>(vin.size()), args);
    }
    /// run_op for a signature whose return token is SCALAR/I64/BOOL, reducing
    /// one column; `ok` (if given) is set false on a NULL/kind/shape mismatch
    /// or unregistered name.
    dftu_scalar run_op_aggregate(const char* name, const dftu_series* in,
                                 const dftu_op_arg* args = nullptr,
                                 bool* ok = nullptr) const {
        const dftu_ext_ops* e = ext(DFTU_EXT_OPS, ops_ext_);
        dftu_scalar out{};
        int run_ok = 0;
        if (e && e->run_aggregate)
            e->run_aggregate(h_->h, name, &in, 1, args, &out, &run_ok);
        if (ok) *ok = run_ok != 0;
        return out;
    }
    /// The registered op named `name` (built-in or user), or NULL if none or
    /// the host lacks the ops group. See dftu_op_find.
    const dftu_op_desc* find_op(const char* name) const {
        const dftu_ext_ops* e = ext(DFTU_EXT_OPS, ops_ext_);
        return e && e->find ? e->find(h_->h, name) : nullptr;
    }
    /// Register a user op in the host's shared registry; see dftu_op_register.
    /// Returns non-zero on a NULL desc/name, an already-registered name, or if
    /// the host lacks the ops group.
    int register_op(const dftu_op_desc* desc) const {
        const dftu_ext_ops* e = ext(DFTU_EXT_OPS, ops_ext_);
        return e && e->register_op ? e->register_op(h_->h, desc) : -1;
    }

    /// Named result channel: emit an opaque blob (the host copies `len` bytes)
    /// or a user-schema Arrow array (the host moves it); both surface from
    /// Plugins::run keyed by name. Best called at on_finalize.
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
    /// Hands a native dataframe result to the host, which takes ownership; -1
    /// if the host lacks the result channel or the emit_frame slot (`df` is
    /// left intact so it frees normally). Same finalize/collision semantics as
    /// emit_result.
    int emit_result_frame(const char* name, OwnedDataFrame&& df) const {
        const dftu_ext_result* e = ext(DFTU_EXT_RESULT, result_ext_);
        if (!e || !e->emit_frame) return -1;
        return e->emit_frame(h_->h, name, df.release());
    }
    /// OwnedLazyFrame overload of emit_result_lazyframe; -1 leaves `lf` intact
    /// so it frees normally.
    int emit_result_lazyframe(const char* name, OwnedLazyFrame&& lf) const {
        const dftu_ext_result* e = ext(DFTU_EXT_RESULT, result_ext_);
        if (!e || !e->emit_lazyframe) return -1;
        return e->emit_lazyframe(h_->h, name, lf.release());
    }

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
    mutable const dftu_ext_sketch* sketch_ext_ = nullptr;
    mutable const dftu_ext_arrow* arrow_ext_ = nullptr;
    mutable const dftu_ext_trace* trace_ext_ = nullptr;
    mutable const dftu_ext_ports* ports_ext_ = nullptr;
    mutable const dftu_ext_result* result_ext_ = nullptr;
    mutable const dftu_ext_agg* agg_ext_ = nullptr;
    mutable const dftu_ext_ops* ops_ext_ = nullptr;
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

/// Non-owning view over a host-owned cross-batch aggregation accumulator (see
/// Host::agg). The host owns the handle (freed at fold teardown), so there is
/// nothing for the plugin to free.
class Agg {
   public:
    Agg() = default;

    /// Fold one batch's key/value columns (looked up by name in `df`) into
    /// this accumulator. Serial per accumulator; one slice's accumulator is
    /// touched by one thread.
    void accumulate(const dftu_dataframe* df) const {
        host_.agg_accumulate(a_, df);
    }
    void accumulate(const OwnedDataFrame& df) const { accumulate(df.handle); }

    /// Borrowed handle; the host retains ownership. Null if unsupported.
    dftu_agg* raw() const noexcept { return a_; }
    explicit operator bool() const noexcept { return a_ != nullptr; }

   private:
    friend class Host;
    Agg(Host host, dftu_agg* a) noexcept : host_(host), a_(a) {}

    Host host_{nullptr};
    dftu_agg* a_ = nullptr;
};

inline Agg Host::agg(const char* name,
                     std::initializer_list<const char*> key_names,
                     std::initializer_list<AggCol> cols) const {
    std::vector<const char*> keys(key_names.begin(), key_names.end());
    std::vector<dftu_agg_col> specs;
    specs.reserve(cols.size());
    for (const AggCol& c : cols) specs.push_back(c.raw());
    dftu_agg* a =
        agg_new(name, keys.data(), static_cast<std::uint32_t>(keys.size()),
                specs.data(), static_cast<std::uint32_t>(specs.size()));
    return Agg{*this, a};
}

inline OwnedDataFrame Host::agg_result(const char* name) const {
    const dftu_ext_agg* e = ext(DFTU_EXT_AGG, agg_ext_);
    return OwnedDataFrame{e && e->agg_result ? e->agg_result(h_->h, name)
                                             : nullptr};
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

    StrId key_id() const noexcept { return StrId{a_->key}; }
    std::string_view key(const Host& h) const { return h.str(key_id()); }

    ArgKind kind() const noexcept { return static_cast<ArgKind>(a_->kind); }
    bool is_i64() const noexcept { return a_->kind == DFTU_ARG_I64; }
    bool is_f64() const noexcept { return a_->kind == DFTU_ARG_F64; }
    bool is_str() const noexcept { return a_->kind == DFTU_ARG_STR; }

    /// Read the accessor matching kind(); another slot holds a stale value.
    std::int64_t i64() const noexcept { return a_->v.i64; }
    double f64() const noexcept { return a_->v.f64; }
    StrId str_id() const noexcept { return StrId{a_->v.str}; }
    std::string_view str(const Host& h) const { return h.str(str_id()); }

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
    Phase phase() const noexcept {
        return static_cast<Phase>(static_cast<dftu_phase>(e_->phase));
    }

    /// Interned id of a string field; absent() (DFTU_STR_NONE) when not
    /// present.
    StrId cat_id() const noexcept { return StrId{e_->cat}; }
    StrId name_id() const noexcept { return StrId{e_->name}; }
    StrId fhash_id() const noexcept { return StrId{e_->fhash}; }
    StrId hhash_id() const noexcept { return StrId{e_->hhash}; }

    /// Whether a string field was present on the event, so a plugin never
    /// compares an id against the raw DFTU_STR_NONE sentinel.
    bool has_cat() const noexcept { return e_->cat != DFTU_STR_NONE; }
    bool has_name() const noexcept { return e_->name != DFTU_STR_NONE; }
    bool has_fhash() const noexcept { return e_->fhash != DFTU_STR_NONE; }
    bool has_hhash() const noexcept { return e_->hhash != DFTU_STR_NONE; }

    /// Resolve a string field to its bytes via `h`; empty when absent. The
    /// returned view is stable for the whole scan.
    std::string_view cat(const Host& h) const { return h.str(cat_id()); }
    std::string_view name(const Host& h) const { return h.str(name_id()); }
    std::string_view fhash(const Host& h) const { return h.str(fhash_id()); }
    std::string_view hhash(const Host& h) const { return h.str(hhash_id()); }

    /// Args are present only when the plugin declared DFTU_NEED_ARGS; otherwise
    /// arg_count() is 0. The views are borrowed for the current on_batch call.
    std::uint32_t arg_count() const noexcept { return e_->arg_count; }
    ArgRange args() const noexcept { return ArgRange{e_->args, e_->arg_count}; }
    Arg arg(std::uint32_t i) const noexcept { return Arg{e_->args[i]}; }

    /// First arg whose interned key matches `key`, or nullopt if none (and when
    /// args were not requested). Keys are flattened to dotted paths.
    std::optional<Arg> find_arg(StrId key) const noexcept {
        for (std::uint32_t i = 0; i < e_->arg_count; ++i)
            if (e_->args[i].key == key.raw()) return Arg{e_->args[i]};
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

template <class Fn>
inline int Host::trace_read(const char* path, Fn&& fn) const {
    auto thunk = [](const void* item, void* ud) {
        const dftu_event& ev = *static_cast<const dftu_event*>(item);
        auto& f = *static_cast<std::decay_t<Fn>*>(ud);
        if constexpr (std::is_invocable_v<std::decay_t<Fn>&, const Event&>)
            f(Event{ev});
        else
            f(ev);
    };
    return trace_read(
        path, thunk,
        const_cast<void*>(static_cast<const void*>(std::addressof(fn))));
}

/// Non-owning view over a compiled dftu_query (see Host::compile). Host-owned
/// and valid for the scan; the plugin never frees it.
class Query {
   public:
    Query() = default;

    explicit operator bool() const noexcept { return q_ != nullptr; }

    bool matches(const dftu_event& e) const {
        return host_.query_matches(q_, e);
    }
    bool matches(const Event& e) const { return matches(e.raw()); }

    /// Borrowed handle; the host retains ownership. Null if unsupported.
    dftu_query* raw() const noexcept { return q_; }

   private:
    friend class Host;
    Query(Host host, dftu_query* q) noexcept : host_(host), q_(q) {}

    Host host_{nullptr};
    dftu_query* q_ = nullptr;
};

inline Query Host::compile(std::string_view src) const {
    return Query{*this, query_compile(src)};
}
inline Query Host::compile(const Expr& e) const {
    return Query{*this, query_compile(e)};
}

/// Move-only RAII owner of a host trace writer (see Host::trace_writer):
/// opens a gzip .pfw.gz for writing, appends events, then closes exactly once,
/// in close() or the destructor, whichever runs first.
class TraceWriter {
   public:
    TraceWriter() = default;
    TraceWriter(TraceWriter&& o) noexcept : host_(o.host_), w_(o.w_) {
        o.w_ = nullptr;
    }
    TraceWriter& operator=(TraceWriter&& o) noexcept {
        if (this != &o) {
            close();
            host_ = o.host_;
            w_ = o.w_;
            o.w_ = nullptr;
        }
        return *this;
    }
    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;
    ~TraceWriter() { close(); }

    explicit operator bool() const noexcept { return w_ != nullptr; }

    int write(std::span<const dftu_event> evs) const {
        return w_ ? host_.trace_write(w_, evs.data(),
                                      static_cast<std::uint32_t>(evs.size()))
                  : -1;
    }
    int write(const Batch& b) const {
        return write(
            std::span<const dftu_event>{b.raw().events, b.raw().count});
    }
    /// Idempotent: a no-op returning 0 once already closed.
    int close() {
        int rc = w_ ? host_.trace_close(w_) : 0;
        w_ = nullptr;
        return rc;
    }

    /// Borrowed handle; the TraceWriter retains ownership. Null if unsupported.
    dftu_trace_writer* raw() const noexcept { return w_; }

   private:
    friend class Host;
    TraceWriter(Host host, dftu_trace_writer* w) noexcept
        : host_(host), w_(w) {}

    Host host_{nullptr};
    dftu_trace_writer* w_ = nullptr;
};

inline TraceWriter Host::trace_writer(const char* path) const {
    return TraceWriter{*this, trace_open_write(path)};
}

}  // namespace dftracer::utils::plugins

#endif /* DFTRACER_UTILS_PLUGINS_PLUGIN_MAP_H */
