#ifndef DFTRACER_UTILS_PLUGINS_ABI_H
#define DFTRACER_UTILS_PLUGINS_ABI_H

/** @file
 * Stable C ABI for dftracer-utils plugins; include only this header.
 */

#include <dftracer/utils/core/common/export.h>
#include <dftracer/utils/dataframe/agg_op_codes.h> /* DFTU_AGG_* for dftu_agg_col */
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#include <sys/socket.h> /* struct sockaddr, socklen_t for dftu_io::accept */
#include <sys/uio.h>    /* struct iovec for the vectored dftu_io ops */

#ifdef __cplusplus
extern "C" {
#endif

/** The standard Arrow C Data Interface structs (defined in arrow_abi.h); only
   opaque pointers to them cross this ABI. */
struct ArrowArray;
struct ArrowSchema;

#define DFTRACER_PLUGIN_ABI_VERSION 1u

/** Interned string id; resolve for bytes stable for the whole scan.
   DFTU_STR_NONE marks an absent field. */
typedef uint32_t dftu_str;
#define DFTU_STR_NONE ((dftu_str)0xFFFFFFFFu)

typedef enum {
    DFTU_T_VOID = 0,
    DFTU_T_F64,
    DFTU_T_I64,
    DFTU_T_STR,
    DFTU_T_EVENT,
    DFTU_T_BATCH,
    DFTU_T_BYTES,
    DFTU_T_TABLE,
    /** Fixed-width map-key/column scalar types; appended, never renumbered. */
    DFTU_T_I8,
    DFTU_T_I16,
    DFTU_T_I32,
    DFTU_T_U8,
    DFTU_T_U16,
    DFTU_T_U32,
    DFTU_T_U64,
    /** Float key components are bit-cast into the int64 key slot (exact bit
       equality) and materialize as float32/float64; F64 reuses its enumerator
       above for an f64 key. */
    DFTU_T_F32,
    /** A column handle carried by value: the compose value is a `dftu_series*`
       pointer (size = sizeof(pointer)), passed zero-copy - only the handle
       moves, never the column data. Ownership moves to the receiver unless the
       op documents a borrow; the final owner frees with `dftu_series_free`.
       DFTU_T_TABLE is the matching tag for a `dftu_dataframe*` handle. */
    DFTU_T_SERIES
} dftu_type;

/** ABI-stable phase values; the host adapter maps these to the internal enum.
 */
typedef enum {
    DFTU_PH_UNKNOWN = 0,
    DFTU_PH_COMPLETE,
    DFTU_PH_COUNTER,
    DFTU_PH_AGGREGATED,
    DFTU_PH_METADATA
} dftu_phase;

/** OR the flags and return from dftu_plugin.needs to request scan extraction.
 */
enum {
    DFTU_NEED_ARGS = 1u << 0,
    DFTU_NEED_FHASH = 1u << 1,
    DFTU_NEED_HHASH = 1u << 2
};

typedef enum {
    DFTU_ARG_F64 = 0,
    DFTU_ARG_I64 = 1,
    DFTU_ARG_STR = 2
} dftu_arg_kind;

/** A flat, interned arg; nested args are flattened to dotted keys. */
typedef struct {
    dftu_str key;
    dftu_arg_kind kind;
    union {
        double f64;
        int64_t i64;
        dftu_str str;
    } v;
} dftu_arg;

/** One parsed event; strings are interned ids. `args` is NULL unless the plugin
   declared DFTU_NEED_ARGS, and is valid only for the current on_batch call. */
typedef struct {
    dftu_str cat, name, fhash, hhash;
    uint64_t pid, tid, ts, dur;
    uint8_t phase;
    uint8_t has_dur;
    uint32_t arg_count;
    const dftu_arg* args;
} dftu_event;

/** Valid for the on_batch call only; never retain the pointer. */
typedef struct {
    const dftu_event* events;
    uint32_t count;
} dftu_batch;

typedef struct dftu_host dftu_host;
typedef struct dftu_task dftu_task;     /**< async node; awaited exactly once */
typedef struct dftu_writer dftu_writer; /**< parallel output; host-owned */
typedef struct dftu_query dftu_query;   /**< compiled filter; scan-lifetime */
typedef struct dftu_sketch
    dftu_sketch;                /**< quantile accumulator; plugin-owned */
typedef struct dftu_trace_writer
    dftu_trace_writer;          /**< trace output; host-owned */
typedef struct dftu_op dftu_op; /**< composable async op node; scan-lifetime */
/** A columnar batch/table handle for the vectorized fold seam. The concrete
   type is the dataframe engine's (dftracer/utils/dataframe/abi.h); a plugin
   that uses it includes that header for the dftu_dataframe and dftu_series
   column ops. */
typedef struct dftu_dataframe dftu_dataframe;
/** A deferred columnar plan handle (the lazy counterpart to dftu_dataframe).
   The concrete type is the dataframe engine's (dftracer/utils/dataframe/abi.h).
 */
typedef struct dftu_lazyframe dftu_lazyframe;
/** A single column handle for the vectorized fold seam. The concrete type is
   the dataframe engine's (dftracer/utils/dataframe/abi.h); a plugin that uses
   it includes that header for the column ops. */
typedef struct dftu_series dftu_series;
/** A registered dataframe op record and its call-site operand bag. The
   concrete types are the dataframe engine's (dftracer/utils/dataframe/abi.h).
 */
typedef struct dftu_op_desc dftu_op_desc;
typedef struct dftu_op_arg dftu_op_arg;
/** A tagged scalar value. The concrete type is the dataframe engine's
   (dftracer/utils/dataframe/abi.h); a plugin that reads its fields includes
   that header. Passed by pointer here so this header need not define it. */
typedef struct dftu_scalar dftu_scalar;

typedef struct {
    uint64_t count;
    double min, max, mean, p50, p90, p95, p99;
} dftu_quantiles;

/** Stable stat subset, since struct stat layout is platform-dependent. */
typedef struct {
    uint64_t size;
    uint64_t mtime_ns;
    uint32_t mode;
} dftu_stat;

/** dftu_bytes is a borrowed span; dftu_hex16 is a caller-owned 16-hex-digit
   buffer, no NUL. */
typedef struct {
    const char* ptr;
    uint32_t len;
} dftu_bytes;

typedef struct {
    char c[16];
} dftu_hex16;

/** Work function for spawn()/then(); arg is plugin-owned. */
typedef void (*dftu_work_fn)(void* arg);

/** Per-item sink; `item` points at the producer's item struct, borrowed for the
   call only. `ud` is plugin-owned. */
typedef void (*dftu_stream_item_fn)(const void* item, void* ud);

/** Each call returns a dftu_task to compose/await; the out-slot must outlive
 * it.
 */
typedef struct dftu_io {
    dftu_task* (*open)(void* h, const char* path, int flags, int mode,
                       int* out_fd);
    dftu_task* (*close)(void* h, int fd, int* out_rc);
    dftu_task* (*read)(void* h, int fd, void* buf, uint64_t len,
                       int64_t* out_n);
    dftu_task* (*write)(void* h, int fd, const void* buf, uint64_t len,
                        int64_t* out_n);
    dftu_task* (*pread)(void* h, int fd, void* buf, uint64_t len, uint64_t off,
                        int64_t* out_n);
    dftu_task* (*pwrite)(void* h, int fd, const void* buf, uint64_t len,
                         uint64_t off, int64_t* out_n);
    dftu_task* (*fsync)(void* h, int fd, int* out_rc);
    dftu_task* (*ftruncate)(void* h, int fd, uint64_t len, int* out_rc);
    dftu_task* (*fstat)(void* h, int fd, dftu_stat* out);
    /** Appended after the initial 9 ops; append-only within dft.ext.io@1, so a
       host predating one of these leaves its slot NULL. The vectored ops take a
       borrowed iovec array valid for the await; out_n receives the byte count
       (negative errno on failure). */
    dftu_task* (*readv)(void* h, int fd, const struct iovec* iov, int iovcnt,
                        int64_t* out_n);
    dftu_task* (*writev)(void* h, int fd, const struct iovec* iov, int iovcnt,
                         int64_t* out_n);
    dftu_task* (*preadv)(void* h, int fd, const struct iovec* iov, int iovcnt,
                         uint64_t off, int64_t* out_n);
    dftu_task* (*pwritev)(void* h, int fd, const struct iovec* iov, int iovcnt,
                          uint64_t off, int64_t* out_n);
    /** Reposition the file offset; whence is SEEK_SET/CUR/END. out_off receives
       the resulting absolute offset, or a negative errno. */
    dftu_task* (*lseek)(void* h, int fd, int64_t off, int whence,
                        int64_t* out_off);
    /** Zero-copy transfer of `count` bytes from in_fd (a regular file) to
       out_fd starting at `off`; out_n receives the bytes sent. */
    dftu_task* (*sendfile)(void* h, int out_fd, int in_fd, uint64_t off,
                           uint64_t count, int64_t* out_n);
    /** Accept a connection on a listening socket; addr/addrlen may be NULL.
       out_fd receives the client fd, or a negative errno. */
    dftu_task* (*accept)(void* h, int fd, struct sockaddr* addr,
                         socklen_t* addrlen, int* out_fd);
    /** Socket receive/send; flags are the recv(2)/send(2) flags. out_n receives
       the byte count, or a negative errno. */
    dftu_task* (*recv)(void* h, int fd, void* buf, uint64_t len, int flags,
                       int64_t* out_n);
    dftu_task* (*send)(void* h, int fd, const void* buf, uint64_t len,
                       int flags, int64_t* out_n);
} dftu_io;

/** Optional host-service groups, fetched by id via dftu_host::get_extension.
   Ids follow `dftu.ext.<name>@<major>`; a struct is append-only within one
   major.
 */
#define DFTU_EXT_IO "dftu.ext.io@1" /**< the dftu_io struct above */
#define DFTU_EXT_CORO "dftu.ext.coro@1"
#define DFTU_EXT_QUERY "dftu.ext.query@1"
#define DFTU_EXT_COMPOSE "dftu.ext.compose@1"
#define DFTU_EXT_WRITER "dftu.ext.writer@1"
#define DFTU_EXT_SKETCH "dftu.ext.sketch@1"
#define DFTU_EXT_ARROW "dftu.ext.arrow@1"
#define DFTU_EXT_TRACE "dftu.ext.trace@1"
#define DFTU_EXT_PORTS "dftu.ext.ports@1"
#define DFTU_EXT_RESULT "dftu.ext.result@1"
#define DFTU_EXT_AGG "dftu.ext.agg@1"
#define DFTU_EXT_OPS "dftu.ext.ops@1"

typedef struct dftu_ext_coro {
    dftu_task* (*spawn)(void* h, dftu_work_fn fn, void* arg);
    dftu_task* (*when_all)(void* h, dftu_task* const* ts, uint32_t n);
    dftu_task* (*when_any)(void* h, dftu_task* const* ts, uint32_t n);
    dftu_task* (*then)(void* h, dftu_task* t, dftu_work_fn fn, void* arg);
    /** Drive a coroutine: host awaits each dftu_task step() yields until NULL.
     */
    dftu_task* (*drive)(void* h, dftu_task* (*step)(void* coro), void* coro);
    /** Run a synchronous blocking call fn(arg) inline while releasing the
       worker's run-permit for its duration, so a raw block (a legacy blocking
       library, a call with no async form) does not starve the elastic pool.
       Prefer the async dftu_task path for anything that has an awaitable form.
     */
    void (*run_blocking)(void* h, dftu_work_fn fn, void* arg);
} dftu_ext_coro;

typedef struct dftu_ext_query {
    dftu_query* (*query_compile)(void* h, const char* src, uint32_t len);
    int (*query_matches)(void* h, const dftu_query* q, const dftu_event* e);
} dftu_ext_query;

/** A leaf op: transform `in_size` bytes at `in` into `out_size` bytes at `out`,
   returning a dftu_task the host drives to completion (NULL = ran inline).
   Writes *rc (0 ok, <0 error). `in`/`out` are POD value buffers, borrowed for
   the await.
 */
typedef dftu_task* (*dftu_op_fn)(void* state, const void* in, void* out,
                                 int* rc);

/** Compose async ops as first-class handles, the value-typed tier over
   dftu_ext_coro's task combinators. An op transforms a POD input value into a
   POD output value; the byte sizes are carried on the handle so `then` can
   thread an intermediate and check out_size(a) == in_size(b). Ops are
   scan-lifetime (freed at fold teardown); free_op is an optional early release.
   Values cross as void*+size, the same erasure the util registry uses. */
typedef struct dftu_ext_compose {
    /** A leaf op wrapping `fn` with owned `state` (freed via free_state at op
       teardown). in_ty/out_ty are the value types (DFTU_T_BYTES for an opaque
       POD); in_size/out_size are their byte sizes, sized so `then` can thread
       an intermediate. */
    dftu_op* (*make_op)(void* h, dftu_op_fn fn, void* state,
                        void (*free_state)(void*), dftu_type in_ty,
                        uint32_t in_size, dftu_type out_ty, uint32_t out_size);
    /** Pipe: run `a`, feed its output as `b`'s input. Requires the piped value
       to match by type and size: out_ty(a)==in_ty(b) and
       out_size(a)==in_size(b); NULL on mismatch. */
    dftu_op* (*then)(void* h, dftu_op* a, dftu_op* b);
    /** Join: run all `n` ops on the same input, output is their outputs
       concatenated in order (out_size = sum of child out_sizes). */
    dftu_op* (*when_all)(void* h, dftu_op* const* ops, uint32_t n);
    /** Race: run all `n` ops on the same input, output is the first to finish
       (all ops must share one out_size). */
    dftu_op* (*when_any)(void* h, dftu_op* const* ops, uint32_t n);
    /** Run `op` over `in`, writing `out` and *rc. `in`/`out` must outlive the
       await; sized by op's in_size/out_size. */
    dftu_task* (*run)(void* h, dftu_op* op, const void* in, void* out, int* rc);
    /** Optional early release; ops are otherwise freed at fold teardown. */
    void (*free_op)(void* h, dftu_op* op);
} dftu_ext_compose;

typedef struct dftu_ext_writer {
    dftu_task* (*merge_shards)(void* h, const char* target,
                               const char* const* shards, uint32_t n);
    dftu_writer* (*writer_create)(void* h, const char* path,
                                  uint32_t num_workers, int gzip);
    dftu_task* (*writer_open)(void* h, dftu_writer* w);
    dftu_task* (*writer_chunk)(void* h, dftu_writer* w, uint32_t worker,
                               const void* data, uint64_t len);
    dftu_task* (*writer_close)(void* h, dftu_writer* w);
} dftu_ext_writer;

typedef struct dftu_ext_sketch {
    /** Mergeable quantile accumulator; the plugin must sketch_free what it
       creates. add's w is the sample weight. */
    dftu_sketch* (*sketch_create)(void* h);
    void (*sketch_add)(void* h, dftu_sketch* s, double v, double w);
    void (*sketch_merge)(void* h, dftu_sketch* into, const dftu_sketch* other);
    dftu_quantiles (*sketch_result)(void* h, const dftu_sketch* s);
    void (*sketch_free)(void* h, dftu_sketch* s);
} dftu_ext_sketch;

typedef struct dftu_ext_arrow {
    /** Build an Arrow record batch (cat,name,pid,tid,ts,dur,phase) from the
       events; the plugin owns the result and must call out->release and
       out_schema->release. */
    int (*batch_to_arrow)(void* h, const dftu_batch* b, struct ArrowArray* out,
                          struct ArrowSchema* out_schema);
    /** Write a plugin-provided Arrow batch to an IPC file. 0 ok, -1 on error.
     */
    int (*arrow_write_ipc)(void* h, struct ArrowArray* a, struct ArrowSchema* s,
                           const char* path);
    /** Read the first record batch of an Arrow IPC file; the plugin owns *out
       and *out_schema and must call their release. 0 ok, -1 on error. */
    int (*arrow_read_ipc)(void* h, const char* path, struct ArrowArray* out,
                          struct ArrowSchema* out_schema);
} dftu_ext_arrow;

typedef struct dftu_ext_trace {
    /** dftracer trace writer: open a gzip .pfw.gz, append events, then close.
       The index is built lazily on first read, not at close. */
    dftu_trace_writer* (*trace_open_write)(void* h, const char* path);
    int (*trace_write)(void* h, dftu_trace_writer* w, const dftu_event* evs,
                       uint32_t n);
    int (*trace_close)(void* h, dftu_trace_writer* w);
    /** Scan a trace file (auto-indexed) and call on_event per event, ids
       interned into the host scan table. 0 on success. */
    int (*trace_read)(void* h, const char* path, dftu_stream_item_fn on_event,
                      void* ud);
} dftu_ext_trace;

/** Batch-scoped ports for intra-batch producer -> consumer communication,
   fetched via dftu_host::get_extension(DFTU_EXT_PORTS). A port is a name; a
   producer and a consumer are wired by naming the same one, and a name that
   ever needs a version carries it in the name. The bus resets between batches.
   A consume result is borrowed until the current on_batch returns (copy to
   retain) and NULL if the producer has not published this batch. The host runs
   a producer before every consumer of its ports; that order comes from the
   plugins' declared dftu_plugin::provides / dftu_plugin::consumes, not from
   registration order.
 */
typedef struct dftu_ext_ports {
    /** Stable key for the port named `name` (ASCII [a-z0-9._-]). The "dftu."
       namespace belongs to the host and is refused to plugins. */
    uint64_t (*port_key)(void* h, const char* name);
    void (*publish)(void* h, uint64_t key, const void* data, uint32_t len);
    const void* (*consume)(void* h, uint64_t key, uint32_t* out_len);
} dftu_ext_ports;

/** Named result channel, fetched via dftu_host::get_extension(DFTU_EXT_RESULT).
   The host moves opaque bytes / user-schema Arrow and never interprets them;
   Plugins::run returns the collected results to the caller keyed by name. */
typedef struct dftu_ext_result {
    /** Emit a named opaque result; the host COPIES len bytes. Intended for
       on_finalize (called on the merged master fold); if called concurrently
       from on_batch the host serializes writes, last-writer-wins on a name. */
    void (*emit)(void* h, const char* name, const void* data, uint64_t len);
    /** Emit a named Arrow result; the host MOVES the array/schema (like
       ArrowArrayMove/ArrowSchemaMove). 0 ok, -1 on error. Same
       finalize/collision semantics. */
    int (*emit_arrow)(void* h, const char* name, struct ArrowArray* a,
                      struct ArrowSchema* s);
    /** Emit a named native dataframe result; the host TAKES OWNERSHIP of the
       dftu_dataframe handle (do not free it after). The plugin ABI is internal,
       so a columnar result crosses it as our own dataframe with no Arrow
       round-trip; the Arrow edge is the external (Python) reader's concern. 0
       ok, -1 on error. Same finalize/collision semantics. Appended after
       emit_arrow; a host predating it leaves the slot NULL. */
    int (*emit_frame)(void* h, const char* name, dftu_dataframe* df);
    /** Emit a named deferred dataframe result; the host TAKES OWNERSHIP of the
       dftu_lazyframe handle (do not free it after) and collects it at the
       Python edge. The plan MUST be self-contained - its source an in-memory
       frame or a re-openable source (dftu_dataframe_lazy of a materialized
       frame is the supported construction); a plan referencing the plugin's
       per-scan/executor state, which dies at finalize, is a plugin bug. 0 ok,
       -1 on error. Same finalize/collision semantics. Appended after
       emit_frame; a host predating it leaves the slot NULL. */
    int (*emit_lazyframe)(void* h, const char* name, dftu_lazyframe* lf);
} dftu_ext_result;

/** Host-owned cross-batch aggregation accumulator, fetched via
   dftu_host::get_extension(DFTU_EXT_AGG). This is the ONE accumulator a plugin
   gets: it wraps the dataframe engine's mergeable AggState, so a keyed map is
   an accumulator with key columns, a scalar handle is one with zero key
   columns, and the reduction vocabulary is the engine's agg op table. The host
   merges same-named accumulators across worker slices and finalizes each at
   scan end to a native dataframe (the key columns, then one column per
   aggregate), returned to run() under `name`. */
typedef struct dftu_agg dftu_agg;

/** One aggregate for a dftu_ext_agg accumulator. `op` is a DFTU_AGG_* code (the
   dftu_agg_op enum, kept a fixed-width int at the seam for ABI stability); a
   code outside the DFTU_AGG_* range makes agg_new return NULL. `value` names
   the value column in each accumulated batch dataframe (NULL for
   DFTU_AGG_COUNT, the group row count); `out` names the result column; `param`
   is the op's scalar parameter (a quantile in [0,1] for DFTU_AGG_PCT, k for
   DFTU_AGG_TOPK and friends, 0 otherwise); `by` names the op's second input
   column (the ordering column for DFTU_AGG_ARGMAX/ARGMIN/TOPK, the dur column
   for the occupancy ops, x for the co-moment ops; NULL when the op takes one
   input). All names are borrowed for the agg_new call only. */
typedef struct dftu_agg_col {
    int32_t op;
    const char* value;
    const char* out;
    double param;
    const char* by;
} dftu_agg_col;

typedef struct dftu_ext_agg {
    /** Get-or-create a named accumulator grouping by the `key_n` columns named
       in `key_names` and computing each of `spec_n` aggregates. Returns a
       stable handle owned by the host (freed at fold teardown, never by the
       plugin); NULL on a bad op code, a missing output name, or allocation
       failure, or a `name` in the host's own "dftu." namespace, which is
       refused to plugins. A name seen before returns the existing handle and
       ignores the new spec. Safe from any slice thread on that slice's host. */
    dftu_agg* (*agg_new)(void* h, const char* name,
                         const char* const* key_names, uint32_t key_n,
                         const dftu_agg_col* specs, uint32_t spec_n);
    /** Fold one batch into `a`: each key and value column is looked up by name
       in `df` (host-owned, borrowed for the call) and accumulated. A batch
       missing any referenced column is skipped. Serial per accumulator; one
       slice's accumulator is touched by one thread. */
    void (*agg_accumulate)(void* h, dftu_agg* a, const dftu_dataframe* df);
    /** The cross-worker-merged, finalized result of the accumulator named
       `name` - any plugin's, which is how one plugin reads another's whole-scan
       aggregate. Call at on_finalize; the host finalizes the producer first,
       from the plugins' declared dftu_plugin::provides / dftu_plugin::consumes
       rather than registration order. Returns a NEW owned dataframe the caller
       frees with dftu_dataframe_free, or NULL if the producer created no such
       accumulator (an empty scan). */
    dftu_dataframe* (*agg_result)(void* h, const char* name);
} dftu_ext_agg;

/** Host-service group exposing the dataframe engine's op registry, fetched via
   dftu_host::get_extension(DFTU_EXT_OPS). Each slot is a thin name-keyed
   forwarder to the matching dftu_op_run / dftu_op_find / dftu_op_register
   function (dftracer/utils/dataframe/abi.h): a plugin looks an op up by name
   and runs it on Series/DataFrame handles it already holds, with no need to
   link the dataframe C ABI itself. */
typedef struct dftu_ext_ops {
    /** find(name) then dftu_op_run: `in` are `n_in` borrowed input columns,
       `args` supplies the op's other operands (NULL if none). Returns a new
       owned column (free with dftu_series_free), or NULL if `name` is
       unknown or the call mismatches the op's kind/arity/shape. */
    dftu_series* (*run)(void* h, const char* name, const dftu_series* const* in,
                        uint32_t n_in, const dftu_op_arg* args);
    /** find(name) then dftu_op_run_aggregate on one column: `in[0]` is the
       reduced column (n_in must be 1). Writes the reduction to *out (see
       dftu_op_run_aggregate); *ok is set to 0 on a NULL/kind/shape mismatch or
       unknown name (leaving *out zeroed), 1 otherwise. */
    void (*run_aggregate)(void* h, const char* name,
                          const dftu_series* const* in, uint32_t n_in,
                          const dftu_op_arg* args, dftu_scalar* out, int* ok);
    /** find(name) then dftu_op_run_frame: `in` are `n_in` borrowed input
       dataframes. Returns a new owned dataframe (free with
       dftu_dataframe_free), or NULL if `name` is unknown or the call
       mismatches the op's kind/arity/shape. */
    dftu_dataframe* (*run_frame)(void* h, const char* name,
                                 const dftu_dataframe* const* in, uint32_t n_in,
                                 const dftu_op_arg* args);
    /** The registered op named `name` (built-in or user), or NULL if none. See
       dftu_op_find. */
    const dftu_op_desc* (*find)(void* h, const char* name);
    /** Register a user op; see dftu_op_register. A plugin op must be named
       `<plugin>.<name>`: the bare namespace holds the host's built-in ops
       (`add`, `sum`, ...) and the "dftu." prefix its internal ones, so both
       belong to the host and are refused here, and a host op can never be
       shadowed. Returns 0 on success, non-zero if `desc`/its name is NULL, the
       name is already registered, or the name is one the host keeps. */
    int (*register_op)(void* h, const dftu_op_desc* desc);
} dftu_ext_ops;

/** Severity for dftu_host::log; higher is more severe. Mirrors the host's own
   logger levels, so a plugin's line is gated by the same threshold. */
typedef enum {
    DFTU_LOG_TRACE = 0,
    DFTU_LOG_DEBUG = 1,
    DFTU_LOG_INFO = 2,
    DFTU_LOG_WARN = 3,
    DFTU_LOG_ERROR = 4
} dftu_log_level;

/** Services the host lends the plugin; all calls are safe from any slice
 * thread.
 */
struct dftu_host {
    uint32_t abi_version;
    void* h;

    /** Fetch an optional host-service group by id (DFTU_EXT_*); NULL if absent.
     */
    const void* (*get_extension)(void* h, const char* ext_id);

    const char* (*resolve)(void* h, dftu_str id, uint32_t* out_len);
    dftu_str (*intern)(void* h, const char* s, uint32_t len);

    /** Emit a diagnostic line at `level` (a dftu_log_level); `s`/`n` need no
     * NUL.
     */
    void (*log)(void* h, uint8_t level, const char* s, uint32_t n);
};

/** Plugin config tree; host-owned and valid for the plugin's whole lifetime. */
typedef enum {
    DFTU_VAL_NULL = 0,
    DFTU_VAL_BOOL,
    DFTU_VAL_I64,
    DFTU_VAL_F64,
    DFTU_VAL_STR,
    DFTU_VAL_ARRAY,
    DFTU_VAL_OBJECT
} dftu_value_kind;

typedef struct dftu_value dftu_value;

typedef struct {
    const char* key; /**< NUL-terminated */
    uint32_t key_len;
    const dftu_value* value;
} dftu_member;

struct dftu_value {
    dftu_value_kind kind;
    uint32_t count;      /**< STR: byte length; ARRAY/OBJECT: child count */
    union {
        uint32_t b;      /**< BOOL: 0 or 1 */
        int64_t i64;     /**< I64 */
        double f64;      /**< F64 */
        const char* str; /**< STR: `count` bytes, NUL-terminated */
        const dftu_value* items;    /**< ARRAY: `count` values */
        const dftu_member* members; /**< OBJECT: `count` members */
    } as;
};

static inline const dftu_value* dftu_obj_get(const dftu_value* obj,
                                             const char* key) {
    if (!obj || obj->kind != DFTU_VAL_OBJECT) return NULL;
    for (uint32_t i = 0; i < obj->count; ++i)
        if (strcmp(obj->as.members[i].key, key) == 0)
            return obj->as.members[i].value;
    return NULL;
}
static inline int64_t dftu_as_i64(const dftu_value* v, int64_t dflt) {
    if (!v) return dflt;
    if (v->kind == DFTU_VAL_I64) return v->as.i64;
    if (v->kind == DFTU_VAL_F64) return (int64_t)v->as.f64;
    if (v->kind == DFTU_VAL_BOOL) return (int64_t)v->as.b;
    return dflt;
}
static inline double dftu_as_f64(const dftu_value* v, double dflt) {
    if (!v) return dflt;
    if (v->kind == DFTU_VAL_F64) return v->as.f64;
    if (v->kind == DFTU_VAL_I64) return (double)v->as.i64;
    return dflt;
}
static inline int dftu_as_bool(const dftu_value* v, int dflt) {
    if (!v) return dflt;
    if (v->kind == DFTU_VAL_BOOL) return (int)v->as.b;
    if (v->kind == DFTU_VAL_I64) return v->as.i64 != 0;
    return dflt;
}
static inline const char* dftu_as_str(const dftu_value* v, uint32_t* out_len) {
    if (v && v->kind == DFTU_VAL_STR) {
        if (out_len) *out_len = v->count;
        return v->as.str;
    }
    if (out_len) *out_len = 0;
    return NULL;
}

/** A plugin: a data-parallel fold; one slice per worker, merged then finalized.
   on_batch and on_finalize return NULL when handled synchronously, else a task
   the host awaits. The dftu_batch b stays valid until that returned task
   completes (for a synchronous NULL return, only during the call). */
typedef struct dftu_plugin {
    uint32_t abi_version;
    void* self; /**< read-only config, shared across slices */

    uint32_t (*needs)(void* self);         /**< OR of DFTU_NEED_* */
    const char* (*plan_query)(void* self); /**< coarse filter, or NULL = all */
    void* (*make_slice)(void* self);
    dftu_task* (*on_batch)(void* slice, const dftu_batch* b,
                           const dftu_host* host);
    void (*merge)(void* into, void* other);
    dftu_task* (*on_finalize)(void* slice, const dftu_host* host);
    void (*destroy_slice)(void* slice); /**< one call per make_slice */
    void (*destroy)(void* self);        /**< plugin teardown; frees self */

    /** Optional vectorized-fold seam: when set, the host hands each batch as a
       dftu_dataframe (its events materialized into columns) instead of calling
       on_batch per event, so the fold runs SIMD column ops in-scan. A plugin
       sets EITHER on_batch OR this. Must be synchronous (return NULL); `df` is
       owned by the host and valid only for the call. */
    dftu_task* (*on_batch_columns)(void* slice, const dftu_dataframe* df,
                                   const dftu_host* host);

    /** The names this plugin produces, as a NULL-terminated array that outlives
       the plugin; NULL = none. One namespace covers both edge kinds: a
       dftu_ext_ports port it publishes and a dftu_ext_agg accumulator it
       creates. Two plugins providing the same name is a load error. */
    const char* const* (*provides)(void* self);
    /** The names this plugin reads, as a NULL-terminated array that outlives
       the plugin; NULL = none: ports it consumes and accumulators it fetches
       with dftu_ext_agg::agg_result. The host runs every provider of a
       consumed name first, and rejects the set when no loaded plugin provides
       one or when the resulting graph has a cycle. */
    const char* const* (*consumes)(void* self);
} dftu_plugin;

/** The one symbol the loader resolves via dlsym; config is NULL when none
 * given.
 */
typedef dftu_plugin* (*dftu_plugin_factory)(const dftu_value* config);
#define DFTRACER_PLUGIN_FACTORY_SYMBOL "dftracer_plugin"

#ifdef __cplusplus
}
#endif

#endif /* DFTRACER_UTILS_PLUGINS_ABI_H */
