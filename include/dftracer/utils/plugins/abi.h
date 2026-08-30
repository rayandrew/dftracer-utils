#ifndef DFTRACER_UTILS_PLUGINS_ABI_H
#define DFTRACER_UTILS_PLUGINS_ABI_H

/** @file
 * Stable C ABI for dftracer-utils plugins; include only this header.
 */

#include <dftracer/utils/core/common/export.h>
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
typedef struct dftu_stream
    dftu_stream;                /**< pulled utility stream; plugin-owned */
typedef struct dftu_op dftu_op; /**< composable async op node; scan-lifetime */
/** A columnar batch/table handle for the vectorized fold seam. The concrete
   type is the dataframe engine's (dftracer/utils/dataframe/abi.h); a plugin
   that uses it includes that header for the dftu_dataframe and dftu_series
   column ops. */
typedef struct dftu_dataframe dftu_dataframe;

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

/** A registered host utility; `run` reads *in (in_tag), writes *out (out_tag),
   returns 0 on success. Call via the typed wrappers in utilities.h. Any
   borrowed pointers in *out (strings, spans) stay valid only until the next
   run() of the same utility on the same thread; copy them out to keep them. */
typedef struct {
    const char* name;
    uint32_t name_len;
    dftu_type in_tag, out_tag;
    int (*run)(void* self, const void* in, void* out);
    void* self;
} dftu_utility;

/** Work function for spawn()/then(); arg is plugin-owned. */
typedef void (*dftu_work_fn)(void* arg);

/** Per-item sink for run_stream; `item` points at the utility's generated C
   output struct, borrowed for the call only. `ud` is plugin-owned. */
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
#define DFTU_EXT_UTIL "dftu.ext.util@1"
#define DFTU_EXT_WRITER "dftu.ext.writer@1"
#define DFTU_EXT_SKETCH "dftu.ext.sketch@1"
#define DFTU_EXT_ARROW "dftu.ext.arrow@1"
#define DFTU_EXT_TRACE "dftu.ext.trace@1"
#define DFTU_EXT_COMMS "dftu.ext.comms@1"
#define DFTU_EXT_PORTS "dftu.ext.ports@1"
#define DFTU_EXT_HANDLES "dftu.ext.handles@1"
#define DFTU_EXT_RESULT "dftu.ext.result@1"
#define DFTU_EXT_MAP "dftu.ext.map@1"

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

typedef struct dftu_ext_util {
    const dftu_utility* (*find_by_id)(void* h, uint32_t util_id);

    /** Run a streaming utility: calls on_item once per yielded item, in_data
       host-owned. 0 on success, -1 on failure. Single-value utilities deliver
       exactly one item. */
    int (*run_stream)(void* h, uint32_t util_id, const void* in_data,
                      dftu_stream_item_fn on_item, void* ud);

    /** Lazy dftu_task running a single-output utility on the current executor;
       marshals in_data in and out_data out and writes *out_rc. in_data and
       out_data must outlive the await. A stream or unknown util_id sets
       *out_rc = -1. */
    dftu_task* (*util_run_async)(void* h, uint32_t util_id, const void* in_data,
                                 void* out_data, int* out_rc);

    /** Lazy dftu_task driving a streaming utility on the current executor,
       firing on_item per item as produced; *out_rc set on completion. in_data,
       on_item, and ud must outlive the await. */
    dftu_task* (*util_run_stream_async)(void* h, uint32_t util_id,
                                        const void* in_data,
                                        dftu_stream_item_fn on_item, void* ud,
                                        int* out_rc);

    /** Instantiate a utility's stream; NULL if util_id is not a stream utility.
       The plugin owns the handle and must util_stream_close it exactly once,
       within the same scan it was opened in. */
    dftu_stream* (*util_stream_open)(void* h, uint32_t util_id,
                                     const void* in_data);
    /** Task resuming the stream one step. *out_item is the marshalled C item,
       BORROWED and valid only until the next util_stream_next or
       util_stream_close, NULL at end-of-stream; *out_rc is status. Await from
       the same executor context; no concurrent next on one stream. */
    dftu_task* (*util_stream_next)(void* h, dftu_stream* s,
                                   const void** out_item, int* out_rc);
    /** Safe only between completed next() calls, never with one in flight. */
    void (*util_stream_close)(void* h, dftu_stream* s);
} dftu_ext_util;

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
    /** A leaf op wrapping registered host utility `util_id` (a single-value
       utility, id < DFTU_UTIL__COUNT), so then()/when_all can pipe host
       utilities. The value crosses as the utility's generated C in/out struct;
       NULL for a stream-only or unknown id. Runs the utility asynchronously on
       the executor (co_await), never blocking a worker. */
    dftu_op* (*util_op)(void* h, uint32_t util_id);
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

/** Capability identity and versioning. A capability is a namespaced id plus a
   semantic version; a requirement is an id plus a version constraint. Ids are
   ASCII [a-z0-9._-]; the `dftu.` prefix is reserved for the host. */
typedef struct {
    uint16_t major, minor, patch;
} dftu_version;

typedef struct {
    const char* id; /**< NUL-terminated capability id */
    dftu_version ver;
} dftu_capability;

typedef enum {
    DFTU_VER_GE =
        0,       /**< >=X.Y.Z; a bare id parses to >=0.0.0, i.e. any version */
    DFTU_VER_GT, /**< >X.Y.Z */
    DFTU_VER_LE, /**< <=X.Y.Z */
    DFTU_VER_LT, /**< <X.Y.Z */
    DFTU_VER_EQ, /**< =X.Y.Z */
    DFTU_VER_CARET, /**< ^X.Y.Z: compatible within major (0.y treats minor as
                      the breaking axis, 0.0.z as exact), the sane default */
    DFTU_VER_TILDE  /**< ~X.Y.Z: compatible within minor (same major and minor)
                     */
} dftu_ver_op;

typedef struct {
    const char* id; /**< NUL-terminated capability id */
    dftu_ver_op op;
    dftu_version ver;
    int required;   /**< nonzero: unmet is a load error; zero: graceful fallback
                     */
} dftu_requirement;

static inline int dftu_version_cmp(dftu_version a, dftu_version b) {
    if (a.major != b.major) return a.major < b.major ? -1 : 1;
    if (a.minor != b.minor) return a.minor < b.minor ? -1 : 1;
    if (a.patch != b.patch) return a.patch < b.patch ? -1 : 1;
    return 0;
}

/** Parse up to three dot-separated components; missing ones are 0. Rejects a
   component above 65535, non-digits, and trailing junk. 0 ok, -1 malformed. */
static inline int dftu_version_parse(const char* s, uint32_t len,
                                     dftu_version* out) {
    dftu_version v = {0, 0, 0};
    uint16_t* parts[3];
    uint32_t i = 0;
    int part = 0;
    if (len == 0) return -1;
    parts[0] = &v.major;
    parts[1] = &v.minor;
    parts[2] = &v.patch;
    for (;;) {
        uint32_t n = 0;
        int digits = 0;
        for (; i < len && s[i] >= '0' && s[i] <= '9'; ++i) {
            n = n * 10u + (uint32_t)(s[i] - '0');
            if (n > 0xFFFFu) return -1;
            ++digits;
        }
        if (!digits) return -1; /* empty component, including a trailing dot */
        *parts[part++] = (uint16_t)n;
        if (i == len) break;
        if (s[i] != '.' || part == 3) return -1; /* junk, or too many parts */
        ++i;
    }
    *out = v;
    return 0;
}

/** Does a provider version satisfy a requirement's op and version? */
static inline int dftu_version_satisfies(dftu_version have, dftu_ver_op op,
                                         dftu_version want) {
    int c = dftu_version_cmp(have, want);
    switch (op) {
        case DFTU_VER_GE:
            return c >= 0;
        case DFTU_VER_GT:
            return c > 0;
        case DFTU_VER_LE:
            return c <= 0;
        case DFTU_VER_LT:
            return c < 0;
        case DFTU_VER_EQ:
            return c == 0;
        case DFTU_VER_CARET:
            if (c < 0) return 0;
            if (want.major > 0) return have.major == want.major;
            if (want.minor > 0)
                return have.major == 0 && have.minor == want.minor;
            return have.major == 0 && have.minor == 0 &&
                   have.patch == want.patch;
        case DFTU_VER_TILDE:
            return c >= 0 && have.major == want.major &&
                   have.minor == want.minor;
        default:
            return 0;
    }
}

/** Parse "id@MAJOR.MINOR.PATCH" (a bare id yields version 0.0.0). The id is
   copied NUL-terminated into id_buf. 0 ok, -1 on malformed input or too-small
   id_buf. */
static inline int dftu_capability_parse(const char* s, char* id_buf,
                                        uint32_t id_buf_cap,
                                        dftu_capability* out) {
    uint32_t i = 0, vlen = 0;
    if (!s || !id_buf || !out) return -1;
    while (s[i] && s[i] != '@') ++i;
    if (i == 0 || i >= id_buf_cap) return -1;
    memcpy(id_buf, s, i);
    id_buf[i] = '\0';
    out->id = id_buf;
    if (!s[i]) {
        out->ver.major = out->ver.minor = out->ver.patch = 0;
        return 0;
    }
    ++i; /* skip '@' */
    while (s[i + vlen]) ++vlen;
    return dftu_version_parse(s + i, vlen, &out->ver);
}

/** A provider satisfies a requirement iff the ids match and the version does.
 */
static inline int dftu_capability_satisfies(const dftu_capability* cap,
                                            const dftu_requirement* req) {
    if (!cap || !req || !cap->id || !req->id) return 0;
    if (strcmp(cap->id, req->id) != 0) return 0;
    return dftu_version_satisfies(cap->ver, req->op, req->ver);
}

/** Blessed host-owned capability ids. These are reserved dft. ids: a plugin may
   REQUIRE one, but the reserved-namespace rule forbids a plugin PROVIDING it.
   A host provider is wired per id only once a real consumer needs it. */
#define DFTU_CAP_EVENTS                                                \
    "dftu.cap.events" /* the raw per-batch events already delivered to \
                       * on_batch                                      \
                       */
#define DFTU_CAP_RESOLVED_EVENTS \
    "dftu.cap.resolved_events" /* events with their string fields resolved */
#define DFTU_CAP_TIME_WINDOW \
    "dftu.cap.time_window"     /* the scan's [min,max] timestamp range */

/** Parse "id", "id>=1.2", "id^1.2", "id~1.0", "id=1.2.3", "id<2", "id>1.0", or
   "id<=1.4". The id (everything before the first of > < = ^ ~) is copied
   NUL-terminated into id_buf; a bare id yields >=0.0.0. out->required is left
   untouched. 0 ok, -1 on malformed input or too-small id_buf. */
static inline int dftu_requirement_parse(const char* s, char* id_buf,
                                         uint32_t id_buf_cap,
                                         dftu_requirement* out) {
    uint32_t i = 0, j, vlen = 0;
    if (!s || !id_buf || !out) return -1;
    while (s[i] && s[i] != '>' && s[i] != '<' && s[i] != '=' && s[i] != '^' &&
           s[i] != '~')
        ++i;
    if (i == 0 || i >= id_buf_cap) return -1;
    memcpy(id_buf, s, i);
    id_buf[i] = '\0';
    out->id = id_buf;
    if (!s[i]) {
        out->op = DFTU_VER_GE;
        out->ver.major = out->ver.minor = out->ver.patch = 0;
        return 0;
    }
    j = i;
    if (s[j] == '>' && s[j + 1] == '=') {
        out->op = DFTU_VER_GE;
        j += 2;
    } else if (s[j] == '>') {
        out->op = DFTU_VER_GT;
        j += 1;
    } else if (s[j] == '<' && s[j + 1] == '=') {
        out->op = DFTU_VER_LE;
        j += 2;
    } else if (s[j] == '<') {
        out->op = DFTU_VER_LT;
        j += 1;
    } else if (s[j] == '=') {
        out->op = DFTU_VER_EQ;
        j += 1;
    } else if (s[j] == '^') {
        out->op = DFTU_VER_CARET;
        j += 1;
    } else if (s[j] == '~') {
        out->op = DFTU_VER_TILDE;
        j += 1;
    } else {
        return -1;
    }
    while (s[j + vlen]) ++vlen;
    return dftu_version_parse(s + j, vlen, &out->ver);
}

/** Plugin-side capability negotiation, fetched via dftu_plugin::get_extension.
   Fill functions return the count; if it exceeds max the fill is partial and
   the return is the count needed. resolve runs once after every plugin
   declared. */
typedef struct dftu_plugin_comms {
    uint32_t (*provides)(void* self, dftu_capability* out, uint32_t max);
    uint32_t (*require_caps)(void* self, dftu_requirement* out, uint32_t max);
    void (*resolve)(void* self, const dftu_host* host);
} dftu_plugin_comms;

/** Host-side registry queries, fetched via
   dftu_host::get_extension(DFTU_EXT_COMMS) during resolve. */
typedef struct dftu_ext_comms {
    /** Providers of cap_id at any version; 0 means absent. */
    uint32_t (*provider_count)(void* h, const char* cap_id);
    /** Highest provider version satisfying req into *out_ver; 0 ok, -1 if none.
     */
    int (*provider_best)(void* h, const dftu_requirement* req,
                         dftu_version* out_ver);
} dftu_ext_comms;

/** Batch-scoped ports for intra-batch producer -> consumer communication,
   fetched via dftu_host::get_extension(DFTU_EXT_PORTS). The bus resets between
   batches. A consume result is borrowed until the current on_batch returns
   (copy to retain) and NULL if the producer has not published this batch. A
   producer must run before its consumer in the fuse (registration/CLI) order.
 */
typedef struct dftu_ext_ports {
    uint64_t (*port_key)(void* h, const char* cap_id);
    void (*publish)(void* h, uint64_t key, const void* data, uint32_t len);
    const void* (*consume)(void* h, uint64_t key, uint32_t* out_len);
} dftu_ext_ports;

/** Cross-worker mergeable handles: a fixed monoid vocabulary the host merges
   across every worker slice of a plugin. Each named handle accumulates during
   the scan and is read at finalize. */
typedef enum {
    DFTU_MONOID_COUNTER = 0, /**< u64 sum */
    DFTU_MONOID_SUM_F64,     /**< double sum */
    DFTU_MONOID_MIN_F64,     /**< double min */
    DFTU_MONOID_MAX_F64,     /**< double max */
    DFTU_MONOID_SKETCH,      /**< DDSketch quantiles */
    /** Appended after SKETCH; never renumber. These take add_u64 and yield u64.
     */
    DFTU_MONOID_MIN_U64,   /**< u64 min */
    DFTU_MONOID_MAX_U64,   /**< u64 max */
    DFTU_MONOID_BOOL_AND,  /**< logical AND; nonzero add is true, result 0/1 */
    DFTU_MONOID_BOOL_OR,   /**< logical OR; nonzero add is true, result 0/1 */
    DFTU_MONOID_BITSET_OR, /**< u64 bitwise OR */
    DFTU_MONOID_DISTINCT,  /**< approx distinct count of added values (u64) */
    /** Collects the distinct interned-string ids added via add_u64; has no
       scalar result, materializes to a list<string> column (ids resolved to
       labels). */
    DFTU_MONOID_SET_STR,
    /** Ordered list of interned strings, sorted by a caller order key;
       materializes to a list<string> column. */
    DFTU_MONOID_LIST_STR,
    /** Unordered distinct set of raw int64 values added via add_u64;
       materializes to a list<int64> column, sorted ascending. */
    DFTU_MONOID_SET_I64,
    /** Ordered list of raw int64 values, sorted by a caller order key;
       materializes to a list<int64> column. */
    DFTU_MONOID_LIST_I64,
    /** Typed min/max; materializes at the element width (signed/unsigned via
       add_u64 as int64/uint64 bits, float via add_f64). Appended, never
       renumbered. */
    DFTU_MONOID_MIN_I8,
    DFTU_MONOID_MIN_I16,
    DFTU_MONOID_MIN_I32,
    DFTU_MONOID_MIN_I64,
    DFTU_MONOID_MIN_U8,
    DFTU_MONOID_MIN_U16,
    DFTU_MONOID_MIN_U32,
    DFTU_MONOID_MIN_F32,
    DFTU_MONOID_MAX_I8,
    DFTU_MONOID_MAX_I16,
    DFTU_MONOID_MAX_I32,
    DFTU_MONOID_MAX_I64,
    DFTU_MONOID_MAX_U8,
    DFTU_MONOID_MAX_U16,
    DFTU_MONOID_MAX_U32,
    DFTU_MONOID_MAX_F32,
    /** Argmin/argmax (min-by/max-by): keep `payload` from the contribution
       whose f64 `by` is extreme, fed via map_add_argby_at. Equal `by` keeps the
       smaller payload id. */
    DFTU_MONOID_ARGMIN_I64, /**< payload materializes as an int64 column */
    DFTU_MONOID_ARGMAX_I64,
    DFTU_MONOID_ARGMIN_STR, /**< payload is an interned id, resolved at
                             * materialize
                             */
    DFTU_MONOID_ARGMAX_STR,
    /** Moment stats fed via map_add_f64_at, each one double column.
       VARIANCE/STDDEV are the sample (n-1) statistics, 0 for n<2. */
    DFTU_MONOID_MEAN,
    DFTU_MONOID_VARIANCE,
    DFTU_MONOID_STDDEV,
    /** Bounded top-k / bottom-k: keep the k payloads at the k largest (TOPK) or
       smallest (BOTTOMK) f64 `by` keys, fed via map_add_topk_at. Payloads emit
       in `by` order, payload id breaking ties; _I64 -> list<int64>, _STR ->
       list<string>. */
    DFTU_MONOID_TOPK_I64,
    DFTU_MONOID_TOPK_STR,
    DFTU_MONOID_BOTTOMK_I64,
    DFTU_MONOID_BOTTOMK_STR,
    /** Approximate heavy-hitters (SpaceSaving): the k most frequent values with
       approximate counts, in at most k counters, fed via
       map_add_approx_topk_at. Materializes to list<struct<value, count>>
       ordered by count descending, value id breaking ties. */
    DFTU_MONOID_APPROX_TOPK_I64,
    DFTU_MONOID_APPROX_TOPK_STR,
    /** Deterministic mergeable sample: keep the k items with the smallest
       hash(item) (bottom-k / KMV), not a reservoir. Samples DISTINCT items
       uniformly; a unique per-row value makes it a uniform row sample, fed via
       map_add_sample_at. Materializes to a list column (list<int64> or
       list<string>) sorted by item. */
    DFTU_MONOID_SAMPLE_I64,
    DFTU_MONOID_SAMPLE_STR,
    /** Argmin-row / argmax-row (full-row min-by/max-by, DISTINCT ON): keep the
       whole payload ROW from the contribution whose f64 `by` is extreme.
       Created only via map_new_argrow and fed via map_add_argrow; a `by` tie
       keeps the lexicographically smaller row. Materializes to payload_n typed
       columns. */
    DFTU_MONOID_ARGMIN_ROW,
    DFTU_MONOID_ARGMAX_ROW,
    /** Higher moments fed via map_add_f64_at, each one double column.
       Population skewness, 0 for n<3; population excess kurtosis, 0 for n<4;
       both 0 when the variance is 0. */
    DFTU_MONOID_SKEWNESS,
    DFTU_MONOID_KURTOSIS,
    /** Two-variable co-moment stats fed via map_add_xy_at (x independent, y
       dependent), each one double column: COVAR_POP/COVAR_SAMP (over n / n-1),
       CORR, and the OLS fit of y on x (REGR_SLOPE, REGR_INTERCEPT, REGR_R2). 0
       for n<2 or a zero-variance denominator. */
    DFTU_MONOID_CORR,
    DFTU_MONOID_COVAR_POP,
    DFTU_MONOID_COVAR_SAMP,
    DFTU_MONOID_REGR_SLOPE,
    DFTU_MONOID_REGR_INTERCEPT,
    DFTU_MONOID_REGR_R2
} dftu_monoid_kind;

typedef struct dftu_handle
    dftu_handle; /**< host-owned per-plugin named handle */

typedef struct {
    dftu_monoid_kind kind;
    union {
        uint64_t u64;
        double f64;
        dftu_quantiles quant;
    } as;
} dftu_monoid_value;

/** Fetched via dftu_host::get_extension(DFTU_EXT_HANDLES). */
typedef struct dftu_ext_handles {
    /** Get-or-create a named per-plugin handle of `kind`; call during on_batch.
       The host merges same-named handles across worker slices automatically. An
       existing handle of a different kind is returned unchanged. */
    dftu_handle* (*shared_get)(void* h, const char* cap_id,
                               dftu_monoid_kind kind);
    /** COUNTER/MIN_U64/MAX_U64/BOOL_AND/BOOL_OR/BITSET_OR/DISTINCT. */
    void (*add_u64)(void* h, dftu_handle* hd, uint64_t v);
    /** SUM/MIN/MAX (w ignored) or SKETCH (w is the sample weight). */
    void (*add_f64)(void* h, dftu_handle* hd, double v, double w);
    /** Read the cross-worker-merged result by cap id; call at on_finalize. 0
       and fills *out, or -1 if no plugin produced that handle. The producer
       must finalize before the consumer, i.e. be registered first. */
    int (*result)(void* h, const char* cap_id, dftu_monoid_value* out);
} dftu_ext_handles;

/** Named result channel, fetched via dftu_host::get_extension(DFTU_EXT_RESULT).
   The host moves opaque bytes / user-schema Arrow and never interprets them;
   PluginHost::run returns the collected results to the caller keyed by name. */
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
} dftu_ext_result;

/** Join kind for map_declare_join; the host maps this to its internal enum. */
typedef enum {
    DFTU_JOIN_INNER = 0,
    DFTU_JOIN_LEFT,
    DFTU_JOIN_RIGHT,
    DFTU_JOIN_FULL
} dftu_join_type;

/** Host-owned per-plugin mergeable map, fetched via
   dftu_host::get_extension(DFTU_EXT_MAP). Keys are a fixed tuple of fixed-width
   integer (I8..I64, U8..U64), STR, or BYTES components, each passed in an int64
   key slot (integers by value/bit pattern, STR/BYTES as a dftu_str id). The
   value is one scalar monoid or a product of them. The host merges same-named
   maps across worker slices and materializes each at finalize to an Arrow table
   [key columns, then one value column per component], returned to run() under
   `name`. */
typedef struct dftu_map dftu_map;

/** One component contribution for map_add_row: apply `value` to component
   `comp`, as an f64 add when `is_f64` is nonzero, else a u64 add. */
typedef struct {
    uint32_t comp;
    uint32_t is_f64;
    union {
        uint64_t u;
        double f;
    } value;
} dftu_row_val;

typedef struct dftu_ext_map {
    /** Get-or-create a named map; NULL if any key type is not a fixed-width
       integer (I8..I64, U8..U64) or STR, or the value monoid has no scalar
       result (SKETCH). key_types/key_n fix the schema. */
    dftu_map* (*map_new)(void* h, const char* name, const dftu_type* key_types,
                         uint32_t key_n, dftu_monoid_kind value);
    /** Add a contribution at `key` (key_n int64s); routed to the value monoid.
     */
    void (*map_add_u64)(void* h, dftu_map* m, const int64_t* key, uint64_t v);
    void (*map_add_f64)(void* h, dftu_map* m, const int64_t* key, double v);
    /** Get-or-create a map whose value is a product of value_n monoids ->
       value_n value columns. */
    dftu_map* (*map_new_product)(void* h, const char* name,
                                 const dftu_type* key_types, uint32_t key_n,
                                 const dftu_monoid_kind* values,
                                 uint32_t value_n);
    /** Add to value component `comp` (0-based); map_add_u64/map_add_f64 target
       component 0. */
    void (*map_add_u64_at)(void* h, dftu_map* m, const int64_t* key,
                           uint32_t comp, uint64_t v);
    void (*map_add_f64_at)(void* h, dftu_map* m, const int64_t* key,
                           uint32_t comp, double v);
    /** Append (order_key, element id) to an ordered-list value component; the
       list materializes sorted by order_key, element id breaking ties. */
    void (*map_add_ordered_at)(void* h, dftu_map* m, const int64_t* key,
                               uint32_t comp, int64_t order_key,
                               uint64_t element);
    /** Materialize result rows sorted by key (default is unordered hash order).
     */
    void (*map_set_ordered)(void* h, dftu_map* m, int ordered);
    /** Get-or-create a nested-preserved map: the value at each outer key is
       itself a map (inner keys -> value monoids). Materializes to one row per
       outer key: the outer key columns, then a nested "value" column of
       list<struct<inner_keys.., values..>>. NULL if any key type is unsupported
       or a value monoid is not a materializable scalar/product (collections,
       ordered, and SKETCH rejected). */
    dftu_map* (*map_new_nested)(
        void* h, const char* name, const dftu_type* outer_key_types,
        uint32_t outer_key_n, const dftu_type* inner_key_types,
        uint32_t inner_key_n, const dftu_monoid_kind* values, uint32_t value_n);
    /** Add a contribution to value component `comp` at (outer_key, inner_key);
       outer_key is outer_key_n int64s, inner_key is inner_key_n int64s. */
    void (*map_add_nested_u64)(void* h, dftu_map* m, const int64_t* outer_key,
                               const int64_t* inner_key, uint32_t comp,
                               uint64_t v);
    void (*map_add_nested_f64)(void* h, dftu_map* m, const int64_t* outer_key,
                               const int64_t* inner_key, uint32_t comp,
                               double v);
    /** Contribute (by, payload) to an ARGMIN/ARGMAX value component `comp`;
       kind decides min vs max, _I64/_STR decides payload materialization. Equal
       `by` keeps the smaller payload. */
    void (*map_add_argby_at)(void* h, dftu_map* m, const int64_t* key,
                             uint32_t comp, double by, int64_t payload);
    /** Contribute (by, payload) to a bounded TOPK/BOTTOMK value component
       `comp`, keeping the k payloads at the k extreme `by` keys. k is passed on
       every add (constant per component). Payloads emit in `by` order, smaller
       payload id breaking ties. */
    void (*map_add_topk_at)(void* h, dftu_map* m, const int64_t* key,
                            uint32_t comp, uint32_t k, double by,
                            int64_t payload);
    /** Observe `value` for an APPROX_TOPK heavy-hitters component `comp`
       (SpaceSaving); k is the counter capacity, passed on every add.
       Materializes to list<struct<value, count>> ordered by count descending,
       value id breaking ties. */
    void (*map_add_approx_topk_at)(void* h, dftu_map* m, const int64_t* key,
                                   uint32_t comp, uint32_t k, int64_t value);
    /** Observe `item` for a bottom-k-by-hash SAMPLE component `comp`; k is the
       sample size, passed on every add. Samples DISTINCT items; a unique
       per-row item makes it a uniform row sample. Materializes to a list column
       sorted by item. */
    void (*map_add_sample_at)(void* h, dftu_map* m, const int64_t* key,
                              uint32_t comp, uint32_t k, int64_t item);
    /** Get-or-create a map whose single value is an ARGMIN_ROW (is_max==0) /
       ARGMAX_ROW (is_max!=0) over a payload row of payload_n typed columns.
       NULL if any key or payload type is unsupported (only I8..I64, U8..U64,
       STR, BYTES). The value is created only here, never as a component. */
    dftu_map* (*map_new_argrow)(void* h, const char* name,
                                const dftu_type* key_types, uint32_t key_n,
                                int is_max, const dftu_type* payload_types,
                                uint32_t payload_n);
    /** Contribute (by, payload-row) at `key`; keeps the whole payload row at
       the extreme `by`. payload is payload_n int64 slots encoded like key
       slots. A `by` tie keeps the lexicographically smaller row. */
    void (*map_add_argrow)(void* h, dftu_map* m, const int64_t* key, double by,
                           const int64_t* payload, uint32_t payload_n);
    /** Declare a join run at finalize on the merged master maps: join left_name
       and right_name on their shared key tuple, emitting map out_name. By name
       because per-slice dftu_map* handles do not survive the merge. */
    void (*map_declare_join)(void* h, const char* out_name,
                             const char* left_name, const char* right_name,
                             dftu_join_type type);
    /** Contribute (x, y) to a two-variable co-moment value component `comp` (a
       CORR, COVAR, or REGR monoid); x is the independent variable, y the
       dependent. */
    void (*map_add_xy_at)(void* h, dftu_map* m, const int64_t* key,
                          uint32_t comp, double x, double y);
    /** Get-or-create a map whose single value is a DDSketch quantile estimator.
       Values are fed via map_add_f64 (weight 1). Materializes to a `count`
       int64 column followed by one f64 column per requested quantile in `qs`
       (each in [0,1]), named p<q*100> (p50, p90, p99, ..). NULL if any key type
       is unsupported or nq is 0. The value is created only here, never as a
       component. Appended to this struct after map_add_xy_at; a host that
       predates it leaves the slot NULL. */
    dftu_map* (*map_new_sketch)(void* h, const char* name,
                                const dftu_type* key_types, uint32_t key_n,
                                const double* qs, uint32_t nq);
    /** Get-or-create a fused map: one physical product keyed by `key_types`,
       whose `value_n` scalar components each materialize as a SEPARATE named
       result table (`out_names[i]`, key columns + a `value` column), never as a
       single product table. Lets several same-key maps share one hash lookup
       via map_add_row while staying invisible to the caller. Components must be
       scalar monoids fed by u64/f64 adds (count, sum, min/max, mean, variance,
       distinct, bool, bitset); collections, arg-by, arg-row, and SKETCH are
       rejected. NULL if any key/value type is unsupported. Appended after
       map_new_sketch; a host that predates it leaves the slot NULL. */
    dftu_map* (*map_new_fused)(void* h, const char* name,
                               const dftu_type* key_types, uint32_t key_n,
                               const char* const* out_names,
                               const dftu_monoid_kind* values,
                               uint32_t value_n);
    /** Add a whole row at `key` with ONE hash lookup: apply each of `n`
       contributions to its component. A dftu_row_val is {comp, is_f64, value};
       is_f64 picks map_add_f64 vs map_add_u64 semantics on that component.
       Intended for a map_new_fused map but valid on any product map. */
    void (*map_add_row)(void* h, dftu_map* m, const int64_t* key,
                        const dftu_row_val* vals, uint32_t n);
} dftu_ext_map;

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

    /** Optional capability discovery, symmetric to dftu_host; NULL if none. */
    const void* (*get_extension)(void* self, const char* ext_id);

    /** Optional vectorized-fold seam: when set, the host hands each batch as a
       dftu_dataframe (its events materialized into columns) instead of calling
       on_batch per event, so the fold runs SIMD column ops in-scan. A plugin
       sets EITHER on_batch OR this. Must be synchronous (return NULL); `df` is
       owned by the host and valid only for the call. */
    dftu_task* (*on_batch_columns)(void* slice, const dftu_dataframe* df,
                                   const dftu_host* host);
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
