#include <dftracer/utils/core/common/byte_view.h>
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/constants.h>
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/field_ref.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/coro/when_all.h>
#include <dftracer/utils/core/coro/when_any.h>
#include <dftracer/utils/core/coro/yield.h>
#include <dftracer/utils/core/io/ops.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/agg_expr.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/map_grouping_arrow.h>
#include <dftracer/utils/plugins/map_join_arrow.h>
#include <dftracer/utils/plugins/map_unnest_arrow.h>
#include <dftracer/utils/plugins/utility_registry.h>
#include <dftracer/utils/trace/schema.h>
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <dftracer/utils/trace/views/native_row_fold.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/utilities/common/serialization/binary_codec.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>
#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>
#include <dftracer/utils/utilities/fileio/parallel/merge.h>
#include <simdjson.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>
#include <dftracer/utils/utilities/common/arrow/column_builder.h>
#endif
#ifdef DFTRACER_UTILS_ENABLE_ARROW_IPC
#include <dftracer/utils/utilities/common/arrow/ipc_reader.h>
#include <dftracer/utils/utilities/common/arrow/ipc_writer.h>
#endif

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <new>
#include <numeric>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <variant>
#include <vector>

namespace dftracer::utils::plugins {

// Per-slice engine-backed aggregation accumulator (dft.ext.agg). value_names
// are the distinct value/by columns the specs reference, ordered so each spec's
// value_col/by_col indexes into it.
struct AggAccum {
    std::string name;
    std::vector<std::string> key_names;
    std::vector<std::string> value_names;
    dataframe::AggStatePtr state;
};

namespace {

using query::Query;
using trace::RecordPhase;
namespace parallel = utilities::fileio::parallel;

coro::CoroTask<void>* as_task(::dftu_task* t) {
    return reinterpret_cast<coro::CoroTask<void>*>(t);
}

std::uint8_t map_phase(RecordPhase phase) {
    switch (phase) {
        case RecordPhase::COMPLETE:
            return DFTU_PH_COMPLETE;
        case RecordPhase::COUNTER:
            return DFTU_PH_COUNTER;
        case RecordPhase::METADATA:
            return DFTU_PH_METADATA;
        case RecordPhase::AGGREGATED:
            return DFTU_PH_AGGREGATED;
        case RecordPhase::UNKNOWN:
        default:
            return DFTU_PH_UNKNOWN;
    }
}

const char* host_resolve(void* h, dftu_str id, std::uint32_t* out_len) {
    if (out_len) *out_len = 0;
    if (id == DFTU_STR_NONE ||
        id >= dftracer::utils::StringIntern::FAST_CAPACITY)
        return nullptr;
    auto& intern = static_cast<PluginFold*>(h)->intern_table();
    std::string_view sv = intern.resolve(id);
    if (out_len) *out_len = static_cast<std::uint32_t>(sv.size());
    // Interned bytes are stable for the scan, so they outlive the call.
    return sv.data();
}

dftu_str host_intern(void* h, const char* s, std::uint32_t len) {
    if (!s) return DFTU_STR_NONE;
    auto& intern = static_cast<PluginFold*>(h)->intern_table();
    try {
        return intern.get_or_insert(std::string_view{s, len});
    } catch (...) {
        return DFTU_STR_NONE;
    }
}

// Route a plugin's log line through our logger, tagged with the plugin name, so
// it honors the configured (compile-time and runtime) level.
void host_log(void* h, std::uint8_t level, const char* s, std::uint32_t n) {
    const auto* pf = static_cast<const PluginFold*>(h);
    const char* name =
        pf && !pf->plugin_name().empty() ? pf->plugin_name().c_str() : "plugin";
    const int len = static_cast<int>(n);
    const char* msg = s ? s : "";
    switch (level) {
        case DFTU_LOG_ERROR:
            DFTRACER_UTILS_LOG_ERROR("[plugin:%s] %.*s", name, len, msg);
            break;
        case DFTU_LOG_WARN:
            DFTRACER_UTILS_LOG_WARN("[plugin:%s] %.*s", name, len, msg);
            break;
        case DFTU_LOG_DEBUG:
            DFTRACER_UTILS_LOG_DEBUG("[plugin:%s] %.*s", name, len, msg);
            break;
        case DFTU_LOG_TRACE:
            DFTRACER_UTILS_LOG_TRACE("[plugin:%s] %.*s", name, len, msg);
            break;
        default:
            DFTRACER_UTILS_LOG_INFO("[plugin:%s] %.*s", name, len, msg);
            break;
    }
}

// Each op builds a lazy CoroTask; the caller's out-slot must outlive the task.
::dftu_task* io_open(void* h, const char* path, int flags, int mode,
                     int* out_fd) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](const char* p, int fl, int md, int* ofd) -> coro::CoroTask<void> {
            int fd = static_cast<int>(
                co_await io::open(p, fl, static_cast<mode_t>(md)));
            if (ofd) *ofd = fd;
        }(path, flags, mode, out_fd));
}

::dftu_task* io_close(void* h, int fd, int* out_rc) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, int* orc) -> coro::CoroTask<void> {
            int rc = static_cast<int>(co_await io::close(f));
            if (orc) *orc = rc;
        }(fd, out_rc));
}

::dftu_task* io_read(void* h, int fd, void* buf, std::uint64_t len,
                     std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, void* b, std::uint64_t l,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::read(f, b, static_cast<std::size_t>(l));
            if (on) *on = n;
        }(fd, buf, len, out_n));
}

::dftu_task* io_write(void* h, int fd, const void* buf, std::uint64_t len,
                      std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const void* b, std::uint64_t l,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::write(f, b, static_cast<std::size_t>(l));
            if (on) *on = n;
        }(fd, buf, len, out_n));
}

::dftu_task* io_pread(void* h, int fd, void* buf, std::uint64_t len,
                      std::uint64_t off, std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, void* b, std::uint64_t l, std::uint64_t o,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::pread(f, b, static_cast<std::size_t>(l),
                                           static_cast<off_t>(o));
            if (on) *on = n;
        }(fd, buf, len, off, out_n));
}

::dftu_task* io_pwrite(void* h, int fd, const void* buf, std::uint64_t len,
                       std::uint64_t off, std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const void* b, std::uint64_t l, std::uint64_t o,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::pwrite(f, b, static_cast<std::size_t>(l),
                                            static_cast<off_t>(o));
            if (on) *on = n;
        }(fd, buf, len, off, out_n));
}

::dftu_task* io_fsync(void* h, int fd, int* out_rc) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, int* orc) -> coro::CoroTask<void> {
            int rc = static_cast<int>(co_await io::fsync(f));
            if (orc) *orc = rc;
        }(fd, out_rc));
}

::dftu_task* io_ftruncate(void* h, int fd, std::uint64_t len, int* out_rc) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, std::uint64_t l, int* orc) -> coro::CoroTask<void> {
            int rc = static_cast<int>(
                co_await io::ftruncate(f, static_cast<off_t>(l)));
            if (orc) *orc = rc;
        }(fd, len, out_rc));
}

::dftu_task* io_fstat(void* h, int fd, dftu_stat* out) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, dftu_stat* o) -> coro::CoroTask<void> {
            struct stat st{};
            ssize_t rc = co_await io::fstat(f, &st);
            if (o && rc == 0) {
                o->size = static_cast<std::uint64_t>(st.st_size);
#if defined(__APPLE__)
                const struct timespec& mt = st.st_mtimespec;
#else
                const struct timespec& mt = st.st_mtim;
#endif
                o->mtime_ns =
                    static_cast<std::uint64_t>(mt.tv_sec) * 1000000000ull +
                    static_cast<std::uint64_t>(mt.tv_nsec);
                o->mode = static_cast<std::uint32_t>(st.st_mode);
            }
        }(fd, out));
}

::dftu_task* io_readv(void* h, int fd, const struct iovec* iov, int iovcnt,
                      std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const struct iovec* v, int c,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::readv(f, v, c);
            if (on) *on = n;
        }(fd, iov, iovcnt, out_n));
}

::dftu_task* io_writev(void* h, int fd, const struct iovec* iov, int iovcnt,
                       std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const struct iovec* v, int c,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::writev(f, v, c);
            if (on) *on = n;
        }(fd, iov, iovcnt, out_n));
}

::dftu_task* io_preadv(void* h, int fd, const struct iovec* iov, int iovcnt,
                       std::uint64_t off, std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const struct iovec* v, int c, std::uint64_t o,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::preadv(f, v, c, static_cast<off_t>(o));
            if (on) *on = n;
        }(fd, iov, iovcnt, off, out_n));
}

::dftu_task* io_pwritev(void* h, int fd, const struct iovec* iov, int iovcnt,
                        std::uint64_t off, std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const struct iovec* v, int c, std::uint64_t o,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::pwritev(f, v, c, static_cast<off_t>(o));
            if (on) *on = n;
        }(fd, iov, iovcnt, off, out_n));
}

::dftu_task* io_lseek(void* h, int fd, std::int64_t off, int whence,
                      std::int64_t* out_off) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, std::int64_t o, int w,
           std::int64_t* oo) -> coro::CoroTask<void> {
            off_t r = co_await io::lseek(f, static_cast<off_t>(o), w);
            if (oo) *oo = static_cast<std::int64_t>(r);
        }(fd, off, whence, out_off));
}

::dftu_task* io_sendfile(void* h, int out_fd, int in_fd, std::uint64_t off,
                         std::uint64_t count, std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int ofd, int ifd, std::uint64_t o, std::uint64_t c,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n = co_await io::sendfile(ofd, ifd, static_cast<off_t>(o),
                                              static_cast<std::size_t>(c));
            if (on) *on = n;
        }(out_fd, in_fd, off, count, out_n));
}

::dftu_task* io_accept(void* h, int fd, struct sockaddr* addr,
                       socklen_t* addrlen, int* out_fd) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, struct sockaddr* a, socklen_t* al,
           int* ofd) -> coro::CoroTask<void> {
            int cfd = static_cast<int>(co_await io::accept(f, a, al));
            if (ofd) *ofd = cfd;
        }(fd, addr, addrlen, out_fd));
}

::dftu_task* io_recv(void* h, int fd, void* buf, std::uint64_t len, int flags,
                     std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, void* b, std::uint64_t l, int fl,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n =
                co_await io::recv(f, b, static_cast<std::size_t>(l), fl);
            if (on) *on = n;
        }(fd, buf, len, flags, out_n));
}

::dftu_task* io_send(void* h, int fd, const void* buf, std::uint64_t len,
                     int flags, std::int64_t* out_n) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](int f, const void* b, std::uint64_t l, int fl,
           std::int64_t* on) -> coro::CoroTask<void> {
            ssize_t n =
                co_await io::send(f, b, static_cast<std::size_t>(l), fl);
            if (on) *on = n;
        }(fd, buf, len, flags, out_n));
}

const dftu_io g_io = {io_open,   io_close,  io_read,      io_write, io_pread,
                      io_pwrite, io_fsync,  io_ftruncate, io_fstat, io_readv,
                      io_writev, io_preadv, io_pwritev,   io_lseek, io_sendfile,
                      io_accept, io_recv,   io_send};

// yield() reschedules fn onto a pool worker, so when_all over several spawns
// runs them concurrently instead of inline and serial.
::dftu_task* host_spawn(void* h, dftu_work_fn fn, void* arg) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](dftu_work_fn f, void* a) -> coro::CoroTask<void> {
            co_await coro::yield();
            if (f) f(a);
        }(fn, arg));
}

::dftu_task* host_when_all(void* h, ::dftu_task* const* ts, std::uint32_t n) {
    std::vector<coro::CoroTask<void>*> kids;
    kids.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) kids.push_back(as_task(ts[i]));
    return static_cast<PluginFold*>(h)->emplace_task(
        [](std::vector<coro::CoroTask<void>*> ks) -> coro::CoroTask<void> {
            std::vector<coro::CoroTask<void>> v;
            v.reserve(ks.size());
            for (auto* k : ks) v.push_back(std::move(*k));
            co_await coro::when_all(std::move(v));
        }(std::move(kids)));
}

::dftu_task* host_when_any(void* h, ::dftu_task* const* ts, std::uint32_t n) {
    std::vector<coro::CoroTask<void>*> kids;
    kids.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) kids.push_back(as_task(ts[i]));
    return static_cast<PluginFold*>(h)->emplace_task(
        [](std::vector<coro::CoroTask<void>*> ks) -> coro::CoroTask<void> {
            // when_any needs a non-void result, so wrap each void child.
            std::vector<coro::CoroTask<int>> v;
            v.reserve(ks.size());
            for (auto* k : ks)
                v.push_back(
                    [](coro::CoroTask<void>* kk) -> coro::CoroTask<int> {
                        co_await std::move(*kk);
                        co_return 0;
                    }(k));
            co_await coro::when_any(std::move(v));
        }(std::move(kids)));
}

::dftu_task* host_drive(void* h, ::dftu_task* (*step)(void*), void* coro) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](::dftu_task* (*s)(void*), void* c) -> coro::CoroTask<void> {
            while (::dftu_task* t = s(c)) co_await *as_task(t);
        }(step, coro));
}

::dftu_task* host_then(void* h, ::dftu_task* t, dftu_work_fn fn, void* arg) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](coro::CoroTask<void>* pre, dftu_work_fn f,
           void* a) -> coro::CoroTask<void> {
            coro::CoroTask<void> p = std::move(*pre);
            co_await p;
            if (f) f(a);
        }(as_task(t), fn, arg));
}

void host_run_blocking(void* /*h*/, dftu_work_fn fn, void* arg) {
    if (!fn) return;
    BlockingRegion block;
    fn(arg);
}

::dftu_op* host_compose_make(void* h, dftu_op_fn fn, void* state,
                             void (*free_state)(void*), dftu_type in_ty,
                             std::uint32_t in_size, dftu_type out_ty,
                             std::uint32_t out_size) {
    return static_cast<PluginFold*>(h)->compose_make(
        fn, state, free_state, in_ty, in_size, out_ty, out_size);
}
::dftu_op* host_compose_then(void* h, ::dftu_op* a, ::dftu_op* b) {
    return static_cast<PluginFold*>(h)->compose_then(a, b);
}
::dftu_op* host_compose_when_all(void* h, ::dftu_op* const* ops,
                                 std::uint32_t n) {
    return static_cast<PluginFold*>(h)->compose_when_all(ops, n);
}
::dftu_op* host_compose_when_any(void* h, ::dftu_op* const* ops,
                                 std::uint32_t n) {
    return static_cast<PluginFold*>(h)->compose_when_any(ops, n);
}
::dftu_task* host_compose_run(void* h, ::dftu_op* op, const void* in, void* out,
                              int* rc) {
    return static_cast<PluginFold*>(h)->compose_run(op, in, out, rc);
}
::dftu_op* host_compose_util_op(void* h, std::uint32_t util_id) {
    return static_cast<PluginFold*>(h)->compose_util_op(util_id);
}
// Ops are arena-owned (scan-lifetime); early release is a no-op.
void host_compose_free_op(void* /*h*/, ::dftu_op* /*op*/) {}

::dftu_task* host_util_run_async(void* h, std::uint32_t util_id,
                                 const void* in_data, void* out_data,
                                 int* out_rc) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](std::uint32_t id, const void* in, void* out,
           int* orc) -> coro::CoroTask<void> {
            int rc = co_await registry_run_async(id, in, out);
            if (orc) *orc = rc;
        }(util_id, in_data, out_data, out_rc));
}

::dftu_task* host_util_run_stream_async(void* h, std::uint32_t util_id,
                                        const void* in_data,
                                        dftu_stream_item_fn on_item, void* ud,
                                        int* out_rc) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](std::uint32_t id, const void* in, dftu_stream_item_fn oi, void* u,
           int* orc) -> coro::CoroTask<void> {
            int rc = co_await registry_run_stream_async(id, in, oi, u);
            if (orc) *orc = rc;
        }(util_id, in_data, on_item, ud, out_rc));
}

::dftu_stream* host_util_stream_open(void* h, std::uint32_t util_id,
                                     const void* in_data) {
    (void)h;
    return reinterpret_cast<::dftu_stream*>(
        registry_stream_open(util_id, in_data));
}

::dftu_task* host_util_stream_next(void* h, ::dftu_stream* s,
                                   const void** out_item, int* out_rc) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](StreamDriver* d, const void** oi, int* orc) -> coro::CoroTask<void> {
            int rc = co_await d->next(d, oi);
            if (orc) *orc = rc;
        }(reinterpret_cast<StreamDriver*>(s), out_item, out_rc));
}

void host_util_stream_close(void*, ::dftu_stream* s) {
    if (!s) return;
    auto* d = reinterpret_cast<StreamDriver*>(s);
    d->destroy(d);
}

::dftu_task* host_merge_shards(void* h, const char* target,
                               const char* const* shards, std::uint32_t n) {
    std::string tgt(target ? target : "");
    std::vector<std::string> sv;
    sv.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i)
        sv.emplace_back(shards[i] ? shards[i] : "");
    return static_cast<PluginFold*>(h)->emplace_task(
        [](std::string t, std::vector<std::string> s) -> coro::CoroTask<void> {
            co_await parallel::merge_shards(t, s);
        }(std::move(tgt), std::move(sv)));
}

::dftu_writer* host_writer_create(void* h, const char* path,
                                  std::uint32_t num_workers, int gzip) {
    return static_cast<PluginFold*>(h)->create_writer(path, num_workers, gzip);
}

::dftu_task* host_writer_open(void* h, ::dftu_writer* w) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](PluginWriter* pw) -> coro::CoroTask<void> {
            co_await pw->writer->open(pw->path, pw->num_workers, pw->gzip,
                                      nullptr);
        }(reinterpret_cast<PluginWriter*>(w)));
}

::dftu_task* host_writer_chunk(void* h, ::dftu_writer* w, std::uint32_t worker,
                               const void* data, std::uint64_t len) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](PluginWriter* pw, std::uint32_t wk, const void* d,
           std::uint64_t l) -> coro::CoroTask<void> {
            ByteView bv(static_cast<const unsigned char*>(d),
                        static_cast<std::size_t>(l));
            co_await pw->writer->write_chunk(static_cast<std::size_t>(wk), bv);
        }(reinterpret_cast<PluginWriter*>(w), worker, data, len));
}

::dftu_task* host_writer_close(void* h, ::dftu_writer* w) {
    return static_cast<PluginFold*>(h)->emplace_task(
        [](PluginWriter* pw) -> coro::CoroTask<void> {
            co_await pw->writer->close();
        }(reinterpret_cast<PluginWriter*>(w)));
}

// DDSketch lacks a mean, so the box tracks the running weighted sum.
struct SketchBox {
    utilities::common::statistics::DDSketch sketch;
    double sum = 0.0;
};

::dftu_sketch* host_sketch_create(void*) {
    return reinterpret_cast<::dftu_sketch*>(new (std::nothrow) SketchBox());
}

void host_sketch_add(void*, ::dftu_sketch* s, double v, double w) {
    if (!s) return;
    auto* b = reinterpret_cast<SketchBox*>(s);
    b->sketch.add(v, w);
    b->sum += v * w;
}

void host_sketch_merge(void*, ::dftu_sketch* into, const ::dftu_sketch* other) {
    if (!into || !other) return;
    auto* a = reinterpret_cast<SketchBox*>(into);
    const auto* b = reinterpret_cast<const SketchBox*>(other);
    a->sketch.merge(b->sketch);
    a->sum += b->sum;
}

::dftu_quantiles host_sketch_result(void*, const ::dftu_sketch* s) {
    ::dftu_quantiles q{};
    if (!s) return q;
    const auto* b = reinterpret_cast<const SketchBox*>(s);
    q.count = b->sketch.count();
    q.min = b->sketch.min();
    q.max = b->sketch.max();
    q.mean = q.count ? b->sum / static_cast<double>(q.count) : 0.0;
    q.p50 = b->sketch.quantile(0.5);
    q.p90 = b->sketch.quantile(0.9);
    q.p95 = b->sketch.quantile(0.95);
    q.p99 = b->sketch.quantile(0.99);
    return q;
}

void host_sketch_free(void*, ::dftu_sketch* s) {
    delete reinterpret_cast<SketchBox*>(s);
}

// SKETCH yields a quantile struct, not a scalar, so it has no scalar column;
// map_new_sketch materializes it to a count column plus one column per
// quantile.
bool monoid_has_scalar(dftu_monoid_kind kind) {
    return kind != DFTU_MONOID_SKETCH;
}

bool monoid_is_quantiles(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SKETCH;
}

// Two-variable co-moment monoids fed by map_add_xy_at, not a single u64/f64
// add.
bool monoid_is_xy(dftu_monoid_kind kind) {
    switch (kind) {
        case DFTU_MONOID_CORR:
        case DFTU_MONOID_COVAR_POP:
        case DFTU_MONOID_COVAR_SAMP:
        case DFTU_MONOID_REGR_SLOPE:
        case DFTU_MONOID_REGR_INTERCEPT:
        case DFTU_MONOID_REGR_R2:
            return true;
        default:
            return false;
    }
}

// A component that map_add_row can drive: a scalar monoid fed by a single
// u64/f64 add. Excludes collections, arg-row, sketch, and the two-variable
// co-moment monoids.
bool monoid_is_fused_eligible(dftu_monoid_kind kind) {
    return monoid_has_scalar(kind) && !monoid_is_argrow(kind) &&
           !monoid_is_quantiles(kind) && !monoid_is_xy(kind);
}

// Column label for a requested quantile: p50, p90, p99, p99_9 (dot -> '_').
std::string quantile_column_name(double q) {
    double pct = q * 100.0;
    long long whole = static_cast<long long>(pct + 0.5);
    double diff = pct - static_cast<double>(whole);
    std::string name = "p";
    if (diff < 1e-9 && diff > -1e-9) {
        name += std::to_string(whole);
    } else {
        char buf[32];
        std::snprintf(buf, sizeof(buf), "%.4g", pct);
        for (char* c = buf; *c; ++c) name += (*c == '.') ? '_' : *c;
    }
    return name;
}

bool monoid_is_f64(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SUM_F64 || kind == DFTU_MONOID_MIN_F64 ||
           kind == DFTU_MONOID_MAX_F64;
}

bool monoid_is_set(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SET_STR || kind == DFTU_MONOID_SET_I64;
}

bool monoid_is_list(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_LIST_STR || kind == DFTU_MONOID_LIST_I64;
}

// Also the TOPK/BOTTOMK/SAMPLE _STR kinds, which emit their kept ids as
// list<string>.
bool monoid_is_str_collection(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SET_STR || kind == DFTU_MONOID_LIST_STR ||
           kind == DFTU_MONOID_TOPK_STR || kind == DFTU_MONOID_BOTTOMK_STR ||
           kind == DFTU_MONOID_SAMPLE_STR;
}

bool monoid_is_i64_collection(dftu_monoid_kind kind) {
    return kind == DFTU_MONOID_SET_I64 || kind == DFTU_MONOID_LIST_I64 ||
           kind == DFTU_MONOID_TOPK_I64 || kind == DFTU_MONOID_BOTTOMK_I64 ||
           kind == DFTU_MONOID_SAMPLE_I64;
}

// State grows per element (bounded to k for TOPK/BOTTOMK), so the footprint
// counter bumps on every add, not only on a new-key insert.
bool monoid_is_variable(dftu_monoid_kind kind) {
    return monoid_is_set(kind) || monoid_is_list(kind) ||
           monoid_is_topk(kind) || monoid_is_approx_topk(kind) ||
           monoid_is_sample(kind);
}

namespace codec = utilities::common::serialization;

// 256 partitions (one hash byte), ClickHouse's default, so one oversized
// partition is rare without recursive re-partition.
constexpr std::uint32_t MAP_SPILL_PART_BITS = 8;
// The fuse fan-out cap (fold.cpp: min(units, 16)); the budget is split across
// it so peak stays ~budget.
constexpr std::size_t MAP_SPILL_WORKERS = 16;
// Test the budget on a new key or every this-many adds, so the hot path pays a
// counter bump not a comparison.
constexpr std::size_t MAP_SPILL_CHECK_STRIDE = 4096;
constexpr std::size_t MAP_STREAM_CHUNK_ROWS = 65536;

// MEM_BUDGET is an explicit global byte budget (0 = off) and wins; else
// AUTO_SPILL=1 uses compute_memory_budget. SPILL_DIR overrides $TMPDIR.
struct MapSpillEnv {
    bool enabled = false;
    std::size_t share = 0;
    std::string dir;
    // STREAM=1 surfaces map results as a streamed per-partition sequence
    // instead of one eager batch.
    bool stream = false;
};

MapSpillEnv read_map_spill_env() {
    MapSpillEnv c;
    if (const char* s = std::getenv("DFTRACER_PLUGIN_MAP_STREAM");
        s && std::atoi(s) != 0)
        c.stream = true;
    std::size_t budget = 0;
    if (const char* b = std::getenv("DFTRACER_PLUGIN_MAP_MEM_BUDGET")) {
        budget = static_cast<std::size_t>(std::strtoull(b, nullptr, 10));
    } else if (const char* a = std::getenv("DFTRACER_PLUGIN_MAP_AUTO_SPILL");
               a && std::atoi(a) != 0) {
        budget = dftracer::utils::compute_memory_budget(0);
    }
    if (budget == 0) return c;
    c.enabled = true;
    c.share = std::max<std::size_t>(1, budget / MAP_SPILL_WORKERS);
    if (const char* d = std::getenv("DFTRACER_PLUGIN_MAP_SPILL_DIR"); d && *d)
        c.dir = d;
    return c;
}

// Write one partition to a sorted run file in view_spill's framing (big-endian
// length prefix per record, key ints then MonoidAccumulator::serialize),
// keeping one spill dialect. Throws on any I/O failure.
void write_run(const MapAccum& m, const MapAccum::EntriesMap& part,
               const std::string& path) {
    using Entry = std::pair<const std::vector<std::int64_t>,
                            std::vector<MonoidAccumulator>>;
    std::vector<const Entry*> ents;
    ents.reserve(part.size());
    for (const auto& kv : part) ents.push_back(&kv);
    std::sort(ents.begin(), ents.end(), [](const Entry* a, const Entry* b) {
        return a->first < b->first;
    });
    std::ofstream os(path, std::ios::binary);
    if (!os)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: cannot open run file " + path);
    std::string rec, hdr;
    for (const Entry* e : ents) {
        rec.clear();
        for (std::uint32_t i = 0; i < m.key_n; ++i)
            codec::put_be64(rec, static_cast<std::uint64_t>(e->first[i]));
        for (const MonoidAccumulator& mon : e->second) mon.serialize(rec);
        hdr.clear();
        codec::put_be64(hdr, rec.size());
        os.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        os.write(rec.data(), static_cast<std::streamsize>(rec.size()));
    }
    os.flush();
    if (!os)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: write failed (disk full?) " + path);
}

// Stream one run back, merging each partial into m by key. partition_of(key) is
// deterministic, so a key spilled from partition p reloads into partition p.
void read_run_into(MapAccum& m, const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: cannot reopen run " + path);
    std::string buf;
    char hdr[8];
    while (is.read(hdr, 8)) {
        std::uint64_t len = 0;
        for (int i = 0; i < 8; ++i)
            len = (len << 8) | static_cast<std::uint8_t>(hdr[i]);
        buf.resize(len);
        if (len && !is.read(buf.data(), static_cast<std::streamsize>(len)))
            throw dftracer::utils::DFTUtilsException(
                dftracer::utils::ErrorCode::IO,
                "plugin map spill: truncated run " + path);
        codec::BinaryReader br(buf);
        std::vector<std::int64_t> key(m.key_n);
        for (std::uint32_t i = 0; i < m.key_n; ++i)
            key[i] = static_cast<std::int64_t>(br.be64());
        std::string_view rest = br.remaining();
        const std::byte* p = reinterpret_cast<const std::byte*>(rest.data());
        std::size_t n = rest.size();
        std::vector<MonoidAccumulator>& into = m.touch(key);
        for (std::size_t c = 0; c < m.value_kinds.size(); ++c) {
            MonoidAccumulator mon(m.value_kinds[c]);
            std::size_t used = mon.deserialize(p, n);
            if (used == 0)
                throw dftracer::utils::DFTUtilsException(
                    dftracer::utils::ErrorCode::PARSE,
                    "plugin map spill: corrupt run record in " + path);
            p += used;
            n -= used;
            if (c < into.size()) into[c].merge(mon);
        }
    }
}

// Fixed-width integer, float, STR, or BYTES; other ABI types are rejected at
// map creation.
bool key_type_supported(dftu_type t) {
    switch (t) {
        case DFTU_T_I8:
        case DFTU_T_I16:
        case DFTU_T_I32:
        case DFTU_T_I64:
        case DFTU_T_U8:
        case DFTU_T_U16:
        case DFTU_T_U32:
        case DFTU_T_U64:
        case DFTU_T_F32:
        case DFTU_T_F64:
        case DFTU_T_STR:
        case DFTU_T_BYTES:
            return true;
        default:
            return false;
    }
}

// STR and BYTES both ride the int64 key slot as an interned dftu_str id,
// differing only in the materialized column type (utf8 vs binary); every other
// path treats the slot as a plain int64 id.
bool key_type_interned(dftu_type t) {
    return t == DFTU_T_STR || t == DFTU_T_BYTES;
}

bool key_type_unsigned(dftu_type t) {
    return t == DFTU_T_U8 || t == DFTU_T_U16 || t == DFTU_T_U32 ||
           t == DFTU_T_U64;
}

bool key_type_float(dftu_type t) { return t == DFTU_T_F32 || t == DFTU_T_F64; }

// Recover a float key component bit-cast into the int64 slot (f32 from the low
// 32 bits, f64 from all 64).
double key_bits_to_double(dftu_type t, std::int64_t bits) {
    if (t == DFTU_T_F32) {
        std::uint32_t lo =
            static_cast<std::uint32_t>(static_cast<std::uint64_t>(bits));
        float f;
        std::memcpy(&f, &lo, sizeof(f));
        return static_cast<double>(f);
    }
    double d;
    std::memcpy(&d, &bits, sizeof(d));
    return d;
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
utilities::common::arrow::ColumnType key_column_type(dftu_type t) {
    namespace arr = utilities::common::arrow;
    switch (t) {
        case DFTU_T_STR:
            return arr::ColumnType::STRING;
        case DFTU_T_BYTES:
            return arr::ColumnType::BINARY;
        case DFTU_T_I8:
            return arr::ColumnType::INT8;
        case DFTU_T_I16:
            return arr::ColumnType::INT16;
        case DFTU_T_I32:
            return arr::ColumnType::INT32;
        case DFTU_T_U8:
            return arr::ColumnType::UINT8;
        case DFTU_T_U16:
            return arr::ColumnType::UINT16;
        case DFTU_T_U32:
            return arr::ColumnType::UINT32;
        case DFTU_T_U64:
            return arr::ColumnType::UINT64;
        case DFTU_T_F32:
            return arr::ColumnType::FLOAT32;
        case DFTU_T_F64:
            return arr::ColumnType::DOUBLE;
        default:
            return arr::ColumnType::INT64;
    }
}

// Append a non-STR key component, recovering unsigned and float values from the
// int64 slot bits.
void append_scalar_key(utilities::common::arrow::RecordBatchBuilder& builder,
                       std::uint32_t col, dftu_type t, std::int64_t bits) {
    if (key_type_float(t))
        builder.append_double(col, key_bits_to_double(t, bits));
    else if (key_type_unsigned(t))
        builder.append_uint64(col, static_cast<std::uint64_t>(bits));
    else
        builder.append_int64(col, bits);
}

// Typed min/max materialize at their element width; MIN/MAX_U64 keep the
// historical int64 column, f64-result monoids double, every other scalar int64.
utilities::common::arrow::ColumnType monoid_value_column_type(
    dftu_monoid_kind k) {
    namespace arr = utilities::common::arrow;
    switch (k) {
        case DFTU_MONOID_MIN_I8:
        case DFTU_MONOID_MAX_I8:
            return arr::ColumnType::INT8;
        case DFTU_MONOID_MIN_I16:
        case DFTU_MONOID_MAX_I16:
            return arr::ColumnType::INT16;
        case DFTU_MONOID_MIN_I32:
        case DFTU_MONOID_MAX_I32:
            return arr::ColumnType::INT32;
        case DFTU_MONOID_MIN_U8:
        case DFTU_MONOID_MAX_U8:
            return arr::ColumnType::UINT8;
        case DFTU_MONOID_MIN_U16:
        case DFTU_MONOID_MAX_U16:
            return arr::ColumnType::UINT16;
        case DFTU_MONOID_MIN_U32:
        case DFTU_MONOID_MAX_U32:
            return arr::ColumnType::UINT32;
        case DFTU_MONOID_MIN_F32:
        case DFTU_MONOID_MAX_F32:
            return arr::ColumnType::FLOAT32;
        case DFTU_MONOID_MEAN:
        case DFTU_MONOID_VARIANCE:
        case DFTU_MONOID_STDDEV:
        case DFTU_MONOID_SKEWNESS:
        case DFTU_MONOID_KURTOSIS:
        case DFTU_MONOID_CORR:
        case DFTU_MONOID_COVAR_POP:
        case DFTU_MONOID_COVAR_SAMP:
        case DFTU_MONOID_REGR_SLOPE:
        case DFTU_MONOID_REGR_INTERCEPT:
        case DFTU_MONOID_REGR_R2:
            return arr::ColumnType::DOUBLE;
        case DFTU_MONOID_ARGMIN_I64:
        case DFTU_MONOID_ARGMAX_I64:
            return arr::ColumnType::INT64;
        case DFTU_MONOID_ARGMIN_STR:
        case DFTU_MONOID_ARGMAX_STR:
            return arr::ColumnType::STRING;
        default:
            return monoid_is_f64(k) ? arr::ColumnType::DOUBLE
                                    : arr::ColumnType::INT64;
    }
}
#endif

std::string_view resolve_id(dftracer::utils::StringIntern& intern,
                            dftu_str id) {
    if (id == DFTU_STR_NONE ||
        id >= dftracer::utils::StringIntern::FAST_CAPACITY)
        return {};
    return intern.resolve(id);
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
// Shared column emission for the map materializers. materialize_map passes an
// empty prefix for byte-identical output; materialize_joined disambiguates its
// two value sides with "l_"/"r_".

// STR resolves to utf8, BYTES to binary, every other width to
// append_scalar_key.
void append_key_cell(utilities::common::arrow::RecordBatchBuilder& builder,
                     std::uint32_t col, dftu_type t, std::int64_t bits,
                     dftracer::utils::StringIntern& intern) {
    if (t == DFTU_T_STR)
        builder.append_string(col,
                              resolve_id(intern, static_cast<dftu_str>(bits)));
    else if (t == DFTU_T_BYTES)
        builder.append_binary(col,
                              resolve_id(intern, static_cast<dftu_str>(bits)));
    else
        append_scalar_key(builder, col, t, bits);
}

void append_key_specs(std::vector<utilities::common::arrow::ColumnSpec>& specs,
                      const std::vector<dftu_type>& key_types,
                      std::uint32_t key_n) {
    for (std::uint32_t i = 0; i < key_n; ++i)
        specs.push_back(
            {"k" + std::to_string(i), key_column_type(key_types[i])});
}

// An arg-row spreads across its payload columns and a sketch across its count +
// quantile columns; every other monoid is one.
std::uint32_t value_column_count(const std::vector<dftu_monoid_kind>& kinds,
                                 const std::vector<dftu_type>& payload_types,
                                 const std::vector<double>& quantile_qs,
                                 int only_comp = -1) {
    if (only_comp >= 0) return 1;  // a fused map emits one scalar per table
    std::uint32_t n = 0;
    for (dftu_monoid_kind k : kinds) {
        if (monoid_is_argrow(k))
            n += static_cast<std::uint32_t>(payload_types.size());
        else if (monoid_is_quantiles(k))
            n += 1u + static_cast<std::uint32_t>(quantile_qs.size());
        else
            n += 1u;
    }
    return n;
}

// Append the Arrow column specs for a flat value schema. `prefix` disambiguates
// a join's left/right value columns; "" reproduces materialize_map's naming.
void append_value_specs(
    std::vector<utilities::common::arrow::ColumnSpec>& specs,
    const std::vector<dftu_monoid_kind>& kinds,
    const std::vector<dftu_type>& payload_types,
    const std::vector<double>& quantile_qs, const std::string& prefix,
    int only_comp = -1) {
    namespace arr = utilities::common::arrow;
    if (only_comp >= 0) {  // a fused component: one scalar column named "value"
        specs.push_back(
            {prefix + "value", monoid_value_column_type(kinds[only_comp])});
        return;
    }
    const std::uint32_t value_n = static_cast<std::uint32_t>(kinds.size());
    for (std::uint32_t c = 0; c < value_n; ++c) {
        // arg-row expands to one column per payload component (p0..).
        if (monoid_is_argrow(kinds[c])) {
            for (std::uint32_t p = 0; p < payload_types.size(); ++p)
                specs.push_back({prefix + "p" + std::to_string(p),
                                 key_column_type(payload_types[p])});
            continue;
        }
        // sketch expands to a count column plus one column per quantile.
        if (monoid_is_quantiles(kinds[c])) {
            specs.push_back({prefix + "count", arr::ColumnType::INT64});
            for (double q : quantile_qs)
                specs.push_back({prefix + quantile_column_name(q),
                                 arr::ColumnType::DOUBLE});
            continue;
        }
        std::string col_name =
            prefix +
            (value_n == 1 ? std::string("value") : "v" + std::to_string(c));
        // APPROX_TOPK materializes to a list<struct<value, count>> column.
        if (monoid_is_approx_topk(kinds[c])) {
            arr::ColumnSpec spec;
            spec.name = std::move(col_name);
            spec.type = arr::ColumnType::STRUCT_LIST;
            spec.fields.push_back({"value",
                                   monoid_approx_topk_is_str(kinds[c])
                                       ? arr::ColumnType::STRING
                                       : arr::ColumnType::INT64,
                                   {}});
            spec.fields.push_back({"count", arr::ColumnType::INT64, {}});
            specs.push_back(std::move(spec));
            continue;
        }
        arr::ColumnType ct = arr::ColumnType::INT64;
        if (monoid_is_str_collection(kinds[c]))
            ct = arr::ColumnType::STRING_LIST;
        else if (monoid_is_i64_collection(kinds[c]))
            ct = arr::ColumnType::INT64_LIST;
        else
            ct = monoid_value_column_type(kinds[c]);
        specs.push_back({std::move(col_name), ct});
    }
}

// Emit one scalar monoid value cell at `col`.
void append_scalar_value(utilities::common::arrow::RecordBatchBuilder& builder,
                         std::uint32_t col, const MonoidAccumulator& mon,
                         dftu_monoid_kind kind,
                         dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    dftu_monoid_value v = mon.to_value();
    switch (monoid_value_column_type(kind)) {
        case arr::ColumnType::FLOAT32:
        case arr::ColumnType::DOUBLE:
            builder.append_double(col, v.as.f64);
            break;
        case arr::ColumnType::UINT8:
        case arr::ColumnType::UINT16:
        case arr::ColumnType::UINT32:
            builder.append_uint64(col, v.as.u64);
            break;
        case arr::ColumnType::STRING:
            builder.append_string(
                col, resolve_id(intern, static_cast<dftu_str>(v.as.u64)));
            break;
        default:
            builder.append_int64(col, static_cast<std::int64_t>(v.as.u64));
            break;
    }
}

// Append the value cells of one row starting at `first_col`, one per flat value
// column. `mons` are this side's Monoids; `kinds`/`payload_types` its schema.
void append_value_cells(utilities::common::arrow::RecordBatchBuilder& builder,
                        std::uint32_t first_col,
                        const std::vector<MonoidAccumulator>& mons,
                        const std::vector<dftu_monoid_kind>& kinds,
                        const std::vector<dftu_type>& payload_types,
                        const std::vector<double>& quantile_qs,
                        dftracer::utils::StringIntern& intern,
                        int only_comp = -1) {
    namespace arr = utilities::common::arrow;
    if (only_comp >= 0) {  // a fused component: just its scalar at first_col
        append_scalar_value(builder, first_col, mons[only_comp],
                            kinds[only_comp], intern);
        return;
    }
    const std::uint32_t value_n = static_cast<std::uint32_t>(kinds.size());
    for (std::uint32_t c = 0; c < value_n; ++c) {
        // A never-added arg-row nulls every payload column.
        if (monoid_is_argrow(kinds[c])) {
            const MonoidAccumulator& mon = mons[c];
            const bool has = mon.argrow_has();
            const std::vector<std::int64_t>& pl = mon.argrow_payload();
            for (std::uint32_t p = 0; p < payload_types.size(); ++p) {
                const std::uint32_t col = first_col + c + p;
                if (!has) {
                    builder.append_null(col);
                    continue;
                }
                append_key_cell(builder, col, payload_types[p], pl[p], intern);
            }
            continue;
        }
        // sketch emits count, then the value at each requested quantile.
        if (monoid_is_quantiles(kinds[c])) {
            const MonoidAccumulator& mon = mons[c];
            std::uint32_t col = first_col + c;
            builder.append_int64(col++,
                                 static_cast<std::int64_t>(mon.sketch_count()));
            for (double q : quantile_qs)
                builder.append_double(col++, mon.sketch_quantile(q));
            continue;
        }
        if (monoid_is_approx_topk(kinds[c])) {
            const bool is_str = monoid_approx_topk_is_str(kinds[c]);
            std::vector<std::vector<arr::StructCell>> structs;
            for (const auto& [value, count] : mons[c].approx_topk_entries()) {
                std::vector<arr::StructCell> cells(2);
                if (is_str)
                    cells[0].str =
                        resolve_id(intern, static_cast<dftu_str>(value));
                else
                    cells[0].i64 = value;
                cells[1].i64 = static_cast<std::int64_t>(count);
                structs.push_back(std::move(cells));
            }
            builder.append_struct_list(first_col + c, structs);
            continue;
        }
        if (monoid_is_str_collection(kinds[c])) {
            std::vector<std::string_view> labels;
            auto emit_id = [&](std::int64_t id) {
                std::string_view s =
                    resolve_id(intern, static_cast<dftu_str>(id));
                if (!s.empty()) labels.push_back(s);
            };
            if (monoid_is_topk(kinds[c])) {
                for (std::int64_t id : mons[c].topk_elements()) emit_id(id);
            } else if (monoid_is_list(kinds[c])) {
                for (std::int64_t id : mons[c].ordered_elements()) emit_id(id);
            } else if (monoid_is_sample(kinds[c])) {
                for (std::int64_t id : mons[c].sample_items()) emit_id(id);
                std::sort(labels.begin(), labels.end());
            } else {
                for (std::int64_t id : mons[c].elements()) emit_id(id);
                std::sort(labels.begin(), labels.end());
            }
            builder.append_string_list(first_col + c, labels);
            continue;
        }
        if (monoid_is_i64_collection(kinds[c])) {
            std::vector<std::int64_t> vals;
            if (monoid_is_topk(kinds[c]))
                vals = mons[c].topk_elements();
            else if (monoid_is_list(kinds[c]))
                vals = mons[c].ordered_elements();
            else if (monoid_is_sample(kinds[c]))
                vals = mons[c].sample_items();
            else
                vals = mons[c].sorted_elements();
            builder.append_int64_list(first_col + c, vals);
            continue;
        }
        append_scalar_value(builder, first_col + c, mons[c], kinds[c], intern);
    }
}

// Null every value column of a flat schema (an outer-join row whose side had no
// key); append_null covers scalar, list, struct, and arg-row payload columns.
void null_value_cells(utilities::common::arrow::RecordBatchBuilder& builder,
                      std::uint32_t first_col,
                      const std::vector<dftu_monoid_kind>& kinds,
                      const std::vector<dftu_type>& payload_types,
                      const std::vector<double>& quantile_qs) {
    const std::uint32_t n =
        value_column_count(kinds, payload_types, quantile_qs);
    for (std::uint32_t i = 0; i < n; ++i) builder.append_null(first_col + i);
}
#endif

void append_json_escaped(std::string& out, std::string_view s) {
    for (char c : s) {
        switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    char b[8];
                    std::snprintf(b, sizeof(b), "\\u%04x",
                                  static_cast<unsigned>(c));
                    out += b;
                } else {
                    out += c;
                }
        }
    }
}

void serialize_event(std::string& out, dftracer::utils::StringIntern& intern,
                     const dftu_event& e, std::uint64_t id) {
    out += "{\"id\":";
    out += std::to_string(id);
    out += ",\"name\":\"";
    append_json_escaped(out, resolve_id(intern, e.name));
    out += "\",\"cat\":\"";
    append_json_escaped(out, resolve_id(intern, e.cat));
    out += "\",\"ph\":";
    out += std::to_string(static_cast<int>(e.phase));
    out += ",\"pid\":";
    out += std::to_string(e.pid);
    out += ",\"tid\":";
    out += std::to_string(e.tid);
    out += ",\"ts\":";
    out += std::to_string(e.ts);
    if (e.has_dur) {
        out += ",\"dur\":";
        out += std::to_string(e.dur);
    }
    if (e.arg_count && e.args) {
        out += ",\"args\":{";
        for (std::uint32_t i = 0; i < e.arg_count; ++i) {
            if (i) out += ',';
            const dftu_arg& a = e.args[i];
            out += '"';
            append_json_escaped(out, resolve_id(intern, a.key));
            out += "\":";
            if (a.kind == DFTU_ARG_STR) {
                out += '"';
                append_json_escaped(out, resolve_id(intern, a.v.str));
                out += '"';
            } else if (a.kind == DFTU_ARG_I64) {
                out += std::to_string(a.v.i64);
            } else {
                char b[32];
                std::snprintf(b, sizeof(b), "%.17g", a.v.f64);
                out += b;
            }
        }
        out += '}';
    }
    out += "}\n";
}

int host_batch_to_arrow(void* h, const dftu_batch* b, ::ArrowArray* out,
                        ::ArrowSchema* out_schema) {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    if (!b || !out || !out_schema) return -1;
    namespace arr = utilities::common::arrow;
    auto& intern = static_cast<PluginFold*>(h)->intern_table();
    try {
        arr::RecordBatchBuilder builder;
        builder.declare_schema({{"cat", arr::ColumnType::STRING},
                                {"name", arr::ColumnType::STRING},
                                {"pid", arr::ColumnType::UINT64},
                                {"tid", arr::ColumnType::UINT64},
                                {"ts", arr::ColumnType::UINT64},
                                {"dur", arr::ColumnType::UINT64},
                                {"phase", arr::ColumnType::INT64}});
        builder.reserve(b->count);
        for (std::uint32_t i = 0; i < b->count; ++i) {
            const dftu_event& e = b->events[i];
            builder.append_string(0, resolve_id(intern, e.cat));
            builder.append_string(1, resolve_id(intern, e.name));
            builder.append_uint64(2, e.pid);
            builder.append_uint64(3, e.tid);
            builder.append_uint64(4, e.ts);
            if (e.has_dur)
                builder.append_uint64(5, e.dur);
            else
                builder.append_null(5);
            builder.append_int64(6, static_cast<std::int64_t>(e.phase));
            builder.end_row();
        }
        arr::ArrowExportResult res = builder.finish();
        if (!res.valid()) return -1;
        ArrowArrayMove(res.get_array(), out);
        ArrowSchemaMove(res.get_schema(), out_schema);
        return 0;
    } catch (...) {
        return -1;
    }
#else
    (void)h;
    (void)b;
    (void)out;
    (void)out_schema;
    return -1;
#endif
}

int host_arrow_write_ipc(void*, ::ArrowArray* a, ::ArrowSchema* s,
                         const char* path) {
#ifdef DFTRACER_UTILS_ENABLE_ARROW_IPC
    if (!a || !s || !path || !a->release || !s->release) return -1;
    namespace arr = utilities::common::arrow;
    try {
        nanoarrow::UniqueSchema schema;
        ArrowSchemaMove(s, schema.get());
        nanoarrow::UniqueArray array;
        ArrowArrayMove(a, array.get());
        arr::ArrowExportResult res(std::move(schema), std::move(array));
        int rc = -1;
        std::string p(path);
        dftracer::utils::default_runtime().run_blocking(
            "dft-plugin-ipc", [&](CoroScope&) -> coro::CoroTask<void> {
                arr::IpcWriter writer;
                if (co_await writer.open(p) != 0) co_return;
                if (co_await writer.write_batch(res) != 0) co_return;
                if (co_await writer.close() != 0) co_return;
                rc = 0;
            });
        ArrowSchemaMove(res.get_schema(), s);
        ArrowArrayMove(res.get_array(), a);
        return rc;
    } catch (...) {
        return -1;
    }
#else
    (void)a;
    (void)s;
    (void)path;
    return -1;
#endif
}

// Read the first record batch of an IPC file into out/out_schema; the caller
// must release both.
int host_arrow_read_ipc(void*, const char* path, ::ArrowArray* out,
                        ::ArrowSchema* out_schema) {
#ifdef DFTRACER_UTILS_ENABLE_ARROW_IPC
    if (!path || !out || !out_schema) return -1;
    namespace arr = utilities::common::arrow;
    try {
        arr::IpcReader reader;
        if (reader.open(std::string(path)) != 0) return -1;
        if (reader.num_batches() == 0) return -1;
        arr::ArrowExportResult res = reader.read_batch(0);
        if (!res.get_array() || !res.get_schema() ||
            !res.get_array()->release || !res.get_schema()->release)
            return -1;
        ArrowArrayMove(res.get_array(), out);
        ArrowSchemaMove(res.get_schema(), out_schema);
        return 0;
    } catch (...) {
        return -1;
    }
#else
    (void)path;
    (void)out;
    (void)out_schema;
    return -1;
#endif
}

::dftu_trace_writer* host_trace_open_write(void* h, const char* path) {
    return static_cast<PluginFold*>(h)->create_trace_writer(path);
}

int host_trace_write(void* h, ::dftu_trace_writer* w, const dftu_event* evs,
                     std::uint32_t n) {
    if (!w || (!evs && n)) return -1;
    auto* tw = reinterpret_cast<PluginTraceWriter*>(w);
    auto& intern = static_cast<PluginFold*>(h)->intern_table();
    try {
        for (std::uint32_t i = 0; i < n; ++i)
            serialize_event(tw->buffer, intern, evs[i], tw->next_id++);
        return 0;
    } catch (...) {
        return -1;
    }
}

int host_trace_close(void* h, ::dftu_trace_writer* w) {
    return static_cast<PluginFold*>(h)->close_trace_writer(w);
}

const dftu_ext_writer g_writer = {host_merge_shards, host_writer_create,
                                  host_writer_open, host_writer_chunk,
                                  host_writer_close};

const dftu_ext_sketch g_sketch = {host_sketch_create, host_sketch_add,
                                  host_sketch_merge, host_sketch_result,
                                  host_sketch_free};

const dftu_ext_arrow g_arrow = {host_batch_to_arrow, host_arrow_write_ipc,
                                host_arrow_read_ipc};

int host_trace_read(void* h, const char* path, dftu_stream_item_fn on_event,
                    void* ud) {
    if (!path || !on_event) return -1;
    auto& intern = static_cast<PluginFold*>(h)->intern_table();
    namespace views = trace::views;
    try {
        std::string p(path);
        dftracer::utils::default_runtime().run_blocking(
            "dft-plugin-trace-read", [&](CoroScope&) -> coro::CoroTask<void> {
                simdjson::dom::parser parser;
                std::vector<dftu_arg> args;
                views::View v = views::View::from_file(p);
                co_await v.for_each_batch(
                    [&](std::size_t,
                        const std::vector<std::string_view>& events) {
                        for (std::string_view ev : events) {
                            simdjson::padded_string ps(ev);
                            simdjson::dom::element root;
                            if (parser.parse(ps).get(root)) continue;
                            auto fe = views::detail::extract_fold_event(
                                root, intern, /*needs_args=*/true);
                            dftu_event ce{};
                            ce.cat = fe.cat_id;
                            ce.name = fe.name_id;
                            ce.fhash = fe.fhash_id;
                            ce.hhash = fe.hhash_id;
                            ce.pid = fe.pid;
                            ce.tid = fe.tid;
                            ce.ts = fe.ts;
                            ce.dur = fe.dur;
                            ce.phase = map_phase(fe.phase);
                            ce.has_dur = fe.has_dur ? 1 : 0;
                            args.clear();
                            for (const auto& [key, val] : fe.args) {
                                dftu_arg a{};
                                a.key = key;
                                switch (val.index()) {
                                    case 0:
                                        a.kind = DFTU_ARG_F64;
                                        a.v.f64 = std::get<double>(val);
                                        break;
                                    case 1:
                                        a.kind = DFTU_ARG_I64;
                                        a.v.i64 = std::get<std::int64_t>(val);
                                        break;
                                    default:
                                        a.kind = DFTU_ARG_STR;
                                        a.v.str = std::get<std::uint32_t>(val);
                                        break;
                                }
                                args.push_back(a);
                            }
                            ce.arg_count =
                                static_cast<std::uint32_t>(args.size());
                            ce.args = args.empty() ? nullptr : args.data();
                            on_event(&ce, ud);
                        }
                    },
                    /*num_slots=*/1, /*limit=*/0);
            });
        return 0;
    } catch (...) {
        return -1;
    }
}

const dftu_ext_trace g_trace = {host_trace_open_write, host_trace_write,
                                host_trace_close, host_trace_read};

const dftu_utility* host_find_by_id(void*, std::uint32_t id) {
    return registry_find(id);
}

int host_run_stream(void*, std::uint32_t id, const void* in,
                    dftu_stream_item_fn on_item, void* ud) {
    return registry_run_stream(id, in, on_item, ud);
}

dftu_query* host_query_compile(void* h, const char* src, std::uint32_t len) {
    return static_cast<PluginFold*>(h)->compile_query(src, len);
}

int host_query_matches(void* h, const dftu_query* q, const dftu_event* e) {
    if (!q || !e) return 0;
    try {
        return static_cast<PluginFold*>(h)->match_query(
            *reinterpret_cast<const Query*>(q), *e);
    } catch (...) {
        return 0;
    }
}

std::uint64_t host_port_key(void*, const char* cap_id) {
    return cap_id ? dftracer::utils::hash::fnv1a_hash(cap_id) : 0;
}

void host_port_publish(void* h, std::uint64_t key, const void* data,
                       std::uint32_t len) {
    static_cast<PluginFold*>(h)->port_publish(key, data, len);
}

const void* host_port_consume(void* h, std::uint64_t key,
                              std::uint32_t* out_len) {
    return static_cast<PluginFold*>(h)->port_consume(key, out_len);
}

const dftu_ext_ports g_ports = {host_port_key, host_port_publish,
                                host_port_consume};

::dftu_handle* host_handle_get(void* h, const char* cap_id,
                               dftu_monoid_kind kind) {
    return reinterpret_cast<::dftu_handle*>(
        static_cast<PluginFold*>(h)->handle_get(cap_id, kind));
}

void host_handle_add_u64(void*, ::dftu_handle* hd, std::uint64_t v) {
    if (hd) reinterpret_cast<MonoidAccumulator*>(hd)->add_u64(v);
}

void host_handle_add_f64(void*, ::dftu_handle* hd, double v, double w) {
    if (hd) reinterpret_cast<MonoidAccumulator*>(hd)->add_f64(v, w);
}

int host_handle_result(void* h, const char* cap_id, dftu_monoid_value* out) {
    return static_cast<PluginFold*>(h)->handle_result(cap_id, out);
}

const dftu_ext_handles g_handles = {host_handle_get, host_handle_add_u64,
                                    host_handle_add_f64, host_handle_result};

void host_result_emit(void* h, const char* name, const void* data,
                      std::uint64_t len) {
    static_cast<PluginFold*>(h)->result_emit(name, data, len);
}

int host_result_emit_arrow(void* h, const char* name, ::ArrowArray* a,
                           ::ArrowSchema* s) {
    return static_cast<PluginFold*>(h)->result_emit_arrow(name, a, s);
}

int host_result_emit_frame(void* h, const char* name, ::dftu_dataframe* df) {
    return static_cast<PluginFold*>(h)->result_emit_frame(name, df);
}

const dftu_ext_result g_result = {host_result_emit, host_result_emit_arrow,
                                  host_result_emit_frame};

::dftu_map* host_map_new(void* h, const char* name, const dftu_type* key_types,
                         std::uint32_t key_n, dftu_monoid_kind value) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get(name, key_types, key_n, value));
}

void host_map_add_u64(void* h, ::dftu_map* m, const std::int64_t* key,
                      std::uint64_t v) {
    static_cast<PluginFold*>(h)->map_add_u64(reinterpret_cast<MapAccum*>(m),
                                             key, v);
}

void host_map_add_f64(void* h, ::dftu_map* m, const std::int64_t* key,
                      double v) {
    static_cast<PluginFold*>(h)->map_add_f64(reinterpret_cast<MapAccum*>(m),
                                             key, v);
}

::dftu_map* host_map_new_product(void* h, const char* name,
                                 const dftu_type* key_types,
                                 std::uint32_t key_n,
                                 const dftu_monoid_kind* values,
                                 std::uint32_t value_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_product(name, key_types, key_n,
                                                     values, value_n));
}

void host_map_add_u64_at(void* h, ::dftu_map* m, const std::int64_t* key,
                         std::uint32_t comp, std::uint64_t v) {
    static_cast<PluginFold*>(h)->map_add_u64_at(reinterpret_cast<MapAccum*>(m),
                                                key, comp, v);
}

void host_map_add_f64_at(void* h, ::dftu_map* m, const std::int64_t* key,
                         std::uint32_t comp, double v) {
    static_cast<PluginFold*>(h)->map_add_f64_at(reinterpret_cast<MapAccum*>(m),
                                                key, comp, v);
}

void host_map_add_ordered_at(void* h, ::dftu_map* m, const std::int64_t* key,
                             std::uint32_t comp, std::int64_t order_key,
                             std::uint64_t element) {
    static_cast<PluginFold*>(h)->map_add_ordered_at(
        reinterpret_cast<MapAccum*>(m), key, comp, order_key, element);
}

void host_map_set_ordered(void* h, ::dftu_map* m, int ordered) {
    static_cast<PluginFold*>(h)->map_set_ordered(reinterpret_cast<MapAccum*>(m),
                                                 ordered);
}

::dftu_map* host_map_new_nested(void* h, const char* name,
                                const dftu_type* outer_key_types,
                                std::uint32_t outer_key_n,
                                const dftu_type* inner_key_types,
                                std::uint32_t inner_key_n,
                                const dftu_monoid_kind* values,
                                std::uint32_t value_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_nested(
            name, outer_key_types, outer_key_n, inner_key_types, inner_key_n,
            values, value_n));
}

void host_map_add_nested_u64(void* h, ::dftu_map* m,
                             const std::int64_t* outer_key,
                             const std::int64_t* inner_key, std::uint32_t comp,
                             std::uint64_t v) {
    static_cast<PluginFold*>(h)->map_add_nested_u64(
        reinterpret_cast<MapAccum*>(m), outer_key, inner_key, comp, v);
}

void host_map_add_nested_f64(void* h, ::dftu_map* m,
                             const std::int64_t* outer_key,
                             const std::int64_t* inner_key, std::uint32_t comp,
                             double v) {
    static_cast<PluginFold*>(h)->map_add_nested_f64(
        reinterpret_cast<MapAccum*>(m), outer_key, inner_key, comp, v);
}

void host_map_add_argby_at(void* h, ::dftu_map* m, const std::int64_t* key,
                           std::uint32_t comp, double by,
                           std::int64_t payload) {
    static_cast<PluginFold*>(h)->map_add_argby_at(
        reinterpret_cast<MapAccum*>(m), key, comp, by, payload);
}

void host_map_add_topk_at(void* h, ::dftu_map* m, const std::int64_t* key,
                          std::uint32_t comp, std::uint32_t k, double by,
                          std::int64_t payload) {
    static_cast<PluginFold*>(h)->map_add_topk_at(reinterpret_cast<MapAccum*>(m),
                                                 key, comp, k, by, payload);
}

void host_map_add_approx_topk_at(void* h, ::dftu_map* m,
                                 const std::int64_t* key, std::uint32_t comp,
                                 std::uint32_t k, std::int64_t value) {
    static_cast<PluginFold*>(h)->map_add_approx_topk_at(
        reinterpret_cast<MapAccum*>(m), key, comp, k, value);
}

void host_map_add_sample_at(void* h, ::dftu_map* m, const std::int64_t* key,
                            std::uint32_t comp, std::uint32_t k,
                            std::int64_t item) {
    static_cast<PluginFold*>(h)->map_add_sample_at(
        reinterpret_cast<MapAccum*>(m), key, comp, k, item);
}

::dftu_map* host_map_new_argrow(void* h, const char* name,
                                const dftu_type* key_types, std::uint32_t key_n,
                                int is_max, const dftu_type* payload_types,
                                std::uint32_t payload_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_argrow(
            name, key_types, key_n, is_max, payload_types, payload_n));
}

void host_map_add_argrow(void* h, ::dftu_map* m, const std::int64_t* key,
                         double by, const std::int64_t* payload,
                         std::uint32_t payload_n) {
    static_cast<PluginFold*>(h)->map_add_argrow(reinterpret_cast<MapAccum*>(m),
                                                key, by, payload, payload_n);
}

void host_map_declare_join(void* h, const char* out_name, const char* left_name,
                           const char* right_name, dftu_join_type type) {
    static_cast<PluginFold*>(h)->declare_join(out_name, left_name, right_name,
                                              type);
}

void host_map_add_xy_at(void* h, ::dftu_map* m, const std::int64_t* key,
                        std::uint32_t comp, double x, double y) {
    static_cast<PluginFold*>(h)->map_add_xy_at(reinterpret_cast<MapAccum*>(m),
                                               key, comp, x, y);
}

::dftu_map* host_map_new_sketch(void* h, const char* name,
                                const dftu_type* key_types, std::uint32_t key_n,
                                const double* qs, std::uint32_t nq) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_sketch(name, key_types, key_n, qs,
                                                    nq));
}

::dftu_map* host_map_new_fused(void* h, const char* name,
                               const dftu_type* key_types, std::uint32_t key_n,
                               const char* const* out_names,
                               const dftu_monoid_kind* values,
                               std::uint32_t value_n) {
    return reinterpret_cast<::dftu_map*>(
        static_cast<PluginFold*>(h)->map_get_fused(name, key_types, key_n,
                                                   out_names, values, value_n));
}

void host_map_add_row(void* h, ::dftu_map* m, const std::int64_t* key,
                      const ::dftu_row_val* vals, std::uint32_t n) {
    static_cast<PluginFold*>(h)->map_add_row(reinterpret_cast<MapAccum*>(m),
                                             key, vals, n);
}

::dftu_agg* host_agg_new(void* h, const char* name,
                         const char* const* key_names, std::uint32_t key_n,
                         const ::dftu_agg_col* specs, std::uint32_t spec_n) {
    return static_cast<PluginFold*>(h)->agg_new(name, key_names, key_n, specs,
                                                spec_n);
}

void host_agg_accumulate(void* h, ::dftu_agg* a, const ::dftu_dataframe* df) {
    static_cast<PluginFold*>(h)->agg_accumulate(a, df);
}

const dftu_ext_agg g_agg = {host_agg_new, host_agg_accumulate};

const dftu_ext_map g_map = {host_map_new,
                            host_map_add_u64,
                            host_map_add_f64,
                            host_map_new_product,
                            host_map_add_u64_at,
                            host_map_add_f64_at,
                            host_map_add_ordered_at,
                            host_map_set_ordered,
                            host_map_new_nested,
                            host_map_add_nested_u64,
                            host_map_add_nested_f64,
                            host_map_add_argby_at,
                            host_map_add_topk_at,
                            host_map_add_approx_topk_at,
                            host_map_add_sample_at,
                            host_map_new_argrow,
                            host_map_add_argrow,
                            host_map_declare_join,
                            host_map_add_xy_at,
                            host_map_new_sketch,
                            host_map_new_fused,
                            host_map_add_row};

const dftu_ext_coro g_coro = {host_spawn, host_when_all, host_when_any,
                              host_then,  host_drive,    host_run_blocking};

const dftu_ext_query g_query = {host_query_compile, host_query_matches};

const dftu_ext_compose g_compose = {
    host_compose_make,     host_compose_then, host_compose_when_all,
    host_compose_when_any, host_compose_run,  host_compose_util_op,
    host_compose_free_op};

const dftu_ext_util g_util = {host_find_by_id,       host_run_stream,
                              host_util_run_async,   host_util_run_stream_async,
                              host_util_stream_open, host_util_stream_next,
                              host_util_stream_close};

// Stateless ext tables share one instance across all folds (each fn takes h).
const void* host_get_extension(void*, const char* ext_id) {
    if (!ext_id) return nullptr;
    if (std::strcmp(ext_id, DFTU_EXT_IO) == 0) return &g_io;
    if (std::strcmp(ext_id, DFTU_EXT_CORO) == 0) return &g_coro;
    if (std::strcmp(ext_id, DFTU_EXT_QUERY) == 0) return &g_query;
    if (std::strcmp(ext_id, DFTU_EXT_COMPOSE) == 0) return &g_compose;
    if (std::strcmp(ext_id, DFTU_EXT_UTIL) == 0) return &g_util;
    if (std::strcmp(ext_id, DFTU_EXT_WRITER) == 0) return &g_writer;
    if (std::strcmp(ext_id, DFTU_EXT_SKETCH) == 0) return &g_sketch;
    if (std::strcmp(ext_id, DFTU_EXT_ARROW) == 0) return &g_arrow;
    if (std::strcmp(ext_id, DFTU_EXT_TRACE) == 0) return &g_trace;
    if (std::strcmp(ext_id, DFTU_EXT_PORTS) == 0) return &g_ports;
    if (std::strcmp(ext_id, DFTU_EXT_HANDLES) == 0) return &g_handles;
    if (std::strcmp(ext_id, DFTU_EXT_RESULT) == 0) return &g_result;
    if (std::strcmp(ext_id, DFTU_EXT_MAP) == 0) return &g_map;
    if (std::strcmp(ext_id, DFTU_EXT_AGG) == 0) return &g_agg;
    return nullptr;
}

}  // namespace

::dftu_task* PluginFold::emplace_task(coro::CoroTask<void>&& task) {
    try {
        task_arena_.push_back(
            std::make_unique<coro::CoroTask<void>>(std::move(task)));
        return reinterpret_cast<::dftu_task*>(task_arena_.back().get());
    } catch (...) {
        return nullptr;
    }
}

namespace {
// Evaluate a compose graph over `in`, writing `out`; returns the leaf rc, with
// the first non-zero short-circuiting a pipe. Intermediate/child buffers are
// sized by the ops' declared value sizes.
coro::CoroTask<int> eval_compose(PluginFold* self, ComposeOp* op,
                                 const void* in, void* out) {
    using Kind = ComposeOp::Kind;
    if (op->kind == Kind::LEAF) {
        int rc = 0;
        ::dftu_task* t = op->fn ? op->fn(op->state, in, out, &rc) : nullptr;
        if (t) co_await *as_task(t);
        co_return rc;
    }
    if (op->kind == Kind::THEN) {
        ComposeOp* a = op->children[0];
        ComposeOp* b = op->children[1];
        std::vector<char> mid(a->out_size);
        int rc = co_await eval_compose(self, a, in, mid.data());
        if (rc != 0) co_return rc;
        co_return co_await eval_compose(self, b, mid.data(), out);
    }
    if (op->kind == Kind::WHEN_ALL) {
        // Each child writes its own disjoint slice of the concatenated output.
        std::vector<coro::CoroTask<int>> kids;
        kids.reserve(op->children.size());
        std::size_t off = 0;
        for (ComposeOp* c : op->children) {
            char* slot = static_cast<char*>(out) + off;
            kids.push_back(eval_compose(self, c, in, slot));
            off += c->out_size;
        }
        std::vector<int> rcs = co_await coro::when_all(std::move(kids));
        for (int r : rcs)
            if (r != 0) co_return r;
        co_return 0;
    }
    // WHEN_ANY: each child writes its own buffer; copy the winner into out.
    std::vector<std::vector<char>> bufs(op->children.size());
    std::vector<coro::CoroTask<int>> kids;
    kids.reserve(op->children.size());
    for (std::size_t i = 0; i < op->children.size(); ++i) {
        bufs[i].resize(op->children[i]->out_size);
        kids.push_back(eval_compose(self, op->children[i], in, bufs[i].data()));
    }
    auto winner = co_await coro::when_any(std::move(kids));
    if (op->out_size) std::memcpy(out, bufs[winner.index].data(), op->out_size);
    co_return winner.result;
}

// A compose leaf wrapping a registered host utility. Runs it through the async
// util path so it co_awaits cooperatively on the executor rather than blocking
// a worker like the synchronous run() would.
struct UtilLeafState {
    PluginFold* self;
    std::uint32_t util_id;
};
::dftu_task* util_leaf_thunk(void* state, const void* in, void* out, int* rc) {
    auto* s = static_cast<UtilLeafState*>(state);
    return host_util_run_async(s->self, s->util_id, in, out, rc);
}
void util_leaf_free(void* state) { delete static_cast<UtilLeafState*>(state); }
}  // namespace

::dftu_op* PluginFold::compose_make(dftu_op_fn fn, void* state,
                                    void (*free_state)(void*), dftu_type in_ty,
                                    std::uint32_t in_size, dftu_type out_ty,
                                    std::uint32_t out_size) {
    try {
        compose_ops_.emplace_back();
        ComposeOp& op = compose_ops_.back();
        op.kind = ComposeOp::Kind::LEAF;
        op.fn = fn;
        op.state = state;
        op.free_state = free_state;
        op.in_ty = in_ty;
        op.out_ty = out_ty;
        op.in_size = in_size;
        op.out_size = out_size;
        return reinterpret_cast<::dftu_op*>(&op);
    } catch (...) {
        if (free_state) free_state(state);  // never leak the plugin's state
        return nullptr;
    }
}

::dftu_op* PluginFold::compose_then(::dftu_op* a, ::dftu_op* b) {
    if (!a || !b) return nullptr;
    ComposeOp* oa = reinterpret_cast<ComposeOp*>(a);
    ComposeOp* ob = reinterpret_cast<ComposeOp*>(b);
    if (oa->out_ty != ob->in_ty || oa->out_size != ob->in_size) {
        // Return null per the C ABI, but log so a mistyped pipe is not silent.
        DFTRACER_UTILS_LOG_WARN(
            "compose then: cannot pipe, value mismatch (a out ty=%d/%uB -> b "
            "in "
            "ty=%d/%uB)",
            static_cast<int>(oa->out_ty), oa->out_size,
            static_cast<int>(ob->in_ty), ob->in_size);
        return nullptr;
    }
    try {
        compose_ops_.emplace_back();
        ComposeOp& op = compose_ops_.back();
        op.kind = ComposeOp::Kind::THEN;
        op.in_ty = oa->in_ty;
        op.out_ty = ob->out_ty;
        op.in_size = oa->in_size;
        op.out_size = ob->out_size;
        op.children = {oa, ob};
        return reinterpret_cast<::dftu_op*>(&op);
    } catch (...) {
        return nullptr;
    }
}

::dftu_op* PluginFold::compose_when_all(::dftu_op* const* ops,
                                        std::uint32_t n) {
    if (!ops || n == 0) return nullptr;
    std::vector<ComposeOp*> kids;
    kids.reserve(n);
    ComposeOp* first = reinterpret_cast<ComposeOp*>(ops[0]);
    std::uint32_t out_total = 0;
    for (std::uint32_t i = 0; i < n; ++i) {
        ComposeOp* c = reinterpret_cast<ComposeOp*>(ops[i]);
        // All fan the same input; the concatenated output is an opaque blob.
        if (!c || c->in_ty != first->in_ty || c->in_size != first->in_size) {
            DFTRACER_UTILS_LOG_WARN(
                "compose when_all: op %u does not share the input type/size "
                "(want ty=%d/%uB)",
                i, static_cast<int>(first->in_ty), first->in_size);
            return nullptr;
        }
        kids.push_back(c);
        out_total += c->out_size;
    }
    try {
        compose_ops_.emplace_back();
        ComposeOp& op = compose_ops_.back();
        op.kind = ComposeOp::Kind::WHEN_ALL;
        op.in_ty = first->in_ty;
        op.out_ty = DFTU_T_BYTES;
        op.in_size = first->in_size;
        op.out_size = out_total;
        op.children = std::move(kids);
        return reinterpret_cast<::dftu_op*>(&op);
    } catch (...) {
        return nullptr;
    }
}

::dftu_op* PluginFold::compose_when_any(::dftu_op* const* ops,
                                        std::uint32_t n) {
    if (!ops || n == 0) return nullptr;
    ComposeOp* first = reinterpret_cast<ComposeOp*>(ops[0]);
    std::vector<ComposeOp*> kids;
    kids.reserve(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        ComposeOp* c = reinterpret_cast<ComposeOp*>(ops[i]);
        if (!c || c->in_ty != first->in_ty || c->in_size != first->in_size ||
            c->out_ty != first->out_ty || c->out_size != first->out_size) {
            DFTRACER_UTILS_LOG_WARN(
                "compose when_any: op %u does not share the input+output "
                "type/size of the racers",
                i);
            return nullptr;  // racers must share one input and output type/size
        }
        kids.push_back(c);
    }
    try {
        compose_ops_.emplace_back();
        ComposeOp& op = compose_ops_.back();
        op.kind = ComposeOp::Kind::WHEN_ANY;
        op.in_ty = first->in_ty;
        op.out_ty = first->out_ty;
        op.in_size = first->in_size;
        op.out_size = first->out_size;
        op.children = std::move(kids);
        return reinterpret_cast<::dftu_op*>(&op);
    } catch (...) {
        return nullptr;
    }
}

::dftu_task* PluginFold::compose_run(::dftu_op* op, const void* in, void* out,
                                     int* rc) {
    if (!op) {
        if (rc) *rc = -1;
        return nullptr;
    }
    ComposeOp* graph = reinterpret_cast<ComposeOp*>(op);
    return emplace_task([](PluginFold* self, ComposeOp* g, const void* i,
                           void* o, int* r) -> coro::CoroTask<void> {
        int res = co_await eval_compose(self, g, i, o);
        if (r) *r = res;
    }(this, graph, in, out, rc));
}

::dftu_op* PluginFold::compose_util_op(std::uint32_t util_id) {
    const dftu_utility* u = registry_find(util_id);
    std::uint32_t in_sz = 0, out_sz = 0;
    if (!u || !u->run || !registry_run_sizes(util_id, &in_sz, &out_sz))
        return nullptr;
    // compose_make frees the state via util_leaf_free on its own failure path.
    auto* st = new UtilLeafState{this, util_id};
    return compose_make(&util_leaf_thunk, st, &util_leaf_free, u->in_tag, in_sz,
                        u->out_tag, out_sz);
}

::dftu_writer* PluginFold::create_writer(const char* path,
                                         std::uint32_t num_workers, int gzip) {
    try {
        auto pw = std::make_unique<PluginWriter>();
        pw->writer = parallel::make_sharded_writer();
        pw->path = path ? path : "";
        pw->num_workers = num_workers ? num_workers : 1;
        pw->gzip = gzip != 0;
        writers_.push_back(std::move(pw));
        return reinterpret_cast<::dftu_writer*>(writers_.back().get());
    } catch (...) {
        return nullptr;
    }
}

::dftu_trace_writer* PluginFold::create_trace_writer(const char* path) {
    if (!path) return nullptr;
    try {
        auto tw = std::make_unique<PluginTraceWriter>();
        tw->path = path;
        trace_writers_.push_back(std::move(tw));
        return reinterpret_cast<::dftu_trace_writer*>(
            trace_writers_.back().get());
    } catch (...) {
        return nullptr;
    }
}

// Compress the buffered NDJSON into whole-line gzip members and write a
// re-indexable .pfw.gz. The index is built lazily on first read, not here.
int PluginFold::close_trace_writer(::dftu_trace_writer* w) {
    auto* tw = reinterpret_cast<PluginTraceWriter*>(w);
    if (!tw) return -1;
    namespace pfw = utilities::fileio::parallel;
    namespace cmp = utilities::fileio::compress;
    int result = -1;
    try {
        constexpr std::size_t DEFAULT_FLUSH =
            constants::indexer::DEFAULT_CHECKPOINT_SIZE;
        constexpr std::size_t HEADROOM = 1 * 1024 * 1024;
        auto cw = pfw::make_writer_for_path(
            {tw->path, 1, DEFAULT_FLUSH, HEADROOM, /*gzip=*/true});
        const std::size_t member_size = cw.sizing.flush_threshold;
        auto writer = std::move(cw.writer);
        const std::string& buf = tw->buffer;
        bool ok = true;
        dftracer::utils::default_runtime().run_blocking(
            "dft-plugin-trace-write",
            [&](CoroScope& scope) -> coro::CoroTask<void> {
                if (co_await writer->open(tw->path, 1, true, &scope) != 0) {
                    ok = false;
                    co_return;
                }
                cmp::GzipMemberCompressor comp(6);
                std::vector<std::uint8_t> scratch;
                auto emit = [&](std::size_t begin,
                                std::size_t len) -> coro::CoroTask<bool> {
                    if (!comp.compress_member_into(scratch, buf.data() + begin,
                                                   len))
                        co_return false;
                    co_return co_await writer->write_chunk(
                        0,
                        ByteView(reinterpret_cast<const char*>(scratch.data()),
                                 scratch.size())) == 0;
                };
                std::size_t pos = 0;
                while (buf.size() - pos >= member_size) {
                    std::size_t cut = pos + member_size;
                    while (cut < buf.size() && buf[cut] != '\n') ++cut;
                    if (cut >= buf.size()) break;
                    ++cut;
                    if (!co_await emit(pos, cut - pos)) {
                        ok = false;
                        co_return;
                    }
                    pos = cut;
                }
                if (pos < buf.size() && !co_await emit(pos, buf.size() - pos))
                    ok = false;
                if (co_await writer->close() != 0) ok = false;
                if (ok && cw.layout.layout == pfw::FileLayout::SHARDED) {
                    auto shards = writer->output_paths();
                    if (co_await pfw::merge_shards(tw->path, shards) != 0)
                        ok = false;
                }
            });
        result = ok ? 0 : -1;
    } catch (...) {
        result = -1;
    }
    for (auto it = trace_writers_.begin(); it != trace_writers_.end(); ++it) {
        if (it->get() == tw) {
            trace_writers_.erase(it);
            break;
        }
    }
    return result;
}

PluginFold::PluginFold(const dftu_plugin* plugin,
                       dftracer::utils::StringIntern& intern,
                       SharedResultRegistry* results,
                       NamedResultRegistry* named_results,
                       std::string plugin_name)
    : plugin_(plugin),
      plugin_name_(std::move(plugin_name)),
      intern_(&intern),
      slice_(plugin->make_slice(plugin->self)),
      results_(results),
      named_results_(named_results) {
    host_.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    host_.h = this;
    host_.get_extension = host_get_extension;
    host_.resolve = host_resolve;
    host_.intern = host_intern;
    host_.log = host_log;

    // Spill is opt-in via env; unset keeps map_spill_enabled_ false. slice()
    // re-runs this constructor, so each worker inherits the same config.
    MapSpillEnv sp = read_map_spill_env();
    if (sp.enabled) {
        map_spill_enabled_ = true;
        map_spill_share_ = sp.share;
        map_spill_dir_root_ = sp.dir;
        map_part_bits_ = MAP_SPILL_PART_BITS;
    }
    if (sp.stream) {
        map_stream_enabled_ = true;
        map_stream_chunk_rows_ = MAP_STREAM_CHUNK_ROWS;
    }

    // An invalid filter is logged and left unset rather than aborting.
    const char* q =
        plugin_->plan_query ? plugin_->plan_query(plugin_->self) : nullptr;
    if (q && *q) {
        auto r = Query::from_string(q);
        if (r)
            query_ = std::move(*r);
        else
            DFTRACER_UTILS_LOG_ERROR("Plugin query '%s' is invalid: %s", q,
                                     r.error().message.c_str());
    }
}

PluginFold::~PluginFold() {
    // An exceptional unwind that skipped the post-materialize cleanup still
    // removes every spill dir this fold owns.
    remove_spill_dirs();
    if (slice_) plugin_->destroy_slice(slice_);
}

void PluginFold::set_map_spill(std::size_t share_bytes,
                               const std::string& dir) {
    map_spill_enabled_ = share_bytes > 0;
    map_spill_share_ = share_bytes;
    if (!dir.empty()) map_spill_dir_root_ = dir;
    if (map_spill_enabled_) map_part_bits_ = MAP_SPILL_PART_BITS;
}

void PluginFold::set_map_stream(bool enabled, std::size_t chunk_rows) {
    map_stream_enabled_ = enabled;
    map_stream_chunk_rows_ = chunk_rows ? chunk_rows : MAP_STREAM_CHUNK_ROWS;
}

const std::string& PluginFold::ensure_spill_dir() {
    if (!map_spill_cur_dir_.empty()) return map_spill_cur_dir_;
    static std::atomic<std::uint64_t> seq{0};
    const std::string root = map_spill_dir_root_.empty()
                                 ? fs::temp_directory_path().string()
                                 : map_spill_dir_root_;
    std::string dir = root + "/dftplugmap_" + std::to_string(seq.fetch_add(1)) +
                      "_" +
                      std::to_string(reinterpret_cast<std::uintptr_t>(this));
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec)
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::IO,
            "plugin map spill: cannot create spill dir " + dir);
    map_spill_cur_dir_ = std::move(dir);
    map_spill_dirs_.push_back(map_spill_cur_dir_);
    return map_spill_cur_dir_;
}

void PluginFold::remove_spill_dirs() {
    std::error_code ec;
    for (const auto& d : map_spill_dirs_) fs::remove_all(d, ec);
    map_spill_dirs_.clear();
    map_spill_cur_dir_.clear();
}

void PluginFold::account_add(MapAccum& m, const std::vector<std::int64_t>& key,
                             std::uint32_t comp, bool inserted) {
    if (!map_spill_enabled_) return;
    const std::uint32_t p = m.partition_of(key);
    if (p >= m.part_bytes.size()) return;
    std::size_t delta = 0;
    if (inserted)
        delta += MapAccum::KEY_OVERHEAD_BYTES + sizeof(std::int64_t) * m.key_n +
                 m.value_base_bytes;
    if (comp < m.value_kinds.size() && monoid_is_variable(m.value_kinds[comp]))
        delta += 16;
    m.part_bytes[p] += delta;
    m.footprint += delta;
}

void PluginFold::spill_partition(MapAccum& m, std::uint32_t p) {
    if (p >= m.parts.size() || m.parts[p].empty()) return;
    const std::string& dir = ensure_spill_dir();
    std::string path = dir + "/" + std::to_string(map_spill_run_seq_++);
    write_run(m, m.parts[p], path);
    m.runs[p].push_back(std::move(path));
    m.footprint -= std::min(m.footprint, m.part_bytes[p]);
    m.part_bytes[p] = 0;
    m.parts[p].clear();
    ++map_spill_count_;
}

void PluginFold::note_and_maybe_spill(MapAccum& m, bool inserted) {
    if (!map_spill_enabled_ || map_spill_failed_) return;
    if (!inserted && ++m.adds_since_check < MAP_SPILL_CHECK_STRIDE) return;
    m.adds_since_check = 0;
    if (m.footprint <= map_spill_share_) return;
    // Hysteresis: drain the largest partitions down to a low-water mark (~75%
    // of the share) so one more add does not immediately re-trigger a spill.
    const std::size_t low = map_spill_share_ - map_spill_share_ / 4;
    try {
        while (m.footprint > low) {
            std::uint32_t best = m.partitions();
            std::size_t best_bytes = 0;
            for (std::uint32_t p = 0; p < m.parts.size(); ++p)
                if (!m.parts[p].empty() && m.part_bytes[p] >= best_bytes) {
                    best_bytes = m.part_bytes[p];
                    best = p;
                }
            if (best == m.partitions()) break;
            spill_partition(m, best);
        }
    } catch (const std::exception& e) {
        // Fail the run loudly rather than emit a partial result; materialize
        // refuses to emit after this.
        map_spill_failed_ = true;
        DFTRACER_UTILS_LOG_ERROR("Plugin map '%s' spill failed: %s",
                                 m.name.c_str(), e.what());
    }
}

void PluginFold::reload_runs(MapAccum& m) {
    if (!m.has_runs()) return;
    for (std::uint32_t p = 0; p < m.runs.size(); ++p) {
        for (const std::string& path : m.runs[p]) read_run_into(m, path);
        m.runs[p].clear();
    }
}

::dftu_query* PluginFold::compile_query(const char* src, std::uint32_t len) {
    if (!src) return nullptr;
    auto r = Query::from_string(std::string_view{src, len});
    if (!r) return nullptr;
    try {
        compiled_queries_.push_back(std::move(*r));
    } catch (...) {
        return nullptr;
    }
    return reinterpret_cast<::dftu_query*>(&compiled_queries_.back());
}

int PluginFold::match_query(const Query& q, const dftu_event& e) {
    match_qmap_.clear();
    for (std::string_view f : q.fields()) {
        if (f == "cat") {
            if (e.cat != DFTU_STR_NONE)
                match_qmap_[f] = std::string(intern_->resolve(e.cat));
        } else if (f == "name") {
            if (e.name != DFTU_STR_NONE)
                match_qmap_[f] = std::string(intern_->resolve(e.name));
        } else if (f == "fhash") {
            if (e.fhash != DFTU_STR_NONE)
                match_qmap_[f] = std::string(intern_->resolve(e.fhash));
        } else if (f == "hhash") {
            if (e.hhash != DFTU_STR_NONE)
                match_qmap_[f] = std::string(intern_->resolve(e.hhash));
        } else if (f == "pid") {
            match_qmap_[f] = static_cast<double>(e.pid);
        } else if (f == "tid") {
            match_qmap_[f] = static_cast<double>(e.tid);
        } else if (f == "ts") {
            match_qmap_[f] = static_cast<double>(e.ts);
        } else if (f == "dur") {
            if (e.has_dur) match_qmap_[f] = static_cast<double>(e.dur);
        } else {
            std::string_view key = strip_args_prefix(f);
            for (std::uint32_t i = 0; i < e.arg_count; ++i) {
                const dftu_arg& a = e.args[i];
                if (a.key == DFTU_STR_NONE || intern_->resolve(a.key) != key)
                    continue;
                if (a.kind == DFTU_ARG_F64)
                    match_qmap_[f] = a.v.f64;
                else if (a.kind == DFTU_ARG_I64)
                    match_qmap_[f] = static_cast<double>(a.v.i64);
                else if (a.v.str != DFTU_STR_NONE)
                    match_qmap_[f] = std::string(intern_->resolve(a.v.str));
                break;
            }
        }
    }
    return q.evaluate(match_qmap_) ? 1 : 0;
}

// Field coverage must match match_query() so plan_query and query_matches
// agree.
bool PluginFold::passes_query(const FoldEvent& ev) {
    trace::views::detail::PodSource src(ev, *intern_);
    qmap_.clear();
    for (std::string_view f : query_->fields()) {
        if (f == "cat" || f == "name" || f == "fhash" || f == "hhash") {
            std::string s = src.value(f);
            if (!s.empty()) qmap_[f] = std::move(s);
        } else if (auto n = src.number(f)) {
            qmap_[f] = *n;
        } else {
            std::string s = src.value(f);
            if (!s.empty()) qmap_[f] = std::move(s);
        }
    }
    return query_->evaluate(qmap_);
}

void PluginFold::step(const FoldBatch& batch) {
    if (!slice_) return;

    // The previous step's tasks were awaited by the fuse worker before this
    // call, so freeing them now is safe.
    task_arena_.clear();

    if (plugin_->on_batch_columns) {
        step_columns(batch);
        return;
    }
    if (!plugin_->on_batch) return;  // malformed plugin sets neither seam

    const std::size_t count = batch.events.size();
    const bool with_args = needs_args();

    kept_scratch_.clear();
    kept_scratch_.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        if (!query_ || passes_query(batch.events[i]))
            kept_scratch_.push_back(i);
    }

    const std::size_t kept = kept_scratch_.size();
    event_scratch_.clear();
    event_scratch_.resize(kept);

    if (with_args) {
        // Fill arg_scratch_ fully before pointing events at it; growth would
        // otherwise invalidate the pointers.
        arg_scratch_.clear();
        std::vector<std::pair<std::size_t, std::size_t>> ranges(kept);
        for (std::size_t j = 0; j < kept; ++j) {
            const auto& fe = batch.events[kept_scratch_[j]];
            const std::size_t begin = arg_scratch_.size();
            for (const auto& [key_id, val] : fe.args) {
                dftu_arg a{};
                a.key = key_id;
                switch (val.index()) {
                    case 0:
                        a.kind = DFTU_ARG_F64;
                        a.v.f64 = std::get<double>(val);
                        break;
                    case 1:
                        a.kind = DFTU_ARG_I64;
                        a.v.i64 = std::get<std::int64_t>(val);
                        break;
                    default:
                        a.kind = DFTU_ARG_STR;
                        a.v.str = std::get<std::uint32_t>(val);
                        break;
                }
                arg_scratch_.push_back(a);
            }
            ranges[j] = {begin, fe.args.size()};
        }
        for (std::size_t j = 0; j < kept; ++j) {
            const auto& fe = batch.events[kept_scratch_[j]];
            dftu_event& ce = event_scratch_[j];
            ce.cat = fe.cat_id;
            ce.name = fe.name_id;
            ce.fhash = fe.fhash_id;
            ce.hhash = fe.hhash_id;
            ce.pid = fe.pid;
            ce.tid = fe.tid;
            ce.ts = fe.ts;
            ce.dur = fe.dur;
            ce.phase = map_phase(fe.phase);
            ce.has_dur = fe.has_dur ? 1 : 0;
            ce.arg_count = static_cast<std::uint32_t>(ranges[j].second);
            ce.args = ranges[j].second ? arg_scratch_.data() + ranges[j].first
                                       : nullptr;
        }
    } else {
        for (std::size_t j = 0; j < kept; ++j) {
            const auto& fe = batch.events[kept_scratch_[j]];
            dftu_event& ce = event_scratch_[j];
            ce.cat = fe.cat_id;
            ce.name = fe.name_id;
            ce.fhash = fe.fhash_id;
            ce.hhash = fe.hhash_id;
            ce.pid = fe.pid;
            ce.tid = fe.tid;
            ce.ts = fe.ts;
            ce.dur = fe.dur;
            ce.phase = map_phase(fe.phase);
            ce.has_dur = fe.has_dur ? 1 : 0;
            ce.arg_count = 0;
            ce.args = nullptr;
        }
    }

    cbatch_ = dftu_batch{event_scratch_.data(),
                         static_cast<std::uint32_t>(event_scratch_.size())};
    pending_ = plugin_->on_batch(slice_, &cbatch_, &host_);
}

// Vectorized-fold seam: materialize the (query-passing, non-metadata) events of
// this batch into a native DataFrame and hand it across the ABI as a
// dftu_dataframe, so the plugin runs SIMD column ops instead of a per-event
// loop. build_row_frame is the same columnar materializer NativeRowFold uses.
void PluginFold::step_columns(const FoldBatch& batch) {
    namespace views = dftracer::utils::trace::views::detail;
    col_scratch_.clear();
    for (const auto& fe : batch.events) {
        if (fe.phase == trace::RecordPhase::METADATA ||
            fe.phase == trace::RecordPhase::UNKNOWN)
            continue;
        if (query_ && !passes_query(fe)) continue;
        col_scratch_.push_back(fe);
    }
    if (col_scratch_.empty()) return;

    dataframe::DataFrame df =
        views::build_row_frame(col_scratch_, *intern_, {}, 1.0, nullptr);

    // dftu_dataframe_new takes ownership of the column handles, so pass shared
    // copies (a refcount bump, no data copy); df keeps its own.
    std::vector<dftu_series*> handles;
    handles.reserve(df.columns.size());
    std::vector<const char*> names;
    names.reserve(df.names.size());
    for (const auto& c : df.columns)
        handles.push_back(dftu_series_share(c.handle()));
    for (const auto& n : df.names) names.push_back(n.c_str());

    dftu_dataframe* cdf =
        dftu_dataframe_new(names.data(), handles.data(),
                           static_cast<std::int32_t>(handles.size()));
    // The seam is contractually synchronous (abi.h: must return NULL); a
    // returned task would reference the now-freed cdf and cannot be driven.
    ::dftu_task* t = plugin_->on_batch_columns(slice_, cdf, &host_);
    dftu_dataframe_free(cdf);
    if (t)
        DFTRACER_UTILS_LOG_ERROR(
            "[plugin:%s] on_batch_columns returned a non-null task; the seam "
            "is synchronous, dropping it",
            plugin_name().empty() ? "plugin" : plugin_name().c_str());
    pending_ = nullptr;
}

MonoidAccumulator* PluginFold::handle_get(const char* cap_id,
                                          dftu_monoid_kind kind) {
    if (!cap_id) return nullptr;
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(cap_id);
    auto it = handle_index_.find(key);
    if (it != handle_index_.end()) return &handles_[it->second];
    try {
        handles_.emplace_back(kind);
        handle_index_.emplace(key, handles_.size() - 1);
        return &handles_.back();
    } catch (...) {
        return nullptr;
    }
}

int PluginFold::handle_result(const char* cap_id,
                              dftu_monoid_value* out) const {
    if (!cap_id || !results_) return -1;
    auto it = results_->values.find(dftracer::utils::hash::fnv1a_hash(cap_id));
    if (it == results_->values.end()) return -1;
    if (out) *out = it->second;
    return 0;
}

void PluginFold::result_emit(const char* name, const void* data,
                             std::uint64_t len) {
    if (named_results_) named_results_->emit_blob(name, data, len);
}

int PluginFold::result_emit_arrow(const char* name, ::ArrowArray* a,
                                  ::ArrowSchema* s) {
    return named_results_ ? named_results_->emit_arrow(name, a, s) : -1;
}

int PluginFold::result_emit_frame(const char* name, ::dftu_dataframe* df) {
    if (!named_results_) return -1;
    named_results_->emit_frame(name, df);
    return 0;
}

MapAccum* PluginFold::map_get(const char* name, const dftu_type* key_types,
                              std::uint32_t key_n, dftu_monoid_kind value) {
    return map_get_product(name, key_types, key_n, &value, 1);
}

::dftu_agg* PluginFold::agg_new(const char* name, const char* const* key_names,
                                std::uint32_t key_n,
                                const ::dftu_agg_col* specs,
                                std::uint32_t spec_n) {
    if (!name || (key_n && !key_names) || spec_n == 0 || !specs) return nullptr;
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    auto it = agg_index_.find(key);
    if (it != agg_index_.end())
        return reinterpret_cast<::dftu_agg*>(aggs_[it->second].get());

    std::vector<std::string> value_names;
    auto column_index = [&](const char* col) -> std::int32_t {
        if (!col) return -1;
        for (std::size_t i = 0; i < value_names.size(); ++i)
            if (value_names[i] == col) return static_cast<std::int32_t>(i);
        value_names.emplace_back(col);
        return static_cast<std::int32_t>(value_names.size() - 1);
    };

    std::vector<dataframe::AggSpec> aspecs;
    aspecs.reserve(spec_n);
    for (std::uint32_t i = 0; i < spec_n; ++i) {
        if (specs[i].op < DFTU_AGG_COUNT || specs[i].op > DFTU_AGG_SET_UNION) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin agg '%s' aggregate %u has an out-of-range op code %d",
                name, i, specs[i].op);
            return nullptr;
        }
        if (!specs[i].out || !specs[i].out[0]) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin agg '%s' aggregate %u is missing an output name", name,
                i);
            return nullptr;
        }
        dataframe::AggSpec s;
        s.op = static_cast<dataframe::AggOp>(specs[i].op);
        s.value_col = column_index(specs[i].value);
        s.out = specs[i].out;
        s.param = specs[i].param;
        s.by_col = column_index(specs[i].by);
        aspecs.push_back(std::move(s));
    }

    try {
        auto acc = std::make_unique<AggAccum>();
        acc->name = name;
        acc->key_names.reserve(key_n);
        for (std::uint32_t i = 0; i < key_n; ++i)
            acc->key_names.emplace_back(key_names[i] ? key_names[i] : "");
        acc->value_names = std::move(value_names);
        acc->state = dataframe::agg_new(std::move(aspecs));
        AggAccum* raw = acc.get();
        aggs_.push_back(std::move(acc));
        agg_index_.emplace(key, aggs_.size() - 1);
        return reinterpret_cast<::dftu_agg*>(raw);
    } catch (...) {
        return nullptr;
    }
}

void PluginFold::agg_accumulate(::dftu_agg* a, const ::dftu_dataframe* df) {
    if (!a || !df) return;
    AggAccum& acc = *reinterpret_cast<AggAccum*>(a);
    if (!acc.state) return;

    // Series wraps and frees each shared handle; a missing column yields a null
    // handle, so skip the whole batch rather than accumulate a partial key.
    std::vector<dataframe::Series> owned;
    owned.reserve(acc.key_names.size() + acc.value_names.size());
    auto column = [&](const std::string& n) -> const dataframe::Series* {
        dftu_series* h = dftu_dataframe_column(df, n.c_str());
        if (!h) return nullptr;
        owned.emplace_back(h);
        return &owned.back();
    };

    std::vector<const dataframe::Series*> keys;
    keys.reserve(acc.key_names.size());
    for (const std::string& n : acc.key_names) {
        const dataframe::Series* c = column(n);
        if (!c) return;
        keys.push_back(c);
    }
    std::vector<const dataframe::Series*> values;
    values.reserve(acc.value_names.size());
    for (const std::string& n : acc.value_names) {
        const dataframe::Series* c = column(n);
        if (!c) return;
        values.push_back(c);
    }

    try {
        dataframe::agg_accumulate(*acc.state, keys, values);
    } catch (...) {
        DFTRACER_UTILS_LOG_ERROR("Plugin agg '%s' accumulate failed",
                                 acc.name.c_str());
    }
}

MapAccum* PluginFold::map_get_product(const char* name,
                                      const dftu_type* key_types,
                                      std::uint32_t key_n,
                                      const dftu_monoid_kind* values,
                                      std::uint32_t value_n) {
    if (!name || (key_n && !key_types) || value_n == 0 || !values)
        return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin map '%s' key component %u must be a fixed-width "
                "integer "
                "(I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < value_n; ++i) {
        // ARGMIN_ROW/ARGMAX_ROW are whole-value, created only via
        // map_new_argrow, never a product component.
        if (monoid_is_argrow(values[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin map '%s' value component %u is an arg-row monoid; "
                "create it via map_new_argrow, not as a product component",
                name, i);
            return nullptr;
        }
        if (!monoid_has_scalar(values[i]) && !monoid_is_set(values[i]) &&
            !monoid_is_list(values[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin map '%s' value component %u is not materializable "
                "(SKETCH)",
                name, i);
            return nullptr;
        }
    }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    auto it = map_index_.find(key);
    if (it != map_index_.end()) return &maps_[it->second];
    try {
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(values, values + value_n);
        for (std::uint32_t i = 0; i < value_n; ++i)
            m.value_base_bytes += MonoidAccumulator(values[i]).state_bytes();
        m.set_part_bits(map_part_bits_);
        maps_.push_back(std::move(m));
        map_index_.emplace(key, maps_.size() - 1);
        return &maps_.back();
    } catch (...) {
        return nullptr;
    }
}

MapAccum* PluginFold::map_get_nested(const char* name,
                                     const dftu_type* outer_key_types,
                                     std::uint32_t outer_key_n,
                                     const dftu_type* inner_key_types,
                                     std::uint32_t inner_key_n,
                                     const dftu_monoid_kind* values,
                                     std::uint32_t value_n) {
    if (!name || outer_key_n == 0 || inner_key_n == 0 || !outer_key_types ||
        !inner_key_types || value_n == 0 || !values)
        return nullptr;
    for (std::uint32_t i = 0; i < outer_key_n; ++i)
        if (!key_type_supported(outer_key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin nested map '%s' outer key component %u must be a "
                "fixed-width integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < inner_key_n; ++i)
        if (!key_type_supported(inner_key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin nested map '%s' inner key component %u must be a "
                "fixed-width integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < value_n; ++i)
        if (!monoid_has_scalar(values[i]) || monoid_is_set(values[i]) ||
            monoid_is_list(values[i]) || monoid_is_topk(values[i]) ||
            monoid_is_approx_topk(values[i]) || monoid_is_sample(values[i]) ||
            monoid_is_argrow(values[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin nested map '%s' inner value component %u must be a "
                "scalar or product monoid (collections, ordered lists, top-k, "
                "approx-top-k, sample, arg-row and SKETCH are not supported "
                "this phase)",
                name, i);
            return nullptr;
        }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    auto it = map_index_.find(key);
    if (it != map_index_.end()) return &maps_[it->second];
    try {
        MapAccum m;
        m.name = name;
        m.key_n = outer_key_n + inner_key_n;
        m.key_types.assign(outer_key_types, outer_key_types + outer_key_n);
        m.key_types.insert(m.key_types.end(), inner_key_types,
                           inner_key_types + inner_key_n);
        m.value_kinds.assign(values, values + value_n);
        for (std::uint32_t i = 0; i < value_n; ++i)
            m.value_base_bytes += MonoidAccumulator(values[i]).state_bytes();
        m.nested_inner_n = inner_key_n;
        m.inner_key_types.assign(inner_key_types,
                                 inner_key_types + inner_key_n);
        m.set_part_bits(map_part_bits_);
        maps_.push_back(std::move(m));
        map_index_.emplace(key, maps_.size() - 1);
        return &maps_.back();
    } catch (...) {
        return nullptr;
    }
}

void PluginFold::map_add_nested_u64(MapAccum* m, const std::int64_t* outer_key,
                                    const std::int64_t* inner_key,
                                    std::uint32_t comp, std::uint64_t v) {
    if (!m || m->nested_inner_n == 0 || !outer_key || !inner_key) return;
    try {
        const std::uint32_t outer_n = m->key_n - m->nested_inner_n;
        std::vector<std::int64_t> key;
        key.reserve(m->key_n);
        key.insert(key.end(), outer_key, outer_key + outer_n);
        key.insert(key.end(), inner_key, inner_key + m->nested_inner_n);
        map_add_u64_at(m, key.data(), comp, v);
    } catch (...) {
    }
}

void PluginFold::map_add_nested_f64(MapAccum* m, const std::int64_t* outer_key,
                                    const std::int64_t* inner_key,
                                    std::uint32_t comp, double v) {
    if (!m || m->nested_inner_n == 0 || !outer_key || !inner_key) return;
    try {
        const std::uint32_t outer_n = m->key_n - m->nested_inner_n;
        std::vector<std::int64_t> key;
        key.reserve(m->key_n);
        key.insert(key.end(), outer_key, outer_key + outer_n);
        key.insert(key.end(), inner_key, inner_key + m->nested_inner_n);
        map_add_f64_at(m, key.data(), comp, v);
    } catch (...) {
    }
}

void PluginFold::map_add_u64(MapAccum* m, const std::int64_t* key,
                             std::uint64_t v) {
    map_add_u64_at(m, key, 0, v);
}

void PluginFold::map_add_f64(MapAccum* m, const std::int64_t* key, double v) {
    map_add_f64_at(m, key, 0, v);
}

void PluginFold::map_add_u64_at(MapAccum* m, const std::int64_t* key,
                                std::uint32_t comp, std::uint64_t v) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_u64(v);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_f64_at(MapAccum* m, const std::int64_t* key,
                                std::uint32_t comp, double v) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_f64(v, 1.0);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_ordered_at(MapAccum* m, const std::int64_t* key,
                                    std::uint32_t comp, std::int64_t order_key,
                                    std::uint64_t element) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_ordered(
            order_key, static_cast<std::int64_t>(element));
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_argby_at(MapAccum* m, const std::int64_t* key,
                                  std::uint32_t comp, double by,
                                  std::int64_t payload) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_argby(by, payload);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_xy_at(MapAccum* m, const std::int64_t* key,
                               std::uint32_t comp, double x, double y) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[comp].add_xy(x, y);
        account_add(*m, k, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

MapAccum* PluginFold::map_get_argrow(const char* name,
                                     const dftu_type* key_types,
                                     std::uint32_t key_n, int is_max,
                                     const dftu_type* payload_types,
                                     std::uint32_t payload_n) {
    if (!name || (key_n && !key_types) || payload_n == 0 || !payload_types)
        return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin arg-row map '%s' key component %u must be a "
                "fixed-width "
                "integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    // The payload row reuses the key encoding, so it accepts the same supported
    // component types.
    for (std::uint32_t i = 0; i < payload_n; ++i)
        if (!key_type_supported(payload_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin arg-row map '%s' payload component %u must be a "
                "fixed-width integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    auto it = map_index_.find(key);
    if (it != map_index_.end()) return &maps_[it->second];
    try {
        const dftu_monoid_kind vk =
            is_max ? DFTU_MONOID_ARGMAX_ROW : DFTU_MONOID_ARGMIN_ROW;
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(1, vk);
        m.payload_types.assign(payload_types, payload_types + payload_n);
        m.value_base_bytes += MonoidAccumulator(vk).state_bytes();
        m.set_part_bits(map_part_bits_);
        maps_.push_back(std::move(m));
        map_index_.emplace(key, maps_.size() - 1);
        return &maps_.back();
    } catch (...) {
        return nullptr;
    }
}

MapAccum* PluginFold::map_get_sketch(const char* name,
                                     const dftu_type* key_types,
                                     std::uint32_t key_n, const double* qs,
                                     std::uint32_t nq) {
    if (!name || (key_n && !key_types) || nq == 0 || !qs) return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin sketch map '%s' key component %u must be a fixed-width "
                "integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < nq; ++i)
        if (!(qs[i] >= 0.0 && qs[i] <= 1.0)) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin sketch map '%s' quantile %u = %f is outside [0, 1]",
                name, i, qs[i]);
            return nullptr;
        }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    auto it = map_index_.find(key);
    if (it != map_index_.end()) return &maps_[it->second];
    try {
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(1, DFTU_MONOID_SKETCH);
        m.quantile_qs.assign(qs, qs + nq);
        m.value_base_bytes +=
            MonoidAccumulator(DFTU_MONOID_SKETCH).state_bytes();
        m.set_part_bits(map_part_bits_);
        maps_.push_back(std::move(m));
        map_index_.emplace(key, maps_.size() - 1);
        return &maps_.back();
    } catch (...) {
        return nullptr;
    }
}

MapAccum* PluginFold::map_get_fused(const char* name,
                                    const dftu_type* key_types,
                                    std::uint32_t key_n,
                                    const char* const* out_names,
                                    const dftu_monoid_kind* values,
                                    std::uint32_t value_n) {
    if (!name || (key_n && !key_types) || value_n == 0 || !values || !out_names)
        return nullptr;
    for (std::uint32_t i = 0; i < key_n; ++i)
        if (!key_type_supported(key_types[i])) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin fused map '%s' key component %u must be a fixed-width "
                "integer (I8..I64, U8..U64), STR, or BYTES",
                name, i);
            return nullptr;
        }
    for (std::uint32_t i = 0; i < value_n; ++i) {
        if (!monoid_is_fused_eligible(values[i]) || !out_names[i]) {
            DFTRACER_UTILS_LOG_ERROR(
                "Plugin fused map '%s' component %u is not a fused-eligible "
                "scalar monoid",
                name, i);
            return nullptr;
        }
    }
    std::uint64_t key = dftracer::utils::hash::fnv1a_hash(name);
    auto it = map_index_.find(key);
    if (it != map_index_.end()) return &maps_[it->second];
    try {
        MapAccum m;
        m.name = name;
        m.key_n = key_n;
        m.key_types.assign(key_types, key_types + key_n);
        m.value_kinds.assign(values, values + value_n);
        for (std::uint32_t i = 0; i < value_n; ++i) {
            m.value_base_bytes += MonoidAccumulator(values[i]).state_bytes();
            m.fused_out_names.emplace_back(out_names[i]);
        }
        m.set_part_bits(map_part_bits_);
        maps_.push_back(std::move(m));
        map_index_.emplace(key, maps_.size() - 1);
        return &maps_.back();
    } catch (...) {
        return nullptr;
    }
}

void PluginFold::map_add_row(MapAccum* m, const std::int64_t* key,
                             const ::dftu_row_val* vals, std::uint32_t n) {
    if (!m || (m->key_n && !key) || (n && !vals)) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        // One lookup for the whole row; components are updated in place, so no
        // spill occurs until after the row is applied and the reference is
        // done.
        std::vector<MonoidAccumulator>& into = m->touch(k, inserted);
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t comp = vals[i].comp;
            if (comp >= into.size()) continue;
            if (vals[i].is_f64)
                into[comp].add_f64(vals[i].value.f, 1.0);
            else
                into[comp].add_u64(vals[i].value.u);
            account_add(*m, k, comp, inserted && i == 0);
        }
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_argrow(MapAccum* m, const std::int64_t* key, double by,
                                const std::int64_t* payload,
                                std::uint32_t payload_n) {
    if (!m || (m->key_n && !key) || m->value_kinds.empty() ||
        (payload_n && !payload))
        return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> k(key, key + m->key_n);
        m->touch(k, inserted)[0].add_argrow(by, payload, payload_n);
        account_add(*m, k, 0, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_topk_at(MapAccum* m, const std::int64_t* key,
                                 std::uint32_t comp, std::uint32_t k, double by,
                                 std::int64_t payload) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> kk(key, key + m->key_n);
        m->touch(kk, inserted)[comp].add_topk(k, by, payload);
        account_add(*m, kk, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_approx_topk_at(MapAccum* m, const std::int64_t* key,
                                        std::uint32_t comp, std::uint32_t k,
                                        std::int64_t value) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> kk(key, key + m->key_n);
        m->touch(kk, inserted)[comp].add_approx_topk(k, value);
        account_add(*m, kk, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_add_sample_at(MapAccum* m, const std::int64_t* key,
                                   std::uint32_t comp, std::uint32_t k,
                                   std::int64_t item) {
    if (!m || (m->key_n && !key) || comp >= m->value_kinds.size()) return;
    bool inserted = false;
    try {
        std::vector<std::int64_t> kk(key, key + m->key_n);
        m->touch(kk, inserted)[comp].add_sample(k, item);
        account_add(*m, kk, comp, inserted);
    } catch (...) {
        return;
    }
    note_and_maybe_spill(*m, inserted);
}

void PluginFold::map_set_ordered(MapAccum* m, int ordered) {
    if (m) m->ordered = ordered != 0;
}

void PluginFold::declare_join(const char* out_name, const char* left_name,
                              const char* right_name, dftu_join_type type) {
    if (!out_name || !left_name || !right_name) return;
    for (const DeclaredJoin& j : joins_)
        if (j.out_name == out_name) return;
    joins_.push_back({out_name, left_name, right_name, type});
}

void PluginFold::merge(Fold& other) {
    if (!slice_) return;
    auto& o = static_cast<PluginFold&>(other);
    if (o.slice_) plugin_->merge(slice_, o.slice_);
    // Merge named handles by key; fuse folds each worker slice into the master.
    for (const auto& [key, idx] : o.handle_index_) {
        const MonoidAccumulator& src = o.handles_[idx];
        auto it = handle_index_.find(key);
        if (it == handle_index_.end()) {
            handles_.emplace_back(src.kind());
            handle_index_.emplace(key, handles_.size() - 1);
            handles_.back().merge(src);
        } else {
            handles_[it->second].merge(src);
        }
    }
    // Fold each named map's entries into this by key, merging a colliding key's
    // monoid.
    for (const auto& [key, idx] : o.map_index_) {
        MapAccum& src = o.maps_[idx];
        auto it = map_index_.find(key);
        MapAccum* dst;
        if (it == map_index_.end()) {
            MapAccum m;
            m.name = src.name;
            m.key_n = src.key_n;
            m.key_types = src.key_types;
            m.value_kinds = src.value_kinds;
            m.value_base_bytes = src.value_base_bytes;
            m.nested_inner_n = src.nested_inner_n;
            m.inner_key_types = src.inner_key_types;
            m.payload_types = src.payload_types;
            m.quantile_qs = src.quantile_qs;
            m.fused_out_names = src.fused_out_names;
            m.set_part_bits(src.part_bits);
            maps_.push_back(std::move(m));
            map_index_.emplace(key, maps_.size() - 1);
            dst = &maps_.back();
        } else {
            dst = &maps_[it->second];
        }
        dst->ordered = dst->ordered || src.ordered;
        // dst carries src's part_bits, so src partition p folds into dst
        // partition p.
        for (const MapAccum::EntriesMap& part : src.parts) {
            for (const auto& [k, mons] : part) {
                bool inserted = false;
                std::vector<MonoidAccumulator>& into = dst->touch(k, inserted);
                for (std::size_t c = 0; c < mons.size() && c < into.size(); ++c)
                    into[c].merge(mons[c]);
                if (map_spill_enabled_) {
                    const std::uint32_t p = dst->partition_of(k);
                    std::size_t delta =
                        inserted ? MapAccum::KEY_OVERHEAD_BYTES +
                                       sizeof(std::int64_t) * dst->key_n +
                                       dst->value_base_bytes
                                 : 0;
                    for (std::size_t c = 0;
                         c < mons.size() && c < dst->value_kinds.size(); ++c)
                        if (monoid_is_variable(dst->value_kinds[c]))
                            delta += mons[c].state_bytes();
                    dst->part_bytes[p] += delta;
                    dst->footprint += delta;
                }
            }
        }
        // Adopt the worker's spilled runs so reload merges them at materialize;
        // partition indices align via dst's part_bits.
        if (dst->runs.size() == src.runs.size())
            for (std::uint32_t p = 0; p < src.runs.size(); ++p)
                for (auto& r : src.runs[p])
                    dst->runs[p].push_back(std::move(r));
        // Force a budget check now that a whole worker folded in.
        note_and_maybe_spill(*dst, /*inserted=*/true);
    }
    for (const DeclaredJoin& j : o.joins_)
        declare_join(j.out_name.c_str(), j.left_name.c_str(),
                     j.right_name.c_str(), j.type);
    // Adopt the worker's spill dirs so its destructor does not delete runs this
    // fold still needs.
    for (auto& d : o.map_spill_dirs_) map_spill_dirs_.push_back(std::move(d));
    o.map_spill_dirs_.clear();
    o.map_spill_cur_dir_.clear();
    if (o.map_spill_failed_) map_spill_failed_ = true;

    // Fold each worker's aggregation accumulators into this master by name via
    // the engine's single merge path (agg_merge). An accumulator new to the
    // master is moved in whole; a shared name merges the two AggStates.
    for (auto& [key, idx] : o.agg_index_) {
        std::unique_ptr<AggAccum>& src = o.aggs_[idx];
        if (!src || !src->state) continue;
        auto it = agg_index_.find(key);
        if (it == agg_index_.end()) {
            aggs_.push_back(std::move(src));
            agg_index_.emplace(key, aggs_.size() - 1);
        } else if (aggs_[it->second]->state) {
            dataframe::agg_merge(*aggs_[it->second]->state, *src->state);
        }
    }
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
void PluginFold::materialize_map(
    MapAccum& m, std::size_t max_rows_per_batch,
    std::vector<utilities::common::arrow::ArrowExportResult>& out,
    int only_comp) {
    namespace arr = utilities::common::arrow;
    using NestedEntryRef = const std::pair<const std::vector<std::int64_t>,
                                           std::vector<MonoidAccumulator>>*;
    // Compare key components [off, off+n): STR by resolved label, float by
    // value, unsigned by magnitude, signed by value. Returns -1/0/1.
    auto cmp_range = [&](const MapAccum& acc,
                         const std::vector<std::int64_t>& a,
                         const std::vector<std::int64_t>& b, std::uint32_t off,
                         std::uint32_t n) -> int {
        for (std::uint32_t i = 0; i < n; ++i) {
            const std::uint32_t k = off + i;
            const dftu_type t = acc.key_types[k];
            if (key_type_interned(t)) {
                // STR and BYTES both sort byte-lexicographically on their
                // resolved span.
                std::string_view sa =
                    resolve_id(*intern_, static_cast<dftu_str>(a[k]));
                std::string_view sb =
                    resolve_id(*intern_, static_cast<dftu_str>(b[k]));
                if (sa != sb) return sa < sb ? -1 : 1;
            } else if (a[k] != b[k]) {
                if (key_type_float(t)) {
                    double da = key_bits_to_double(t, a[k]);
                    double db = key_bits_to_double(t, b[k]);
                    if (da != db) return da < db ? -1 : 1;
                    return a[k] < b[k] ? -1 : 1;
                }
                if (key_type_unsigned(t))
                    return static_cast<std::uint64_t>(a[k]) <
                                   static_cast<std::uint64_t>(b[k])
                               ? -1
                               : 1;
                return a[k] < b[k] ? -1 : 1;
            }
        }
        return 0;
    };

    // One Arrow batch per chunk of output rows; max_rows == 0 emits a single
    // batch. append_unit fills output row u, nunits is the output-row count
    // (groups for a nested map). A fresh builder per chunk avoids reset()-reuse
    // across the list/struct column types.
    auto emit_chunks = [&](const std::vector<arr::ColumnSpec>& specs,
                           std::size_t nunits, std::size_t max_rows,
                           const auto& append_unit) {
        auto emit = [&](std::size_t lo, std::size_t hi) {
            auto builder = std::make_unique<arr::RecordBatchBuilder>();
            builder->declare_schema(specs);
            builder->reserve(hi - lo);
            for (std::size_t u = lo; u < hi; ++u) append_unit(*builder, u);
            arr::ArrowExportResult res = builder->finish();
            if (res.valid()) out.push_back(std::move(res));
        };
        if (max_rows == 0 || nunits == 0) {
            emit(0, nunits);
        } else {
            for (std::size_t lo = 0; lo < nunits; lo += max_rows)
                emit(lo, std::min(nunits, lo + max_rows));
        }
    };

    // Reload any spilled runs so the map materializes identically to the
    // never-spilled path.
    reload_runs(m);
    const std::size_t max = max_rows_per_batch;
    if (m.nested_inner_n > 0) {
        const std::uint32_t inner_n = m.nested_inner_n;
        const std::uint32_t outer_n = m.key_n - inner_n;
        const std::uint32_t nvalue =
            static_cast<std::uint32_t>(m.value_kinds.size());
        std::vector<arr::ColumnSpec> specs;
        specs.reserve(outer_n + 1);
        append_key_specs(specs, m.key_types, outer_n);
        arr::ColumnSpec nested;
        nested.name = "value";
        nested.type = arr::ColumnType::STRUCT_LIST;
        for (std::uint32_t j = 0; j < inner_n; ++j)
            nested.fields.push_back({"ik" + std::to_string(j),
                                     key_column_type(m.inner_key_types[j]),
                                     {}});
        for (std::uint32_t c = 0; c < nvalue; ++c)
            nested.fields.push_back(
                {nvalue == 1 ? std::string("value") : "v" + std::to_string(c),
                 monoid_value_column_type(m.value_kinds[c]),
                 {}});
        specs.push_back(std::move(nested));

        std::vector<NestedEntryRef> rows;
        rows.reserve(m.total_entries());
        for (const auto& part : m.parts)
            for (const auto& kv : part) rows.push_back(&kv);
        // Always sort (outer then inner), not only when m.ordered: a nested
        // list has no intrinsic element order, so sorting is what makes the
        // rows reproducible across the merge's arbitrary hash order.
        std::sort(
            rows.begin(), rows.end(), [&](NestedEntryRef a, NestedEntryRef b) {
                int c = cmp_range(m, a->first, b->first, 0, outer_n);
                if (c != 0) return c < 0;
                return cmp_range(m, a->first, b->first, outer_n, inner_n) < 0;
            });
        // One output row per outer key. Chunk by whole groups so an outer group
        // is never split across batches.
        std::vector<std::pair<std::size_t, std::size_t>> groups;
        for (std::size_t gi = 0; gi < rows.size();) {
            std::size_t gj = gi;
            while (gj < rows.size() &&
                   cmp_range(m, rows[gi]->first, rows[gj]->first, 0, outer_n) ==
                       0)
                ++gj;
            groups.push_back({gi, gj});
            gi = gj;
        }
        auto append_group = [&](arr::RecordBatchBuilder& builder,
                                std::size_t g) {
            const std::size_t gi = groups[g].first;
            const std::size_t gj = groups[g].second;
            const std::vector<std::int64_t>& okey = rows[gi]->first;
            for (std::uint32_t o = 0; o < outer_n; ++o)
                append_key_cell(builder, o, m.key_types[o], okey[o], *intern_);
            std::vector<std::vector<arr::StructCell>> inner;
            inner.reserve(gj - gi);
            for (std::size_t r = gi; r < gj; ++r) {
                const std::vector<std::int64_t>& key = rows[r]->first;
                const std::vector<MonoidAccumulator>& mons = rows[r]->second;
                std::vector<arr::StructCell> cells(inner_n + nvalue);
                for (std::uint32_t jj = 0; jj < inner_n; ++jj) {
                    const dftu_type t = m.inner_key_types[jj];
                    const std::int64_t bits = key[outer_n + jj];
                    arr::StructCell& cell = cells[jj];
                    if (key_type_interned(t))
                        // STR and BYTES both resolve to a byte span; the field
                        // column type (utf8 vs binary) decides output.
                        cell.str =
                            resolve_id(*intern_, static_cast<dftu_str>(bits));
                    else if (key_type_float(t))
                        cell.f64 = key_bits_to_double(t, bits);
                    else if (key_type_unsigned(t))
                        cell.u64 = static_cast<std::uint64_t>(bits);
                    else
                        cell.i64 = bits;
                }
                for (std::uint32_t c = 0; c < nvalue; ++c) {
                    dftu_monoid_value v = mons[c].to_value();
                    arr::StructCell& cell = cells[inner_n + c];
                    switch (monoid_value_column_type(m.value_kinds[c])) {
                        case arr::ColumnType::FLOAT32:
                        case arr::ColumnType::DOUBLE:
                            cell.f64 = v.as.f64;
                            break;
                        case arr::ColumnType::UINT8:
                        case arr::ColumnType::UINT16:
                        case arr::ColumnType::UINT32:
                            cell.u64 = v.as.u64;
                            break;
                        case arr::ColumnType::STRING:
                            cell.str = resolve_id(
                                *intern_, static_cast<dftu_str>(v.as.u64));
                            break;
                        default:
                            cell.i64 = static_cast<std::int64_t>(v.as.u64);
                            break;
                    }
                }
                inner.push_back(std::move(cells));
            }
            builder.append_struct_list(outer_n, inner);
            builder.end_row();
        };
        emit_chunks(specs, groups.size(), max, append_group);
        return;
    }
    const std::uint32_t value_n =
        static_cast<std::uint32_t>(m.value_kinds.size());
    std::vector<arr::ColumnSpec> specs;
    specs.reserve(m.key_n + value_n);
    append_key_specs(specs, m.key_types, m.key_n);
    append_value_specs(specs, m.value_kinds, m.payload_types, m.quantile_qs, "",
                       only_comp);
    using EntryRef = const std::pair<const std::vector<std::int64_t>,
                                     std::vector<MonoidAccumulator>>*;
    std::vector<EntryRef> rows;
    rows.reserve(m.total_entries());
    for (const auto& part : m.parts)
        for (const auto& kv : part) rows.push_back(&kv);
    if (m.ordered) {
        // Resolve each interned key component once, then sort rows: interned by
        // resolved bytes, I64 by value.
        std::vector<std::vector<std::string_view>> labels(rows.size());
        for (std::size_t r = 0; r < rows.size(); ++r) {
            labels[r].resize(m.key_n);
            for (std::uint32_t i = 0; i < m.key_n; ++i)
                if (key_type_interned(m.key_types[i]))
                    labels[r][i] = resolve_id(
                        *intern_, static_cast<dftu_str>(rows[r]->first[i]));
        }
        std::vector<std::size_t> idx(rows.size());
        std::iota(idx.begin(), idx.end(), std::size_t{0});
        std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
            const auto& ka = rows[a]->first;
            const auto& kb = rows[b]->first;
            for (std::uint32_t i = 0; i < m.key_n; ++i) {
                if (key_type_interned(m.key_types[i])) {
                    if (labels[a][i] != labels[b][i])
                        return labels[a][i] < labels[b][i];
                } else if (ka[i] != kb[i]) {
                    if (key_type_float(m.key_types[i])) {
                        double da = key_bits_to_double(m.key_types[i], ka[i]);
                        double db = key_bits_to_double(m.key_types[i], kb[i]);
                        if (da != db) return da < db;
                        // Equal value, differing bits (e.g. -0.0): fall back to
                        // bit order for determinism.
                        return ka[i] < kb[i];
                    }
                    if (key_type_unsigned(m.key_types[i]))
                        return static_cast<std::uint64_t>(ka[i]) <
                               static_cast<std::uint64_t>(kb[i]);
                    return ka[i] < kb[i];
                }
            }
            return false;
        });
        std::vector<EntryRef> sorted;
        sorted.reserve(rows.size());
        for (std::size_t r : idx) sorted.push_back(rows[r]);
        rows.swap(sorted);
    }

    auto append_row = [&](arr::RecordBatchBuilder& builder, std::size_t ri) {
        EntryRef row = rows[ri];
        const std::vector<std::int64_t>& key = row->first;
        const std::vector<MonoidAccumulator>& mons = row->second;
        for (std::uint32_t i = 0; i < m.key_n; ++i)
            append_key_cell(builder, i, m.key_types[i], key[i], *intern_);
        append_value_cells(builder, m.key_n, mons, m.value_kinds,
                           m.payload_types, m.quantile_qs, *intern_, only_comp);
        builder.end_row();
    };
    emit_chunks(specs, rows.size(), max, append_row);
}

// Column order: key columns k0.., then LEFT value columns ("l_"), then RIGHT
// ("r_"); the prefixes stop the two sides colliding on a shared value name.
// Rows follow the JoinedMap's key order; an absent OUTER-join side nulls its
// values.
static JoinType to_join_type(dftu_join_type t) {
    switch (t) {
        case DFTU_JOIN_LEFT:
            return JoinType::LEFT;
        case DFTU_JOIN_RIGHT:
            return JoinType::RIGHT;
        case DFTU_JOIN_FULL:
            return JoinType::FULL;
        case DFTU_JOIN_INNER:
        default:
            return JoinType::INNER;
    }
}

utilities::common::arrow::ArrowExportResult materialize_joined(
    const JoinedMap& j, dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    arr::ArrowExportResult empty;
    if (!j.valid) return empty;

    std::vector<arr::ColumnSpec> specs;
    append_key_specs(specs, j.key_types, j.key_n);
    append_value_specs(specs, j.left_schema.value_kinds,
                       j.left_schema.payload_types, j.left_schema.quantile_qs,
                       "l_");
    if (!j.left_only)
        append_value_specs(specs, j.right_schema.value_kinds,
                           j.right_schema.payload_types,
                           j.right_schema.quantile_qs, "r_");

    const std::uint32_t left_first = j.key_n;
    const std::uint32_t right_first =
        left_first + value_column_count(j.left_schema.value_kinds,
                                        j.left_schema.payload_types,
                                        j.left_schema.quantile_qs);

    arr::RecordBatchBuilder builder;
    builder.declare_schema(specs);
    builder.reserve(j.rows.size());
    for (const JoinedRow& row : j.rows) {
        for (std::uint32_t i = 0; i < j.key_n; ++i)
            append_key_cell(builder, i, j.key_types[i], row.key[i], intern);
        if (row.left_present)
            append_value_cells(
                builder, left_first, row.left_values, j.left_schema.value_kinds,
                j.left_schema.payload_types, j.left_schema.quantile_qs, intern);
        else
            null_value_cells(builder, left_first, j.left_schema.value_kinds,
                             j.left_schema.payload_types,
                             j.left_schema.quantile_qs);
        if (!j.left_only) {
            if (row.right_present)
                append_value_cells(builder, right_first, row.right_values,
                                   j.right_schema.value_kinds,
                                   j.right_schema.payload_types,
                                   j.right_schema.quantile_qs, intern);
            else
                null_value_cells(
                    builder, right_first, j.right_schema.value_kinds,
                    j.right_schema.payload_types, j.right_schema.quantile_qs);
        }
        builder.end_row();
    }
    return builder.finish();
}

utilities::common::arrow::ArrowExportResult materialize_exploded(
    const ExplodedRows& e, dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    arr::ArrowExportResult empty;
    if (!e.valid) return empty;

    std::vector<arr::ColumnSpec> specs;
    append_key_specs(specs, e.key_types, e.key_n);
    // Nested maps reject a SKETCH value, so the exploded rows carry no
    // quantiles.
    const std::vector<double> no_qs;
    append_value_specs(specs, e.value_kinds, e.payload_types, no_qs, "");
    const std::uint32_t elem_col =
        e.key_n + value_column_count(e.value_kinds, e.payload_types, no_qs);
    specs.push_back({e.elem_name, key_column_type(e.elem_type)});

    arr::RecordBatchBuilder builder;
    builder.declare_schema(specs);
    builder.reserve(e.rows.size());
    for (const ExplodedRow& row : e.rows) {
        for (std::uint32_t i = 0; i < e.key_n; ++i)
            append_key_cell(builder, i, e.key_types[i], row.key[i], intern);
        append_value_cells(builder, e.key_n, row.kept_values, e.value_kinds,
                           e.payload_types, no_qs, intern);
        if (row.elem_null)
            builder.append_null(elem_col);
        else
            append_key_cell(builder, elem_col, e.elem_type, row.elem, intern);
        builder.end_row();
    }
    return builder.finish();
}

utilities::common::arrow::ArrowExportResult materialize_grouping_sets(
    const std::vector<MapAccum>& groupings,
    const std::vector<std::vector<std::uint32_t>>& keep_sets,
    const MapAccum& original, dftracer::utils::StringIntern& intern) {
    namespace arr = utilities::common::arrow;
    arr::ArrowExportResult empty;
    if (groupings.size() != keep_sets.size() || original.key_n == 0)
        return empty;

    std::vector<arr::ColumnSpec> specs;
    append_key_specs(specs, original.key_types, original.key_n);
    append_value_specs(specs, original.value_kinds, original.payload_types,
                       original.quantile_qs, "");
    const std::uint32_t value_first = original.key_n;
    const std::uint32_t gid_col =
        original.key_n + value_column_count(original.value_kinds,
                                            original.payload_types,
                                            original.quantile_qs);
    // grouping_id is the keep-set index, not a SQL GROUPING() null-mask
    // bitmask.
    specs.push_back({"grouping_id", arr::ColumnType::INT64});

    arr::RecordBatchBuilder builder;
    builder.declare_schema(specs);
    std::size_t total = 0;
    for (const MapAccum& g : groupings) total += g.total_entries();
    builder.reserve(total);

    using Entry = std::pair<const std::vector<std::int64_t>,
                            std::vector<MonoidAccumulator>>;
    std::vector<std::int32_t> slot(original.key_n);
    for (std::size_t i = 0; i < groupings.size(); ++i) {
        const MapAccum& g = groupings[i];
        const std::vector<std::uint32_t>& ks = keep_sets[i];
        // slot[d] = the regrouped-key position of original component d, or -1
        // when this set dropped d (its column is nulled).
        for (std::uint32_t d = 0; d < original.key_n; ++d) slot[d] = -1;
        for (std::uint32_t j = 0; j < ks.size(); ++j)
            if (ks[j] < original.key_n)
                slot[ks[j]] = static_cast<std::int32_t>(j);

        std::vector<const Entry*> ents;
        ents.reserve(g.total_entries());
        for (const MapAccum::EntriesMap& part : g.parts)
            for (const auto& e : part) ents.push_back(&e);
        std::sort(ents.begin(), ents.end(), [](const Entry* a, const Entry* b) {
            return a->first < b->first;
        });

        for (const Entry* e : ents) {
            for (std::uint32_t d = 0; d < original.key_n; ++d) {
                if (slot[d] < 0)
                    builder.append_null(d);
                else
                    append_key_cell(builder, d, original.key_types[d],
                                    e->first[slot[d]], intern);
            }
            append_value_cells(builder, value_first, e->second,
                               original.value_kinds, original.payload_types,
                               original.quantile_qs, intern);
            builder.append_int64(gid_col, static_cast<std::int64_t>(i));
            builder.end_row();
        }
    }
    return builder.finish();
}

void PluginFold::collect_map_batches_streaming(
    MapAccum& m,
    std::vector<utilities::common::arrow::ArrowExportResult>& out) {
    // Ordered/nested maps must order globally, so reload the whole map and
    // stream the sorted result in row-capped chunks. A key (flat) or outer
    // group (nested) lives in exactly one partition, so the union never
    // re-merges a monoid and a global sort matches merging the K sorted
    // partition streams.
    if (m.ordered || m.nested_inner_n > 0) {
        const std::size_t chunk = map_stream_chunk_rows_
                                      ? map_stream_chunk_rows_
                                      : MAP_STREAM_CHUNK_ROWS;
        materialize_map(m, chunk, out);
        if (m.partitions() > map_stream_max_resident_parts_)
            map_stream_max_resident_parts_ = m.partitions();
        return;
    }
    // Unordered: each key lives in one partition, so materialize one at a time
    // and free it before the next. Row order across partitions is unspecified.
    for (std::uint32_t p = 0; p < m.partitions(); ++p) {
        if (m.parts[p].empty() && m.runs[p].empty()) continue;
        MapAccum tmp;
        tmp.name = m.name;
        tmp.key_n = m.key_n;
        tmp.key_types = m.key_types;
        tmp.value_kinds = m.value_kinds;
        tmp.payload_types = m.payload_types;
        tmp.quantile_qs = m.quantile_qs;
        tmp.fused_out_names = m.fused_out_names;
        tmp.value_base_bytes = m.value_base_bytes;
        tmp.set_part_bits(0);
        tmp.parts[0] = std::move(m.parts[p]);
        tmp.runs[0] = std::move(m.runs[p]);
        materialize_map(tmp, 0, out);
        m.parts[p].clear();
        m.runs[p].clear();
        if (map_stream_max_resident_parts_ < 1)
            map_stream_max_resident_parts_ = 1;
    }
}
#endif

void PluginFold::materialize_maps() {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    if (!named_results_) return;
    if (map_spill_failed_) {
        DFTRACER_UTILS_LOG_ERROR(
            "Plugin map spill failed earlier; refusing to emit maps rather "
            "than "
            "a silently partial result");
        return;
    }
    namespace arr = utilities::common::arrow;
    map_stream_batches_ = 0;
    map_stream_max_resident_parts_ = 0;
    // Emit one collected map (as `name`): a single batch as an Arrow table,
    // multiple as a streamed RecordBatchReader (see result_to_py).
    auto emit_named = [&](const char* name,
                          std::vector<arr::ArrowExportResult>& batches) {
        std::vector<arr::ArrowExportResult> valid;
        valid.reserve(batches.size());
        for (auto& b : batches)
            if (b.valid()) valid.push_back(std::move(b));
        if (valid.empty()) return;
        map_stream_batches_ += valid.size();
        if (valid.size() == 1) {
            named_results_->emit_arrow(name, valid[0].get_array(),
                                       valid[0].get_schema());
        } else {
            OwnedArrowBatches ob;
            ob.batches.reserve(valid.size());
            for (auto& b : valid) {
                OwnedArrow o;
                ArrowArrayMove(b.get_array(), &o.array);
                ArrowSchemaMove(b.get_schema(), &o.schema);
                ob.batches.push_back(std::move(o));
            }
            named_results_->emit_arrow_batches(name, std::move(ob));
        }
    };
    for (MapAccum& m : maps_) {
        try {
            // A fused map splits into one named table per component, so the
            // caller sees the separate maps it declared, never the product.
            if (!m.fused_out_names.empty()) {
                for (std::uint32_t c = 0; c < m.fused_out_names.size(); ++c) {
                    std::vector<arr::ArrowExportResult> batches;
                    materialize_map(m, 0, batches, static_cast<int>(c));
                    emit_named(m.fused_out_names[c].c_str(), batches);
                }
                continue;
            }
            std::vector<arr::ArrowExportResult> batches;
            if (map_stream_enabled_)
                collect_map_batches_streaming(m, batches);
            else
                materialize_map(m, 0, batches);
            emit_named(m.name.c_str(), batches);
        } catch (...) {
        }
    }
    auto find_map = [&](const std::string& name) -> MapAccum* {
        auto it = map_index_.find(dftracer::utils::hash::fnv1a_hash(name));
        return it == map_index_.end() ? nullptr : &maps_[it->second];
    };
    for (const DeclaredJoin& dj : joins_) {
        try {
            MapAccum* left = find_map(dj.left_name);
            MapAccum* right = find_map(dj.right_name);
            if (!left || !right) {
                DFTRACER_UTILS_LOG_ERROR(
                    "Plugin join '%s' skipped: input map '%s' not found",
                    dj.out_name.c_str(),
                    !left ? dj.left_name.c_str() : dj.right_name.c_str());
                continue;
            }
            JoinedMap jm = join_maps(*left, *right, to_join_type(dj.type));
            if (!jm.valid) {
                DFTRACER_UTILS_LOG_ERROR(
                    "Plugin join '%s' skipped: maps '%s' and '%s' do not share "
                    "a key schema",
                    dj.out_name.c_str(), dj.left_name.c_str(),
                    dj.right_name.c_str());
                continue;
            }
            arr::ArrowExportResult res = materialize_joined(jm, *intern_);
            if (res.valid())
                named_results_->emit_arrow(dj.out_name.c_str(), res.get_array(),
                                           res.get_schema());
        } catch (...) {
        }
    }
#endif
}

void PluginFold::materialize_aggs() {
    if (!named_results_) return;
    for (std::unique_ptr<AggAccum>& acc : aggs_) {
        if (!acc || !acc->state) continue;
        try {
            dataframe::DataFrame out =
                dataframe::agg_finalize(*acc->state, acc->key_names);
            // Move the columns into a dftu_dataframe handle (the engine's own
            // ABI boundary type) so the result crosses as our DataFrame, no
            // Arrow round-trip.
            std::vector<dftu_series*> handles;
            handles.reserve(out.columns.size());
            std::vector<const char*> names;
            names.reserve(out.names.size());
            for (dataframe::Series& c : out.columns)
                handles.push_back(c.release());
            for (const std::string& n : out.names) names.push_back(n.c_str());
            dftu_dataframe* h =
                dftu_dataframe_new(names.data(), handles.data(),
                                   static_cast<std::int32_t>(handles.size()));
            if (h) named_results_->emit_frame(acc->name.c_str(), h);
        } catch (...) {
        }
    }
}

coro::CoroTask<bool> PluginFold::finalize(const CoverageSet&) {
    // Publish this fold's merged handles before on_finalize so a consumer
    // finalizing later in fold order can read them via result().
    if (results_)
        for (const auto& [key, idx] : handle_index_)
            results_->values[key] = handles_[idx].to_value();
    materialize_maps();
    materialize_aggs();
    // Runs reloaded and emitted; drop the temp dirs (the destructor repeats
    // this on an exceptional unwind).
    remove_spill_dirs();
    if (slice_) {
        if (::dftu_task* t = plugin_->on_finalize(slice_, &host_))
            co_await *as_task(t);
        task_arena_.clear();
    }
    co_return false;
}

}  // namespace dftracer::utils::plugins
