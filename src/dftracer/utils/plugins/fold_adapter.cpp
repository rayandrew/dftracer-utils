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
#include <dftracer/utils/plugins/fold_adapter_ext.h>
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

// 256 partitions (one hash byte), ClickHouse's default, so one oversized
// partition is rare without recursive re-partition.
constexpr std::uint32_t MAP_SPILL_PART_BITS = 8;

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
    if (std::strcmp(ext_id, DFTU_EXT_HANDLES) == 0)
        return detail::handles_ext_vtable();
    if (std::strcmp(ext_id, DFTU_EXT_RESULT) == 0)
        return detail::result_ext_vtable();
    if (std::strcmp(ext_id, DFTU_EXT_MAP) == 0) return detail::map_ext_vtable();
    if (std::strcmp(ext_id, DFTU_EXT_AGG) == 0) return detail::agg_ext_vtable();
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

void PluginFold::merge(Fold& other) {
    if (!slice_) return;
    auto& o = static_cast<PluginFold&>(other);
    if (o.slice_) plugin_->merge(slice_, o.slice_);
    // Merge named handles by key; fuse folds each worker slice into the master.
    for (const auto& [key, idx] : o.handles_.index()) {
        const MonoidAccumulator& src = o.handles_[idx];
        MonoidAccumulator* dst = handles_.find(key);
        if (!dst) dst = handles_.push(key, MonoidAccumulator(src.kind()));
        dst->merge(src);
    }
    // Fold each named map's entries into this by key, merging a colliding key's
    // monoid.
    for (const auto& [key, idx] : o.maps_.index()) {
        MapAccum& src = o.maps_[idx];
        MapAccum* dst = maps_.find(key);
        if (!dst) {
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
            dst = maps_.push(key, std::move(m));
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
    for (const auto& [key, idx] : o.aggs_.index()) {
        std::unique_ptr<AggAccum>& src = o.aggs_[idx];
        if (!src || !src->state) continue;
        std::unique_ptr<AggAccum>* dst = aggs_.find(key);
        if (!dst) {
            aggs_.push(key, std::move(src));
        } else if ((*dst)->state) {
            dataframe::agg_merge(*(*dst)->state, *src->state);
        }
    }
}

coro::CoroTask<bool> PluginFold::finalize(const CoverageSet&) {
    // Publish this fold's merged handles before on_finalize so a consumer
    // finalizing later in fold order can read them via result().
    if (results_)
        for (const auto& [key, idx] : handles_.index())
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
