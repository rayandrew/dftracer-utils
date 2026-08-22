/* A real C consumer of the dftu_ext_trace ABI: open a .pfw.gz, append events,
 * close, then scan it back - all through the raw vtable from C. */
#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>

static void count_cb(const void* item, void* ud) {
    (void)item;
    ++*(int*)ud;
}

/* Write `n` events to `path`, then read them back; *out_read gets the count. */
int dftu_test_trace_roundtrip(const dftu_host* h, const char* path,
                              const dftu_event* evs, uint32_t n,
                              int* out_read) {
    const dftu_ext_trace* t =
        (const dftu_ext_trace*)h->get_extension(h->h, DFTU_EXT_TRACE);
    if (!t) return -1;
    dftu_trace_writer* w = t->trace_open_write(h->h, path);
    if (!w) return -1;
    int wrote = t->trace_write(h->h, w, evs, n);
    if (t->trace_close(h->h, w) != 0 || wrote != 0) return -1;

    int count = 0;
    if (t->trace_read(h->h, path, count_cb, &count) != 0) return -1;
    *out_read = count;
    return 0;
}
