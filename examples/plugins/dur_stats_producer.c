/* Example producer plugin: each batch it accumulates every event's duration
 * into a shared DDSketch handle and counts the events into a shared COUNTER
 * handle, both named under its provided capability com.example.dur_stats@1.0.0.
 * The host merges these handles across all worker slices, so at finalize a
 * consumer reads whole-scan stats no single slice ever saw.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o dur_stats_producer.so dur_stats_producer.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdlib.h>
#include <string.h>

#define DUR_STATS_CAP "com.example.dur_stats"
#define DUR_COUNT_HANDLE "com.example.dur_stats.count"
#define DUR_SKETCH_HANDLE "com.example.dur_stats.sketch"

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) {
    (void)self;
    return calloc(1, 1);
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    (void)slice;
    const dftu_ext_handles* hd =
        (const dftu_ext_handles*)host->get_extension(host->h, DFTU_EXT_HANDLES);
    if (hd) {
        dftu_handle* cnt =
            hd->shared_get(host->h, DUR_COUNT_HANDLE, DFTU_MONOID_COUNTER);
        dftu_handle* skt =
            hd->shared_get(host->h, DUR_SKETCH_HANDLE, DFTU_MONOID_SKETCH);
        hd->add_u64(host->h, cnt, b->count);
        for (uint32_t i = 0; i < b->count; ++i)
            hd->add_f64(host->h, skt, (double)b->events[i].dur, 1.0);
    }
    return NULL;
}

static void merge(void* into, void* other) {
    (void)into;
    (void)other;
}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {
    (void)slice;
    (void)host;
    return NULL;
}

static void destroy_slice(void* slice) { free(slice); }
static void destroy(void* self) { (void)self; }

static uint32_t provides(void* self, dftu_capability* out, uint32_t max) {
    (void)self;
    if (max >= 1) {
        out[0].id = DUR_STATS_CAP;
        out[0].ver.major = 1;
        out[0].ver.minor = 0;
        out[0].ver.patch = 0;
    }
    return 1;
}

static const dftu_plugin_comms g_comms = {provides, NULL, NULL};

static const void* get_extension(void* self, const char* ext_id) {
    (void)self;
    if (ext_id && strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_comms;
    return NULL;
}

static dftu_plugin g_plugin;

DFTU_PLUGIN_EXPORT dftu_plugin* dftracer_plugin(const dftu_value* config) {
    (void)config;
    g_plugin.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    g_plugin.self = NULL;
    g_plugin.needs = needs;
    g_plugin.plan_query = NULL;
    g_plugin.make_slice = make_slice;
    g_plugin.on_batch = on_batch;
    g_plugin.merge = merge;
    g_plugin.on_finalize = on_finalize;
    g_plugin.destroy_slice = destroy_slice;
    g_plugin.destroy = destroy;
    g_plugin.get_extension = get_extension;
    return &g_plugin;
}
