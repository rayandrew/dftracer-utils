/* Example consumer plugin: optionally requires com.example.dur_stats>=1.0 and
 * in resolve() learns whether the producer is present. At finalize it reads the
 * producer's cross-worker-merged COUNTER and DDSketch handles from the shared
 * registry and logs the whole-scan count and quantiles (WIRED), or the
 * standalone fallback when no producer published them.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o dur_stats_consumer.so dur_stats_consumer.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DUR_STATS_CAP "com.example.dur_stats"
#define DUR_COUNT_HANDLE "com.example.dur_stats.count"
#define DUR_SKETCH_HANDLE "com.example.dur_stats.sketch"

typedef struct {
    int wired;
} Cfg;

static Cfg g_cfg;

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) { return self; }

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    (void)slice;
    (void)b;
    (void)host;
    return NULL;
}

static void merge(void* into, void* other) {
    (void)into;
    (void)other;
}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {
    const Cfg* cfg = (const Cfg*)slice;
    const dftu_ext_handles* hd =
        (const dftu_ext_handles*)host->get_extension(host->h, DFTU_EXT_HANDLES);
    dftu_monoid_value cnt, skt;
    char line[192];
    int n;
    if (cfg && cfg->wired && hd &&
        hd->result(host->h, DUR_COUNT_HANDLE, &cnt) == 0 &&
        hd->result(host->h, DUR_SKETCH_HANDLE, &skt) == 0)
        n = snprintf(line, sizeof line,
                     "dur_stats_consumer: WIRED count=%llu p50=%.3f min=%.3f "
                     "max=%.3f",
                     (unsigned long long)cnt.as.u64, skt.as.quant.p50,
                     skt.as.quant.min, skt.as.quant.max);
    else
        n = snprintf(line, sizeof line,
                     "dur_stats_consumer: STANDALONE no producer");
    if (n > 0) host->log(host->h, DFTU_LOG_INFO, line, (uint32_t)n);
    return NULL;
}

static void destroy_slice(void* slice) { (void)slice; }
static void destroy(void* self) { (void)self; }

static dftu_requirement dur_stats_requirement(void) {
    dftu_requirement r;
    r.id = DUR_STATS_CAP;
    r.op = DFTU_VER_GE;
    r.ver.major = 1;
    r.ver.minor = 0;
    r.ver.patch = 0;
    r.required = 0;
    return r;
}

static uint32_t require_caps(void* self, dftu_requirement* out, uint32_t max) {
    (void)self;
    if (max >= 1) out[0] = dur_stats_requirement();
    return 1;
}

static void resolve(void* self, const dftu_host* host) {
    Cfg* cfg = (Cfg*)self;
    const dftu_ext_comms* c =
        (const dftu_ext_comms*)host->get_extension(host->h, DFTU_EXT_COMMS);
    dftu_requirement req = dur_stats_requirement();
    dftu_version v;
    cfg->wired = c && c->provider_best(host->h, &req, &v) == 0;
}

static const dftu_plugin_comms g_comms = {NULL, require_caps, resolve};

static const void* get_extension(void* self, const char* ext_id) {
    (void)self;
    if (ext_id && strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_comms;
    return NULL;
}

static dftu_plugin g_plugin;

DFTU_PLUGIN_EXPORT dftu_plugin* dftracer_plugin(const dftu_value* config) {
    (void)config;
    g_cfg.wired = 0;
    g_plugin.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    g_plugin.self = &g_cfg;
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
