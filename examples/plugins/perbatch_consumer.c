/* Example consumer plugin: optionally requires com.example.perbatch_dur>=1.0
 * and in resolve() discovers whether the producer is present. Each batch it
 * consumes the producer's per-batch value on the shared port and accumulates
 * it, logging a total at finalize. With no producer every consume returns NULL,
 * so it logs the standalone/no-data fallback instead.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o perbatch_consumer.so perbatch_consumer.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define PERBATCH_CAP "com.example.perbatch_dur"

typedef struct {
    int wired;
} Cfg;

typedef struct {
    const Cfg* cfg;
    uint64_t total;
    uint64_t batches;
} Slice;

static Cfg g_cfg;

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) {
    Slice* s = (Slice*)calloc(1, sizeof(Slice));
    if (s) s->cfg = (const Cfg*)self;
    return s;
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    (void)b;
    Slice* s = (Slice*)slice;
    const dftu_ext_ports* p =
        (const dftu_ext_ports*)host->get_extension(host->h, DFTU_EXT_PORTS);
    if (p) {
        uint32_t len = 0;
        const void* d =
            p->consume(host->h, p->port_key(host->h, PERBATCH_CAP), &len);
        if (d && len == (uint32_t)sizeof(uint64_t)) {
            s->total += *(const uint64_t*)d;
            s->batches++;
        }
    }
    return NULL;
}

static void merge(void* into, void* other) {
    Slice* a = (Slice*)into;
    const Slice* b = (const Slice*)other;
    a->total += b->total;
    a->batches += b->batches;
}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {
    const Slice* s = (const Slice*)slice;
    char line[128];
    int n;
    if (s->cfg && s->cfg->wired && s->batches > 0)
        n = snprintf(line, sizeof line,
                     "perbatch_consumer: WIRED total=%llu batches=%llu",
                     (unsigned long long)s->total,
                     (unsigned long long)s->batches);
    else
        n = snprintf(line, sizeof line,
                     "perbatch_consumer: STANDALONE no producer");
    if (n > 0) host->log(host->h, DFTU_LOG_INFO, line, (uint32_t)n);
    return NULL;
}

static void destroy_slice(void* slice) { free(slice); }
static void destroy(void* self) { (void)self; }

static dftu_requirement perbatch_requirement(void) {
    dftu_requirement r;
    r.id = PERBATCH_CAP;
    r.op = DFTU_VER_GE;
    r.ver.major = 1;
    r.ver.minor = 0;
    r.ver.patch = 0;
    r.required = 0;
    return r;
}

static uint32_t require_caps(void* self, dftu_requirement* out, uint32_t max) {
    (void)self;
    if (max >= 1) out[0] = perbatch_requirement();
    return 1;
}

static void resolve(void* self, const dftu_host* host) {
    Cfg* cfg = (Cfg*)self;
    const dftu_ext_comms* c =
        (const dftu_ext_comms*)host->get_extension(host->h, DFTU_EXT_COMMS);
    dftu_requirement req = perbatch_requirement();
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
