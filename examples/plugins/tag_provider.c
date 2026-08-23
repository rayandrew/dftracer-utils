/* Example provider plugin: declares the capability com.example.tag@1.0.0 so a
 * peer can discover it by capability rather than by identity. The fold work is
 * trivial (an event count); the point is the declare side of the lifecycle.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o tag_provider.so tag_provider.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint64_t events;
} Counter;

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) {
    (void)self;
    return calloc(1, sizeof(Counter));
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    Counter* c = (Counter*)slice;
    (void)host;
    c->events += b->count;
    return NULL;
}

static void merge(void* into, void* other) {
    ((Counter*)into)->events += ((const Counter*)other)->events;
}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {
    const Counter* c = (const Counter*)slice;
    char line[96];
    int n = snprintf(line, sizeof line, "tag_provider: events=%llu",
                     (unsigned long long)c->events);
    if (n > 0) host->log(host->h, DFTU_LOG_INFO, line, (uint32_t)n);
    return NULL;
}

static void destroy_slice(void* slice) { free(slice); }
static void destroy(void* self) { (void)self; }

static uint32_t provides(void* self, dftu_capability* out, uint32_t max) {
    (void)self;
    if (max >= 1) {
        out[0].id = "com.example.tag";
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
