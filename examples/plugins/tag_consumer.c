/* Example consumer plugin: optionally requires com.example.tag>=1.0 and, in
 * resolve(), discovers the provider by capability. It logs WIRED when a
 * satisfying provider is present and STANDALONE when none is, demonstrating
 * capability-based peer discovery with graceful fallback.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o tag_consumer.so tag_consumer.c
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
    (void)host;
    ((Counter*)slice)->events += b->count;
    return NULL;
}

static void merge(void* into, void* other) {
    ((Counter*)into)->events += ((const Counter*)other)->events;
}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {
    (void)slice;
    (void)host;
    return NULL;
}

static void destroy_slice(void* slice) { free(slice); }
static void destroy(void* self) { (void)self; }

static dftu_requirement tag_requirement(void) {
    dftu_requirement r;
    r.id = "com.example.tag";
    r.op = DFTU_VER_GE;
    r.ver.major = 1;
    r.ver.minor = 0;
    r.ver.patch = 0;
    r.required = 0;
    return r;
}

static uint32_t require_caps(void* self, dftu_requirement* out, uint32_t max) {
    (void)self;
    if (max >= 1) out[0] = tag_requirement();
    return 1;
}

static void resolve(void* self, const dftu_host* host) {
    (void)self;
    const dftu_ext_comms* c =
        (const dftu_ext_comms*)host->get_extension(host->h, DFTU_EXT_COMMS);
    dftu_requirement req = tag_requirement();
    dftu_version v;
    char line[128];
    int n;
    if (c && c->provider_best(host->h, &req, &v) == 0) {
        n = snprintf(line, sizeof line, "tag_consumer: WIRED %s %u.%u.%u",
                     req.id, v.major, v.minor, v.patch);
    } else {
        n = snprintf(line, sizeof line, "tag_consumer: STANDALONE %s", req.id);
    }
    if (n > 0) host->log(host->h, DFTU_LOG_INFO, line, (uint32_t)n);
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
