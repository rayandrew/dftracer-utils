/* Example dftracer-utils plugin in pure C: a weighted process<->file bipartite
 * graph. Each file event contributes 1 to the host-owned mergeable map at key
 * {pid, fhash}; the host merges the map across workers and materializes it to
 * an Arrow table [k0, k1, value]. PluginHost::run returns it to Python as a
 * pyarrow.Table under "process_file_edges", from which the caller builds an
 * adjacency matrix in bulk (numpy + scipy) with no per-edge Python.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_file_edges.so process_file_edges.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>
#include <stdlib.h>

static uint32_t needs(void* self) {
    (void)self;
    return DFTU_NEED_FHASH;
}

/* The map lives host-side; the slice only marks make_slice non-null so the fold
 * delivers batches. */
static void* make_slice(void* self) {
    (void)self;
    return calloc(1, 1);
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    const dftu_ext_map* map =
        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);
    static const dftu_type key_types[2] = {DFTU_T_I64, DFTU_T_I64};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new) return NULL;
    m = map->map_new(host->h, "process_file_edges", key_types, 2,
                     DFTU_MONOID_COUNTER);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[2];
        if (e->fhash == DFTU_STR_NONE) continue;
        key[0] = (int64_t)e->pid;
        key[1] = (int64_t)e->fhash;
        map->map_add_u64(host->h, m, key, 1);
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
    return &g_plugin;
}
