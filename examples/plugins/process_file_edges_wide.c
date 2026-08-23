/* Example dftracer-utils plugin in pure C: a process<->file bipartite graph
 * whose edges carry two attributes. Each file event contributes to a product
 * map keyed {pid, fhash} with values (COUNTER, SUM_F64): +1 to component 0 and
 * +dur to component 1. The host merges the map across workers and materializes
 * it to an Arrow table [k0, k1, v0, v1], surfaced from PluginHost::run under
 * "process_file_edges_wide".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_file_edges_wide.so process_file_edges_wide.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>
#include <stdlib.h>

static uint32_t needs(void* self) {
    (void)self;
    return DFTU_NEED_FHASH;
}

static void* make_slice(void* self) {
    (void)self;
    return calloc(1, 1);
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    const dftu_ext_map* map =
        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);
    static const dftu_type key_types[2] = {DFTU_T_I64, DFTU_T_I64};
    static const dftu_monoid_kind values[2] = {DFTU_MONOID_COUNTER,
                                               DFTU_MONOID_SUM_F64};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new_product) return NULL;
    m = map->map_new_product(host->h, "process_file_edges_wide", key_types, 2,
                             values, 2);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[2];
        if (e->fhash == DFTU_STR_NONE) continue;
        key[0] = (int64_t)e->pid;
        key[1] = (int64_t)e->fhash;
        map->map_add_u64_at(host->h, m, key, 0, 1);
        map->map_add_f64_at(host->h, m, key, 1, (double)e->dur);
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
