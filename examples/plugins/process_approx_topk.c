/* Example dftracer-utils plugin in pure C: the 5 most FREQUENT files each
 * process touched, approximate, in bounded memory. Each event observes its
 * fhash into an APPROX_TOPK_STR value at key {pid} with k=5 (the SpaceSaving
 * counter capacity, also the output size). Unlike TOPK, which ranks payloads by
 * a provided `by` key, this ranks values by their own frequency and keeps at
 * most k (value, count, error) counters. The host merges the heavy-hitter
 * sketch across workers and materializes it to an Arrow table
 * [k0 : int64 (pid), value : list<struct<value : string, count : int64>>
 * (the file labels with approximate counts, most frequent first)]. Returned to
 * Python as a pyarrow.Table under "process_approx_topk".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_approx_topk.so process_approx_topk.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>
#include <stdlib.h>

#define TOPK 5u

static uint32_t needs(void* self) {
    (void)self;
    return 0u;
}

static void* make_slice(void* self) {
    (void)self;
    return calloc(1, 1);
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    const dftu_ext_map* map =
        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);
    static const dftu_type key_types[1] = {DFTU_T_I64};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new || !map->map_add_approx_topk_at) return NULL;
    m = map->map_new(host->h, "process_approx_topk", key_types, 1,
                     DFTU_MONOID_APPROX_TOPK_STR);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        if (e->fhash == DFTU_STR_NONE) continue;
        key[0] = (int64_t)e->pid;
        /* k is passed on every add; the monoid records it and stays bounded. */
        map->map_add_approx_topk_at(host->h, m, key, 0, TOPK,
                                    (int64_t)e->fhash);
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
