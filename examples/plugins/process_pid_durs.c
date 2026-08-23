/* Example dftracer-utils plugin in pure C: the ts-ordered SEQUENCE of event
 * durations each process emitted. Each event appends (e->ts, e->dur) to the
 * host-owned mergeable map at key {pid}, whose value is a LIST_I64 monoid. The
 * host merges the map across workers (list concat) and materializes it sorted
 * by ts to an Arrow table [k0 : int64 (pid), value : list<int64> (the raw
 * durations in ts order)]. PluginHost::run returns it to Python as a
 * pyarrow.Table under "process_pid_durs".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_pid_durs.so process_pid_durs.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>
#include <stdlib.h>

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
    if (!map || !map->map_new || !map->map_add_ordered_at) return NULL;
    m = map->map_new(host->h, "process_pid_durs", key_types, 1,
                     DFTU_MONOID_LIST_I64);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        key[0] = (int64_t)e->pid;
        map->map_add_ordered_at(host->h, m, key, 0, (int64_t)e->ts,
                                (uint64_t)e->dur);
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
