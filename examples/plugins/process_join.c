/* Example dftracer-utils plugin in pure C: declares a native map-to-map join.
 * Each slice builds two host-owned mergeable maps keyed {pid}: "pid_counts"
 * (COUNTER of events per process) and "pid_durs" (SUM_F64 of event durations
 * per process, only for events that carry one). It then declares a LEFT join
 * of the two into "joined". The host merges each map across workers and, at
 * finalize, joins the merged masters on {pid} and emits the result as an extra
 * named Arrow table [k0 : int64 (pid), l_v0 : int64 (count), r_v0 : f64 (dur
 * sum)]; a pid present in counts but absent from durs nulls r_v0. The input
 * maps still materialize as "pid_counts" and "pid_durs"; the join is additive.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_join.so process_join.c
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
    dftu_map* counts;
    dftu_map* durs;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new || !map->map_declare_join) return NULL;
    counts =
        map->map_new(host->h, "pid_counts", key_types, 1, DFTU_MONOID_COUNTER);
    durs = map->map_new(host->h, "pid_durs", key_types, 1, DFTU_MONOID_SUM_F64);
    if (!counts || !durs) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        key[0] = (int64_t)e->pid;
        map->map_add_u64(host->h, counts, key, 1);
        if (e->has_dur) map->map_add_f64(host->h, durs, key, (double)e->dur);
    }
    map->map_declare_join(host->h, "joined", "pid_counts", "pid_durs",
                          DFTU_JOIN_LEFT);
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
