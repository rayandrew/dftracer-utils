/* Example dftracer-utils plugin in pure C: per process, the file touched at the
 * longest event (argmax-by-duration) and the sample variance of its event
 * durations. Each event contributes to a host-owned mergeable product map at
 * key {pid} whose two value components are: 0: ARGMAX_STR over `by = dur`,
 * payload = fhash (the file's interned id), resolved to a string at
 * materialize; 1: VARIANCE of `dur` (a FieldStat power-sum moment). The host
 * merges the map across workers (argmax keeps the extreme, variance merges the
 * power sums) and materializes it to an Arrow table [k0 : int64 (pid), v0 :
 * utf8 (busiest file), v1 : float64 (dur variance)], returned to Python as a
 * pyarrow.Table under "process_pid_argstats".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_pid_argstats.so process_pid_argstats.c
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
    static const dftu_monoid_kind values[2] = {DFTU_MONOID_ARGMAX_STR,
                                               DFTU_MONOID_VARIANCE};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new_product || !map->map_add_argby_at ||
        !map->map_add_f64_at)
        return NULL;
    m = map->map_new_product(host->h, "process_pid_argstats", key_types, 1,
                             values, 2);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        double dur;
        if (e->fhash == DFTU_STR_NONE) continue;
        key[0] = (int64_t)e->pid;
        dur = (double)e->dur;
        map->map_add_argby_at(host->h, m, key, 0, dur, (int64_t)e->fhash);
        map->map_add_f64_at(host->h, m, key, 1, dur);
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
