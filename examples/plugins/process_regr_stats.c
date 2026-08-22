/* Example dftracer-utils plugin in pure C: per process, the linear fit of event
 * duration (y, dependent) on timestamp (x, independent) and the shape of the
 * duration distribution. Each event contributes to a host-owned mergeable
 * product map at key {pid} whose five value components are: 0: REGR_SLOPE and
 * 1: REGR_INTERCEPT of the ordinary-least-squares fit y = slope*x + intercept;
 * 2: CORR, the Pearson correlation of x and y; 3: SKEWNESS and 4: KURTOSIS
 * (population, excess) of the durations. The host merges the map across workers
 * (co-moment power sums and FieldStat power sums both field-add) and
 * materializes it to an Arrow table [k0 : int64 (pid), v0..v4 : float64],
 * returned to Python as a pyarrow.Table under "process_regr_stats".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_regr_stats.so process_regr_stats.c
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
    static const dftu_monoid_kind values[5] = {
        DFTU_MONOID_REGR_SLOPE, DFTU_MONOID_REGR_INTERCEPT, DFTU_MONOID_CORR,
        DFTU_MONOID_SKEWNESS, DFTU_MONOID_KURTOSIS};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new_product || !map->map_add_xy_at ||
        !map->map_add_f64_at)
        return NULL;
    m = map->map_new_product(host->h, "process_regr_stats", key_types, 1,
                             values, 5);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        double x, y;
        if (!e->has_dur) continue;
        key[0] = (int64_t)e->pid;
        x = (double)e->ts;
        y = (double)e->dur;
        map->map_add_xy_at(host->h, m, key, 0, x, y);
        map->map_add_xy_at(host->h, m, key, 1, x, y);
        map->map_add_xy_at(host->h, m, key, 2, x, y);
        map->map_add_f64_at(host->h, m, key, 3, y);
        map->map_add_f64_at(host->h, m, key, 4, y);
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
