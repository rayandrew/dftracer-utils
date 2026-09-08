/* Example producer plugin: a ZERO-KEY dft.ext.agg accumulator, the AggState
 * form of a scalar handle. Each batch arrives as a column dataframe and is
 * folded whole into one accumulator named com.example.dur_stats, computing the
 * whole-scan event count and the duration median, min and max. A consumer
 * reads it back by that name. The host merges the accumulator across
 * every worker slice, so a consumer reads stats no single slice ever saw.
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o dur_stats_producer.so dur_stats_producer.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdlib.h>
#include <string.h>

#define DUR_STATS_PORT "com.example.dur_stats"

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) {
    (void)self;
    return calloc(1, 1);
}

static dftu_task* on_batch_columns(void* slice, const dftu_dataframe* df,
                                   const dftu_host* host) {
    const dftu_ext_agg* agg =
        (const dftu_ext_agg*)host->get_extension(host->h, DFTU_EXT_AGG);
    static const dftu_agg_col specs[4] = {
        {DFTU_AGG_COUNT, NULL, "count", 0.0, NULL},
        {DFTU_AGG_PCT, "dur", "p50", 0.5, NULL},
        {DFTU_AGG_MIN, "dur", "min", 0.0, NULL},
        {DFTU_AGG_MAX, "dur", "max", 0.0, NULL}};
    dftu_agg* a;
    (void)slice;
    if (!agg || !agg->agg_new) return NULL;
    /* No key columns: the whole scan is one group, i.e. a scalar handle. */
    a = agg->agg_new(host->h, DUR_STATS_PORT, NULL, 0, specs, 4);
    if (a) agg->agg_accumulate(host->h, a, df);
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
    g_plugin.on_batch = NULL;
    g_plugin.merge = merge;
    g_plugin.on_finalize = on_finalize;
    g_plugin.destroy_slice = destroy_slice;
    g_plugin.destroy = destroy;
    g_plugin.on_batch_columns = on_batch_columns;
    return &g_plugin;
}
