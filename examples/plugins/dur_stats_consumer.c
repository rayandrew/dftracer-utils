/* Example consumer plugin: optionally requires com.example.dur_stats>=1.0 and
 * in resolve() learns whether the producer is present. At finalize it asks
 * dft.ext.agg for the producer's cross-worker-merged accumulator by name; the
 * host hands back the finalized result as a one-row dataframe, which this
 * plugin reads with the dataframe C ABI and logs (WIRED). With no producer
 * registered the name is unknown and it logs the standalone fallback.
 *
 * The producer must be registered before this plugin: agg_result only sees an
 * accumulator whose owning fold has already finalized.
 *
 * Build: cc -std=c11 -shared -fPIC -I<repo>/include \
 *           -o dur_stats_consumer.so dur_stats_consumer.c \
 *           -ldftracer_utils_dataframe
 */

#include <dftracer/utils/plugins/abi.h>
/* Reading a result frame needs the engine's column ops, so this example links
 * the dataframe C ABI; the other examples link nothing. */
#include <dftracer/utils/dataframe/abi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DUR_STATS_CAP "com.example.dur_stats"

typedef struct {
    int wired;
} Cfg;

static Cfg g_cfg;

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) { return self; }

static void merge(void* into, void* other) {
    (void)into;
    (void)other;
}

/* Row 0 of `name` as a double; *ok stays untouched on success and is cleared
 * when the column is absent or of an unhandled type. */
static double cell(const dftu_dataframe* df, const char* name, int* ok) {
    dftu_series* col = dftu_dataframe_column(df, name);
    dftu_series* flat;
    const void* data;
    double out = 0.0;
    if (!col) {
        *ok = 0;
        return 0.0;
    }
    flat = dftu_series_materialize(col);
    dftu_series_free(col);
    if (!flat) {
        *ok = 0;
        return 0.0;
    }
    data = dftu_series_data(flat);
    if (!data || dftu_series_length(flat) < 1) {
        *ok = 0;
    } else {
        switch (dftu_series_type(flat)) {
            case DFTU_TYPE_INT64:
                out = (double)((const int64_t*)data)[0];
                break;
            case DFTU_TYPE_UINT64:
                out = (double)((const uint64_t*)data)[0];
                break;
            case DFTU_TYPE_FLOAT64:
                out = ((const double*)data)[0];
                break;
            default:
                *ok = 0;
                break;
        }
    }
    dftu_series_free(flat);
    return out;
}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {
    const Cfg* cfg = (const Cfg*)slice;
    const dftu_ext_agg* agg =
        (const dftu_ext_agg*)host->get_extension(host->h, DFTU_EXT_AGG);
    dftu_dataframe* res = NULL;
    char line[192];
    int n = 0;
    if (cfg && cfg->wired && agg && agg->agg_result)
        res = agg->agg_result(host->h, DUR_STATS_CAP);
    if (res) {
        int ok = 1;
        double count = cell(res, "count", &ok);
        double p50 = cell(res, "p50", &ok);
        double lo = cell(res, "min", &ok);
        double hi = cell(res, "max", &ok);
        if (ok)
            n = snprintf(line, sizeof line,
                         "dur_stats_consumer: WIRED count=%llu p50=%.3f "
                         "min=%.3f max=%.3f",
                         (unsigned long long)count, p50, lo, hi);
        dftu_dataframe_free(res);
    }
    if (n == 0)
        n = snprintf(line, sizeof line,
                     "dur_stats_consumer: STANDALONE no producer");
    if (n > 0) host->log(host->h, DFTU_LOG_INFO, line, (uint32_t)n);
    return NULL;
}

static void destroy_slice(void* slice) { (void)slice; }
static void destroy(void* self) { (void)self; }

static dftu_requirement dur_stats_requirement(void) {
    dftu_requirement r;
    r.id = DUR_STATS_CAP;
    r.op = DFTU_VER_GE;
    r.ver.major = 1;
    r.ver.minor = 0;
    r.ver.patch = 0;
    r.required = 0;
    return r;
}

static uint32_t require_caps(void* self, dftu_requirement* out, uint32_t max) {
    (void)self;
    if (max >= 1) out[0] = dur_stats_requirement();
    return 1;
}

static void resolve(void* self, const dftu_host* host) {
    Cfg* cfg = (Cfg*)self;
    const dftu_ext_comms* c =
        (const dftu_ext_comms*)host->get_extension(host->h, DFTU_EXT_COMMS);
    dftu_requirement req = dur_stats_requirement();
    dftu_version v;
    cfg->wired = c && c->provider_best(host->h, &req, &v) == 0;
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
    g_cfg.wired = 0;
    g_plugin.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    g_plugin.self = &g_cfg;
    g_plugin.needs = needs;
    g_plugin.plan_query = NULL;
    g_plugin.make_slice = make_slice;
    g_plugin.on_batch = NULL;
    g_plugin.merge = merge;
    g_plugin.on_finalize = on_finalize;
    g_plugin.destroy_slice = destroy_slice;
    g_plugin.destroy = destroy;
    g_plugin.get_extension = get_extension;
    return &g_plugin;
}
