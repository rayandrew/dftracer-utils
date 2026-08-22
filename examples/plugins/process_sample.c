/* Example dftracer-utils plugin in pure C: a deterministic sample of up to 5
 * FILES each process touched. Each event contributes item=e->fhash to a
 * SAMPLE_STR value at key {pid} with k=5; the monoid keeps the five distinct
 * fhash ids with the smallest hash(item) - a bottom-k (KMV / min-hash) sample,
 * NOT an Algorithm-R reservoir. Bottom-k is deterministic (a stable hash makes
 * the kept set a pure function of the input) and mergeable (union then keep the
 * five smallest), so the parallel fold is reproducible. The host merges the
 * sample across workers and materializes it to an Arrow table [k0 : int64
 * (pid), value : list<string> (up to five file labels, sorted by label)].
 * PluginHost::run returns it to Python as a pyarrow.Table under
 * "process_sample".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_sample.so process_sample.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>
#include <stdlib.h>

#define SAMPLE_K 5u

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
    if (!map || !map->map_new || !map->map_add_sample_at) return NULL;
    m = map->map_new(host->h, "process_sample", key_types, 1,
                     DFTU_MONOID_SAMPLE_STR);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        if (e->fhash == DFTU_STR_NONE) continue;
        key[0] = (int64_t)e->pid;
        /* k is passed on every add; the monoid records it and stays bounded. */
        map->map_add_sample_at(host->h, m, key, 0, SAMPLE_K, (int64_t)e->fhash);
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
