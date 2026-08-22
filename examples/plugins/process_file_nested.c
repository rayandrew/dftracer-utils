/* Example dftracer-utils plugin in pure C: per process, a nested-preserved map
 * of file -> access count. Each file event contributes 1 at outer key {pid},
 * inner key {fhash}; the host merges across workers and materializes it to ONE
 * Arrow row per pid: a "k0" (pid) column plus a nested "value" column of
 * list<struct<ik0, value>> holding that pid's per-file counts. PluginHost::run
 * returns it to Python as a pyarrow.Table under "process_file_nested".
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_file_nested.so process_file_nested.c
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
    static const dftu_type outer_types[1] = {DFTU_T_I64};
    static const dftu_type inner_types[1] = {DFTU_T_STR};
    static const dftu_monoid_kind values[1] = {DFTU_MONOID_COUNTER};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new_nested) return NULL;
    m = map->map_new_nested(host->h, "process_file_nested", outer_types, 1,
                            inner_types, 1, values, 1);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        int64_t outer[1];
        int64_t inner[1];
        if (e->fhash == DFTU_STR_NONE) continue;
        outer[0] = (int64_t)e->pid;
        inner[0] = (int64_t)e->fhash;
        map->map_add_nested_u64(host->h, m, outer, inner, 0, 1);
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
