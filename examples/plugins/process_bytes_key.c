/* Example dftracer-utils plugin in pure C: a count map keyed on a BYTES
 * component - an opaque, composite byte blob rather than an int or a string.
 *
 * Each event is keyed by a small composite: the 8-byte little-endian pid
 * followed by the 8-byte little-endian tid, packed into one 16-byte buffer.
 * The plugin interns that buffer via host->intern(ptr, len) - intern is
 * length-based and byte-safe, so an embedded NUL is preserved - and passes the
 * returned dftu_str id in the int64 key slot, exactly like a STR key. The host
 * resolves the id back to its raw bytes at materialization, so the column is
 * Arrow binary (format "z"), not utf8. The result table is
 * [k0 : binary (pid|tid blob), value : int64] under "process_bytes_key".
 *
 * A BYTES key rides the same interned-id path as a STR key; the only
 * difference is the materialized column type (binary vs utf8).
 *
 * Build: cc -std=c99 -shared -fPIC -I<repo>/include \
 *           -o process_bytes_key.so process_bytes_key.c
 */

#include <dftracer/utils/plugins/abi.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static uint32_t needs(void* self) {
    (void)self;
    return 0;
}

static void* make_slice(void* self) {
    (void)self;
    return calloc(1, 1);
}

static void put_le64(unsigned char* p, uint64_t v) {
    int i;
    for (i = 0; i < 8; ++i) p[i] = (unsigned char)((v >> (8 * i)) & 0xFFu);
}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                           const dftu_host* host) {
    const dftu_ext_map* map =
        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);
    static const dftu_type key_types[1] = {DFTU_T_BYTES};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new) return NULL;
    m = map->map_new(host->h, "process_bytes_key", key_types, 1,
                     DFTU_MONOID_COUNTER);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {
        const dftu_event* e = &b->events[i];
        unsigned char blob[16];
        dftu_str id;
        int64_t key[1];
        put_le64(blob, e->pid);
        put_le64(blob + 8, e->tid);
        id = host->intern(host->h, (const char*)blob, (uint32_t)sizeof(blob));
        if (id == DFTU_STR_NONE) continue;
        key[0] = (int64_t)id;
        map->map_add_u64(host->h, m, key, 1);
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
