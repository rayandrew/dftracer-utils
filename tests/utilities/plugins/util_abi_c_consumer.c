/* A real C consumer of the dftu_ext_util ABI: invoke a registered host utility
 * (fnv1a) through its generated C wrapper, which resolves the utility by id and
 * marshals the call - the same path a C plugin uses. */
#include <dftracer/utils/plugins/dftu_generated_utilities.h>
#include <stdint.h>

/* Hash `data` with the host's fnv1a utility; 0 ok, -1 on failure. */
int dftu_test_util_fnv1a(const dftu_host* h, const char* data, uint32_t len,
                         uint64_t* out) {
    dftu_bytes in;
    in.ptr = data;
    in.len = len;
    uint64_t result = 0;
    if (dftu_util_fnv1a(h, &in, &result) != 0) return -1;
    *out = result;
    return 0;
}

/* The util registry is reachable by id, and find_by_id exposes the metadata. */
int dftu_test_util_find(const dftu_host* h) {
    const dftu_ext_util* u =
        (const dftu_ext_util*)h->get_extension(h->h, DFTU_EXT_UTIL);
    if (!u || !u->find_by_id) return -1;
    const dftu_utility* util = u->find_by_id(h->h, (uint32_t)DFTU_UTIL_FNV1A);
    return (util && util->run) ? 1 : 0;
}
