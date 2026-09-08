#ifndef DFTRACER_UTILS_PLUGINS_ABI_ARROW_H
#define DFTRACER_UTILS_PLUGINS_ABI_ARROW_H

/** @file
 * dftu.ext.arrow: Arrow IPC file I/O lent to a plugin. Optional service
 * group, fetched via dftu_host::get_extension(DFTU_EXT_ARROW). Include
 * dftracer/utils/plugins/abi.h rather than this file directly.
 */

#include <dftracer/utils/plugins/abi/core.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DFTU_EXT_ARROW "dftu.ext.arrow@1"

typedef struct dftu_ext_arrow {
    /** Write a plugin-provided Arrow batch to an IPC file. 0 ok, -1 on error.
     */
    int (*arrow_write_ipc)(void* h, struct ArrowArray* a, struct ArrowSchema* s,
                           const char* path);
    /** Read the first record batch of an Arrow IPC file; the plugin owns *out
       and *out_schema and must call their release. 0 ok, -1 on error. */
    int (*arrow_read_ipc)(void* h, const char* path, struct ArrowArray* out,
                          struct ArrowSchema* out_schema);
} dftu_ext_arrow;

#ifdef __cplusplus
}
#endif

#endif /* DFTRACER_UTILS_PLUGINS_ABI_ARROW_H */
