#ifndef DFTRACER_UTILS_PLUGINS_ARROW_ABI_H
#define DFTRACER_UTILS_PLUGINS_ARROW_ABI_H

/* The standard Arrow C Data Interface (stable across Arrow versions). Guarded
   by ARROW_C_DATA_INTERFACE so it coexists with a real arrow header. */

#ifdef __cplusplus
extern "C" {
#endif

#ifndef ARROW_C_DATA_INTERFACE
#define ARROW_C_DATA_INTERFACE

#include <stdint.h>

#define ARROW_FLAG_DICTIONARY_ORDERED 1
#define ARROW_FLAG_NULLABLE 2
#define ARROW_FLAG_MAP_KEYS_SORTED 4

struct ArrowSchema {
    const char* format;
    const char* name;
    const char* metadata;
    int64_t flags;
    int64_t n_children;
    struct ArrowSchema** children;
    struct ArrowSchema* dictionary;
    void (*release)(struct ArrowSchema*);
    void* private_data;
};

struct ArrowArray {
    int64_t length;
    int64_t null_count;
    int64_t offset;
    int64_t n_buffers;
    int64_t n_children;
    const void** buffers;
    struct ArrowArray** children;
    struct ArrowArray* dictionary;
    void (*release)(struct ArrowArray*);
    void* private_data;
};

#endif /* ARROW_C_DATA_INTERFACE */

#ifdef __cplusplus
}
#endif

#endif /* DFTRACER_UTILS_PLUGINS_ARROW_ABI_H */
