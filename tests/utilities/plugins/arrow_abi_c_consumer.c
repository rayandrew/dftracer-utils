/* A real C consumer of the dftu_ext_arrow ABI: convert an event batch into an
 * Arrow record batch through the raw vtable, read its row count, and release it
 * - the Arrow C Data Interface structs are plain C. */
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/arrow_abi.h>
#include <stdint.h>

/* Convert `b` to Arrow and return the row count (out->length); -1 on failure.
 * Releases the produced array and schema. */
int64_t dftu_test_arrow_batch(const dftu_host* h, const dftu_batch* b) {
    const dftu_ext_arrow* ar =
        (const dftu_ext_arrow*)h->get_extension(h->h, DFTU_EXT_ARROW);
    if (!ar || !ar->batch_to_arrow) return -1;
    struct ArrowArray arr;
    struct ArrowSchema sch;
    if (ar->batch_to_arrow(h->h, b, &arr, &sch) != 0) return -1;
    int64_t n = arr.length;
    if (arr.release) arr.release(&arr);
    if (sch.release) sch.release(&sch);
    return n;
}
