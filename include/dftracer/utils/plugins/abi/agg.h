#ifndef DFTRACER_UTILS_PLUGINS_ABI_AGG_H
#define DFTRACER_UTILS_PLUGINS_ABI_AGG_H

/** @file
 * dftu.ext.agg: the host-owned cross-batch aggregation accumulator lent to a
 * plugin. Optional service group, fetched via
 * dftu_host::get_extension(DFTU_EXT_AGG). Include
 * dftracer/utils/plugins/abi.h rather than this file directly.
 */

#include <dftracer/utils/dataframe/agg_op_codes.h> /* DFTU_AGG_* for dftu_agg_col */
#include <dftracer/utils/plugins/abi/core.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DFTU_EXT_AGG "dftu.ext.agg@1"

/** Host-owned cross-batch aggregation accumulator. This is the ONE
   accumulator a plugin gets: it wraps the dataframe engine's mergeable
   AggState, so a keyed map is an accumulator with key columns, a scalar
   handle is one with zero key columns, and the reduction vocabulary is the
   engine's agg op table. The host merges same-named accumulators across
   worker slices and finalizes each at scan end to a native dataframe (the key
   columns, then one column per aggregate), returned to run() under `name`. */
typedef struct dftu_agg dftu_agg;

/** One aggregate for a dftu_ext_agg accumulator. `op` is a DFTU_AGG_* code (the
   dftu_agg_op enum, kept a fixed-width int at the seam for ABI stability); a
   code outside the DFTU_AGG_* range makes agg_new return NULL. `value` names
   the value column in each accumulated batch dataframe (NULL for
   DFTU_AGG_COUNT, the group row count); `out` names the result column; `param`
   is the op's scalar parameter (a quantile in [0,1] for DFTU_AGG_PCT, k for
   DFTU_AGG_TOPK and friends, 0 otherwise); `by` names the op's second input
   column (the ordering column for DFTU_AGG_ARGMAX/ARGMIN/TOPK, the dur column
   for the occupancy ops, x for the co-moment ops; NULL when the op takes one
   input). All names are borrowed for the agg_new call only. */
typedef struct dftu_agg_col {
    int32_t op;
    const char* value;
    const char* out;
    double param;
    const char* by;
} dftu_agg_col;

typedef struct dftu_ext_agg {
    /** Get-or-create a named accumulator grouping by the `key_n` columns named
       in `key_names` and computing each of `spec_n` aggregates. Returns a
       stable handle owned by the host (freed at fold teardown, never by the
       plugin); NULL on a bad op code, a missing output name, or allocation
       failure, or a `name` in the host's own "dftu." namespace, which is
       refused to plugins. A name seen before returns the existing handle and
       ignores the new spec. Safe from any slice thread on that slice's host. */
    dftu_agg* (*agg_new)(void* h, const char* name,
                         const char* const* key_names, uint32_t key_n,
                         const dftu_agg_col* specs, uint32_t spec_n);
    /** Fold one batch into `a`: each key and value column is looked up by name
       in `df` (host-owned, borrowed for the call) and accumulated. A batch
       missing any referenced column is skipped. Serial per accumulator; one
       slice's accumulator is touched by one thread. */
    void (*agg_accumulate)(void* h, dftu_agg* a, const dftu_dataframe* df);
    /** The cross-worker-merged, finalized result of the accumulator named
       `name` - any plugin's, which is how one plugin reads another's whole-scan
       aggregate. Call at on_finalize; the host finalizes the producer first,
       from the plugins' declared dftu_plugin::provides / dftu_plugin::consumes
       rather than registration order. Returns a NEW owned dataframe the caller
       frees with dftu_dataframe_free, or NULL if the producer created no such
       accumulator (an empty scan). */
    dftu_dataframe* (*agg_result)(void* h, const char* name);
} dftu_ext_agg;

#ifdef __cplusplus
}
#endif

#endif /* DFTRACER_UTILS_PLUGINS_ABI_AGG_H */
