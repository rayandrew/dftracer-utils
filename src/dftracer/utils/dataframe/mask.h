#ifndef DFTRACER_UTILS_DATAFRAME_MASK_H
#define DFTRACER_UTILS_DATAFRAME_MASK_H

#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/query/ast.h>

// The post-materialization predicate backend: evaluate a query AST against an
// in-memory columnar batch as a SIMD boolean mask. One of the query language's
// physical backends (the scan-time evaluator in `query` matches events during a
// scan; this matches an already-materialized dataframe::DataFrame), living in
// `dataframe` so the `query` language stays independent of the dataframe
// engine.
namespace dftracer::utils::dataframe {

/// Evaluate `node` over `batch`, returning a bit-packed Bool mask column
/// (length
/// == batch row count). Lowers the columnar subset to dataframe kernels:
/// numeric compares, string ==/!=, in / not-in (as OR-of-equals), and
/// and/or/not. Throws DFTUtilsException{query::QueryErrc::Unsupported} for
/// predicates with no columnar lowering (pattern match, ordered string compare,
/// a field absent from the batch) - the caller should use the scan-time
/// evaluator for those.
dftracer::utils::dataframe::Series evaluate_mask(
    const dftracer::utils::query::QueryNode& node,
    const dftracer::utils::dataframe::DataFrame& batch);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_MASK_H
