#ifndef DFTRACER_UTILS_DATAFRAME_AGG_H
#define DFTRACER_UTILS_DATAFRAME_AGG_H

#include <dftracer/utils/dataframe/dataframe.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// The aggregation engine: a mergeable partial group state (`AggState`) with
// synchronous accumulate / merge / finalize. This is the shared kernel both the
// View (async coroutine driver over scanned batches) and the
// post-materialization path (sync parallel_reduce over column chunks) delegate
// to - one aggregation logic, two drivers. accumulate touches one state on one
// thread (no locks); parallelism is the driver's job (per-worker or per-chunk
// partials, then merge).
namespace dftracer::utils::dataframe {

enum class AggOp {
    Count = 0,
    Sum = 1,
    Min = 2,
    Max = 3,
    Mean = 4,
    Var = 5,   ///< sample variance
    Std = 6,   ///< sample standard deviation
    Skew = 7,  ///< population skewness
    Kurt = 8   ///< excess (population) kurtosis
};

/// One aggregate: `op` over the value column at index `value_col` in the values
/// passed to accumulate (ignored for Count), named `out` in the result.
struct AggSpec {
    AggOp op;
    std::int32_t value_col = -1;
    std::string out;
};

class AggState;  // opaque, mergeable partial group state (defined in agg.cpp)

struct AggStateDeleter {
    void operator()(AggState* p) const noexcept;
};
using AggStatePtr = std::unique_ptr<AggState, AggStateDeleter>;

/// A fresh partial for `specs`, grouping by a single key column.
AggStatePtr agg_new(std::vector<AggSpec> specs);

/// Fold rows [begin, end) of a batch (one key column + the value columns the
/// specs reference) into `state`. `end < 0` means the whole column. Serial; one
/// thread per state - the parallel/async drivers give each chunk its own state.
void agg_accumulate(AggState& state, const Series& key,
                    const std::vector<const Series*>& values,
                    std::int64_t begin = 0, std::int64_t end = -1);

/// Combine `other` into `into` (associative; for spill + distributed merge).
void agg_merge(AggState& into, const AggState& other);

/// Materialize the result: the key column (named `key_name`) plus one column
/// per spec, in spec order.
DataFrame agg_finalize(const AggState& state, const std::string& key_name);

/// Serialize a partial to a portable byte blob (distributed partials / spill)
/// and reconstruct it; round-trips exactly.
std::string agg_serialize(const AggState& state);
AggStatePtr agg_deserialize(const std::string& blob);

/// Fused group-by: accumulate the whole batch in one pass over all specs (the
/// sync driver chunks via the parallel_for seam and merges), then finalize.
DataFrame group_agg(const Series& key, const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs, const std::string& key_name);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_AGG_H
