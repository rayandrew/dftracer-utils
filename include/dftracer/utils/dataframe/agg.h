#ifndef DFTRACER_UTILS_DATAFRAME_AGG_H
#define DFTRACER_UTILS_DATAFRAME_AGG_H

#include <dftracer/utils/dataframe/series.h>

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

struct DataFrame;  // dataframe/dataframe.h; only used by value as a return type

enum class AggOp {
    Count = 0,
    Sum = 1,
    Min = 2,
    Max = 3,
    Mean = 4,
    Var = 5,    ///< sample variance
    Std = 6,    ///< sample standard deviation
    Skew = 7,   ///< population skewness
    Kurt = 8,   ///< excess (population) kurtosis
    First = 9,  ///< first non-null value in row order (order-independent merge)
    Last = 10,  ///< last non-null value in row order
    Pct = 11,   ///< a DDSketch quantile (level in AggSpec::param), mergeable
    Hist = 12,  ///< the DDSketch histogram, a list<struct{lo,hi,count}> column
    ArgMax =
        13,  ///< the String repr of `value_col` at the row maximizing `by_col`
    SumSq = 14,  ///< sum of squares (Float64), from the shared FieldStat
    SetUnion =
        15,      ///< distinct String values of `value_col`, sorted and joined
    // Occupancy (time-window reductions over the ts=`value_col`, dur=`by_col`
    // pair; `param` carries the endpoint-snap tolerance occ_cell_us): a
    // per-group +1/-1 endpoint delta-map, mergeable by key-wise add.
    Busy = 16,         ///< exact interval-union length (us) where depth > 0
    Concurrency = 17,  ///< sum(dur) / busy
    Utilization = 18,  ///< busy / (max_end - min_ts)
    Active = 19,       ///< peak overlap depth
    /// Count of non-null (present) values in `value_col`, an Int64 column.
    /// Unlike Count (the group's row count) this reads the field's
    /// FieldStat::n, so it counts only the rows where the value column was
    /// present.
    CountValid = 20
};

/// Occupancy ops take a second (dur) input through `by_col`, like ArgMax.
inline bool agg_uses_by_col(AggOp op) {
    return op == AggOp::ArgMax || op == AggOp::Busy ||
           op == AggOp::Concurrency || op == AggOp::Utilization ||
           op == AggOp::Active;
}

/// One aggregate: `op` over the value column at index `value_col` in the values
/// passed to accumulate (ignored for Count), named `out` in the result.
struct AggSpec {
    AggOp op;
    std::int32_t value_col = -1;
    std::string out;
    double param = 0.0;  ///< Pct: quantile q in [0, 1]; occupancy: occ_cell_us
    std::int32_t by_col = -1;  ///< ArgMax: the column maximized (value_col is
                               ///< the represented field); occupancy: dur
                               ///< (value_col is ts); unused otherwise
};

class AggState;  // opaque, mergeable partial group state (defined in agg.cpp)

struct AggStateDeleter {
    void operator()(AggState* p) const noexcept;
};
using AggStatePtr = std::unique_ptr<AggState, AggStateDeleter>;

/// A fresh partial for `specs`, grouping by N key columns.
AggStatePtr agg_new(std::vector<AggSpec> specs);

/// Fold rows [begin, end) of a batch (the N `keys` columns + the value columns
/// the specs reference) into `state`. The composite key is hashed and compared
/// column-by-column (no string concatenation); each key column keeps its own
/// type. `end < 0` means the whole column. Serial; one thread per state - the
/// parallel/async drivers give each chunk its own state.
void agg_accumulate(AggState& state, const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    std::int64_t begin = 0, std::int64_t end = -1);
/// Single-key convenience: forwards to the N-key form with `keys = {&key}`.
void agg_accumulate(AggState& state, const Series& key,
                    const std::vector<const Series*>& values,
                    std::int64_t begin = 0, std::int64_t end = -1);

/// Combine `other` into `into` (associative; for spill + distributed merge).
void agg_merge(AggState& into, const AggState& other);

/// Materialize the result: one key column per `key_names` (in order, each
/// keeping its original type) plus one column per spec, in spec order.
DataFrame agg_finalize(const AggState& state,
                       const std::vector<std::string>& key_names);
/// Single-key convenience: forwards to the vector form with `{key_name}`.
DataFrame agg_finalize(const AggState& state, const std::string& key_name);

/// Serialize a partial to a portable byte blob (distributed partials / spill)
/// and reconstruct it; round-trips exactly.
std::string agg_serialize(const AggState& state);
AggStatePtr agg_deserialize(const std::string& blob);

/// Number of groups currently held by `state`.
std::int64_t agg_num_groups(const AggState& state);

/// Approximate in-memory bytes held by `state` (spill trigger; not exact).
std::size_t agg_approx_bytes(const AggState& state);

/// Three-way compare of the composite key of group `ga` in `a` against group
/// `gb` in `b`. `a` and `b` must share the same key layout (same group_by
/// keys, e.g. two states built from the same specs).
int agg_key_cmp(const AggState& a, std::int64_t ga, const AggState& b,
                std::int64_t gb);

/// Sort `state`'s groups in place by ascending composite key. Used to write a
/// spill run in the key order a k-way merge needs.
void agg_sort_groups(AggState& state);

/// A fresh state holding only group `g` of `state`, sharing its specs and key
/// layout - serializable via agg_serialize and mergeable via agg_merge into
/// another state. The unit written to and read back from a spill run.
AggStatePtr agg_extract_group(const AggState& state, std::int64_t g);

/// Fused group-by: accumulate the whole batch in one pass over all specs (the
/// sync driver chunks via the parallel_for seam and merges), then finalize.
DataFrame group_agg(const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs,
                    const std::vector<std::string>& key_names);
/// Single-key convenience: forwards to the N-key form.
DataFrame group_agg(const Series& key, const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs, const std::string& key_name);

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_AGG_H
