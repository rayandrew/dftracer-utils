#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGGREGATE_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGGREGATE_H

#include <ankerl/unordered_dense.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/field_stat.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/trace/views/view_plan.h>
#include <dftracer/utils/trace/views/view_resolver.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>
#include <simdjson.h>

#include <cstdint>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

// The group-by/aggregate core shared by every terminal that folds events into
// groups (collect, counters, distributed partials, spill). Pure functions over
// a GroupMap: fold events/cells in, merge partials, and materialize a table.
namespace dftracer::utils::trace::views::detail {

// Unit separator between composite group-key parts (never appears in field
// values, so it round-trips a multi-column key as one map key).
inline constexpr char GROUP_SEP = '\x1f';

// Separator joining a SetUnion group's distinct values into one text-column
// cell; a control char so it does not collide with categorical values.
inline constexpr char SET_SEP = '\x1e';

// The per-field aggregation atom (shared with the aggregation tier). See
// vec/field_stat.h.
using dftracer::utils::dataframe::FieldStat;

// ArgMax(field, by) representative: the `field` text at the group's max `by`.
struct ArgMaxState {
    double by = 0;
    std::string repr;
    bool has = false;
};

// One group's running aggregate. `fields` aligns to AggSchema.fields (the
// distinct agg fields), `argmax` to AggSchema's ArgMax slots; both stay empty
// when unused. `dyn` holds the auto-discovered numeric args (agg_numeric_args).
struct AggAccum {
    std::vector<std::string> keys;  // aligned to plan.group_by (+ time bucket)
    std::uint64_t count = 0;        // group rows (mean/var denominator)
    std::vector<FieldStat> fields;
    std::vector<ArgMaxState> argmax;
    // Per-field DDSketch for percentile/histogram aggs, aligned to
    // AggSchema.field_sketch slots; empty unless a Pct op is present.
    std::vector<utilities::common::statistics::DDSketch> sketches;
    // Distinct string values per SetUnion slot, aligned to AggSchema.spec_set
    // (which maps into these). Sorted set = deterministic joined output.
    std::vector<std::set<std::string>> sets;
    std::map<std::string, FieldStat> dyn;  // sorted for a stable column union
    // Per-arg quantile sketch, keyed like `dyn`; populated only when a dyn Pct
    // reduction is requested (AggSchema.dyn_sketch), so the common counter path
    // pays nothing for it.
    std::map<std::string, utilities::common::statistics::DDSketch> dyn_sketches;
    // Occupancy (the time-window reduction): a sparse endpoint delta-map per
    // group (deltas[s] += 1, deltas[s+dur] -= 1 per event). Finalize sorts the
    // keys and sweeps a running depth: busy = union length where depth > 0
    // (exact), active = max depth. Mergeable by key-wise add.
    ankerl::unordered_dense::map<std::uint64_t, std::int64_t> occ_deltas;
    std::uint64_t occ_cell_us = 0;  // quantization tolerance applied; 0 = exact
    std::uint64_t occ_total = 0;    // sum(dur), for concurrency
    std::uint64_t occ_ts = (std::numeric_limits<std::uint64_t>::max)();
    std::uint64_t occ_te = 0;
};

using GroupMap = ankerl::unordered_dense::map<std::string, AggAccum>;

// Query-derived fold schema: the distinct agg fields and how each AggSpec maps
// onto them. A pure function of the plan, built once per terminal (cached on
// ViewPlan) so the per-event fold touches each field once, not once per spec.
struct AggSchema {
    std::vector<std::string> fields;  // distinct fields to accumulate
    std::vector<int> spec_field;      // plan.agg[i] -> fields index, or -1
    std::vector<int> spec_argmax;     // plan.agg[i] -> argmax slot, or -1
    std::size_t argmax_count = 0;
    std::vector<bool> field_scaled;   // fields[i] is ts/dur/te (time_scale)
    // fields[i] -> DDSketch slot for percentile/histogram aggs, or -1.
    std::vector<int> field_sketch;
    std::size_t sketch_count = 0;
    std::vector<int> spec_set;      // plan.agg[i] -> SetUnion slot, or -1
    std::size_t set_count = 0;
    bool want_occupancy = false;    // any busy/concurrency/utilization spec
    std::uint64_t occ_cell_us = 0;  // optional exact-union tolerance; 0 = exact
    // A per-arg DDSketch is collected only when a numeric_arg_aggs reduction
    // needs quantiles (Pct); otherwise the dyn FieldStat alone serves the
    // reductions and no sketch is allocated.
    bool dyn_sketch = false;
};

// Build the fold schema; ensure_schema memoizes it on the plan. Call
// ensure_schema at a terminal's single-threaded entry, before the parallel
// fold, so worker reads of plan.schema never race the build.
AggSchema make_agg_schema(const ViewPlan& plan);
const AggSchema& ensure_schema(const ViewPlan& plan);

// Finalize one non-ArgMax agg spec from a group's stats (the value column).
double finalize_value(const AggAccum& a, const ViewPlan& plan, std::size_t i);

// Finalize a Hist agg spec: the group's raw histogram buckets (empty if the
// field had no sketch).
std::vector<utilities::common::statistics::HistogramBin> finalize_hist(
    const AggAccum& a, const ViewPlan& plan, std::size_t i);

// simdjson value coercion helpers.
std::optional<double> to_number(simdjson::dom::element e);
std::string to_str(simdjson::dom::element e);
std::string top_or_args_str(simdjson::dom::element root,
                            const std::string& key);
std::optional<double> agg_field(simdjson::dom::element root,
                                const std::string& field);

// Fold one decoded event into `map`. `keybuf` is a caller-owned scratch string
// reused across events, so the hot path allocates no per-event key strings.
void fold_event(GroupMap& map, simdjson::dom::element root,
                const ViewPlan& plan, std::string& keybuf);

// A group_by + agg request a PartialSource may answer from a materialized
// aggregate (per-chunk stats, a summary mipmap, the aggregation tier, the
// viewcache) instead of decoding events. `schema` is the field layout the
// source fills AggAccum.fields against.
struct PartialRequest {
    std::vector<ViewFile> files;
    const AggSchema* schema = nullptr;
    std::vector<GroupKey> group_by;
    std::uint64_t time_bucket_us = 0;
    std::string agg_field;  // the single reduced field, or empty
    bool needs_argmax = false;
    std::string argmax_value;
    std::string argmax_by;
    bool has_window = false;
    double begin = 0;  // microseconds
    double end = 0;
};

// Pluggable materialized-aggregate source. lookup() emits each AggAccum partial
// it can answer (at its own grain) and returns the chunks it fully covered, so
// the executor merges the partials with merge_accum and scans only the
// remainder. handled=false declines entirely (scan everything).
class PartialSource {
   public:
    virtual ~PartialSource() = default;
    struct Result {
        std::vector<std::pair<std::string, std::uint64_t>> covered_chunks;
        bool handled = false;
    };
    virtual Result lookup(
        const PartialRequest& req,
        const std::function<void(AggAccum&&)>& emit) const = 0;
};

// Index of `field` in schema.fields, or -1; a source uses it to pick the
// AggAccum.fields slot to fill.
int schema_field_index(const AggSchema& s, const std::string& field);

// Merge `sa` into `da` (same group key); associative, so it serves per-unit
// partials and the k-way merge of spilled runs alike.
void merge_accum(AggAccum& da, const AggAccum& sa, const ViewPlan& plan);
void merge_maps(GroupMap& dst, const GroupMap& src, const ViewPlan& plan);

// Post-aggregation re-key: relabel FileName/HostName group columns to the
// resolved name (a bijection, so the fold groups on the hash) and merge groups
// that now collide. No-op when the plan has no resolved-name key. Runs on
// distinct groups, so the resolver never touches events.
void resolve_group_keys(GroupMap& map, const ViewPlan& plan);

// Lazily build (and cache on the plan) the index-backed name resolver; null
// when the plan has no resolved-name group key.
const GroupResolver* ensure_resolver(const ViewPlan& plan);

// Feed pid -> rank harvested from PR metadata into the plan's resolver so the
// Rank group key relabels pid groups post-aggregation. Empties `ranks`.
void apply_ranks(const ViewPlan& plan,
                 std::unordered_map<std::uint64_t, std::string>& ranks);

// Resolve one group value from its stored hash to the name for `kind`
// (FilePath/FileName/HostName); returns `hash` unchanged for other kinds.
std::string resolve_group_value(const GroupResolver& r, GroupKey::Kind kind,
                                const std::string& hash);

// Apply `gk`'s value transform to an already-resolved group value. Coarsens
// the grain, so both the scan and the tier must call it or the same query
// answers differently depending on which path served it.
std::string apply_group_transform(const GroupKey& gk, std::string v);

// Aggregate-source fast-path helpers: the single reduced field ({_, false} if
// the aggs mix fields) and the ArgMax spec if any.
std::pair<std::string, bool> source_agg_field(const ViewPlan& plan);
const AggSpec* find_argmax(const ViewPlan& plan);

// Column names and the final materialized result.
std::string group_col_name(const GroupKey& gk);
std::string agg_col_name(const AggSpec& spec);

// Materialize the group map straight into a columnar dataframe::DataFrame
// (count as Int64, integer Sum/Min/Max exact as Int64/Uint64, other value
// columns Float64, text as String, hist as list<struct>).
dftracer::utils::dataframe::DataFrame to_batch(const GroupMap& map,
                                               const ViewPlan& plan);

// Drop every column whose name is not in `select`, preserving result order. A
// no-op when `select` is empty or names a column that is not present. Apply
// only at the final collect() boundary, never to an intermediate that will be
// merged further.
void project_columns(dftracer::utils::dataframe::DataFrame& batch,
                     const std::vector<std::string>& select);

// Post-aggregation finalize: to_batch, then the plan's sort_by/topk/
// offset+limit/select. Apply only at the final result boundary, after all
// merges. Shared by View::collect() and the ViewSession collect branches.
dftracer::utils::dataframe::DataFrame finalize_collect_batch(
    const GroupMap& map, const ViewPlan& plan);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_AGGREGATE_H
