#ifndef DFTRACER_UTILS_TRACE_COMPARATOR_COMPARE_VIEW_H
#define DFTRACER_UTILS_TRACE_COMPARATOR_COMPARE_VIEW_H

#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/core/coro/when_all.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/trace/views/result_join.h>
#include <dftracer/utils/trace/views/view.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::trace::comparator {

/**
 * @brief A comparison of two Views, expressed entirely on the View/DataFrame
 * engine.
 *
 * collect() aggregates the baseline and variant with the same group_by + agg
 * IN PARALLEL (one scan each, run concurrently), joins their result DataFrames
 * on the group key, and appends per metric `m`:
 *   - `delta_<m>`  = variant - baseline
 *   - `pct_<m>`    = 100 * (delta / baseline)
 *
 * View already owns index resolution, the parallel pruned scan, and the
 * aggregation; the delta is just dataframe column math. There is no bespoke
 * aggregator or comparison-tree machinery. Like StatsView, a CompareView is a
 * cheap description of two sources, not a materialized result.
 */
class CompareView {
   public:
    static CompareView of(views::View baseline, views::View variant) {
        return CompareView(std::move(baseline), std::move(variant));
    }

    CompareView& group_by(std::vector<views::GroupKey> keys) {
        group_by_ = std::move(keys);
        return *this;
    }
    CompareView& agg(std::vector<views::AggSpec> specs) {
        agg_ = std::move(specs);
        return *this;
    }

    /// The comparison DataFrame: the group key columns, `l_<m>`/`r_<m>` for
    /// each aggregated metric, plus `delta_<m>` and `pct_<m>`.
    coro::CoroTask<dataframe::DataFrame> collect() const {
        // Both sides aggregate concurrently: two pruned scans, one when_all.
        auto [base, variant] = co_await coro::when_all(
            baseline_.group_by(group_by_).agg(agg_).collect(),
            variant_.group_by(group_by_).agg(agg_).collect());

        dataframe::DataFrame joined = views::join_batches(
            base, variant, static_cast<std::int64_t>(group_by_.size()),
            views::JoinType::FULL);

        // Each numeric metric shows up as an l_<m>/r_<m> pair after the join
        // (the group-key columns keep their names). Derive the metrics from the
        // join output rather than the agg spec, whose out_name may be empty.
        std::vector<std::string> metrics;
        for (const std::string& name : joined.names) {
            if (name.rfind("l_", 0) != 0) continue;
            const std::string m = name.substr(2);
            if (joined.column_index("r_" + m) >= 0) metrics.push_back(m);
        }

        for (const std::string& m : metrics) {
            // The join's output columns are selections; the numeric kernels
            // need flat inputs. Skip non-numeric metrics (e.g. set_union).
            dataframe::Series l = joined.column("l_" + m).materialize();
            if (l.type() == dataframe::TypeId::String) continue;
            dataframe::Series r = joined.column("r_" + m).materialize();
            dataframe::Series delta = r.sub(l);
            dataframe::Series pct = percent_change(delta, l, joined.num_rows());
            joined = joined.with_column("delta_" + m, delta);
            joined = joined.with_column("pct_" + m, pct);
        }
        co_return joined;
    }

   private:
    CompareView(views::View baseline, views::View variant)
        : baseline_(std::move(baseline)), variant_(std::move(variant)) {}

    // 100 * (delta / baseline), columnwise in Float64; a zero baseline yields
    // inf/nan per the division, the honest signal for "new in the variant".
    static dataframe::Series percent_change(const dataframe::Series& delta,
                                            const dataframe::Series& baseline,
                                            std::int64_t n) {
        std::vector<double> hundred(static_cast<std::size_t>(n), 100.0);
        dataframe::Series scale =
            dataframe::Series::flat_f64(hundred.data(), n);
        dataframe::Series d = delta.cast(dataframe::TypeId::Float64);
        dataframe::Series b = baseline.cast(dataframe::TypeId::Float64);
        return d.div(b).mul(scale);
    }

    views::View baseline_;
    views::View variant_;
    std::vector<views::GroupKey> group_by_;
    std::vector<views::AggSpec> agg_;
};

}  // namespace dftracer::utils::trace::comparator

#endif  // DFTRACER_UTILS_TRACE_COMPARATOR_COMPARE_VIEW_H
