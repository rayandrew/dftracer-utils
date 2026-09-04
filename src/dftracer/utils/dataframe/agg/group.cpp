#include <dftracer/utils/dataframe/agg/detail.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/parallel.h>  // parallel_for

#include <cstdint>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

namespace {
constexpr std::int64_t AGG_GRAIN = 1 << 16;
}  // namespace

AggStatePtr group_agg_state(const std::vector<const Series*>& keys,
                            const std::vector<const Series*>& values,
                            std::vector<AggSpec> specs) {
    const std::int64_t n = keys.empty() ? 0 : keys[0]->length();
    if (n <= AGG_GRAIN) {
        auto st = agg_new(std::move(specs));
        agg_accumulate(*st, keys, values);
        return st;
    }
    // Parallel driver: each chunk folds into its own partial (no locks), then
    // the partials merge. parallel_for runs serial when no backend is
    // installed.
    const std::int64_t chunks = (n + AGG_GRAIN - 1) / AGG_GRAIN;
    std::vector<AggStatePtr> partials(static_cast<std::size_t>(chunks));
    parallel_for(n, AGG_GRAIN, [&](std::int64_t b, std::int64_t e) {
        auto st = agg_new(specs);
        agg_accumulate(*st, keys, values, b, e);
        partials[static_cast<std::size_t>(b / AGG_GRAIN)] = std::move(st);
    });
    auto acc = agg_new(std::move(specs));
    for (auto& p : partials)
        if (p) agg_merge(*acc, *p);
    return acc;
}

DataFrame group_agg(const std::vector<const Series*>& keys,
                    const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs,
                    const std::vector<std::string>& key_names) {
    return agg_finalize(*group_agg_state(keys, values, std::move(specs)),
                        key_names);
}

DataFrame group_agg(const Series& key, const std::vector<const Series*>& values,
                    std::vector<AggSpec> specs, const std::string& key_name) {
    const std::vector<const Series*> keys{&key};
    return group_agg(keys, values, std::move(specs),
                     std::vector<std::string>{key_name});
}

}  // namespace dftracer::utils::dataframe
