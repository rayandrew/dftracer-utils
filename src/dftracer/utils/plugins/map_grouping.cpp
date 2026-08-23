#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/plugins/map_grouping.h>

namespace dftracer::utils::plugins {

std::vector<MapAccum> grouping_sets(
    const MapAccum& m,
    const std::vector<std::vector<std::uint32_t>>& keep_sets) {
    std::vector<MapAccum> out;
    if (m.has_runs()) {
        DFTRACER_UTILS_LOG_ERROR(
            "grouping_sets on map '%s' with un-reloaded spilled runs; refusing "
            "to emit a partial result",
            m.name.c_str());
        return out;
    }
    out.reserve(keep_sets.size());
    for (const std::vector<std::uint32_t>& ks : keep_sets)
        out.push_back(
            regroup_map(m, ks.data(), static_cast<std::uint32_t>(ks.size())));
    return out;
}

std::vector<std::vector<std::uint32_t>> cube_sets(std::uint32_t key_n) {
    std::vector<std::vector<std::uint32_t>> out;
    if (key_n > CUBE_MAX_KEY_N) return out;
    const std::uint32_t n_sets = 1u << key_n;
    out.reserve(n_sets);
    for (std::uint32_t mask = 0; mask < n_sets; ++mask) {
        std::vector<std::uint32_t> s;
        for (std::uint32_t i = 0; i < key_n; ++i)
            if (mask & (1u << i)) s.push_back(i);
        out.push_back(std::move(s));
    }
    return out;
}

std::vector<std::vector<std::uint32_t>> rollup_sets(std::uint32_t key_n) {
    std::vector<std::vector<std::uint32_t>> out;
    out.reserve(key_n + 1);
    for (std::uint32_t keep = key_n;; --keep) {
        std::vector<std::uint32_t> s;
        s.reserve(keep);
        for (std::uint32_t i = 0; i < keep; ++i) s.push_back(i);
        out.push_back(std::move(s));
        if (keep == 0) break;
    }
    return out;
}

}  // namespace dftracer::utils::plugins
