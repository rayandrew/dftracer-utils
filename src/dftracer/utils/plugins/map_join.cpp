#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/plugins/map_join.h>

#include <algorithm>
#include <unordered_map>
#include <unordered_set>

namespace dftracer::utils::plugins {

namespace {

JoinValueSchema schema_of(const MapAccum& m) {
    JoinValueSchema s;
    s.value_kinds = m.value_kinds;
    s.nested_inner_n = m.nested_inner_n;
    s.inner_key_types = m.inner_key_types;
    s.payload_types = m.payload_types;
    s.quantile_qs = m.quantile_qs;
    return s;
}

bool same_key_schema(const MapAccum& a, const MapAccum& b) {
    if (a.key_n != b.key_n) return false;
    if (a.key_types.size() != b.key_types.size()) return false;
    for (std::size_t i = 0; i < a.key_types.size(); ++i)
        if (a.key_types[i] != b.key_types[i]) return false;
    return true;
}

// A key -> its value vector, indexed across every partition. A key lives in
// exactly one partition and is unique within it, so the flat index is 1:1.
using ValueIndex = std::unordered_map<std::vector<std::int64_t>,
                                      const std::vector<MonoidAccumulator>*,
                                      MapAccum::KeyHash>;

ValueIndex index_values(const MapAccum& m) {
    ValueIndex idx;
    idx.reserve(m.total_entries());
    for (const MapAccum::EntriesMap& part : m.parts)
        for (const auto& e : part) idx.emplace(e.first, &e.second);
    return idx;
}

}  // namespace

JoinedMap join_maps(const MapAccum& left, const MapAccum& right,
                    JoinType type) {
    JoinedMap out;
    if (left.has_runs() || right.has_runs()) {
        DFTRACER_UTILS_LOG_ERROR(
            "join_maps on a map with un-reloaded spilled runs (left '%s', "
            "right '%s'); refusing to emit a partial result",
            left.name.c_str(), right.name.c_str());
        return out;
    }
    if (!same_key_schema(left, right)) return out;

    const bool semi = type == JoinType::LEFT_SEMI;
    const bool anti = type == JoinType::LEFT_ANTI;

    out.valid = true;
    out.left_only = semi || anti;
    out.key_n = left.key_n;
    out.key_types = left.key_types;
    out.left_schema = schema_of(left);
    out.right_schema = schema_of(right);

    const bool want_left_only =
        type == JoinType::LEFT || type == JoinType::FULL || anti;
    const bool want_right_only =
        type == JoinType::RIGHT || type == JoinType::FULL;

    ValueIndex right_idx = index_values(right);
    std::unordered_set<std::vector<std::int64_t>, MapAccum::KeyHash> matched;

    for (const MapAccum::EntriesMap& part : left.parts) {
        for (const auto& e : part) {
            auto it = right_idx.find(e.first);
            if (it != right_idx.end()) {
                if (anti) continue;
                JoinedRow row;
                row.key = e.first;
                row.left_present = true;
                row.left_values = e.second;
                if (!semi) {
                    row.right_present = true;
                    row.right_values = *it->second;
                }
                out.rows.push_back(std::move(row));
                if (want_right_only) matched.insert(e.first);
            } else if (want_left_only) {
                JoinedRow row;
                row.key = e.first;
                row.left_present = true;
                row.left_values = e.second;
                out.rows.push_back(std::move(row));
            }
        }
    }

    if (want_right_only) {
        for (const MapAccum::EntriesMap& part : right.parts) {
            for (const auto& e : part) {
                if (matched.count(e.first)) continue;
                JoinedRow row;
                row.key = e.first;
                row.right_present = true;
                row.right_values = e.second;
                out.rows.push_back(std::move(row));
            }
        }
    }

    std::sort(
        out.rows.begin(), out.rows.end(),
        [](const JoinedRow& a, const JoinedRow& b) { return a.key < b.key; });
    return out;
}

MapAccum regroup_map(const MapAccum& m, const std::uint32_t* keep_components,
                     std::uint32_t n_keep) {
    MapAccum out;
    if (m.has_runs()) {
        DFTRACER_UTILS_LOG_ERROR(
            "regroup_map on map '%s' with un-reloaded spilled runs; refusing "
            "to emit a partial result",
            m.name.c_str());
        return out;
    }
    for (std::uint32_t i = 0; i < n_keep; ++i)
        if (keep_components[i] >= m.key_n) return out;

    out.name = m.name;
    out.key_n = n_keep;
    out.key_types.reserve(n_keep);
    for (std::uint32_t i = 0; i < n_keep; ++i)
        out.key_types.push_back(m.key_types[keep_components[i]]);
    out.value_kinds = m.value_kinds;
    out.nested_inner_n = m.nested_inner_n;
    out.inner_key_types = m.inner_key_types;
    out.payload_types = m.payload_types;
    out.ordered = m.ordered;

    std::vector<std::int64_t> rk(n_keep);
    for (const MapAccum::EntriesMap& part : m.parts) {
        for (const auto& e : part) {
            for (std::uint32_t i = 0; i < n_keep; ++i)
                rk[i] = e.first[keep_components[i]];
            bool inserted = false;
            std::vector<MonoidAccumulator>& dst = out.touch(rk, inserted);
            if (inserted) {
                dst = e.second;
            } else {
                for (std::size_t c = 0; c < dst.size(); ++c)
                    dst[c].merge(e.second[c]);
            }
        }
    }
    return out;
}

JoinedMap fk_join(const MapAccum& left, const std::uint32_t* left_key_cols,
                  const MapAccum& right, const std::uint32_t* right_key_cols,
                  std::uint32_t n_join_key, JoinType how) {
    JoinedMap out;
    if (n_join_key == 0) return out;
    // Guard here too: a spilled side regrouped below would collapse to a valid
    // empty map and leak a silently partial join, which join_maps cannot catch.
    if (left.has_runs() || right.has_runs()) {
        DFTRACER_UTILS_LOG_ERROR(
            "fk_join on a map with un-reloaded spilled runs (left '%s', "
            "right '%s'); refusing to emit a partial result",
            left.name.c_str(), right.name.c_str());
        return out;
    }
    for (std::uint32_t i = 0; i < n_join_key; ++i) {
        if (left_key_cols[i] >= left.key_n) return out;
        if (right_key_cols[i] >= right.key_n) return out;
        if (left.key_types[left_key_cols[i]] !=
            right.key_types[right_key_cols[i]])
            return out;
    }

    auto is_full_key = [](const MapAccum& m, const std::uint32_t* cols,
                          std::uint32_t n) {
        if (m.key_n != n) return false;
        for (std::uint32_t i = 0; i < n; ++i)
            if (cols[i] != i) return false;
        return true;
    };

    const bool left_full = is_full_key(left, left_key_cols, n_join_key);
    const bool right_full = is_full_key(right, right_key_cols, n_join_key);
    MapAccum left_rg, right_rg;
    if (!left_full) left_rg = regroup_map(left, left_key_cols, n_join_key);
    if (!right_full) right_rg = regroup_map(right, right_key_cols, n_join_key);
    return join_maps(left_full ? left : left_rg, right_full ? right : right_rg,
                     how);
}

}  // namespace dftracer::utils::plugins
