#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/plugins/map_unnest.h>

#include <algorithm>
#include <utility>

namespace dftracer::utils::plugins {

namespace {

bool is_unnestable(dftu_monoid_kind k) {
    return k == DFTU_MONOID_SET_STR || k == DFTU_MONOID_SET_I64 ||
           k == DFTU_MONOID_LIST_STR || k == DFTU_MONOID_LIST_I64 ||
           monoid_is_topk(k) || monoid_is_sample(k);
}

bool elem_is_str(dftu_monoid_kind k) {
    return k == DFTU_MONOID_SET_STR || k == DFTU_MONOID_LIST_STR ||
           monoid_topk_is_str(k) || monoid_sample_is_str(k);
}

std::vector<std::int64_t> elements_of(const MonoidAccumulator& mon) {
    const dftu_monoid_kind k = mon.kind();
    if (monoid_is_topk(k)) return mon.topk_elements();
    if (monoid_is_sample(k)) return mon.sample_items();
    if (k == DFTU_MONOID_LIST_STR || k == DFTU_MONOID_LIST_I64)
        return mon.ordered_elements();
    return mon.sorted_elements();
}

}  // namespace

ExplodedRows unnest_map(const MapAccum& m, std::uint32_t value_component,
                        bool keep_empty) {
    ExplodedRows out;
    if (m.has_runs()) {
        DFTRACER_UTILS_LOG_ERROR(
            "unnest_map on map '%s' with un-reloaded spilled runs; refusing to "
            "emit a partial result",
            m.name.c_str());
        return out;
    }
    if (value_component >= m.value_kinds.size()) return out;
    const dftu_monoid_kind ek = m.value_kinds[value_component];
    if (!is_unnestable(ek)) return out;

    const std::uint32_t value_n =
        static_cast<std::uint32_t>(m.value_kinds.size());
    out.valid = true;
    out.key_n = m.key_n;
    out.key_types = m.key_types;
    out.payload_types = m.payload_types;
    out.elem_type = elem_is_str(ek) ? DFTU_T_STR : DFTU_T_I64;
    out.elem_name = value_n == 1 ? std::string("value")
                                 : "v" + std::to_string(value_component);
    for (std::uint32_t c = 0; c < value_n; ++c)
        if (c != value_component) out.value_kinds.push_back(m.value_kinds[c]);

    using EntryRef = const std::pair<const std::vector<std::int64_t>,
                                     std::vector<MonoidAccumulator>>*;
    std::vector<EntryRef> entries;
    entries.reserve(m.total_entries());
    for (const MapAccum::EntriesMap& part : m.parts)
        for (const auto& e : part) entries.push_back(&e);
    std::sort(entries.begin(), entries.end(),
              [](EntryRef a, EntryRef b) { return a->first < b->first; });

    for (EntryRef e : entries) {
        const std::vector<MonoidAccumulator>& mons = e->second;
        std::vector<MonoidAccumulator> kept;
        kept.reserve(value_n - 1);
        for (std::uint32_t c = 0; c < value_n; ++c)
            if (c != value_component) kept.push_back(mons[c]);
        std::vector<std::int64_t> elems = elements_of(mons[value_component]);
        if (elems.empty()) {
            if (!keep_empty) continue;
            ExplodedRow row;
            row.key = e->first;
            row.kept_values = kept;
            row.elem_null = true;
            out.rows.push_back(std::move(row));
            continue;
        }
        for (std::int64_t v : elems) {
            ExplodedRow row;
            row.key = e->first;
            row.kept_values = kept;
            row.elem = v;
            out.rows.push_back(std::move(row));
        }
    }
    return out;
}

}  // namespace dftracer::utils::plugins
