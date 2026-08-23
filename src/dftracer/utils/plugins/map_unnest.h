#ifndef DFTRACER_UTILS_PLUGINS_MAP_UNNEST_H
#define DFTRACER_UTILS_PLUGINS_MAP_UNNEST_H

#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/monoid.h>

#include <cstdint>
#include <string>
#include <vector>

// Native UNNEST/EXPLODE over a MapAccum: one output row per element of a
// list-valued collection value component. No Arrow dependency, like map_join.h.
namespace dftracer::utils::plugins {

// elem is raw: an int64 for an _I64 collection, an interned dftu_str id for a
// _STR one (resolved at materialize). elem_null marks a keep_empty empty-row.
struct ExplodedRow {
    std::vector<std::int64_t> key;
    std::vector<MonoidAccumulator> kept_values;
    bool elem_null = false;
    std::int64_t elem = 0;
};

struct ExplodedRows {
    // False when value_component is out of range or its kind is not a
    // list-of-scalar collection (struct-list kinds like APPROX_TOPK and scalar
    // kinds are rejected); rows is then empty.
    bool valid = false;
    std::uint32_t key_n = 0;
    std::vector<dftu_type> key_types;
    std::vector<dftu_monoid_kind> value_kinds;
    std::vector<dftu_type> payload_types;
    dftu_type elem_type = DFTU_T_I64;
    std::string elem_name;
    // Ordered by key tuple then by element order (LIST by its order key,
    // SET/SAMPLE by sorted element) for a deterministic result.
    std::vector<ExplodedRow> rows;
};

// Explode `value_component` of `m` into one row per element. Empty collections
// drop (SQL inner unnest) unless keep_empty, which emits one elem_null row.
// Returns an invalid result for a non-collection or out-of-range component.
// Precondition: m fully resident; a spilled input yields an invalid result.
ExplodedRows unnest_map(const MapAccum& m, std::uint32_t value_component,
                        bool keep_empty);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_MAP_UNNEST_H
