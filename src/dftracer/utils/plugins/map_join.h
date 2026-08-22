#ifndef DFTRACER_UTILS_PLUGINS_MAP_JOIN_H
#define DFTRACER_UTILS_PLUGINS_MAP_JOIN_H

#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/monoid.h>

#include <cstdint>
#include <vector>

// Native map-to-map equi join over two MapAccum results; no Arrow dependency,
// so it works in an Arrow-off build. Keys match as interned int64 slots (all
// maps in a run share the process StringIntern); value Monoids are carried
// opaquely.
namespace dftracer::utils::plugins {

enum class JoinType { INNER, LEFT, RIGHT, FULL, LEFT_SEMI, LEFT_ANTI };

// The value schema one side carries, copied opaquely from its MapAccum; the
// join never reads it to match, it travels with the result for a later
// materialize.
struct JoinValueSchema {
    std::vector<dftu_monoid_kind> value_kinds;
    std::uint32_t nested_inner_n = 0;
    std::vector<dftu_type> inner_key_types;
    std::vector<dftu_type> payload_types;
    std::vector<double> quantile_qs;
};

// One joined row. A MonoidAccumulator is always a value and never null, but an
// OUTER join must null the absent side, so presence flags carry that: an absent
// side has present=false and an empty value vector. INNER rows are
// both-present.
struct JoinedRow {
    std::vector<std::int64_t> key;
    bool left_present = false;
    std::vector<MonoidAccumulator> left_values;
    bool right_present = false;
    std::vector<MonoidAccumulator> right_values;
};

struct JoinedMap {
    // False when the two maps do not share a key schema (key_n or key_types
    // differ); rows is then empty.
    bool valid = false;
    // LEFT_SEMI/LEFT_ANTI carry left-only rows; materialize emits key + left
    // value columns only, skipping right_schema.
    bool left_only = false;
    std::uint32_t key_n = 0;
    std::vector<dftu_type> key_types;
    JoinValueSchema left_schema;
    JoinValueSchema right_schema;
    // Sorted by the key tuple (int64 slots, STR/BYTES by interned id) for a
    // deterministic result; a label-sorted export is the materialize's concern.
    std::vector<JoinedRow> rows;
};

// Equi join two same-key-schema maps on the FULL key tuple. Returns an invalid
// JoinedMap (valid == false) when key_n or key_types differ. Value Monoids are
// copied into the rows unread; the caller owns the result.
// Precondition: both inputs must be fully resident (no un-reloaded spilled
// runs); a spilled input yields an invalid result, never a partial one.
JoinedMap join_maps(const MapAccum& left, const MapAccum& right, JoinType type);

// Rollup m to a subset/reorder of its key components: reduce each key to the
// kept components (in the given order) and merge collapsed rows' value Monoids.
// Value schema unchanged; empty result (key_n==0) if a keep index is out of
// range. Precondition: m fully resident; a spilled input yields an empty map.
MapAccum regroup_map(const MapAccum& m, const std::uint32_t* keep_components,
                     std::uint32_t n_keep);

// FK equi join: regroup each side to its join-key columns (skipped when a
// side's columns already are its full key in order), then join_maps on the
// shared key. Invalid if a column is out of range or the join-key types differ
// across sides. Precondition: both inputs fully resident (see join_maps).
JoinedMap fk_join(const MapAccum& left, const std::uint32_t* left_key_cols,
                  const MapAccum& right, const std::uint32_t* right_key_cols,
                  std::uint32_t n_join_key, JoinType how);

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_MAP_JOIN_H
