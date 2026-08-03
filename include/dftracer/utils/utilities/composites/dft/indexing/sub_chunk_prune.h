#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_INDEXING_SUB_CHUNK_PRUNE_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_INDEXING_SUB_CHUNK_PRUNE_H

#include <dftracer/utils/utilities/common/query/ast.h>
#include <dftracer/utils/utilities/composites/dft/indexing/chunk_statistics.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::indexing {

/// Inclusive numeric bound extracted from a query for one field.
struct NumericRange {
    std::uint64_t lo = 0;
    std::uint64_t hi = std::numeric_limits<std::uint64_t>::max();
    bool constrained = false;
};

/// Extract an inclusive [lo, hi] constraint on `field` from a query, but only
/// when the field appears solely inside a top-level AND of comparisons. Any
/// OR/NOT that references the field makes a range skip unsound (a bucket
/// failing the range no longer implies its events are all rejected), so this
/// returns an unconstrained range. Non-negative integer literals only.
inline NumericRange extract_and_range(
    const dftracer::utils::utilities::common::query::QueryNode& root,
    std::string_view field) {
    namespace q = dftracer::utils::utilities::common::query;
    NumericRange r;
    bool sound = true;

    auto lit_u64 = [](const q::LiteralNode& lit, std::uint64_t& out) -> bool {
        return std::visit(
            [&](auto&& v) -> bool {
                using T = std::decay_t<decltype(v)>;
                if constexpr (std::is_same_v<T, std::int64_t>) {
                    if (v < 0) return false;
                    out = static_cast<std::uint64_t>(v);
                    return true;
                } else if constexpr (std::is_same_v<T, std::uint64_t>) {
                    out = v;
                    return true;
                }
                return false;
            },
            lit.value);
    };

    auto refs = [&](const q::QueryNode& n) {
        auto fields = q::collect_fields(n);
        return fields.find(field) != fields.end();
    };

    auto visit = [&](auto&& self, const q::QueryNode& n) -> void {
        std::visit(
            [&](const auto& x) {
                using T = std::decay_t<decltype(x)>;
                if constexpr (std::is_same_v<T, q::CompareNode>) {
                    if (x.field.path != field) return;
                    std::uint64_t v = 0;
                    if (!lit_u64(x.value, v)) {
                        sound = false;
                        return;
                    }
                    switch (x.op) {
                        case q::CompareOp::EQ:
                            r.lo = std::max(r.lo, v);
                            r.hi = std::min(r.hi, v);
                            r.constrained = true;
                            break;
                        case q::CompareOp::GE:
                            r.lo = std::max(r.lo, v);
                            r.constrained = true;
                            break;
                        case q::CompareOp::GT:
                            r.lo = std::max(r.lo, v == UINT64_MAX ? v : v + 1);
                            r.constrained = true;
                            break;
                        case q::CompareOp::LE:
                            r.hi = std::min(r.hi, v);
                            r.constrained = true;
                            break;
                        case q::CompareOp::LT:
                            r.hi = std::min(r.hi, v == 0 ? v : v - 1);
                            r.constrained = true;
                            break;
                        case q::CompareOp::NE:
                            break;
                    }
                } else if constexpr (std::is_same_v<T, q::AndNode>) {
                    self(self, *x.left);
                    self(self, *x.right);
                } else if constexpr (std::is_same_v<T, q::OrNode>) {
                    if (refs(*x.left) || refs(*x.right)) sound = false;
                } else if constexpr (std::is_same_v<T, q::NotNode>) {
                    if (refs(*x.operand)) sound = false;
                }
            },
            n.data);
    };
    visit(visit, root);

    if (!sound) return NumericRange{};
    return r;
}

/// Keep-mask over a member's sub-chunk buckets: entry b is false only when
/// bucket b cannot contain a matching event because its ts and/or dur
/// zone-map lies entirely outside a required range. Returns an empty vector
/// when no bucket can be excluded (the common case; caller keeps the whole
/// member at no per-event cost).
inline std::vector<char> sub_chunk_keep_mask(
    const std::vector<SubChunkZoneMap>& buckets,
    const dftracer::utils::utilities::common::query::QueryNode& root,
    std::string_view ts_field, std::string_view dur_field) {
    if (buckets.empty()) return {};
    NumericRange ts = extract_and_range(root, ts_field);
    NumericRange dur = extract_and_range(root, dur_field);
    if (!ts.constrained && !dur.constrained) return {};

    std::vector<char> keep(buckets.size(), 1);
    bool any_excluded = false;
    for (std::size_t b = 0; b < buckets.size(); ++b) {
        const auto& z = buckets[b];
        bool excluded =
            (ts.constrained &&
             (z.max_timestamp_us < ts.lo || z.min_timestamp_us > ts.hi)) ||
            (dur.constrained &&
             (z.max_duration_us < dur.lo || z.min_duration_us > dur.hi));
        if (excluded) {
            keep[b] = 0;
            any_excluded = true;
        }
    }
    if (!any_excluded) return {};
    return keep;
}

}  // namespace dftracer::utils::utilities::composites::dft::indexing

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_INDEXING_SUB_CHUNK_PRUNE_H
