#ifndef DFTRACER_UTILS_TRACE_VIEWS_FOLD_EVENT_H
#define DFTRACER_UTILS_TRACE_VIEWS_FOLD_EVENT_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/trace/event.h>
#include <simdjson.h>

#include <cstdint>
#include <utility>
#include <variant>
#include <vector>

// The owned event the fold-fusion scan core batches, kept in its own header so
// both the Fold interface and the event accessor depend on the data, not on
// each other.
namespace dftracer::utils::trace::views::detail {

/// An event owned independently of the simdjson parse that produced it: strings
/// are interned to ids, never views into the reused parser buffer, so a whole
/// batch stays valid at once. `args` is filled only for folds that need it.
struct FoldEvent {
    std::uint32_t cat_id = 0xFFFFFFFF;
    std::uint32_t name_id = 0xFFFFFFFF;
    std::uint32_t fhash_id = 0xFFFFFFFF;
    std::uint32_t hhash_id = 0xFFFFFFFF;
    std::uint64_t pid = 0;
    std::uint64_t tid = 0;
    std::uint64_t ts = 0;
    std::uint64_t dur = 0;
    RecordPhase phase = RecordPhase::UNKNOWN;
    bool has_dur = false;

    /// A metric (double for reals, int64 for exact integers so values above
    /// 2^53 survive) or an interned string id (group dimensions), keyed by the
    /// interned arg-name id.
    using ArgValue = std::variant<double, std::int64_t, std::uint32_t>;
    std::vector<std::pair<std::uint32_t, ArgValue>> args;
};

/// Build an owned event from already-parsed scalars + the args element, for
/// callers (like the index parse) that have run DFTracerEvent::parse_scalars
/// already. Every string is interned, so the result outlives `args`'s parser.
FoldEvent build_fold_event(const DFTracerEvent& scalars,
                           simdjson::dom::element args, bool has_args,
                           dftracer::utils::StringIntern& intern,
                           bool needs_args);

/// Parse a DOM object into an owned event. Every string is interned, so the
/// result stays valid after the parser that produced `root` is reused. Args are
/// captured only when `needs_args`.
FoldEvent extract_fold_event(simdjson::dom::element root,
                             dftracer::utils::StringIntern& intern,
                             bool needs_args);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_FOLD_EVENT_H
