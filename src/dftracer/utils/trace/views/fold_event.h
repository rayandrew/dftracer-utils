#ifndef DFTRACER_UTILS_TRACE_VIEWS_FOLD_EVENT_H
#define DFTRACER_UTILS_TRACE_VIEWS_FOLD_EVENT_H

#include <dftracer/utils/core/common/field_ref.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/json/json_value.h>
#include <dftracer/utils/trace/event.h>
#include <simdjson.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
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
    /// Top-level values for schema fields the POD does not carry as a scalar
    /// (type/ph/id), kept apart from `args` so a same-named args key cannot
    /// shadow the real top-level field. Same encoding as `args`.
    std::vector<std::pair<std::uint32_t, ArgValue>> top_fields;

    /// Every scalar leaf of the record, arbitrarily nested, as (interned dotted
    /// path id -> type tag). Filled only when the scan captures schema (the
    /// index build); empty otherwise. Type tag values match
    /// utilities::indexer::ColumnType (1=Int64, 2=Float64, 3=String) so the
    /// index harvest maps them without a lookup. This is what keeps the engine
    /// schemaless: a nested-object or array arg (which `args` drops) still
    /// surfaces here as one leaf per scalar, so no field is silently lost.
    std::vector<std::pair<std::uint32_t, std::uint8_t>> schema_leaves;
};

/// The fixed top-level fields of the trace schema. A bare reference to one of
/// these resolves to the top-level value, never a same-named args key; reach an
/// args field with the same name through the explicit `args.<name>` path.
inline bool is_schema_field(std::string_view f) {
    return f == "name" || f == "cat" || f == "pid" || f == "tid" || f == "ts" ||
           f == "dur" || f == "ph" || f == "id" || f == "type";
}

/// True when `field` addresses a nested value (`a.b`, `a[0]`, `a.0.b`), so it
/// needs path resolution rather than a single object-key lookup.
inline bool is_nested_path(std::string_view field) {
    return field.find_first_of(".[") != std::string_view::npos;
}

/// Resolve a dotted/bracketed path from `root` (`a.b[0].c`, `a.b.0`); `ok` is
/// false if any segment is missing. Rooted at `root` with no args fallback,
/// matching the query evaluator's treatment of dotted paths. Delegates to the
/// single path walker (JsonValue::at) so flat dotted member keys resolve here
/// too; on failure `ok` is false and `root` is returned unchanged.
inline simdjson::dom::element resolve_json_path(simdjson::dom::element root,
                                                std::string_view path,
                                                bool& ok) {
    json::JsonValue v = json::JsonValue(root).at(path);
    ok = v.exists();
    return ok ? v.element() : root;
}

/// Capture a field the POD does not natively carry (top-level type/ph/id, or a
/// nested a.b/a[0]), JSON type preserved. A bare schema field goes to
/// `top_fields` so a same-named args key cannot shadow it; everything else to
/// `args`. No-op when the path is absent.
inline void capture_extra_field(FoldEvent& ev, simdjson::dom::element root,
                                dftracer::utils::StringIntern& intern,
                                const std::string& name) {
    bool ok = false;
    simdjson::dom::element v = resolve_json_path(root, name, ok);
    if (!ok) return;
    const std::uint32_t key_id = intern.get_or_insert(name);
    auto& into = (!is_nested_path(name) && is_schema_field(name))
                     ? ev.top_fields
                     : ev.args;
    for (const auto& [k, existing] : into)
        if (k == key_id) return;
    if (v.is_string()) {
        into.emplace_back(key_id, intern.get_or_insert(v.get_string()));
    } else if (v.is_int64()) {
        into.emplace_back(key_id, static_cast<std::int64_t>(v.get_int64()));
    } else if (v.is_uint64()) {
        const std::uint64_t u = v.get_uint64().value_unsafe();
        if (u <= static_cast<std::uint64_t>(
                     std::numeric_limits<std::int64_t>::max()))
            into.emplace_back(key_id, static_cast<std::int64_t>(u));
        else
            into.emplace_back(key_id, static_cast<double>(u));
    } else if (v.is_double()) {
        into.emplace_back(key_id, v.get_double().value_unsafe());
    }
}

/// Enumerate every scalar leaf of `root` (arbitrarily nested) into
/// ev.schema_leaves for the index build's schemaless column harvest. Args
/// children are bare paths (hostname, pos.x) and other top-level fields keep
/// their name; the axis/structural keys (pid/tid/ts/dur/ph/id) are excluded.
/// Type tags match utilities::indexer::ColumnType. See extract_fold_event.
void capture_schema_leaves(FoldEvent& ev, simdjson::dom::element root,
                           dftracer::utils::StringIntern& intern);

/// Build an owned event from already-parsed scalars + the args element, for
/// callers (like the index parse) that have run DFTracerEvent::parse_scalars
/// already. Every string is interned, so the result outlives `args`'s parser.
FoldEvent build_fold_event(const DFTracerEvent& scalars,
                           simdjson::dom::element args, bool has_args,
                           dftracer::utils::StringIntern& intern,
                           bool needs_args);

/// Parse a DOM object into an owned event. Every string is interned, so the
/// result stays valid after the parser that produced `root` is reused. Args are
/// captured only when `needs_args`. `extra_fields`, if given, names fields the
/// POD does not natively carry (type/ph, a nested a.b/a[0]) to capture into the
/// event. When `capture_schema`, every scalar leaf (arbitrarily nested) is
/// enumerated into `schema_leaves` for the index build's column harvest.
FoldEvent extract_fold_event(
    simdjson::dom::element root, dftracer::utils::StringIntern& intern,
    bool needs_args, const std::vector<std::string>* extra_fields = nullptr,
    bool capture_schema = false);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_FOLD_EVENT_H
