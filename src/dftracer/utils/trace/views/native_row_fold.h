#ifndef DFTRACER_UTILS_TRACE_VIEWS_NATIVE_ROW_FOLD_H
#define DFTRACER_UTILS_TRACE_VIEWS_NATIVE_ROW_FOLD_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/trace/views/fold.h>
#include <dftracer/utils/trace/views/fold_event.h>

#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dftracer::utils::trace::views::detail {

class GroupResolver;

/// Build one native DataFrame from `events`: top-level columns plus every arg
/// (empty `select`) or a projected subset. Arg columns infer their type per key
/// and null-fill absent rows. `fhash`/`hhash` resolve from their dedicated
/// fields; a selected `resolved.*`/`r.*` field resolves through `resolver` (the
/// index name tables), or is all-null when `resolver` is null. Shared by the
/// materialized collect and the streaming chunk builder.
dataframe::DataFrame build_row_frame(
    const std::vector<FoldEvent>& events,
    const dftracer::utils::StringIntern& intern,
    const std::vector<std::string>& select, double time_scale = 1.0,
    const GroupResolver* resolver = nullptr);

/// True if `select` names any resolved.*/r.* field, so the caller should build
/// a GroupResolver (which opens the index name tables) for build_row_frame.
bool select_needs_resolver(const std::vector<std::string>& select);

/// The output column name build_row_frame's select branch gives `sel`: a
/// top-level field or resolved.*/r.* virtual field keeps its own name;
/// fhash/hhash keep their bare name; anything else is a flattened arg and is
/// canonicalized to "args.<key>" (accepting `sel` bare or "args."-prefixed).
/// ViewSource::names() uses this so a streamed morsel's schema always matches
/// what build_row_frame actually emits for the same select.
std::string canonical_row_column_name(std::string_view sel);

/// Collect matching raw events straight into a native DataFrame. The
/// row-query terminal (collect() / stream() with no group_by/agg) folds every
/// event's top-level fields plus its args into columns; `select` projects a
/// subset (top-level names or arg keys, bare or "args."-prefixed), and an empty
/// `select` emits every column (the union of all args seen, null-filled where
/// an event lacks a key). ph="M" metadata is skipped (it carries no event row).
///
/// build() materializes one frame over every accumulated event, so the column
/// union is exact and no cross-slot schema reconciliation is needed. Types are
/// inferred per arg column: int64 unless a real forces Float64, or a string
/// value forces String (numbers stringified).
class NativeRowFold : public Fold {
   public:
    NativeRowFold(const dftracer::utils::StringIntern& intern,
                  std::vector<std::string> select, double time_scale = 1.0,
                  std::shared_ptr<const GroupResolver> resolver = nullptr)
        : intern_(&intern),
          select_(std::move(select)),
          time_scale_(time_scale),
          resolver_(std::move(resolver)) {}

    bool accepts(const ScanShape&) const override { return true; }
    bool needs_args() const override { return true; }

    std::unique_ptr<Fold> slice() const override {
        return std::make_unique<NativeRowFold>(*intern_, select_, time_scale_,
                                               resolver_);
    }

    void step(const FoldBatch& batch) override {
        for (const FoldEvent& ev : batch.events) {
            if (ev.phase == RecordPhase::METADATA ||
                ev.phase == RecordPhase::UNKNOWN)
                continue;
            events_.push_back(ev);
        }
    }

    void seal_unit(const ScanUnit&) override {}
    void drop_unit(const ScanUnit&) override {}

    void merge(Fold& other) override {
        auto& o = static_cast<NativeRowFold&>(other);
        events_.insert(events_.end(),
                       std::make_move_iterator(o.events_.begin()),
                       std::make_move_iterator(o.events_.end()));
        o.events_.clear();
    }

    coro::CoroTask<bool> finalize(const CoverageSet&) override {
        co_return true;
    }

    /// Build the accumulated events into one native DataFrame. Drains events_.
    dataframe::DataFrame build();

   private:
    const dftracer::utils::StringIntern* intern_;
    std::vector<std::string> select_;  // empty = every column
    double time_scale_;                // ts/dur multiplier (1.0 = none)
    std::shared_ptr<const GroupResolver>
        resolver_;                     // resolved.* names, or null
    std::vector<FoldEvent> events_;
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_NATIVE_ROW_FOLD_H
