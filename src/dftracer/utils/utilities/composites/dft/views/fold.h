#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_FOLD_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_FOLD_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/utilities/composites/dft/views/coverage.h>
#include <dftracer/utils/utilities/composites/dft/views/fold_event.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scan.h>

#include <memory>
#include <span>

// The fold-fusion scan core: one traversal parses each event once into an
// owned interned POD and drives every attached fold (banana-split / tupling).
// See the project_fold_fuse_architecture memory for the rationale.
namespace dftracer::utils::utilities::composites::dft::views::detail {

struct FoldBatch {
    std::span<const FoldEvent> events;  // valid for the `step` call only
    const ScanUnit& unit;
};

/// A push Sink over the fused scan, batch-grained so the virtual call amortizes
/// across the batch and the inner loop stays non-virtual. A fold owns its
/// accumulator and any spill; a returning fold exposes a typed result the
/// caller pulls after fuse, a persisting fold writes inside finalize.
class Fold {
   public:
    virtual ~Fold() = default;

    /// The scan must already deliver what this fold needs; it is never widened
    /// to satisfy one.
    virtual bool accepts(const ScanShape& shape) const = 0;

    /// ORed across folds, so args are extracted only when some fold needs them.
    virtual bool needs_args() const { return false; }

    /// One per worker slot, folded lock-free, merged after the fan-out joins.
    virtual std::unique_ptr<Fold> slice() const = 0;

    virtual void step(const FoldBatch& batch) = 0;
    virtual void seal_unit(const ScanUnit& unit) = 0;
    virtual void drop_unit(const ScanUnit& unit) = 0;
    virtual void merge(Fold& slice) = 0;

    /// `covered` bounds what a fold may claim. Returns false to persist
    /// nothing.
    virtual coro::CoroTask<bool> finalize(const CoverageSet& covered) = 0;
};

/// One traversal: decompress, parse each event once into an interned POD batch,
/// and drive every fold with per-worker slices merged at the end. `intern` is
/// shared across workers so interned ids agree and slices merge correctly; the
/// caller owns it and every fold must resolve ids through the same table.
/// Units `covered` already accounts for are skipped (a materialized aggregate
/// answered them). Distribution wraps this per worker and merges via
/// Fold::merge.
coro::CoroTask<ExportStats> fuse(const ViewPlan& plan,
                                 const ViewDefinition& vdef,
                                 std::span<Fold* const> folds,
                                 dftracer::utils::StringIntern& intern,
                                 const CoverageSet* covered = nullptr);

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_FOLD_H
