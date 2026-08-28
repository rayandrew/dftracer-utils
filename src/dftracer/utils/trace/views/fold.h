#ifndef DFTRACER_UTILS_TRACE_VIEWS_FOLD_H
#define DFTRACER_UTILS_TRACE_VIEWS_FOLD_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/trace/views/coverage.h>
#include <dftracer/utils/trace/views/fold_event.h>
#include <dftracer/utils/trace/views/view_scan.h>

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// A plugin fold stashes the dftu_task* its on_batch returned for the async
// driver to await; forward-declared to keep the C ABI out of this header.
struct dftu_task;

// The fold-fusion scan core; see the project_fold_fuse_architecture memory.
namespace dftracer::utils::trace::views::detail {

struct FoldBatch {
    std::span<const FoldEvent> events;  // valid for the `step` call only
    const ScanUnit& unit;
    // Raw NDJSON lines, populated only when a wants_raw() fold is present, in
    // scan order. A raw fold parses these itself (arbitrary-JSON access); a
    // FoldEvent fold ignores them. Valid for the `step` call only.
    std::span<const std::string_view> raw = {};
};

/// Per-worker keyed byte scratchpad so a producer fold can hand a derived value
/// to a later consumer fold in the same batch. Single-threaded, unlocked;
/// consume borrows the bytes until the next clear.
class FoldPortBus {
   public:
    void publish(std::uint64_t key, const void* data, std::uint32_t len);
    /// NULL if `key` was not published since the last clear.
    const void* consume(std::uint64_t key, std::uint32_t* out_len) const;
    void clear();

   private:
    struct Entry {
        std::size_t offset;
        std::uint32_t len;
    };
    std::unordered_map<std::uint64_t, Entry> index_;
    std::vector<std::byte> arena_;
};

/// A push Sink over the fused scan, batch-grained so the virtual call amortizes
/// across the batch. A returning fold exposes a typed result the caller pulls
/// after fuse; a persisting fold writes inside finalize.
class Fold {
   public:
    virtual ~Fold() = default;

    /// The scan must already deliver what this fold needs; it is never widened
    /// to satisfy one.
    virtual bool accepts(const ScanShape& shape) const = 0;

    /// ORed across folds, so args are extracted only when some fold needs them.
    virtual bool needs_args() const { return false; }

    /// A fold that consumes FoldBatch::raw (raw NDJSON lines it parses itself)
    /// rather than the parsed FoldEvent stream. ORed across folds so the scan
    /// keeps raw lines when any is present.
    virtual bool wants_raw() const { return false; }

    /// Extra fields (top-level type/ph/id, or nested a.b/a[0]) this fold needs
    /// captured into each event beyond what the POD carries and beyond flat
    /// args (which needs_args() already covers). Merged with the plan's own
    /// captures.
    virtual std::vector<std::string> extra_captures() const { return {}; }

    /// One per worker slot, folded lock-free, merged after the fan-out joins.
    virtual std::unique_ptr<Fold> slice() const = 0;

    virtual void step(const FoldBatch& batch) = 0;
    virtual void seal_unit(const ScanUnit& unit) = 0;
    virtual void drop_unit(const ScanUnit& unit) = 0;
    virtual void merge(Fold& slice) = 0;

    /// `covered` bounds what a fold may claim. Returns false to persist
    /// nothing.
    virtual coro::CoroTask<bool> finalize(const CoverageSet& covered) = 0;

    /// The async task the last `step` produced, or null when synchronous; the
    /// driver awaits it. Only a plugin fold returns non-null.
    virtual ::dftu_task* take_pending() { return nullptr; }

    /// The fuse driver binds its per-worker bus so a fold may publish/consume
    /// per-batch values; folds that do not communicate ignore it.
    virtual void bind_port_bus(FoldPortBus*) {}
};

/// One traversal driving every fold with per-worker slices merged at the end.
/// `intern` is shared across workers and caller-owned; every fold must resolve
/// ids through it. Units `covered` already accounts for are skipped.
coro::CoroTask<ExportStats> fuse(const ViewPlan& plan,
                                 const ViewDefinition& vdef,
                                 std::span<Fold* const> folds,
                                 dftracer::utils::StringIntern& intern,
                                 const CoverageSet* covered = nullptr,
                                 std::uint64_t limit = 0);

/// Fields the scanner must capture into each event's args for this plan: the
/// non-scalar Field group keys and nested agg fields the POD does not natively
/// carry (type/ph, a.b, a[0]). Empty for a plain plan.
std::vector<std::string> extra_capture_fields(const ViewPlan& plan);

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_FOLD_H
