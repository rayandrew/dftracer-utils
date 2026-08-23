#ifndef DFTRACER_UTILS_PLUGINS_RESULT_REGISTRY_H
#define DFTRACER_UTILS_PLUGINS_RESULT_REGISTRY_H

#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/plugins/owned_arrow.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

// Host-side named result channel backing dft.ext.result. Policy-free: it moves
// opaque bytes or a user-schema Arrow array and never interprets either.
namespace dftracer::utils::plugins {

/// A streamed map result: an ordered sequence of same-schema Arrow batches the
/// registry owns, materialized one partition at a time so peak memory stays
/// near one partition instead of the whole map. Surfaces to Python as a
/// pull-based RecordBatchReader. Each batch is self-contained (STR labels
/// already copied in) so it outlives the intern table and the producing fold.
struct OwnedArrowBatches {
    std::vector<OwnedArrow> batches;
};

using NamedResult =
    std::variant<std::vector<std::byte>, OwnedArrow, OwnedArrowBatches>;

class NamedResultRegistry {
   public:
    void emit_blob(const char* name, const void* data, std::uint64_t len) {
        if (!name) return;
        std::vector<std::byte> bytes(static_cast<std::size_t>(len));
        if (len && data)
            std::memcpy(bytes.data(), data, static_cast<std::size_t>(len));
        std::lock_guard<std::mutex> lock(mutex_);
        results_[name] = std::move(bytes);
    }

    /// Steals *a and *s (zeroing them). -1 on a null argument.
    int emit_arrow(const char* name, ArrowArray* a, ArrowSchema* s) {
        if (!name || !a || !s) return -1;
        OwnedArrow owned;
        owned.array = *a;
        owned.schema = *s;
        *a = ArrowArray{};
        *s = ArrowSchema{};
        std::lock_guard<std::mutex> lock(mutex_);
        results_[name] = std::move(owned);
        return 0;
    }

    /// Take ownership of a multi-batch streamed result. Empty batches are a
    /// no-op (nothing to stream).
    void emit_arrow_batches(const char* name, OwnedArrowBatches&& batches) {
        if (!name || batches.batches.empty()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        results_[name] = std::move(batches);
    }

    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        results_.clear();
    }

    /// Drained after the scan completes, so no lock is needed by the reader.
    std::unordered_map<std::string, NamedResult>& results() { return results_; }

   private:
    std::mutex mutex_;
    std::unordered_map<std::string, NamedResult> results_;
};

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_RESULT_REGISTRY_H
