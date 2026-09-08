#ifndef DFTRACER_UTILS_PLUGINS_RESULT_REGISTRY_H
#define DFTRACER_UTILS_PLUGINS_RESULT_REGISTRY_H

#include <dftracer/utils/dataframe/abi.h>
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

/// A native columnar result carried across the ABI as a dftu_dataframe handle
/// (the dataframe engine's own boundary type), freed via the dataframe C ABI.
/// Lets an engine-backed result (the aggregation accumulator) cross as our own
/// DataFrame with no Arrow round-trip.
struct OwnedDataFrame {
    dftu_dataframe* handle = nullptr;
    OwnedDataFrame() = default;
    explicit OwnedDataFrame(dftu_dataframe* h) noexcept : handle(h) {}
    OwnedDataFrame(OwnedDataFrame&& o) noexcept : handle(o.handle) {
        o.handle = nullptr;
    }
    OwnedDataFrame& operator=(OwnedDataFrame&& o) noexcept {
        if (this != &o) {
            if (handle) dftu_dataframe_free(handle);
            handle = o.handle;
            o.handle = nullptr;
        }
        return *this;
    }
    OwnedDataFrame(const OwnedDataFrame&) = delete;
    OwnedDataFrame& operator=(const OwnedDataFrame&) = delete;
    ~OwnedDataFrame() {
        if (handle) dftu_dataframe_free(handle);
    }

    /// Releases ownership to the caller, who must free it (or hand it to
    /// something that will); the destructor no longer will.
    dftu_dataframe* release() noexcept {
        dftu_dataframe* h = handle;
        handle = nullptr;
        return h;
    }
};

/// A deferred columnar result carried across the ABI as a dftu_lazyframe
/// handle, freed via the dataframe C ABI. Lets a plugin return a lazy plan the
/// host collects at the Python edge. The plan MUST be self-contained (its
/// source an in-memory frame or a re-openable source, e.g. dftu_dataframe_lazy
/// of a materialized frame); referencing the plugin's per-scan/executor
/// context, which dies at finalize, is a plugin bug.
struct OwnedLazyFrame {
    dftu_lazyframe* handle = nullptr;
    OwnedLazyFrame() = default;
    explicit OwnedLazyFrame(dftu_lazyframe* h) noexcept : handle(h) {}
    OwnedLazyFrame(OwnedLazyFrame&& o) noexcept : handle(o.handle) {
        o.handle = nullptr;
    }
    OwnedLazyFrame& operator=(OwnedLazyFrame&& o) noexcept {
        if (this != &o) {
            if (handle) dftu_lazyframe_free(handle);
            handle = o.handle;
            o.handle = nullptr;
        }
        return *this;
    }
    OwnedLazyFrame(const OwnedLazyFrame&) = delete;
    OwnedLazyFrame& operator=(const OwnedLazyFrame&) = delete;
    ~OwnedLazyFrame() {
        if (handle) dftu_lazyframe_free(handle);
    }

    /// Releases ownership to the caller, who must free it (or hand it to
    /// something that will); the destructor no longer will.
    dftu_lazyframe* release() noexcept {
        dftu_lazyframe* h = handle;
        handle = nullptr;
        return h;
    }
};

using NamedResult = std::variant<std::vector<std::byte>, OwnedArrow,
                                 OwnedDataFrame, OwnedLazyFrame>;

class NamedResultRegistry {
   public:
    NamedResultRegistry() = default;

    /// Movable so a run can hand its results back by value. Only safe once the
    /// scan that writes into it has finished; the mutex is not carried over.
    NamedResultRegistry(NamedResultRegistry&& other) noexcept
        : results_(std::move(other.results_)) {}
    NamedResultRegistry& operator=(NamedResultRegistry&& other) noexcept {
        if (this != &other) results_ = std::move(other.results_);
        return *this;
    }

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

    /// Take ownership of a native dftu_dataframe result. A null handle is a
    /// no-op.
    void emit_frame(const char* name, dftu_dataframe* handle) {
        if (!name || !handle) return;
        std::lock_guard<std::mutex> lock(mutex_);
        results_[name] = OwnedDataFrame{handle};
    }

    /// Take ownership of a native dftu_lazyframe (deferred) result; the host
    /// collects it at the Python edge. The plan must be self-contained (see
    /// OwnedLazyFrame). A null handle is a no-op.
    void emit_lazyframe(const char* name, dftu_lazyframe* handle) {
        if (!name || !handle) return;
        std::lock_guard<std::mutex> lock(mutex_);
        results_[name] = OwnedLazyFrame{handle};
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
