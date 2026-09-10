// dftu_provider_register/unregister and dftu_lazyframe_from_provider: a
// process-lifetime, name-keyed registry of Source vtables, mirroring the op
// registry in op_registry.cpp. dftu.svc.providers@0 (plugins/abi/providers.h)
// is a thin gated forwarder into this registry, but the registry itself is
// not plugin-scoped: a non-plugin C caller, or a future language binding,
// registers and reads it the same way a plugin does.

#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/dataframe_handle.h>
#include <dftracer/utils/dataframe/internal/lazyframe_handle.h>
#include <dftracer/utils/dataframe/lazyframe.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>

namespace dftracer::utils::dataframe {
namespace {

struct ProviderEntry {
    ::dftu_source_vt vt;
    void* self;
};

std::mutex& registry_mutex() {
    static std::mutex m;
    return m;
}

std::unordered_map<std::string, ProviderEntry>& providers() {
    static std::unordered_map<std::string, ProviderEntry> m;
    return m;
}

std::optional<ProviderEntry> find_provider(const char* name) {
    if (!name) return std::nullopt;
    std::lock_guard<std::mutex> lock(registry_mutex());
    auto it = providers().find(name);
    if (it == providers().end()) return std::nullopt;
    return it->second;
}

std::string describe(const ::dftu_error& err) {
    return err.message && *err.message ? err.message : "provider cursor failed";
}

// Adapts a registered dftu_cursor_vt as a Cursor. next() follows the same
// dftu_task convention as dftu_plugin::on_batch/on_finalize: a returned task
// is awaited and freed here, the driving side, exactly as dftu_task_run does
// for a task crossing the ABI the same way.
class ProviderCursor final : public Cursor {
   public:
    ProviderCursor(const ::dftu_cursor_vt* vt, void* self)
        : vt_(vt), self_(self) {}

    ~ProviderCursor() override {
        if (vt_->destroy) vt_->destroy(self_);
    }

    ProviderCursor(const ProviderCursor&) = delete;
    ProviderCursor& operator=(const ProviderCursor&) = delete;

    coro::CoroTask<std::optional<Morsel>> next(std::int64_t max_rows) override {
        ::dftu_result_frame out{};
        if (::dftu_task* t = vt_->next(self_, max_rows, &out)) {
            auto* task = reinterpret_cast<coro::CoroTask<void>*>(t);
            co_await *task;
            delete task;
        }
        if (!DFTU_RESULT_OK(out))
            throw std::runtime_error(describe(DFTU_RESULT_ERROR(out)));

        ::dftu_dataframe* frame = DFTU_RESULT_VALUE(out);
        if (!frame) co_return std::nullopt;

        DataFrame df = dataframe_handle_take(frame);
        Morsel m;
        m.rows = df.num_rows();
        m.columns = std::move(df.columns);
        co_return m;
    }

   private:
    const ::dftu_cursor_vt* vt_;
    void* self_;
};

// Ignores req.projection and reports every filter Pushed::No: always sound,
// since the engine re-applies whatever a source did not translate.
class ProviderSource final : public Source {
   public:
    explicit ProviderSource(ProviderEntry entry) : entry_(entry) {}

    Schema schema() const override {
        Schema s;
        const char* const* names = nullptr;
        const std::int32_t n = entry_.vt.schema(entry_.self, &names);
        if (n > 0 && names) {
            s.names.reserve(static_cast<std::size_t>(n));
            for (std::int32_t i = 0; i < n; ++i)
                s.names.emplace_back(names[i] ? names[i] : "");
        }
        return s;
    }

    ScanResult scan(const ScanRequest& req) const override {
        ScanResult r;
        r.filters.assign(req.filters.size(), Pushed::No);
        void* cursor_self = nullptr;
        const ::dftu_cursor_vt* vt = nullptr;
        if (!entry_.vt.scan(entry_.self, &cursor_self, &vt) || !vt) return r;
        r.cursor = std::make_unique<ProviderCursor>(vt, cursor_self);
        return r;
    }

   private:
    ProviderEntry entry_;
};

}  // namespace
}  // namespace dftracer::utils::dataframe

extern "C" {

int dftu_provider_register(const char* name, const dftu_source_vt* vt,
                           void* self) {
    if (!name || !vt) return 1;
    using dftracer::utils::dataframe::ProviderEntry;
    std::lock_guard<std::mutex> lock(
        dftracer::utils::dataframe::registry_mutex());
    auto& reg = dftracer::utils::dataframe::providers();
    if (reg.find(name) != reg.end()) return 1;
    reg.emplace(name, ProviderEntry{*vt, self});
    return 0;
}

int dftu_provider_unregister(const char* name) {
    if (!name) return 1;
    dftracer::utils::dataframe::ProviderEntry removed;
    {
        std::lock_guard<std::mutex> lock(
            dftracer::utils::dataframe::registry_mutex());
        auto& reg = dftracer::utils::dataframe::providers();
        auto it = reg.find(name);
        if (it == reg.end()) return 1;
        removed = it->second;
        reg.erase(it);
    }
    // Outside the lock: destroy() is caller code (a plugin's), which must not
    // run while the registry lock is held.
    if (removed.vt.destroy) removed.vt.destroy(removed.self);
    return 0;
}

dftu_lazyframe* dftu_lazyframe_from_provider(const char* name) {
    std::optional<dftracer::utils::dataframe::ProviderEntry> entry =
        dftracer::utils::dataframe::find_provider(name);
    if (!entry) return nullptr;
    std::shared_ptr<dftracer::utils::dataframe::Source> source =
        std::make_shared<dftracer::utils::dataframe::ProviderSource>(*entry);
    return dftracer::utils::dataframe::lazyframe_handle_wrap(
        dftracer::utils::dataframe::LazyFrame::scan(std::move(source)));
}

}  // extern "C"
