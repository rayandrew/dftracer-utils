// dftu_provider_register/unregister and dftu_lazyframe_from_provider: a
// process-lifetime, name-keyed registry of Source vtables, mirroring the op
// registry in op_registry.cpp. dftu.svc.providers@0 (plugins/abi/providers.h)
// is a thin gated forwarder into this registry, but the registry itself is
// not plugin-scoped: a non-plugin C caller, or a future language binding,
// registers and reads it the same way a plugin does.

#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/internal/dataframe_handle.h>
#include <dftracer/utils/dataframe/internal/expr_handle.h>
#include <dftracer/utils/dataframe/internal/lazyframe_handle.h>
#include <dftracer/utils/dataframe/internal/provider_source.h>
#include <dftracer/utils/dataframe/lazyframe.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

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

// Any dftu_pushed value the host has not defined is untrusted and treated as
// DFTU_PUSHED_NO (always sound: the engine re-applies) rather than trusted,
// since a source has no way to prove an Exact claim at runtime.
Pushed to_pushed(std::int32_t v) {
    switch (v) {
        case DFTU_PUSHED_NO:
            return Pushed::No;
        case DFTU_PUSHED_INEXACT:
            return Pushed::Inexact;
        case DFTU_PUSHED_EXACT:
            return Pushed::Exact;
        default:
            return Pushed::No;
    }
}

std::string join_names(const std::vector<std::string>& names) {
    std::ostringstream out;
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i) out << ", ";
        out << names[i];
    }
    return out.str();
}

// Adapts a registered dftu_cursor_vt as a Cursor. next() follows the same
// dftu_task convention as dftu_plugin::on_batch/on_finalize: a returned task
// is awaited and freed here, the driving side, exactly as dftu_task_run does
// for a task crossing the ABI the same way.
class ProviderCursor final : public Cursor {
   public:
    ProviderCursor(const ::dftu_cursor_vt* vt, void* self,
                   std::vector<std::string> expected_projection)
        : vt_(vt),
          self_(self),
          expected_projection_(std::move(expected_projection)) {}

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
        // Checked on every morsel, not just the first: the check is
        // O(projected columns), negligible next to the morsel's row data, and
        // the contract binds every dftu_dataframe the cursor produces, not
        // only its first.
        if (!expected_projection_.empty() && df.names != expected_projection_)
            throw std::runtime_error(
                "provider scan: frame columns [" + join_names(df.names) +
                "] do not match the requested projection [" +
                join_names(expected_projection_) + "]");

        Morsel m;
        m.rows = df.num_rows();
        m.columns = std::move(df.columns);
        co_return m;
    }

   private:
    const ::dftu_cursor_vt* vt_;
    void* self_;
    std::vector<std::string> expected_projection_;
};

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

        std::vector<const char*> projection;
        projection.reserve(req.projection.size());
        for (const std::string& name : req.projection)
            projection.push_back(name.c_str());

        std::vector<dftu_expr*> owned_filters;
        owned_filters.reserve(req.filters.size());
        std::vector<const dftu_expr*> filters;
        filters.reserve(req.filters.size());
        for (const Expr& f : req.filters) {
            dftu_expr* h = expr_handle_wrap(f);
            owned_filters.push_back(h);
            filters.push_back(h);
        }

        ::dftu_scan_request creq{};
        creq.projection = projection.empty() ? nullptr : projection.data();
        creq.n_projection = static_cast<std::int32_t>(projection.size());
        creq.filters = filters.empty() ? nullptr : filters.data();
        creq.n_filters = static_cast<std::int32_t>(filters.size());
        creq.limit = req.limit;
        creq.memory_budget = req.memory_budget;

        // Pre-filled DFTU_PUSHED_NO: a provider that writes nothing, or writes
        // only some entries, degrades to "the engine re-applies", which is
        // always sound.
        std::vector<std::int32_t> out_pushed(filters.size(), DFTU_PUSHED_NO);

        void* cursor_self = nullptr;
        const ::dftu_cursor_vt* vt = nullptr;
        void* ok = entry_.vt.scan(entry_.self, &creq, out_pushed.data(),
                                  &cursor_self, &vt);

        for (dftu_expr* h : owned_filters) dftu_expr_free(h);

        if (!ok || !vt) return r;

        for (std::size_t i = 0; i < r.filters.size(); ++i)
            r.filters[i] = to_pushed(out_pushed[i]);
        r.cursor =
            std::make_unique<ProviderCursor>(vt, cursor_self, req.projection);
        return r;
    }

   private:
    ProviderEntry entry_;
};

}  // namespace

std::shared_ptr<Source> make_provider_source(const ::dftu_source_vt& vt,
                                             void* self) {
    return std::make_shared<ProviderSource>(ProviderEntry{vt, self});
}

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
