#ifndef DFTRACER_UTILS_PLUGINS_PLUGINS_H
#define DFTRACER_UTILS_PLUGINS_PLUGINS_H

#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/trace/views/view.h>

#include <cstddef>
#include <memory>
#include <string>

// Load plugin shared libraries and run them as folds over one fused scan of a
// View. Free of the C ABI: the ABI types and the test seams live in
// src/dftracer/utils/plugins/plugins_internal.h.
namespace dftracer::utils::plugins {

class ConfigTree;

/// One run's output: the scan counters and the named results the plugins
/// emitted. Owned by the value, so one plugin set can back many runs.
struct PluginRun {
    trace::views::ExportStats stats;
    NamedResultRegistry results;
};

/// An immutable set of loaded plugins. Everything a run needs - dlopen, the ABI
/// gate, capability resolution, the fold order, and the index prune - is
/// settled by Builder::build(), so the set holds no per-run state. Adding a
/// plugin means building a new set.
class Plugins {
   public:
    class Builder {
       public:
        Builder();
        ~Builder();
        Builder(Builder&&) noexcept;
        Builder& operator=(Builder&&) noexcept;

        /// Queue a shared library and the config tree its factory receives.
        Builder& add(std::string path, ConfigTree config);
        Builder& add(std::string path);

        /// dlopen, gate the ABI version, then declare and resolve every
        /// capability, in queue order. The first failure names the plugin and
        /// the cause; nothing is partially built and no plugin outlives the
        /// failed call.
        Result<Plugins> build();

       private:
        struct State;
        std::unique_ptr<State> state_;
    };

    static Builder builder();

    /// The set's private state; nameable so the internal seams can reach it,
    /// defined only in the implementation.
    struct Impl;

    Plugins(Plugins&&) noexcept;
    Plugins& operator=(Plugins&&) noexcept;
    ~Plugins();

    std::size_t size() const;

    /// Drive every plugin as a fold over one fused, pruned scan of `view`.
    coro::CoroTask<Result<PluginRun>> run(const trace::views::View& view) const;

    /// `view` narrowed by the set's prune - the union of the plugins'
    /// plan_query filters, the weakest predicate that still selects every event
    /// any plugin keeps - or `view` unchanged when no prune applies. Only
    /// chunks no plugin wants are skipped, so per-plugin results are the same
    /// either way. Apply it to the base view of a plugins-only session to give
    /// that session the pruning run() does.
    trace::views::View prune(const trace::views::View& view) const;

    /// Attach every plugin to `session` as a fused fold branch, in fold order,
    /// so they co-scan with its other branches. Call before session.execute();
    /// `results` must outlive the session and holds the named results after it.
    void attach(trace::views::ViewSession& session,
                NamedResultRegistry& results) const;

   private:
    explicit Plugins(std::unique_ptr<Impl> impl);
    friend struct PluginsInternalAccess;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_PLUGINS_H
