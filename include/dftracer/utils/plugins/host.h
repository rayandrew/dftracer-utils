#ifndef DFTRACER_UTILS_PLUGINS_HOST_H
#define DFTRACER_UTILS_PLUGINS_HOST_H

#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/query/query.h>
#include <dftracer/utils/trace/views/view.h>

#include <memory>
#include <optional>
#include <string_view>
#include <vector>

struct dftu_value;
struct dftu_plugin;

// Load plugin shared libraries and run them as folds over one fused scan of a
// View. Kept free of the C ABI via a pimpl.
namespace dftracer::utils::plugins {

class ConfigTree;

namespace detail {

/// Union of the loaded plugins' plan_query strings for coarse index pruning:
/// the weakest predicate that still selects every event any plugin's own filter
/// keeps, so per-fold results are unchanged and only chunks/members no plugin
/// wants are skipped. `plan_queries` is one entry per plugin (nullptr/empty =
/// that plugin declared none). nullopt means match all (prune nothing): a
/// plugin wants every event, or a query failed to parse. Identical strings are
/// deduplicated so the union never repeats a term.
std::optional<query::Query> plugin_union_prune_query(
    const std::vector<const char*>& plan_queries);

}  // namespace detail

class PluginHost {
   public:
    PluginHost();
    ~PluginHost();

    PluginHost(const PluginHost&) = delete;
    PluginHost& operator=(const PluginHost&) = delete;

    /// Take ownership of a config tree and return its root; valid until this
    /// PluginHost is destroyed.
    const dftu_value* add_config(ConfigTree tree);

    /// dlopen and instantiate the plugin; false on load, symbol, or ABI-version
    /// failure (already logged).
    bool load(std::string_view path, const dftu_value* config = nullptr);

    /// Test seam: register an already-built plugin without dlopen; the caller
    /// keeps ownership of the struct (no destroy, no dlclose runs for it).
    void inject_plugin(dftu_plugin* plugin);

    /// Declare every plugin's capabilities into an authoritative registry, then
    /// resolve each plugin's requirements against it and call its resolve()
    /// once. False on a reserved-namespace violation or an unmet required
    /// capability (already logged).
    bool resolve();

    /// The fold order resolve() computed: a permutation of plugin indices with
    /// every capability provider before the plugins that require it. Natural
    /// order (0..n-1) until resolve() runs or if a provide/require cycle forced
    /// a fallback.
    std::vector<std::size_t> fold_order() const;

    bool empty() const;
    std::size_t size() const;

    /// Drive every loaded plugin as a fold over one fused scan of `view`.
    /// Named results emitted during the scan are collected into results().
    coro::CoroTask<trace::views::ExportStats> run(
        const trace::views::View& view) const;

    /// The named results the most recent run() collected, keyed by name.
    NamedResultRegistry& results() const;

   private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_HOST_H
