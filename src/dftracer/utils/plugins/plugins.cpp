#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/config.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/plugins.h>
#include <dftracer/utils/plugins/plugins_internal.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dftracer::utils::plugins {

namespace {
using View = trace::views::View;
using ExportStats = trace::views::ExportStats;
using Fold = trace::views::detail::Fold;
using Query = query::Query;
}  // namespace

struct Plugins::Impl {
    struct Loaded {
        void* handle = nullptr;
        dftu_plugin* plugin = nullptr;
        bool owned = true; /* injected test plugins are not owned */
        std::string name;  /* load-path stem, used to tag the plugin's logs */
    };

    ~Impl() {
        // destroy() must run before dlclose unmaps the plugin's code.
        for (auto& p : plugins) {
            if (p.owned && p.plugin && p.plugin->destroy)
                p.plugin->destroy(p.plugin->self);
            if (p.handle) dlclose(p.handle);
        }
    }

    std::vector<Loaded> plugins;
    // deque pins each ConfigTree so factory-held roots stay valid until
    // teardown.
    std::deque<ConfigTree> configs;
    // Providers before requirers; a full permutation of plugin indices.
    std::vector<std::size_t> order;
    // The union prune, settled once at build so run() and attach() agree.
    std::optional<Query> prune;
};

struct Plugins::Builder::State {
    struct Pending {
        std::string path;
        ConfigTree config;
        bool has_config = false;
    };
    std::vector<Pending> pending;
};

/// Private-member seam for plugins_internal.h; keeps the C ABI out of
/// plugins.h.
struct PluginsInternalAccess {
    static Plugins make(std::unique_ptr<Plugins::Impl> impl) {
        return Plugins(std::move(impl));
    }
    static const Plugins::Impl& impl(const Plugins& set) { return *set.impl_; }
};

namespace {

// The path's file stem (drop directory and the final extension), used to tag
// the plugin's log lines. "/x/y/name_edges.so" -> "name_edges".
std::string plugin_name_from_path(std::string_view path) {
    std::size_t slash = path.find_last_of("/\\");
    std::string_view base =
        slash == std::string_view::npos ? path : path.substr(slash + 1);
    std::size_t dot = base.find_last_of('.');
    if (dot != std::string_view::npos && dot != 0) base = base.substr(0, dot);
    return std::string(base);
}

Result<Plugins::Impl::Loaded> load_plugin(const std::string& path,
                                          const dftu_value* config) {
    void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* why = dlerror();
        return make_error(ErrorCode::IO,
                          "plugin '" + path + "' failed to load: " +
                              (why ? why : "unknown dlopen error"));
    }

    auto factory = reinterpret_cast<dftu_plugin_factory>(
        dlsym(handle, DFTRACER_PLUGIN_FACTORY_SYMBOL));
    if (!factory) {
        dlclose(handle);
        return make_error(ErrorCode::NOT_FOUND,
                          "plugin '" + path + "' exports no '" +
                              DFTRACER_PLUGIN_FACTORY_SYMBOL + "' symbol");
    }

    dftu_plugin* plugin = factory(config);
    if (!plugin) {
        dlclose(handle);
        return make_error(ErrorCode::INVALID_ARGUMENT,
                          "plugin '" + path + "' factory returned null");
    }

    if (plugin->abi_version != DFTRACER_PLUGIN_ABI_VERSION) {
        const std::uint32_t got = plugin->abi_version;
        if (plugin->destroy) plugin->destroy(plugin->self);
        dlclose(handle);
        return make_error(ErrorCode::INVALID_ARGUMENT,
                          "plugin '" + path + "' ABI version " +
                              std::to_string(got) + " does not match host " +
                              std::to_string(DFTRACER_PLUGIN_ABI_VERSION));
    }

    return Plugins::Impl::Loaded{handle, plugin, true,
                                 plugin_name_from_path(path)};
}

struct CapProvider {
    dftu_version ver;
    std::size_t plugin_index;
};

struct ResolveCtx {
    std::unordered_map<std::string, std::vector<CapProvider>> caps;
    dftracer::utils::StringIntern intern;
};

std::uint32_t comms_provider_count(void* h, const char* cap_id) {
    if (!cap_id) return 0;
    auto* ctx = static_cast<ResolveCtx*>(h);
    auto it = ctx->caps.find(cap_id);
    return it == ctx->caps.end()
               ? 0u
               : static_cast<std::uint32_t>(it->second.size());
}

int comms_provider_best(void* h, const dftu_requirement* req,
                        dftu_version* out_ver) {
    if (!req || !req->id) return -1;
    auto* ctx = static_cast<ResolveCtx*>(h);
    auto it = ctx->caps.find(req->id);
    if (it == ctx->caps.end()) return -1;
    const dftu_version* best = nullptr;
    for (const auto& p : it->second) {
        dftu_capability cap{req->id, p.ver};
        if (!dftu_capability_satisfies(&cap, req)) continue;
        if (!best || dftu_version_cmp(p.ver, *best) > 0) best = &p.ver;
    }
    if (!best) return -1;
    if (out_ver) *out_ver = *best;
    return 0;
}

const dftu_ext_comms g_comms = {comms_provider_count, comms_provider_best};

const void* resolve_get_extension(void*, const char* ext_id) {
    if (ext_id && std::strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_comms;
    return nullptr;
}

const char* resolve_host_resolve(void* h, dftu_str id, std::uint32_t* out_len) {
    if (out_len) *out_len = 0;
    if (id == DFTU_STR_NONE ||
        id >= dftracer::utils::StringIntern::FAST_CAPACITY)
        return nullptr;
    std::string_view sv = static_cast<ResolveCtx*>(h)->intern.resolve(id);
    if (out_len) *out_len = static_cast<std::uint32_t>(sv.size());
    return sv.data();
}

dftu_str resolve_host_intern(void* h, const char* s, std::uint32_t len) {
    if (!s) return DFTU_STR_NONE;
    try {
        return static_cast<ResolveCtx*>(h)->intern.get_or_insert(
            std::string_view{s, len});
    } catch (...) {
        return DFTU_STR_NONE;
    }
}

// Route a plugin's log line through our logger so it honors the configured
// (compile-time and runtime) level, instead of an unconditional stderr write.
void resolve_host_log(void*, std::uint8_t level, const char* s,
                      std::uint32_t n) {
    const int len = static_cast<int>(n);
    const char* msg = s ? s : "";
    switch (level) {
        case DFTU_LOG_ERROR:
            DFTRACER_UTILS_LOG_ERROR("[plugin] %.*s", len, msg);
            break;
        case DFTU_LOG_WARN:
            DFTRACER_UTILS_LOG_WARN("[plugin] %.*s", len, msg);
            break;
        case DFTU_LOG_DEBUG:
            DFTRACER_UTILS_LOG_DEBUG("[plugin] %.*s", len, msg);
            break;
        case DFTU_LOG_TRACE:
            DFTRACER_UTILS_LOG_TRACE("[plugin] %.*s", len, msg);
            break;
        default:
            DFTRACER_UTILS_LOG_INFO("[plugin] %.*s", len, msg);
            break;
    }
}

const dftu_plugin_comms* plugin_comms(const dftu_plugin* pl) {
    if (!pl->get_extension) return nullptr;
    return static_cast<const dftu_plugin_comms*>(
        pl->get_extension(pl->self, DFTU_EXT_COMMS));
}

// Kahn topological sort of `n` nodes over `from -> to` edges (from must precede
// to). Ready nodes are drained lowest-index first so an independent set keeps
// its natural order. Empty return signals a cycle.
std::vector<std::size_t> topo_sort(
    std::size_t n,
    const std::vector<std::pair<std::size_t, std::size_t>>& edges) {
    std::vector<std::vector<std::size_t>> succ(n);
    std::vector<std::size_t> indeg(n, 0);
    for (const auto& [a, b] : edges) {
        succ[a].push_back(b);
        ++indeg[b];
    }
    std::priority_queue<std::size_t, std::vector<std::size_t>,
                        std::greater<std::size_t>>
        ready;
    for (std::size_t i = 0; i < n; ++i)
        if (indeg[i] == 0) ready.push(i);
    std::vector<std::size_t> order;
    order.reserve(n);
    while (!ready.empty()) {
        std::size_t u = ready.top();
        ready.pop();
        order.push_back(u);
        for (std::size_t v : succ[u])
            if (--indeg[v] == 0) ready.push(v);
    }
    if (order.size() != n) return {};
    return order;
}

// The plugin name a diagnostic should use: the load-path stem, or the index for
// an injected plugin that has none.
std::string plugin_label(const Plugins::Impl& impl, std::size_t i) {
    const std::string& name = impl.plugins[i].name;
    return name.empty() ? ("plugin " + std::to_string(i)) : ("'" + name + "'");
}

/// Declare every plugin's capabilities into an authoritative registry, resolve
/// each plugin's requirements against it, call its resolve() once, and settle
/// the fold order.
Result<void> resolve_capabilities(Plugins::Impl& impl) {
    ResolveCtx ctx;

    for (std::size_t i = 0; i < impl.plugins.size(); ++i) {
        const dftu_plugin* pl = impl.plugins[i].plugin;
        const dftu_plugin_comms* comms = plugin_comms(pl);
        if (!comms || !comms->provides) continue;
        std::uint32_t n = comms->provides(pl->self, nullptr, 0);
        std::vector<dftu_capability> caps(n);
        if (n) comms->provides(pl->self, caps.data(), n);
        for (std::uint32_t k = 0; k < n; ++k) {
            const dftu_capability& c = caps[k];
            if (!c.id) continue;
            // The dftu. namespace is the host's alone; a plugin claiming it is
            // a hard error, not a silent shadow.
            if (std::strncmp(c.id, "dftu.", 5) == 0)
                return make_error(ErrorCode::INVALID_ARGUMENT,
                                  "plugin " + plugin_label(impl, i) +
                                      " declares reserved capability '" + c.id +
                                      "'; the 'dftu.' namespace is host-only");
            ctx.caps[c.id].push_back({c.ver, i});
        }
    }

    dftu_host rhost{};
    rhost.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    rhost.h = &ctx;
    rhost.get_extension = resolve_get_extension;
    rhost.resolve = resolve_host_resolve;
    rhost.intern = resolve_host_intern;
    rhost.log = resolve_host_log;

    // Every provider of a required capability must precede the requirer.
    std::vector<std::pair<std::size_t, std::size_t>> edges;
    for (std::size_t i = 0; i < impl.plugins.size(); ++i) {
        const dftu_plugin* pl = impl.plugins[i].plugin;
        const dftu_plugin_comms* comms = plugin_comms(pl);
        if (!comms) continue;
        if (comms->require_caps) {
            std::uint32_t n = comms->require_caps(pl->self, nullptr, 0);
            std::vector<dftu_requirement> reqs(n);
            if (n) comms->require_caps(pl->self, reqs.data(), n);
            for (std::uint32_t k = 0; k < n; ++k) {
                const dftu_requirement& req = reqs[k];
                if (req.required &&
                    comms_provider_best(&ctx, &req, nullptr) != 0)
                    return make_error(ErrorCode::NOT_FOUND,
                                      "plugin " + plugin_label(impl, i) +
                                          " requires unmet capability '" +
                                          (req.id ? req.id : "") + "'");
                if (!req.id) continue;
                auto it = ctx.caps.find(req.id);
                if (it == ctx.caps.end()) continue;
                for (const auto& p : it->second)
                    if (p.plugin_index != i)
                        edges.emplace_back(p.plugin_index, i);
            }
        }
        if (comms->resolve) comms->resolve(pl->self, &rhost);
    }

    impl.order = topo_sort(impl.plugins.size(), edges);
    if (impl.order.size() != impl.plugins.size()) {
        // Ports are best-effort (consume yields NULL when unpublished), so a
        // provide/require cycle degrades to natural order rather than failing.
        if (!impl.plugins.empty())
            DFTRACER_UTILS_LOG_WARN(
                "Plugin capability graph has a cycle; keeping natural fold "
                "order");
        impl.order.resize(impl.plugins.size());
        for (std::size_t i = 0; i < impl.order.size(); ++i) impl.order[i] = i;
    }
    return {};
}

void settle_prune(Plugins::Impl& impl) {
    std::vector<const char*> plan_queries;
    plan_queries.reserve(impl.plugins.size());
    for (const auto& p : impl.plugins)
        plan_queries.push_back(p.plugin->plan_query
                                   ? p.plugin->plan_query(p.plugin->self)
                                   : nullptr);
    impl.prune = detail::plugin_union_prune_query(plan_queries);
}

}  // namespace

namespace detail {

std::optional<Query> plugin_union_prune_query(
    const std::vector<const char*>& plan_queries) {
    namespace q = query;
    if (plan_queries.empty()) return std::nullopt;
    std::vector<std::string> distinct;
    for (const char* s : plan_queries) {
        if (!s || !*s) return std::nullopt;
        std::string_view sv{s};
        if (!q::try_parse(sv)) {
            DFTRACER_UTILS_LOG_DEBUG(
                "Plugin plan_query '%s' failed to parse; scanning all", s);
            return std::nullopt;
        }
        if (std::find(distinct.begin(), distinct.end(), sv) == distinct.end())
            distinct.emplace_back(sv);
    }
    if (distinct.size() == 1) return q::try_parse(distinct.front());
    std::string uni;
    for (const auto& d : distinct) {
        if (!uni.empty()) uni += " or ";
        uni += '(';
        uni += d;
        uni += ')';
    }
    return q::try_parse(uni);
}

}  // namespace detail

Plugins::Builder::Builder() : state_(std::make_unique<State>()) {}
Plugins::Builder::~Builder() = default;
Plugins::Builder::Builder(Builder&&) noexcept = default;
Plugins::Builder& Plugins::Builder::operator=(Builder&&) noexcept = default;

Plugins::Builder& Plugins::Builder::add(std::string path, ConfigTree config) {
    state_->pending.push_back({std::move(path), std::move(config), true});
    return *this;
}

Plugins::Builder& Plugins::Builder::add(std::string path) {
    state_->pending.push_back({std::move(path), ConfigTree{}, false});
    return *this;
}

Result<Plugins> Plugins::Builder::build() {
    auto impl = std::make_unique<Impl>();
    impl->plugins.reserve(state_->pending.size());
    for (auto& pending : state_->pending) {
        const dftu_value* root = nullptr;
        if (pending.has_config) {
            impl->configs.push_back(std::move(pending.config));
            root = impl->configs.back().root();
        }
        auto loaded = load_plugin(pending.path, root);
        if (!loaded) return unexpected(std::move(loaded).error());
        impl->plugins.push_back(std::move(*loaded));
    }
    auto resolved = resolve_capabilities(*impl);
    if (!resolved) return unexpected(std::move(resolved).error());
    settle_prune(*impl);
    return PluginsInternalAccess::make(std::move(impl));
}

Plugins::Builder Plugins::builder() { return Builder{}; }

Plugins::Plugins(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Plugins::Plugins(Plugins&&) noexcept = default;
Plugins& Plugins::operator=(Plugins&&) noexcept = default;
Plugins::~Plugins() = default;

std::size_t Plugins::size() const { return impl_->plugins.size(); }

View Plugins::prune(const View& view) const {
    return impl_->prune ? view.filter(*impl_->prune) : view;
}

coro::CoroTask<Result<PluginRun>> Plugins::run(const View& view) const {
    dftracer::utils::StringIntern intern;
    PluginRun out;
    // One registry for the whole scan; every plugin's finalize publishes its
    // merged accumulators here and later plugins read them by name.
    SharedResultRegistry shared;
    std::vector<std::unique_ptr<PluginFold>> owned;
    owned.reserve(impl_->plugins.size());
    std::vector<Fold*> folds;
    folds.reserve(impl_->plugins.size());
    for (std::size_t i : impl_->order) {
        owned.push_back(std::make_unique<PluginFold>(
            impl_->plugins[i].plugin, intern, &shared, &out.results,
            impl_->plugins[i].name));
        folds.push_back(owned.back().get());
    }

    const View pruned = prune(view);
    out.stats = co_await pruned.run_folds(folds, intern);
    co_return out;
}

void Plugins::attach(trace::views::ViewSession& session,
                     NamedResultRegistry& results) const {
    namespace views = trace::views;
    // Captured by the factory closures (owned by the session through execute)
    // so it outlives the scan.
    auto shared = std::make_shared<SharedResultRegistry>();
    results.clear();
    NamedResultRegistry* named = &results;
    for (std::size_t i : impl_->order) {
        const dftu_plugin* plugin = impl_->plugins[i].plugin;
        std::string name = impl_->plugins[i].name;
        session.attach_fold_factory(
            [plugin, shared, named, name](dftracer::utils::StringIntern& intern)
                -> std::unique_ptr<views::detail::Fold> {
                return std::make_unique<PluginFold>(plugin, intern,
                                                    shared.get(), named, name);
            },
            []() {});
    }
}

Result<Plugins> build_injected_plugins(std::vector<dftu_plugin*> plugins) {
    auto impl = std::make_unique<Plugins::Impl>();
    impl->plugins.reserve(plugins.size());
    for (dftu_plugin* pl : plugins)
        impl->plugins.push_back({nullptr, pl, false, {}});
    auto resolved = resolve_capabilities(*impl);
    if (!resolved) return unexpected(std::move(resolved).error());
    settle_prune(*impl);
    return PluginsInternalAccess::make(std::move(impl));
}

std::vector<std::size_t> fold_order(const Plugins& set) {
    return PluginsInternalAccess::impl(set).order;
}

}  // namespace dftracer::utils::plugins
