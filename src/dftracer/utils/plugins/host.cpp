#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/config.h>
#include <dftracer/utils/plugins/fold_adapter.h>
#include <dftracer/utils/plugins/host.h>
#include <dlfcn.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <deque>
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

struct PluginHost::Impl {
    struct Loaded {
        void* handle = nullptr;
        dftu_plugin* plugin = nullptr;
        bool owned = true; /* injected test plugins are not owned */
        std::string name;  /* load-path stem, used to tag the plugin's logs */
    };
    std::vector<Loaded> plugins;
    // deque pins each ConfigTree so factory-held roots stay valid until
    // teardown.
    std::deque<ConfigTree> configs;
    // Fold order resolve() computed: providers before requirers. A full
    // permutation once resolve() succeeds, else empty (run() falls back to
    // natural order).
    std::vector<std::size_t> order;
    // Named results the most recent run() collected; outlives run() so the
    // caller can drain them.
    NamedResultRegistry named_results;
};

PluginHost::PluginHost() : impl_(std::make_unique<Impl>()) {}

PluginHost::~PluginHost() {
    // destroy() must run before dlclose unmaps the plugin's code.
    for (auto& p : impl_->plugins) {
        if (p.owned && p.plugin && p.plugin->destroy)
            p.plugin->destroy(p.plugin->self);
        if (p.handle) dlclose(p.handle);
    }
}

const dftu_value* PluginHost::add_config(ConfigTree tree) {
    impl_->configs.push_back(std::move(tree));
    return impl_->configs.back().root();
}

// The path's file stem (drop directory and the final extension), used to tag
// the plugin's log lines. "/x/y/name_edges.so" -> "name_edges".
static std::string plugin_name_from_path(std::string_view path) {
    std::size_t slash = path.find_last_of("/\\");
    std::string_view base =
        slash == std::string_view::npos ? path : path.substr(slash + 1);
    std::size_t dot = base.find_last_of('.');
    if (dot != std::string_view::npos && dot != 0) base = base.substr(0, dot);
    return std::string(base);
}

bool PluginHost::load(std::string_view path, const dftu_value* config) {
    const std::string path_str(path);
    void* handle = dlopen(path_str.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        DFTRACER_UTILS_LOG_ERROR("Failed to load plugin '%s': %s",
                                 path_str.c_str(), dlerror());
        return false;
    }

    auto factory = reinterpret_cast<dftu_plugin_factory>(
        dlsym(handle, DFTRACER_PLUGIN_FACTORY_SYMBOL));
    if (!factory) {
        DFTRACER_UTILS_LOG_ERROR("Plugin '%s' exports no '%s' symbol: %s",
                                 path_str.c_str(),
                                 DFTRACER_PLUGIN_FACTORY_SYMBOL, dlerror());
        dlclose(handle);
        return false;
    }

    dftu_plugin* plugin = factory(config);
    if (!plugin) {
        DFTRACER_UTILS_LOG_ERROR("Plugin '%s' factory returned null",
                                 path_str.c_str());
        dlclose(handle);
        return false;
    }

    if (plugin->abi_version != DFTRACER_PLUGIN_ABI_VERSION) {
        DFTRACER_UTILS_LOG_ERROR(
            "Plugin '%s' ABI version %u does not match host %u",
            path_str.c_str(), plugin->abi_version, DFTRACER_PLUGIN_ABI_VERSION);
        if (plugin->destroy) plugin->destroy(plugin->self);
        dlclose(handle);
        return false;
    }

    impl_->plugins.push_back(
        {handle, plugin, true, plugin_name_from_path(path_str)});
    return true;
}

void PluginHost::inject_plugin(dftu_plugin* plugin) {
    impl_->plugins.push_back({nullptr, plugin, false, {}});
}

namespace {

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

}  // namespace

bool PluginHost::resolve() {
    ResolveCtx ctx;

    for (std::size_t i = 0; i < impl_->plugins.size(); ++i) {
        const dftu_plugin* pl = impl_->plugins[i].plugin;
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
            if (std::strncmp(c.id, "dftu.", 5) == 0) {
                DFTRACER_UTILS_LOG_ERROR(
                    "Plugin %zu declares reserved capability '%s'; the 'dftu.' "
                    "namespace is host-only",
                    i, c.id);
                return false;
            }
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
    for (std::size_t i = 0; i < impl_->plugins.size(); ++i) {
        const dftu_plugin* pl = impl_->plugins[i].plugin;
        const dftu_plugin_comms* comms = plugin_comms(pl);
        if (!comms) continue;
        if (comms->require_caps) {
            std::uint32_t n = comms->require_caps(pl->self, nullptr, 0);
            std::vector<dftu_requirement> reqs(n);
            if (n) comms->require_caps(pl->self, reqs.data(), n);
            for (std::uint32_t k = 0; k < n; ++k) {
                const dftu_requirement& req = reqs[k];
                if (req.required &&
                    comms_provider_best(&ctx, &req, nullptr) != 0) {
                    DFTRACER_UTILS_LOG_ERROR(
                        "Plugin %zu requires unmet capability '%s'", i,
                        req.id ? req.id : "");
                    return false;
                }
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

    impl_->order = topo_sort(impl_->plugins.size(), edges);
    if (impl_->order.empty() && !impl_->plugins.empty()) {
        // Ports are best-effort (consume yields NULL when unpublished), so a
        // provide/require cycle degrades to natural order rather than failing.
        DFTRACER_UTILS_LOG_WARN(
            "Plugin capability graph has a cycle; keeping natural fold order");
        impl_->order.resize(impl_->plugins.size());
        for (std::size_t i = 0; i < impl_->order.size(); ++i)
            impl_->order[i] = i;
    }
    return true;
}

std::vector<std::size_t> PluginHost::fold_order() const {
    if (impl_->order.size() == impl_->plugins.size()) return impl_->order;
    std::vector<std::size_t> natural(impl_->plugins.size());
    for (std::size_t i = 0; i < natural.size(); ++i) natural[i] = i;
    return natural;
}

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

bool PluginHost::empty() const { return impl_->plugins.empty(); }

std::size_t PluginHost::size() const { return impl_->plugins.size(); }

coro::CoroTask<ExportStats> PluginHost::run(const View& view) const {
    dftracer::utils::StringIntern intern;
    // One registry for the whole scan; every plugin's finalize publishes its
    // merged handles here and consumers read them by cap id.
    SharedResultRegistry results;
    impl_->named_results.clear();
    std::vector<std::unique_ptr<PluginFold>> owned;
    owned.reserve(impl_->plugins.size());
    std::vector<Fold*> folds;
    folds.reserve(impl_->plugins.size());
    for (std::size_t i : fold_order()) {
        owned.push_back(std::make_unique<PluginFold>(
            impl_->plugins[i].plugin, intern, &results, &impl_->named_results,
            impl_->plugins[i].name));
        folds.push_back(owned.back().get());
    }

    // Feed the union of every plugin's plan_query into the scan's index prune.
    std::vector<const char*> plan_queries;
    plan_queries.reserve(impl_->plugins.size());
    for (const auto& p : impl_->plugins)
        plan_queries.push_back(p.plugin->plan_query
                                   ? p.plugin->plan_query(p.plugin->self)
                                   : nullptr);
    std::optional<View> filtered;
    if (auto q = detail::plugin_union_prune_query(plan_queries))
        filtered = view.filter(std::move(*q));

    const View& scan_view = filtered ? *filtered : view;
    co_return co_await scan_view.run_folds(folds, intern);
}

void PluginHost::attach_to_session(trace::views::ViewSession& session) const {
    namespace views = trace::views;
    // Captured by the factory closures (owned by the session through execute)
    // so it outlives the scan.
    auto results = std::make_shared<SharedResultRegistry>();
    impl_->named_results.clear();
    NamedResultRegistry* named = &impl_->named_results;
    std::vector<std::size_t> order = fold_order();
    if (order.empty())
        for (std::size_t i = 0; i < impl_->plugins.size(); ++i)
            order.push_back(i);
    for (std::size_t i : order) {
        const dftu_plugin* plugin = impl_->plugins[i].plugin;
        std::string name = impl_->plugins[i].name;
        session.attach_fold_factory(
            [plugin, results, named,
             name](dftracer::utils::StringIntern& intern)
                -> std::unique_ptr<views::detail::Fold> {
                return std::make_unique<PluginFold>(plugin, intern,
                                                    results.get(), named, name);
            },
            []() {});
    }
}

NamedResultRegistry& PluginHost::results() const {
    return impl_->named_results;
}

}  // namespace dftracer::utils::plugins
