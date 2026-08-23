#include <dftracer/utils/call_tree/flamegraph.h>
#include <dftracer/utils/call_tree/internal/call_tree.h>
#include <dftracer/utils/call_tree/internal/node.h>
#include <dftracer/utils/call_tree/internal/process_call_tree.h>

#include <algorithm>
#include <cstdlib>
#include <map>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace dftracer::utils::call_tree {

namespace {

using internal::CallTreeNode;
using internal::ProcessCallTree;

struct FoldAgg {
    std::string name;
    int depth = 0;
    std::uint64_t total = 0;
    std::uint64_t children_total = 0;
};

std::string join_path(const std::vector<std::string>& path) {
    std::string joined;
    for (std::size_t i = 0; i < path.size(); ++i) {
        if (i != 0) joined.push_back('/');
        joined += path[i];
    }
    return joined;
}

void walk(const ProcessCallTree& graph, const CallTreeNode& node,
          std::vector<std::string>& path,
          std::map<std::vector<std::string>, FoldAgg>& agg) {
    path.emplace_back(node.get_name());
    FoldAgg& a = agg[path];
    a.name = path.back();
    a.depth = static_cast<int>(path.size()) - 1;
    a.total += node.get_duration();
    for (std::uint64_t cid : node.get_children()) {
        auto it = graph.calls.find(cid);
        if (it == graph.calls.end()) continue;
        walk(graph, *it->second, path, agg);
    }
    path.pop_back();
}

}  // namespace

FlamegraphProfile aggregate_flamegraph(const CallTree& tree) {
    std::map<std::vector<std::string>, FoldAgg> agg;

    auto& itree = const_cast<CallTree&>(tree).internal_tree();
    for (const auto& key : itree.keys()) {
        ProcessCallTree* graph = itree.get(key);
        if (graph == nullptr) continue;
        std::vector<std::string> path;
        for (std::uint64_t root_id : graph->root_calls) {
            auto it = graph->calls.find(root_id);
            if (it == graph->calls.end()) continue;
            walk(*graph, *it->second, path, agg);
        }
    }

    for (const auto& [path, a] : agg) {
        if (path.size() < 2) continue;
        std::vector<std::string> parent(path.begin(), path.end() - 1);
        agg[parent].children_total += a.total;
    }

    FlamegraphProfile profile;
    std::vector<std::vector<std::string>> ordered;
    ordered.reserve(agg.size());
    for (const auto& [path, a] : agg) ordered.push_back(path);
    std::sort(ordered.begin(), ordered.end(),
              [&](const std::vector<std::string>& l,
                  const std::vector<std::string>& r) {
                  const FoldAgg& la = agg[l];
                  const FoldAgg& ra = agg[r];
                  if (la.depth != ra.depth) return la.depth < ra.depth;
                  if (la.total != ra.total) return la.total > ra.total;
                  return l < r;
              });

    std::map<std::vector<std::string>, std::uint64_t> ids;
    profile.nodes.reserve(ordered.size());
    for (const auto& path : ordered) {
        const FoldAgg& a = agg[path];
        std::uint64_t id = profile.nodes.size() + 1;
        ids[path] = id;
        std::uint64_t parent_id = 0;
        if (path.size() >= 2) {
            std::vector<std::string> parent(path.begin(), path.end() - 1);
            parent_id = ids[parent];
        }
        profile.nodes.push_back({id, parent_id, a.name, join_path(path),
                                 a.depth, a.total, a.total - a.children_total});
        if (path.size() == 1) profile.total += a.total;
    }
    return profile;
}

std::vector<FlamegraphDiff> diff_flamegraphs(const FlamegraphProfile& a,
                                             const FlamegraphProfile& b) {
    std::unordered_map<std::string, std::uint64_t> a_map;
    for (const auto& n : a.nodes) a_map[n.path] = n.total_weight;

    std::vector<FlamegraphDiff> diffs;
    std::unordered_map<std::string, std::uint64_t> b_map;
    for (const auto& n : b.nodes) {
        b_map[n.path] = n.total_weight;
        std::uint64_t aw = 0;
        auto it = a_map.find(n.path);
        if (it != a_map.end()) aw = it->second;
        diffs.push_back({n.path, aw, n.total_weight,
                         static_cast<std::int64_t>(n.total_weight) -
                             static_cast<std::int64_t>(aw)});
    }
    for (const auto& n : a.nodes) {
        if (b_map.find(n.path) != b_map.end()) continue;
        diffs.push_back({n.path, n.total_weight, 0,
                         -static_cast<std::int64_t>(n.total_weight)});
    }

    std::sort(
        diffs.begin(), diffs.end(),
        [](const FlamegraphDiff& l, const FlamegraphDiff& r) {
            std::uint64_t la = static_cast<std::uint64_t>(std::llabs(l.delta));
            std::uint64_t ra = static_cast<std::uint64_t>(std::llabs(r.delta));
            return std::make_tuple(ra, l.path) < std::make_tuple(la, r.path);
        });
    return diffs;
}

}  // namespace dftracer::utils::call_tree
