#include <dftracer/utils/call_tree/critical_path.h>
#include <dftracer/utils/call_tree/internal/call_tree.h>
#include <dftracer/utils/call_tree/internal/node.h>
#include <dftracer/utils/call_tree/internal/process_call_tree.h>

#include <algorithm>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

namespace dftracer::utils::call_tree {

namespace {

struct NameAgg {
    std::uint64_t total_exclusive_us = 0;
    std::uint64_t on_path_root_count = 0;
};

using internal::CallTreeNode;
using internal::ProcessCallTree;

const CallTreeNode* critical_child(const ProcessCallTree& graph,
                                   const CallTreeNode& node) {
    const CallTreeNode* best = nullptr;
    for (std::uint64_t cid : node.get_children()) {
        auto it = graph.calls.find(cid);
        if (it == graph.calls.end()) continue;
        const CallTreeNode* c = it->second.get();
        if (best == nullptr) {
            best = c;
            continue;
        }
        std::uint64_t c_end = c->get_start_time() + c->get_duration();
        std::uint64_t b_end = best->get_start_time() + best->get_duration();
        bool better =
            std::make_tuple(c_end, c->get_duration(), best->get_id()) >
            std::make_tuple(b_end, best->get_duration(), c->get_id());
        if (better) best = c;
    }
    return best;
}

void walk_root(const ProcessCallTree& graph, const CallTreeNode& root,
               std::unordered_map<std::string, NameAgg>& agg,
               utilities::common::statistics::DDSketch& sketch) {
    std::unordered_set<std::string> seen;
    const CallTreeNode* node = &root;
    while (node != nullptr) {
        const CallTreeNode* child = critical_child(graph, *node);
        std::uint64_t exclusive =
            child ? node->get_duration() - child->get_duration()
                  : node->get_duration();
        std::string name(node->get_name());
        NameAgg& a = agg[name];
        a.total_exclusive_us += exclusive;
        if (seen.insert(name).second) a.on_path_root_count += 1;
        node = child;
    }
    sketch.add(static_cast<double>(root.get_duration()));
}

}  // namespace

CriticalPathProfile aggregate_critical_paths(const CallTree& tree) {
    CriticalPathProfile profile;
    std::unordered_map<std::string, NameAgg> agg;

    auto& itree = const_cast<CallTree&>(tree).internal_tree();
    for (const auto& key : itree.keys()) {
        ProcessCallTree* graph = itree.get(key);
        if (graph == nullptr) continue;
        for (std::uint64_t root_id : graph->root_calls) {
            auto it = graph->calls.find(root_id);
            if (it == graph->calls.end()) continue;
            walk_root(*graph, *it->second, agg, profile.duration_sketch);
            profile.total_roots += 1;
        }
    }

    profile.operations.reserve(agg.size());
    for (auto& [name, a] : agg) {
        profile.operations.push_back(
            {name, a.total_exclusive_us, a.on_path_root_count});
    }
    std::sort(profile.operations.begin(), profile.operations.end(),
              [](const CriticalPathOp& l, const CriticalPathOp& r) {
                  return std::tie(r.total_exclusive_us, l.name) <
                         std::tie(l.total_exclusive_us, r.name);
              });

    if (!profile.duration_sketch.empty()) {
        profile.p50_us = profile.duration_sketch.quantile(0.50);
        profile.p90_us = profile.duration_sketch.quantile(0.90);
        profile.p95_us = profile.duration_sketch.quantile(0.95);
        profile.p99_us = profile.duration_sketch.quantile(0.99);
    }
    return profile;
}

}  // namespace dftracer::utils::call_tree
