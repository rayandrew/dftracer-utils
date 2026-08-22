#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/call_tree/call_tree.h>
#include <dftracer/utils/call_tree/flamegraph.h>
#include <dftracer/utils/call_tree/internal/call_tree.h>
#include <dftracer/utils/call_tree/internal/node.h>
#include <dftracer/utils/call_tree/internal/process_call_tree.h>
#include <dftracer/utils/call_tree/internal/process_key.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <string>

using dftracer::utils::call_tree::aggregate_flamegraph;
using dftracer::utils::call_tree::CallTree;
using dftracer::utils::call_tree::diff_flamegraphs;
using dftracer::utils::call_tree::FlamegraphDiff;
using dftracer::utils::call_tree::FlamegraphNode;
using dftracer::utils::call_tree::internal::ProcessKey;

namespace {

struct Span {
    std::uint64_t id;
    std::string name;
    std::uint64_t start;
    std::uint64_t duration;
    int level;
    std::uint64_t parent;  // 0 == root
};

std::deque<std::string>& name_pool() {
    static std::deque<std::string> pool;
    return pool;
}

void add_root(CallTree& tree, const ProcessKey& key,
              const std::vector<Span>& spans) {
    auto& itree = tree.internal_tree();
    auto* graph = &itree[key];
    for (const auto& s : spans) {
        const std::string& name = name_pool().emplace_back(s.name);
        auto node = itree.get_factory().create_node(
            s.id, name, "function", s.start, s.duration, s.level);
        if (s.parent != 0) {
            node->set_parent_id(s.parent);
            graph->calls[s.parent]->add_child(s.id);
        }
        itree.add_call(key, node);
        if (s.parent == 0) graph->root_calls.push_back(s.id);
    }
}

const FlamegraphNode* find_path(const std::vector<FlamegraphNode>& nodes,
                                const std::string& path) {
    for (const auto& n : nodes)
        if (n.path == path) return &n;
    return nullptr;
}

}  // namespace

TEST_CASE("flamegraph - duplicate-named siblings fold into one path") {
    CallTree tree;
    add_root(tree, ProcessKey(1, 1),
             {{1, "main", 0, 100, 0, 0},
              {2, "read", 0, 30, 1, 1},
              {3, "read", 30, 20, 1, 1},
              {4, "write", 50, 40, 1, 1}});

    auto p = aggregate_flamegraph(tree);

    const auto* main = find_path(p.nodes, "main");
    const auto* read = find_path(p.nodes, "main/read");
    const auto* write = find_path(p.nodes, "main/write");
    REQUIRE(main != nullptr);
    REQUIRE(read != nullptr);
    REQUIRE(write != nullptr);

    CHECK(main->total_weight == 100);
    CHECK(main->self_weight == 10);   // 100 - (read 50 + write 40)
    CHECK(read->total_weight == 50);  // two folded siblings: 30 + 20
    CHECK(read->self_weight == 50);
    CHECK(write->total_weight == 40);
    CHECK(p.total == 100);

    CHECK(read->parent_id == main->id);
    CHECK(main->parent_id == 0);
}

TEST_CASE("flamegraph - deep path folds across occurrences") {
    CallTree tree;
    add_root(tree, ProcessKey(1, 1),
             {{1, "main", 0, 100, 0, 0},
              {2, "read", 0, 60, 1, 1},
              {3, "decode", 0, 40, 2, 2}});
    add_root(tree, ProcessKey(2, 1),
             {{10, "main", 0, 50, 0, 0},
              {11, "read", 0, 30, 1, 10},
              {12, "decode", 0, 20, 2, 11}});

    auto p = aggregate_flamegraph(tree);

    const auto* decode = find_path(p.nodes, "main/read/decode");
    REQUIRE(decode != nullptr);
    CHECK(decode->depth == 2);
    CHECK(decode->total_weight == 60);  // 40 + 20
    CHECK(decode->self_weight == 60);

    const auto* read = find_path(p.nodes, "main/read");
    REQUIRE(read != nullptr);
    CHECK(read->depth == 1);
    CHECK(read->total_weight == 90);  // 60 + 30
    CHECK(read->self_weight == 30);   // 90 - 60
    CHECK(decode->parent_id == read->id);
}

TEST_CASE("flamegraph - diff ranks by absolute delta") {
    auto build = [](CallTree& tree, std::uint64_t read_dur) {
        add_root(tree, ProcessKey(1, 1),
                 {{1, "main", 0, 100, 0, 0}, {2, "read", 0, read_dur, 1, 1}});
    };
    CallTree a, b;
    build(a, 50);
    build(b, 80);

    auto pa = aggregate_flamegraph(a);
    auto pb = aggregate_flamegraph(b);
    auto d = diff_flamegraphs(pa, pb);

    const FlamegraphDiff* read = nullptr;
    for (const auto& e : d)
        if (e.path == "main/read") read = &e;
    REQUIRE(read != nullptr);
    CHECK(read->a_weight == 50);
    CHECK(read->b_weight == 80);
    CHECK(read->delta == 30);
    CHECK(d.front().path == "main/read");  // |30| > |main delta 0|
}

TEST_CASE("flamegraph - deterministic and weight-desc ordered") {
    auto build = [](CallTree& tree) {
        add_root(tree, ProcessKey(1, 1),
                 {{1, "main", 0, 100, 0, 0},
                  {2, "read", 0, 30, 1, 1},
                  {3, "read", 30, 20, 1, 1},
                  {4, "write", 50, 40, 1, 1}});
    };
    CallTree t1, t2;
    build(t1);
    build(t2);
    auto p1 = aggregate_flamegraph(t1);
    auto p2 = aggregate_flamegraph(t2);

    REQUIRE(p1.nodes.size() == p2.nodes.size());
    for (std::size_t i = 0; i < p1.nodes.size(); ++i) {
        CHECK(p1.nodes[i].path == p2.nodes[i].path);
        CHECK(p1.nodes[i].total_weight == p2.nodes[i].total_weight);
        CHECK(p1.nodes[i].self_weight == p2.nodes[i].self_weight);
    }

    CHECK(p1.nodes[0].path == "main");       // depth 0 first
    CHECK(p1.nodes[1].path == "main/read");  // 50 > 40 at depth 1
    CHECK(p1.nodes[2].path == "main/write");
}
