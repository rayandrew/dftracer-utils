#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/call_tree/call_tree.h>
#include <dftracer/utils/call_tree/critical_path.h>
#include <dftracer/utils/call_tree/internal/call_tree.h>
#include <dftracer/utils/call_tree/internal/node.h>
#include <dftracer/utils/call_tree/internal/process_call_tree.h>
#include <dftracer/utils/call_tree/internal/process_key.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <deque>
#include <memory>
#include <string>

using dftracer::utils::call_tree::aggregate_critical_paths;
using dftracer::utils::call_tree::CallTree;
using dftracer::utils::call_tree::CriticalPathOp;
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

// CallTreeNode holds names as non-owning views (production interns them),
// so fixture names must outlive the tree; this pool is never freed until exit.
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

const CriticalPathOp* find_op(const std::vector<CriticalPathOp>& ops,
                              const std::string& name) {
    for (const auto& o : ops)
        if (o.name == name) return &o;
    return nullptr;
}

}  // namespace

TEST_CASE("critical path - known tree with latest-ending child") {
    CallTree tree;
    add_root(tree, ProcessKey(1, 1),
             {{1, "root", 0, 100, 0, 0},
              {2, "A", 0, 30, 1, 1},
              {3, "B", 40, 50, 1, 1}});

    auto profile = aggregate_critical_paths(tree);

    CHECK(profile.total_roots == 1);
    const auto* root_op = find_op(profile.operations, "root");
    const auto* b_op = find_op(profile.operations, "B");
    REQUIRE(root_op != nullptr);
    REQUIRE(b_op != nullptr);
    CHECK(root_op->total_exclusive_us == 50);  // 100 - crit_child(B).dur(50)
    CHECK(b_op->total_exclusive_us == 50);     // leaf on path
    CHECK(root_op->on_path_root_count == 1);
    CHECK(b_op->on_path_root_count == 1);
    CHECK(find_op(profile.operations, "A") == nullptr);  // A not on path

    std::uint64_t path_sum = 0;
    for (const auto& op : profile.operations) path_sum += op.total_exclusive_us;
    CHECK(path_sum == 100);  // invariant: exclusive times sum to root duration
}

TEST_CASE("critical path - tie-breaks by duration then id") {
    SUBCASE("equal end time -> larger duration wins") {
        CallTree tree;
        add_root(tree, ProcessKey(1, 1),
                 {{1, "root", 0, 100, 0, 0},
                  {2, "short", 50, 40, 1, 1},
                  {3, "long", 40, 50, 1, 1}});
        auto p = aggregate_critical_paths(tree);
        REQUIRE(find_op(p.operations, "long") != nullptr);
        CHECK(find_op(p.operations, "short") == nullptr);
        CHECK(find_op(p.operations, "long")->total_exclusive_us == 50);
    }
    SUBCASE("equal end and duration -> smaller id wins") {
        CallTree tree;
        add_root(tree, ProcessKey(1, 1),
                 {{1, "root", 0, 100, 0, 0},
                  {5, "hi", 40, 50, 1, 1},
                  {2, "lo", 40, 50, 1, 1}});
        auto p = aggregate_critical_paths(tree);
        CHECK(find_op(p.operations, "lo") != nullptr);
        CHECK(find_op(p.operations, "hi") == nullptr);
    }
}

TEST_CASE("critical path - deep chain sums to root duration") {
    CallTree tree;
    add_root(tree, ProcessKey(2, 2),
             {{1, "L0", 0, 200, 0, 0},
              {2, "gap", 0, 10, 1, 1},
              {3, "L1", 20, 150, 1, 1},
              {4, "L2", 30, 100, 2, 3},
              {5, "L3", 40, 60, 3, 4}});

    auto p = aggregate_critical_paths(tree);
    CHECK(p.total_roots == 1);

    std::uint64_t sum = 0;
    for (const auto& op : p.operations) sum += op.total_exclusive_us;
    CHECK(sum == 200);

    CHECK(find_op(p.operations, "L0")->total_exclusive_us == 50);  // 200-150
    CHECK(find_op(p.operations, "L1")->total_exclusive_us == 50);  // 150-100
    CHECK(find_op(p.operations, "L2")->total_exclusive_us == 40);  // 100-60
    CHECK(find_op(p.operations, "L3")->total_exclusive_us == 60);  // leaf
    CHECK(find_op(p.operations, "gap") == nullptr);
}

TEST_CASE("critical path - multiple roots aggregate names and quantiles") {
    CallTree tree;
    for (std::uint32_t r = 0; r < 10; ++r) {
        std::uint64_t base = static_cast<std::uint64_t>(r) * 10;
        std::uint64_t dur = 100;
        add_root(tree, ProcessKey(1, 1 + r),
                 {{base + 1, "root", 0, dur, 0, 0},
                  {base + 2, "leaf", 10, dur - 30, 1, base + 1}});
    }

    auto p = aggregate_critical_paths(tree);
    CHECK(p.total_roots == 10);

    const auto* root_op = find_op(p.operations, "root");
    const auto* leaf_op = find_op(p.operations, "leaf");
    REQUIRE(root_op != nullptr);
    REQUIRE(leaf_op != nullptr);
    CHECK(root_op->on_path_root_count == 10);
    CHECK(leaf_op->on_path_root_count == 10);
    CHECK(root_op->total_exclusive_us == 10 * 30);  // 100 - 70 per root
    CHECK(leaf_op->total_exclusive_us == 10 * 70);

    CHECK(p.duration_sketch.count() == 10);
    CHECK(p.p50_us == doctest::Approx(100.0).epsilon(0.02));
    CHECK(p.p99_us == doctest::Approx(100.0).epsilon(0.02));

    CHECK(p.operations.front().name == "leaf");  // 700 > 300, sorted desc
}

TEST_CASE("critical path - deterministic across runs") {
    auto build = [](CallTree& tree) {
        add_root(tree, ProcessKey(1, 1),
                 {{1, "root", 0, 100, 0, 0},
                  {2, "A", 0, 30, 1, 1},
                  {3, "B", 40, 50, 1, 1}});
        add_root(tree, ProcessKey(2, 1),
                 {{10, "root", 0, 80, 0, 0}, {11, "C", 5, 60, 1, 10}});
    };
    CallTree t1, t2;
    build(t1);
    build(t2);
    auto p1 = aggregate_critical_paths(t1);
    auto p2 = aggregate_critical_paths(t2);

    REQUIRE(p1.operations.size() == p2.operations.size());
    for (std::size_t i = 0; i < p1.operations.size(); ++i) {
        CHECK(p1.operations[i].name == p2.operations[i].name);
        CHECK(p1.operations[i].total_exclusive_us ==
              p2.operations[i].total_exclusive_us);
        CHECK(p1.operations[i].on_path_root_count ==
              p2.operations[i].on_path_root_count);
    }
    CHECK(p1.total_roots == p2.total_roots);
    CHECK(p1.p50_us == p2.p50_us);
}
