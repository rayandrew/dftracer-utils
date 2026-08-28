#ifndef DFTRACER_UTILS_SERVER_VIZ_CALLTREE_H
#define DFTRACER_UTILS_SERVER_VIZ_CALLTREE_H

// Call-tree (flame) primitives: the folded-tree node/arena, per-event
// folding, arena merge, and node serialization. The scanning worker and
// handler stay in viz_api.cpp. Internal to the server.

#include <dftracer/utils/dataframe/containment.h>
#include <dftracer/utils/dataframe/flame_arena.h>
#include <dftracer/utils/server/viz_internal.h>
#include <simdjson.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dftracer::utils::server {

using json::json_number;

// The folded-tree node/arena and its fold+merge live in the dataframe layer so
// the server, the columnar flamegraph() op, and the CLI share one core.
using dataframe::FlameNode;
using dataframe::fold_flame_node;
using dataframe::merge_flame_arena;

// One scanned event, reduced to what the call-tree needs.
struct FlameEv {
    std::int64_t pid = 0;
    std::int64_t tid = 0;
    double ts = 0;
    double dur = 0;
    std::string name;
};

static bool parse_flame_ev(std::string_view event, FlameEv& out) {
    EventScalars s;
    // No duration: nothing to place in the tree.
    if (!parse_event_scalars(event, s) || !s.has_dur || !s.has_ts) return false;
    out.dur = s.dur;
    out.ts = s.ts;
    out.pid = s.pid;
    out.tid = s.tid;
    out.name.assign(s.name);  // empty when absent/non-string
    return true;
}

static void serialize_flame_node(simdjson::builder::string_builder& sb,
                                 std::vector<FlameNode>& arena,
                                 std::uint32_t idx) {
    FlameNode& n = arena[idx];
    sb.start_object();
    sb.append_key_value("name", n.name);
    sb.append_comma();
    sb.append_key_value("total", n.total);
    sb.append_comma();
    sb.append_key_value("self", n.self < 0 ? 0.0 : n.self);
    sb.append_comma();
    sb.append_key_value("count", n.count);
    std::sort(n.children.begin(), n.children.end(),
              [&arena](std::uint32_t a, std::uint32_t b) {
                  return arena[a].total > arena[b].total;
              });
    sb.append_comma();
    sb.escape_and_append_with_quotes("children");
    sb.append_colon();
    sb.start_array();
    for (std::size_t i = 0; i < n.children.size(); ++i) {
        if (i > 0) sb.append_comma();
        serialize_flame_node(sb, arena, n.children[i]);
    }
    sb.end_array();
    sb.end_object();
}

// Sort one file's events by (pid,tid,ts,dur) and fold each lane into `arena`.
// Lanes never cross files (one pid per rank file), so this is a complete,
// self-contained partial tree for the file.
static void fold_file_events(
    std::vector<FlameNode>& arena,
    ankerl::unordered_dense::map<std::int64_t, std::uint32_t>& proc_of,
    std::vector<std::pair<double, std::uint32_t>>& open,
    std::vector<FlameEv>& evs, bool by_process) {
    std::sort(evs.begin(), evs.end(), [](const FlameEv& a, const FlameEv& b) {
        if (a.pid != b.pid) return a.pid < b.pid;
        if (a.tid != b.tid) return a.tid < b.tid;
        if (a.ts != b.ts) return a.ts < b.ts;
        return a.dur > b.dur;
    });
    std::size_t i = 0;
    while (i < evs.size()) {
        std::int64_t pid = evs[i].pid, tid = evs[i].tid;
        std::uint32_t base = 0;
        if (by_process) {
            auto pit = proc_of.find(pid);
            if (pit == proc_of.end()) {
                base = static_cast<std::uint32_t>(arena.size());
                arena.emplace_back();
                arena[base].name = "P" + std::to_string(pid);
                proc_of.emplace(pid, base);
                arena[0].children.push_back(base);
                arena[0].kids.emplace(arena[base].name, base);
            } else {
                base = pit->second;
            }
        }
        const std::size_t lane_begin = i;
        while (i < evs.size() && evs[i].pid == pid && evs[i].tid == tid) ++i;
        const std::int64_t lane_n = static_cast<std::int64_t>(i - lane_begin);
        dataframe::containment_walk(
            lane_n, [&](std::int64_t k) { return evs[lane_begin + k].ts; },
            [&](std::int64_t k) {
                const FlameEv& ev = evs[lane_begin + k];
                return ev.ts + (ev.dur > 0 ? ev.dur : 0);
            },
            base,
            [&](std::int64_t k, std::int64_t, std::uint32_t parent) {
                const FlameEv& ev = evs[lane_begin + k];
                return fold_flame_node(arena, ev.name, ev.dur, parent);
            },
            open);
    }
}

}  // namespace dftracer::utils::server

#endif  // DFTRACER_UTILS_SERVER_VIZ_CALLTREE_H
