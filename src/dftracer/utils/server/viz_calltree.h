#ifndef DFTRACER_UTILS_SERVER_VIZ_CALLTREE_H
#define DFTRACER_UTILS_SERVER_VIZ_CALLTREE_H

// Call-tree (flame) primitives: the folded-tree node/arena, per-event
// folding, arena merge, and node serialization. The scanning worker and
// handler stay in viz_api.cpp. Internal to the server.

#include <dftracer/utils/server/viz_internal.h>
#include <simdjson.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dftracer::utils::server {

using utilities::common::json::json_number;

// One scanned event, reduced to what the call-tree needs.
struct FlameEv {
    std::int64_t pid = 0;
    std::int64_t tid = 0;
    double ts = 0;
    double dur = 0;
    std::string name;
};

// A node in the merged call tree: identical name-paths across all lanes fold
// into one node. `total` is inclusive; `self` is total minus nested children.
struct FlameNode {
    std::string name;
    double total = 0;
    double self = 0;
    std::uint64_t count = 0;
    dftracer::utils::StringViewMap<std::uint32_t> kids;
    std::vector<std::uint32_t> children;
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

// Fold one event under `base` in `arena` via the open-stack containment walk.
// kids are keyed by owned name copies (StringViewMap), so the source events may
// be discarded afterwards.
static void fold_flame_event(
    std::vector<FlameNode>& arena,
    std::vector<std::pair<double, std::uint32_t>>& open, std::uint32_t base,
    const FlameEv& ev) {
    double e_end = ev.ts + (ev.dur > 0 ? ev.dur : 0);
    while (!open.empty() && open.back().first <= ev.ts) open.pop_back();
    std::uint32_t parent = open.empty() ? base : open.back().second;
    std::uint32_t mi;
    auto it = arena[parent].kids.find(ev.name);
    if (it == arena[parent].kids.end()) {
        mi = static_cast<std::uint32_t>(arena.size());
        arena.emplace_back();
        arena[mi].name = ev.name;
        arena[parent].kids.emplace(ev.name, mi);
        arena[parent].children.push_back(mi);
    } else {
        mi = it->second;
    }
    arena[mi].total += ev.dur;
    arena[mi].self += ev.dur;
    arena[mi].count += 1;
    if (parent != 0) arena[parent].self -= ev.dur;
    open.push_back({e_end, mi});
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
        open.clear();
        for (; i < evs.size() && evs[i].pid == pid && evs[i].tid == tid; ++i)
            fold_flame_event(arena, open, base, evs[i]);
    }
}

// Merge partial tree `src` (subtree si) into `dst` (node di), summing stats per
// name-path. src node names stay alive for the whole merge.
static void merge_flame_arena(std::vector<FlameNode>& dst, std::uint32_t di,
                              const std::vector<FlameNode>& src,
                              std::uint32_t si) {
    dst[di].total += src[si].total;
    dst[di].self += src[si].self;
    dst[di].count += src[si].count;
    for (std::uint32_t sc : src[si].children) {
        std::uint32_t dc;
        auto it = dst[di].kids.find(src[sc].name);
        if (it == dst[di].kids.end()) {
            dc = static_cast<std::uint32_t>(dst.size());
            dst.emplace_back();
            dst[dc].name = src[sc].name;
            dst[di].kids.emplace(src[sc].name, dc);
            dst[di].children.push_back(dc);
        } else {
            dc = it->second;
        }
        merge_flame_arena(dst, dc, src, sc);
    }
}

}  // namespace dftracer::utils::server

#endif  // DFTRACER_UTILS_SERVER_VIZ_CALLTREE_H
