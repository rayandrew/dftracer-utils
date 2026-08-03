#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/utilities/common/json/json_escape.h>
#include <dftracer/utils/utilities/common/query/query.h>
#include <dftracer/utils/utilities/common/query/subsumption.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/mv_store.h>
#include <dftracer/utils/utilities/composites/dft/views/rollup_store.h>
#include <dftracer/utils/utilities/composites/dft/views/view_plan.h>
#include <fcntl.h>
#include <simdjson.h>
#include <sys/file.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <optional>
#include <string>
#include <system_error>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::views::detail {

namespace json = utilities::common::json;
namespace query = utilities::common::query;

namespace {

constexpr const char* VIEWS_DIRNAME = ".dftindex-views";
constexpr const char* MANIFEST_NAME = "manifest";
constexpr std::size_t SLUG_MAX = 80;

struct BaseId {
    std::string path;
    std::uint64_t size = 0;
    std::int64_t mtime = 0;
};

std::optional<BaseId> stat_base(const std::string& path) {
    std::error_code ec;
    const auto sz = fs::file_size(path, ec);
    if (ec) return std::nullopt;
    const auto wt = fs::last_write_time(path, ec);
    if (ec) return std::nullopt;
    return BaseId{path, static_cast<std::uint64_t>(sz),
                  static_cast<std::int64_t>(wt.time_since_epoch().count())};
}

// Base-file identities of the plan, sorted by path so the identity hash and
// the freshness comparison are order-independent.
std::vector<BaseId> plan_bases(const ViewPlan& plan) {
    std::vector<BaseId> out;
    for (const auto& f : plan.files)
        if (auto b = stat_base(f.file_path)) out.push_back(*b);
    std::sort(out.begin(), out.end(),
              [](const BaseId& a, const BaseId& b) { return a.path < b.path; });
    return out;
}

bool allowed(char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '-';
}

void replace_all(std::string& s, const std::string& from,
                 const std::string& to) {
    std::size_t pos = 0;
    while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.size(), to);
        pos += to.size();
    }
}

// Render a predicate's canonical form to a filesystem-safe, readable token.
// Lossy by design (truncated, operators reduced to words); the identity suffix
// and manifest carry the exact query.
std::string sanitize_predicate(const std::string& canonical) {
    std::string s = canonical;
    replace_all(s, " not in ", "-nin-");
    replace_all(s, " not like ", "-notlike-");
    replace_all(s, " not ilike ", "-notilike-");
    replace_all(s, " ilike ", "-ilike-");
    replace_all(s, " like ", "-like-");
    replace_all(s, " >= ", "-ge-");
    replace_all(s, " <= ", "-le-");
    replace_all(s, " == ", "-eq-");
    replace_all(s, " != ", "-ne-");
    replace_all(s, " > ", "-gt-");
    replace_all(s, " < ", "-lt-");
    replace_all(s, " in ", "-in-");
    replace_all(s, " and ", "__");
    replace_all(s, " or ", "--or--");
    replace_all(s, "not (", "not-");

    std::string out;
    out.reserve(s.size());
    for (char c : s) {
        if (allowed(c)) {
            out.push_back(c);
        } else if (!out.empty() && out.back() != '-') {
            out.push_back('-');  // one dash per run of disallowed chars
        }
    }
    while (!out.empty() && out.back() == '-') out.pop_back();
    std::size_t start = 0;
    while (start < out.size() && out[start] == '-') ++start;
    out = out.substr(start);
    if (out.size() > SLUG_MAX) out.resize(SLUG_MAX);
    return out.empty() ? "all" : out;
}

std::string window_tag(const ViewPlan& plan) {
    if (!plan.time_range) return {};
    return "__ts-" + std::to_string(plan.time_range->first) + "-" +
           std::to_string(plan.time_range->second);
}

std::string phase_tag(const ViewPlan& plan) {
    switch (plan.phase) {
        case Phase::Events:
            return "__ph-E";
        case Phase::Counters:
            return "__ph-C";
        case Phase::Any:
            return {};
    }
    return {};
}

// Stable identity of an MV: base files (path+size+mtime), predicate, phase, and
// window. Two materializations collide only if genuinely identical.
std::uint64_t identity_hash(const ViewPlan& plan) {
    std::string k;
    for (const auto& b : plan_bases(plan)) {
        k += b.path;
        k.push_back('|');
        k += std::to_string(b.size);
        k.push_back('|');
        k += std::to_string(b.mtime);
        k.push_back('\n');
    }
    k += "q=";
    k += plan.query ? plan.query->source() : "";
    k += phase_tag(plan);
    k += window_tag(plan);
    return dftracer::utils::hash::fnv1a_hash(k);
}

std::string hex8(std::uint64_t h) {
    char buf[9];
    std::snprintf(buf, sizeof(buf), "%08x",
                  static_cast<unsigned>(h & 0xFFFFFFFFu));
    return std::string(buf);
}

std::string manifest_path(const std::string& dir) {
    return (fs::path(dir) / MANIFEST_NAME).string();
}

struct Manifest {
    std::string query;
    int phase = static_cast<int>(Phase::Any);
    bool has_window = false;
    double win_begin = 0.0;
    double win_end = 0.0;
    std::vector<BaseId> bases;
};

std::optional<Manifest> read_manifest(const std::string& dir) {
    const std::string path = manifest_path(dir);
    std::error_code ec;
    if (!fs::exists(path, ec)) return std::nullopt;
    try {
        simdjson::dom::parser parser;
        simdjson::dom::element root = parser.load(path);
        Manifest m;
        m.query = std::string(std::string_view(root["query"]));
        m.phase = static_cast<int>(std::int64_t(root["phase"]));
        simdjson::dom::element w = root["window"];
        if (!w.is_null()) {
            m.has_window = true;
            m.win_begin = double(w["begin"]);
            m.win_end = double(w["end"]);
        }
        for (simdjson::dom::element b : simdjson::dom::array(root["bases"])) {
            BaseId id;
            id.path = std::string(std::string_view(b["path"]));
            id.size = std::uint64_t(std::int64_t(b["size"]));
            id.mtime = std::int64_t(b["mtime"]);
            m.bases.push_back(std::move(id));
        }
        std::sort(
            m.bases.begin(), m.bases.end(),
            [](const BaseId& a, const BaseId& c) { return a.path < c.path; });
        return m;
    } catch (const simdjson::simdjson_error&) {
        return std::nullopt;
    }
}

// Trace files inside an MV directory (its own index is a subdir, skipped).
std::vector<std::string> view_trace_files(const std::string& dir) {
    std::vector<std::string> out;
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (!it->is_regular_file(ec)) continue;
        const std::string name = it->path().filename().string();
        if (name.size() >= 4 && name.compare(name.size() - 4, 4, ".pfw") == 0)
            out.push_back(it->path().string());
        else if (name.size() >= 7 &&
                 name.compare(name.size() - 7, 7, ".pfw.gz") == 0)
            out.push_back(it->path().string());
    }
    std::sort(out.begin(), out.end());
    return out;
}

// All part files of an MV as ViewFiles, each paired with the index that covers
// it. Parts written directly in `dir` share `dir/.dftindex`; a distributed
// materialize puts each rank's parts in a subdir with its own index.
// `.dftindex` subdirs (RocksDB dirs, not parts) are skipped.
std::vector<ViewFile> gather_view_files(const std::string& dir) {
    namespace dfti = composites::dft::internal;
    std::vector<ViewFile> out;
    const std::string top_idx = dfti::determine_index_path(dir, "");
    for (const auto& t : view_trace_files(dir))
        out.push_back(ViewFile{t, top_idx});
    std::error_code ec;
    for (fs::directory_iterator it(dir, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (!it->is_directory(ec)) continue;
        if (it->path().filename().string().rfind(".dftindex", 0) == 0) continue;
        const std::string sub = it->path().string();
        const std::string sidx = dfti::determine_index_path(sub, "");
        for (const auto& t : view_trace_files(sub))
            out.push_back(ViewFile{t, sidx});
    }
    return out;
}

std::uint64_t total_size(const std::vector<ViewFile>& files) {
    std::uint64_t sum = 0;
    std::error_code ec;
    for (const auto& f : files) {
        const auto sz = fs::file_size(f.file_path, ec);
        if (!ec) sum += sz;
    }
    return sum;
}

// A stored MV can answer `plan` only if its base set is byte-identical and
// unchanged, its phase covers the query's, its window covers the query's, and
// its predicate subsumes the query's.
bool manifest_matches(const Manifest& m, const ViewPlan& plan,
                      const std::vector<BaseId>& q_bases) {
    if (m.bases.size() != q_bases.size()) return false;
    for (std::size_t i = 0; i < m.bases.size(); ++i)
        if (m.bases[i].path != q_bases[i].path ||
            m.bases[i].size != q_bases[i].size ||
            m.bases[i].mtime != q_bases[i].mtime)
            return false;

    if (m.phase != static_cast<int>(Phase::Any) &&
        m.phase != static_cast<int>(plan.phase))
        return false;

    if (m.has_window) {
        if (!plan.time_range) return false;
        if (m.win_begin > plan.time_range->first ||
            m.win_end < plan.time_range->second)
            return false;
    }

    if (m.query.empty()) return true;  // MV keeps every row
    if (!plan.query) return false;     // query keeps every row, MV does not
    auto mv_q = query::try_parse(m.query);
    if (!mv_q) return false;
    return query::query_subsumes(mv_q->root(), plan.query->root());
}

// True if any base file the manifest was built from is now gone or changed, so
// the MV can never be served again and is safe to delete.
bool manifest_is_stale(const Manifest& m) {
    for (const auto& b : m.bases) {
        auto cur = stat_base(b.path);
        if (!cur || cur->size != b.size || cur->mtime != b.mtime) return true;
    }
    return false;
}

}  // namespace

std::string views_root(const ViewPlan& plan) {
    if (!plan.views_root.empty()) return plan.views_root;
    const std::string anchor = rollup_index_path(plan);
    if (anchor.empty()) return {};
    return (fs::path(anchor).parent_path() / VIEWS_DIRNAME).string();
}

std::string view_slug(const ViewPlan& plan) {
    const std::string canonical =
        plan.query ? query::to_string(plan.query->root()) : "all";
    return sanitize_predicate(canonical) + phase_tag(plan) + window_tag(plan) +
           "-" + hex8(identity_hash(plan));
}

bool view_is_fresh(const ViewPlan& plan) {
    const std::string root = views_root(plan);
    if (root.empty()) return false;
    const std::string dir = (fs::path(root) / view_slug(plan)).string();
    std::error_code ec;
    return fs::exists(manifest_path(dir), ec) &&
           !gather_view_files(dir).empty();
}

int lock_view_dir(const std::string& dir) {
    const std::string path = (fs::path(dir) / ".lock").string();
    const int fd = ::open(path.c_str(), O_CREAT | O_RDWR, 0644);
    if (fd < 0) return -1;
    if (::flock(fd, LOCK_EX | LOCK_NB) != 0) {  // another builder holds it
        ::close(fd);
        return -1;
    }
    return fd;  // released (and the advisory lock dropped) by unlock_view_dir
}

void unlock_view_dir(int fd) {
    if (fd < 0) return;
    ::flock(fd, LOCK_UN);
    ::close(fd);
}

std::string materialize_view_dir(const ViewPlan& plan) {
    const std::string root = views_root(plan);
    if (root.empty()) return {};
    const std::string dir = (fs::path(root) / view_slug(plan)).string();
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return {};
    return dir;
}

void register_view(const std::string& dir, const ViewPlan& plan) {
    std::string j = "{\n";
    j += "  \"version\": 1,\n";
    j += "  \"query\": \"";
    json::append_json_escaped(j, plan.query ? plan.query->source() : "");
    j += "\",\n";
    j += "  \"phase\": " + std::to_string(static_cast<int>(plan.phase)) + ",\n";
    if (plan.time_range) {
        j += "  \"window\": {\"begin\": " +
             std::to_string(plan.time_range->first) +
             ", \"end\": " + std::to_string(plan.time_range->second) + "},\n";
    } else {
        j += "  \"window\": null,\n";
    }
    j += "  \"bases\": [";
    const auto bases = plan_bases(plan);
    for (std::size_t i = 0; i < bases.size(); ++i) {
        j += i ? ",\n    {" : "\n    {";
        j += "\"path\": \"";
        json::append_json_escaped(j, bases[i].path);
        j += "\", \"size\": " + std::to_string(bases[i].size) +
             ", \"mtime\": " + std::to_string(bases[i].mtime) + "}";
    }
    j += bases.empty() ? "]\n}\n" : "\n  ]\n}\n";

    std::ofstream out(manifest_path(dir), std::ios::binary | std::ios::trunc);
    out << j;
}

std::optional<std::vector<ViewFile>> find_subsuming_view(const ViewPlan& plan) {
    const std::string root = views_root(plan);
    std::error_code ec;
    if (root.empty() || !fs::is_directory(root, ec)) return std::nullopt;

    const auto q_bases = plan_bases(plan);
    if (q_bases.size() != plan.files.size())
        return std::nullopt;  // unstattable
    const std::uint64_t base_bytes = [&] {
        std::uint64_t s = 0;
        for (const auto& b : q_bases) s += b.size;
        return s;
    }();

    std::vector<ViewFile> best;
    std::uint64_t best_bytes = base_bytes;  // must beat the base to be worth it
    for (fs::directory_iterator it(root, ec), end; !ec && it != end;
         it.increment(ec)) {
        if (!it->is_directory(ec)) continue;
        const std::string dir = it->path().string();
        auto m = read_manifest(dir);
        if (!m) continue;
        // Evict a view built from a base that has since changed or vanished: it
        // can never be served again, and nobody reads a non-matching MV.
        if (manifest_is_stale(*m)) {
            std::error_code rm;
            fs::remove_all(dir, rm);
            continue;
        }
        if (!manifest_matches(*m, plan, q_bases)) continue;

        auto files = gather_view_files(dir);
        if (files.empty()) continue;
        const std::uint64_t bytes = total_size(files);
        if (bytes >= best_bytes)
            continue;  // not smaller than base or a better MV

        best = std::move(files);
        best_bytes = bytes;
    }

    if (best.empty()) return std::nullopt;
    return best;
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
