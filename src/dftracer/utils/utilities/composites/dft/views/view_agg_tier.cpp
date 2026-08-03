#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/utilities/composites/dft/views/view_agg_tier.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/rocksdb/column_families.h>
#include <dftracer/utils/core/rocksdb/database.h>
#include <dftracer/utils/core/rocksdb/db_manager.h>
#include <dftracer/utils/utilities/common/query/ast.h>
#include <dftracer/utils/utilities/common/query/evaluator.h>
#include <dftracer/utils/utilities/common/query/query.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/agg_db_open.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_output.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_serialization.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/system_metrics_serialization.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/view_plan.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>

#include <cctype>
#include <functional>
#include <limits>
#include <memory>
#include <set>
#include <shared_mutex>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <variant>

namespace dftracer::utils::utilities::composites::dft::views::detail {

namespace {

namespace agg = composites::dft::aggregators;
using agg::AggKeyView;
using agg::AggMetricsFullView;
using dftracer::utils::utilities::indexer::IndexDatabase;
using dftracer::utils::utilities::indexer::internal::get_logical_path;

// Fields the tier's stored MetricStats can answer. ts is only its min, te only
// its max, so those ops are gated in answerable().
bool answerable_field(const std::string& f) {
    return f == "dur" || f == "size" || f == "offset" || f == "ts" || f == "te";
}

std::string to_lower_ascii(std::string s) {
    for (char& c : s)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return s;
}

void lower_literal(common::query::LiteralNode& lit) {
    if (auto* s = std::get_if<std::string>(&lit.value)) *s = to_lower_ascii(*s);
}

// A copy of the query AST with `cat` equality/membership literals lowercased.
// The tier stores cat lowercased (the aggregation's canonical form), so a
// filter literal like "POSIX" is folded to match; this also restores the
// pre-View iter_arrow behavior of comparing cat case-insensitively.
common::query::QueryNodePtr clone_cat_lowered(
    const common::query::QueryNode& n) {
    using namespace common::query;
    return std::visit(
        [](const auto& node) -> QueryNodePtr {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, CompareNode>) {
                CompareNode c = node;
                if (c.field.path == "cat") lower_literal(c.value);
                return make_node(std::move(c));
            } else if constexpr (std::is_same_v<T, InNode>) {
                InNode c = node;
                if (c.field.path == "cat")
                    for (auto& e : c.values.elements) lower_literal(e);
                return make_node(std::move(c));
            } else if constexpr (std::is_same_v<T, NotInNode>) {
                NotInNode c = node;
                if (c.field.path == "cat")
                    for (auto& e : c.values.elements) lower_literal(e);
                return make_node(std::move(c));
            } else if constexpr (std::is_same_v<T, AndNode>) {
                return make_node(AndNode{clone_cat_lowered(*node.left),
                                         clone_cat_lowered(*node.right)});
            } else if constexpr (std::is_same_v<T, OrNode>) {
                return make_node(OrNode{clone_cat_lowered(*node.left),
                                        clone_cat_lowered(*node.right)});
            } else if constexpr (std::is_same_v<T, NotNode>) {
                return make_node(NotNode{clone_cat_lowered(*node.operand)});
            } else {
                return make_node(T(node));
            }
        },
        n.data);
}

// A cat predicate is foldable onto the tier's lowercased cat only when it is an
// equality/membership form (whose literal we can lowercase). A like/regex on
// cat can't be case-folded safely, so the tier declines and the scan (raw cat)
// answers it.
bool cat_foldable(const common::query::QueryNode& n) {
    using namespace common::query;
    return std::visit(
        [](const auto& node) -> bool {
            using T = std::decay_t<decltype(node)>;
            if constexpr (std::is_same_v<T, MatchNode>)
                return node.field.path != "cat";
            else if constexpr (std::is_same_v<T, AndNode> ||
                               std::is_same_v<T, OrNode>)
                return cat_foldable(*node.left) && cat_foldable(*node.right);
            else if constexpr (std::is_same_v<T, NotNode>)
                return cat_foldable(*node.operand);
            else
                return true;
        },
        n.data);
}

// Filter fields evaluable from a CF key. cat is answerable only when foldable
// (see cat_foldable): the tier stores it lowercased, so equality/membership
// literals are lowercased to match, but like/regex on cat must scan.
bool query_answerable(const common::query::Query& q) {
    for (std::string_view f : q.fields())
        if (f != "name" && f != "pid" && f != "tid" && f != "fhash" &&
            f != "hhash" && f != "cat")
            return false;
    return cat_foldable(q.root());
}

// Agg ops + reduced fields the tier can serve, shared by the EVENT-only and the
// unified events+profiles paths (which differ only on group-key allowances).
bool aggs_and_fields_answerable(const ViewPlan& plan, const AggSchema& sch) {
    for (const auto& s : plan.agg) {
        if (s.op == AggOp::ArgMax) return false;
        // Only dur/size persist m3/m4 and a DDSketch; skew/kurtosis,
        // percentiles and histograms on any other field must scan.
        if ((s.op == AggOp::Skew || s.op == AggOp::Kurt || s.op == AggOp::Pct ||
             s.op == AggOp::Hist) &&
            s.field != "dur" && s.field != "size")
            return false;
        if (s.op == AggOp::Count && !s.field.empty()) return false;
        if (s.field == "ts" && s.op != AggOp::Min) return false;
        if (s.field == "te" && s.op != AggOp::Max) return false;
    }
    for (const auto& f : sch.fields)
        if (!answerable_field(f)) return false;
    return true;
}

bool answerable(const ViewPlan& plan, const AggSchema& sch) {
    if (plan.time_bucket_us != 0 || plan.time_range || plan.files.empty())
        return false;
    if (plan.query && !query_answerable(*plan.query)) return false;
    for (const auto& gk : plan.group_by)
        if (gk.kind == GroupKey::Kind::Arg) return false;
    return aggs_and_fields_answerable(plan, sch);
}

// Group value from a CF key, byte-identical to view_aggregate group_value().
std::string key_value(const AggKeyView& kv, const GroupKey& gk,
                      char (&fbuf)[::dftracer::utils::hash::HEX64_DIGITS]) {
    switch (gk.kind) {
        case GroupKey::Kind::Name:
            return std::string(kv.name);
        case GroupKey::Kind::Cat:
            return std::string(kv.cat);
        case GroupKey::Kind::Pid:
            return std::to_string(kv.pid);
        case GroupKey::Kind::Tid:
            return std::to_string(kv.tid);
        case GroupKey::Kind::Fhash:
        case GroupKey::Kind::FilePath:
        case GroupKey::Kind::FileName:
            return std::string(agg::fhash_text(kv, fbuf));
        case GroupKey::Kind::Hhash:
        case GroupKey::Kind::HostName:
            return std::string(kv.hhash);
        case GroupKey::Kind::IoCat:
            return std::to_string(static_cast<int>(
                composites::dft::internal::io_category(kv.name)));
        case GroupKey::Kind::AccPat:
            return "0";
        case GroupKey::Kind::Arg:
            // Resolved from the key's extra_keys (e.g. PROFILE epoch/step);
            // requires the key to have been parsed with want_extra_keys.
            for (const auto& [k, v] : kv.extra_keys)
                if (k == gk.arg) return std::string(v);
            return {};
    }
    return {};
}

void fill_field(FieldStat& fs, const std::string& field,
                const AggMetricsFullView& mv) {
    if (field == "dur") {
        fs.n = mv.count;
        fs.sum = static_cast<double>(mv.dur_total);
        fs.sumsq = mv.dur_m2;
        fs.m3 = mv.dur_m3;
        fs.m4 = mv.dur_m4;
        fs.min = static_cast<double>(mv.dur_min);
        fs.max = static_cast<double>(mv.dur_max);
    } else if (field == "size") {
        if (mv.size_total == 0) return;  // no size events: leave absent
        fs.n = mv.count;
        fs.sum = static_cast<double>(mv.size_total);
        fs.sumsq = mv.size_m2;
        fs.m3 = mv.size_m3;
        fs.m4 = mv.size_m4;
        fs.min = static_cast<double>(mv.size_min);
        fs.max = static_cast<double>(mv.size_max);
    } else if (field == "offset") {
        fs.n = mv.count;
        fs.sum = static_cast<double>(mv.offset_total);
        fs.sumsq = mv.offset_m2;
        fs.m3 = mv.offset_m3;
        fs.m4 = mv.offset_m4;
        fs.min = static_cast<double>(mv.offset_min);
        fs.max = static_cast<double>(mv.offset_max);
    } else if (field == "ts") {
        fs.n = mv.count;
        fs.min = fs.max = static_cast<double>(mv.ts);
    } else if (field == "te") {
        fs.n = mv.count;
        fs.min = fs.max = static_cast<double>(mv.te);
    }
}

// Per-index tier state, resolved once and reused across collects. The read-only
// aggregation DB is opened a single time (not per collect), the intern loaded
// once, and the index's file set cached for the coverage check. Immutable once
// built: a rebuilt index (detected by `mtime`) swaps in a new entry, so a
// reader holding the shared_ptr keeps a consistent snapshot.
struct TierCache {
    bool has_tier = false;
    bool has_sketches = false;  // CF stored DDSketches (compute_percentiles)
    bool group_by_file = true;  // CF keys carry fhash
    std::unordered_set<std::string> files;
    std::shared_ptr<agg::AggDbHandle> handle;
    std::int64_t mtime = 0;     // RocksDB CURRENT mtime; 0 = absent
    std::uint64_t interval_us =
        0;                      // CF aggregation interval (stored bucket size)
};

// Mtime of the aggregation DB's CURRENT file, rewritten on every manifest roll,
// so it changes when the index is rebuilt. A cheap stat, not a content read.
std::int64_t current_mtime(const std::string& index_path) {
    std::error_code ec;
    auto t = fs::last_write_time(fs::path(index_path) / "CURRENT", ec);
    if (ec) return 0;
    return static_cast<std::int64_t>(t.time_since_epoch().count());
}

std::shared_ptr<const TierCache> build_tier_cache(const std::string& index_path,
                                                  std::int64_t mtime) {
    auto tc = std::make_shared<TierCache>();
    tc->mtime = mtime;
    try {
        std::string err;
        auto handle = agg::open_agg_db(index_path, err);
        if (handle) {
            std::string cfg;
            const auto key =
                std::string_view(agg::AGG_GLOBAL_CONFIG_KEY,
                                 sizeof(agg::AGG_GLOBAL_CONFIG_KEY) - 1);
            agg::AggGlobalConfig gcfg;
            const bool have_cfg =
                handle->db->get(key, &cfg, rocksdb::cf::AGGREGATION).ok();
            if (have_cfg) gcfg = agg::deserialize_agg_global_config(cfg);
            if (have_cfg) {
                tc->group_by_file = gcfg.group_by_file;
                tc->interval_us = gcfg.time_interval_us;
                IndexDatabase db(index_path, indexer::IndexOpenMode::ReadOnly);
                for (auto& [p, id] : db.query_all_file_info_ids())
                    tc->files.insert(p);
                tc->has_tier = true;
                tc->handle =
                    std::shared_ptr<agg::AggDbHandle>(std::move(handle));
                // Peek one EVENT row to learn whether the tier stored sketches
                // (compute_percentiles); percentile queries need them.
                const auto ck =
                    std::string_view(agg::AGG_GLOBAL_CONFIG_KEY,
                                     sizeof(agg::AGG_GLOBAL_CONFIG_KEY) - 1);
                auto it =
                    tc->handle->db->new_iterator(rocksdb::cf::AGGREGATION);
                for (it->SeekToFirst(); it->Valid(); it->Next()) {
                    const std::string_view kd(it->key().data(),
                                              it->key().size());
                    if (kd == ck) continue;
                    AggKeyView kvp;
                    if (!agg::parse_agg_key_view(kd, tc->handle->agg->intern(),
                                                 kvp))
                        continue;
                    if (kvp.map_type != agg::AggMapType::EVENT) continue;
                    AggMetricsFullView mvp;
                    if (agg::parse_agg_value_full_view(
                            std::string_view(it->value().data(),
                                             it->value().size()),
                            mvp))
                        tc->has_sketches =
                            !mvp.dur_sketch.empty() || !mvp.size_sketch.empty();
                    break;
                }
            }
        }
    } catch (const std::exception&) {
        auto empty = std::make_shared<TierCache>();
        empty->mtime = mtime;  // incomplete/unreadable index: fall back to scan
        return empty;
    }
    return tc;
}

std::shared_mutex g_tier_mtx;
std::unordered_map<std::string, std::shared_ptr<const TierCache>> g_tier_cache;

// The cache holds each index's read-only agg DB open for reuse. It is dropped
// from a rocksdb pre-exit cleanup (see register_pre_exit_cleanup) so the DBs
// close cleanly before RocksDB starts abandoning handles at process exit; a
// plain std::atexit would run too late and leak them via that abandon path.
void clear_tier_cache() {
    std::unique_lock<std::shared_mutex> lk(g_tier_mtx);
    g_tier_cache.clear();
}

void evict_tier_cache(const std::string& index_path) {
    std::unique_lock<std::shared_mutex> lk(g_tier_mtx);
    g_tier_cache.erase(index_path);
}

std::shared_ptr<const TierCache> tier_cache(const std::string& index_path) {
    static const int registered = [] {
        dftracer::utils::rocksdb::register_pre_exit_cleanup(clear_tier_cache);
        // Drop this index's cached agg DB when its manager entry is reset (e.g.
        // a rebuild removing the directory), so the tier does not keep the DB
        // open and block the removal.
        dftracer::utils::rocksdb::register_reset_listener(evict_tier_cache);
        return 0;
    }();
    (void)registered;
    const std::int64_t mtime = current_mtime(index_path);
    {
        std::shared_lock<std::shared_mutex> rlk(g_tier_mtx);
        if (auto it = g_tier_cache.find(index_path);
            it != g_tier_cache.end() && it->second->mtime == mtime)
            return it->second;
    }
    // Build outside the lock: build_tier_cache -> open_agg_db may reset() the
    // index (its merge-operator open falling back), which fires the tier's own
    // reset listener and re-locks g_tier_mtx. Holding it here would self-lock.
    // The build is idempotent, so a concurrent double-build just races to
    // insert; last write wins and both hold the same data for this mtime.
    auto tc = build_tier_cache(index_path, mtime);
    std::unique_lock<std::shared_mutex> wlk(g_tier_mtx);
    if (auto it = g_tier_cache.find(index_path);
        it != g_tier_cache.end() && it->second->mtime == mtime)
        return it->second;
    g_tier_cache[index_path] = tc;  // replaces stale; old readers keep theirs
    return tc;
}

// Whether the plan needs per-file resolution, which a tier built with
// group_by_file=false folded away and cannot reconstruct.
bool plan_needs_file(const ViewPlan& plan) {
    for (const auto& g : plan.group_by)
        if (g.kind == GroupKey::Kind::Fhash ||
            g.kind == GroupKey::Kind::FilePath ||
            g.kind == GroupKey::Kind::FileName)
            return true;
    if (plan.query)
        for (std::string_view f : plan.query->fields())
            if (f == "fhash") return true;
    return false;
}

// The tier cache iff it exists and spans exactly the plan's files (the CF is
// keyed by content fhash, so a subset view cannot filter it exactly). Else
// null.
std::shared_ptr<const TierCache> covering_tier(const ViewPlan& plan) {
    if (plan.files.empty()) return nullptr;
    const std::string& index_path = plan.files.front().index_path;
    if (index_path.empty()) return nullptr;
    auto tc = tier_cache(index_path);
    if (!tc->has_tier) return nullptr;
    if (!tc->group_by_file && plan_needs_file(plan)) return nullptr;
    if (tc->files.size() != plan.files.size()) return nullptr;
    for (const auto& f : plan.files) {
        if (f.index_path != index_path) return nullptr;
        if (!tc->files.count(get_logical_path(f.file_path))) return nullptr;
    }
    return tc;
}

}  // namespace

static void fill_accum(AggAccum& a, const AggSchema& sch,
                       const AggMetricsFullView& mv) {
    a.count = mv.count;
    a.fields.resize(sch.fields.size());
    a.sketches.resize(sch.sketch_count);
    for (std::size_t fi = 0; fi < sch.fields.size(); ++fi) {
        fill_field(a.fields[fi], sch.fields[fi], mv);
        const int sk = sch.field_sketch[fi];
        if (sk < 0) continue;
        const std::string_view blob =
            sch.fields[fi] == "dur" ? mv.dur_sketch : mv.size_sketch;
        if (!blob.empty())
            a.sketches[sk] = common::statistics::DDSketch::deserialize(
                reinterpret_cast<const std::uint8_t*>(blob.data()),
                blob.size());
    }
}

// `eval_root` is the cat-lowered clone of plan.query's AST (see
// clone_cat_lowered), built once by the caller; the tier's cat values are
// lowercased, so cat literals must be too. Non-null whenever plan.query is set.
static bool key_passes_query(
    const ViewPlan& plan, const common::query::QueryNode* eval_root,
    const AggKeyView& kv, char (&fbuf)[::dftracer::utils::hash::HEX64_DIGITS]) {
    if (!plan.query) return true;
    common::query::ValueMap vm;
    for (std::string_view f : plan.query->fields()) {
        if (f == "name")
            vm[f] = std::string(kv.name);
        else if (f == "cat")
            vm[f] = std::string(kv.cat);
        else if (f == "pid")
            vm[f] = kv.pid;
        else if (f == "tid")
            vm[f] = kv.tid;
        else if (f == "hhash")
            vm[f] = std::string(kv.hhash);
        else if (f == "fhash")
            vm[f] = std::string(agg::fhash_text(kv, fbuf));
    }
    return common::query::evaluate(*eval_root, vm);
}

bool agg_tier_collect(const ViewPlan& plan, GroupMap& out) {
    const AggSchema& sch = ensure_schema(plan);
    if (!answerable(plan, sch)) return false;

    const auto tc = covering_tier(plan);
    if (!tc) return false;

    // Percentiles and histograms need the stored DDSketch; if the tier has
    // none, scan instead.
    if (!tc->has_sketches)
        for (const auto& s : plan.agg)
            if (s.op == AggOp::Pct || s.op == AggOp::Hist) return false;

    const auto cfg_key = std::string_view(
        agg::AGG_GLOBAL_CONFIG_KEY, sizeof(agg::AGG_GLOBAL_CONFIG_KEY) - 1);
    const StringIntern& intern = tc->handle->agg->intern();
    auto it = tc->handle->db->new_iterator(rocksdb::cf::AGGREGATION);
    std::string keybuf;
    char fbuf[::dftracer::utils::hash::HEX64_DIGITS];
    const auto eval_root =
        plan.query ? clone_cat_lowered(plan.query->root()) : nullptr;
    for (it->SeekToFirst(); it->Valid(); it->Next()) {
        const std::string_view kdata(it->key().data(), it->key().size());
        if (kdata == cfg_key) continue;
        AggKeyView kv;
        if (!agg::parse_agg_key_view(kdata, intern, kv)) continue;
        if (kv.map_type != agg::AggMapType::EVENT) continue;
        if (!key_passes_query(plan, eval_root.get(), kv, fbuf)) continue;

        AggMetricsFullView mv;
        if (!agg::parse_agg_value_full_view(
                std::string_view(it->value().data(), it->value().size()), mv))
            continue;

        AggAccum a;
        a.keys.reserve(plan.group_by.size());
        keybuf.clear();
        for (const auto& gk : plan.group_by) {
            std::string v = key_value(kv, gk, fbuf);
            keybuf += v;
            keybuf += GROUP_SEP;
            a.keys.push_back(std::move(v));
        }
        fill_accum(a, sch, mv);
        merge_accum(out[keybuf], a, plan);
    }
    return true;
}

// The bucket value for a stored key, matching view_aggregate's scan-path
// formula floor(ts*time_scale/interval)*interval. kv.time_bucket is already the
// stored bucket-start timestamp (compute_time_bucket floors ts to the CF
// interval), so it plays the role of ts here.
static std::int64_t tier_time_bucket(const AggKeyView& kv,
                                     const ViewPlan& plan) {
    const double start_us = static_cast<double>(kv.time_bucket);
    const double interval = static_cast<double>(plan.time_bucket_us);
    return static_cast<std::int64_t>(start_us * plan.time_scale / interval) *
           static_cast<std::int64_t>(plan.time_bucket_us);
}

bool events_profiles_collect(const ViewPlan& plan, ResultTable& out_table,
                             int shard_begin, int shard_end,
                             const ProgressFn* progress) {
    const AggSchema& sch = ensure_schema(plan);
    // time_range() is a window filter the tier cannot evaluate; time_bucket()
    // maps onto the CF's stored bucket grain and is supported.
    if (plan.time_range) return false;
    if (plan.query && !query_answerable(*plan.query)) return false;
    if (!aggs_and_fields_answerable(plan, sch)) return false;

    const auto tc = covering_tier(plan);
    if (!tc) return false;
    if (!tc->has_sketches)
        for (const auto& s : plan.agg)
            if (s.op == AggOp::Pct || s.op == AggOp::Hist) return false;

    const bool has_bucket = plan.time_bucket_us > 0;
    // The tier can only coarsen the stored grain: a requested bucket finer than
    // (or not a multiple of) the CF interval cannot be produced exactly.
    if (has_bucket && tc->interval_us > 0 &&
        (plan.time_bucket_us < tc->interval_us ||
         plan.time_bucket_us % tc->interval_us != 0))
        return false;
    // Any Arg group key (e.g. epoch/step) lives in the PROFILE key's
    // extra_keys.
    bool want_extra = false;
    for (const auto& gk : plan.group_by)
        if (gk.kind == GroupKey::Kind::Arg) want_extra = true;
    // FileName/HostName group values are stored as hashes; resolve to names so
    // downstream consumers (e.g. dfanalyzer POSIX rules, proc_name) get paths.
    const GroupResolver* resolver = ensure_resolver(plan);

    const auto cfg_key = std::string_view(
        agg::AGG_GLOBAL_CONFIG_KEY, sizeof(agg::AGG_GLOBAL_CONFIG_KEY) - 1);
    const StringIntern& intern = tc->handle->agg->intern();
    auto it = tc->handle->db->new_iterator(rocksdb::cf::AGGREGATION);
    std::string keybuf;
    char fbuf[::dftracer::utils::hash::HEX64_DIGITS];
    GroupMap gm;
    if (shard_end <= 0) shard_end = agg::AGG_KEY_NUM_SHARDS;
    const std::size_t total_shards =
        static_cast<std::size_t>(shard_end - shard_begin);
    int reported_shard = shard_begin;
    if (progress && *progress) (*progress)(0, total_shards);
    // Keys begin with a big-endian 2-byte shard prefix, so a shard range is a
    // contiguous key range and no group straddles shards.
    const char seek[2] = {static_cast<char>((shard_begin >> 8) & 0xFF),
                          static_cast<char>(shard_begin & 0xFF)};
    const auto eval_root =
        plan.query ? clone_cat_lowered(plan.query->root()) : nullptr;
    for (it->Seek(std::string_view(seek, 2)); it->Valid(); it->Next()) {
        const std::string_view kdata(it->key().data(), it->key().size());
        if (kdata.size() >= 2) {
            const int shard = (static_cast<unsigned char>(kdata[0]) << 8) |
                              static_cast<unsigned char>(kdata[1]);
            if (shard >= shard_end) break;  // past our range (and special keys)
            if (progress && *progress && shard > reported_shard) {
                reported_shard = shard;
                (*progress)(static_cast<std::size_t>(shard - shard_begin),
                            total_shards);
            }
        }
        if (kdata == cfg_key) continue;
        AggKeyView kv;
        if (!agg::parse_agg_key_view(kdata, intern, kv, want_extra)) continue;
        const bool is_profile = kv.map_type == agg::AggMapType::PROFILE;
        if (kv.map_type != agg::AggMapType::EVENT && !is_profile) continue;
        if (!key_passes_query(plan, eval_root.get(), kv, fbuf)) continue;

        AggMetricsFullView mv;
        if (!agg::parse_agg_value_full_view(
                std::string_view(it->value().data(), it->value().size()), mv))
            continue;

        AggAccum a;
        a.keys.reserve(plan.group_by.size() + 1);
        // Leading "type" key keeps events and profiles in distinct groups even
        // when the caller does not group on epoch/step.
        const char* type_str = is_profile ? "profile" : "event";
        keybuf.clear();
        keybuf += type_str;
        keybuf += GROUP_SEP;
        a.keys.emplace_back(type_str);
        if (has_bucket) {
            std::string b = std::to_string(tier_time_bucket(kv, plan));
            keybuf += b;
            keybuf += GROUP_SEP;
            a.keys.push_back(std::move(b));
        }
        for (const auto& gk : plan.group_by) {
            std::string v = key_value(kv, gk, fbuf);
            if (resolver) v = resolve_group_value(*resolver, gk.kind, v);
            v = apply_group_transform(gk, std::move(v));
            keybuf += v;
            keybuf += GROUP_SEP;
            a.keys.push_back(std::move(v));
        }
        fill_accum(a, sch, mv);
        merge_accum(gm[keybuf], a, plan);
    }

    // Materialize with a leading "type" column, then time_bucket (when
    // bucketing), then the group columns - matching a.keys order above.
    out_table = ResultTable{};
    out_table.group_columns.push_back("type");
    if (has_bucket) out_table.group_columns.push_back("time_bucket");
    for (const auto& gk : plan.group_by)
        out_table.group_columns.push_back(group_col_name(gk));
    const bool count_col = plan.agg.empty();
    if (count_col) {
        out_table.value_columns.push_back("count");
    } else {
        for (const auto& spec : plan.agg) {
            if (spec.op == AggOp::ArgMax)
                out_table.text_columns.push_back(agg_col_name(spec));
            else if (spec.op == AggOp::Hist)
                out_table.hist_columns.push_back(agg_col_name(spec));
            else
                out_table.value_columns.push_back(agg_col_name(spec));
        }
    }
    for (const auto& [k, a] : gm) {
        (void)k;
        ResultRow row;
        row.keys = a.keys;  // [type, group values...]
        if (count_col) {
            row.values.push_back(static_cast<double>(a.count));
        } else {
            for (std::size_t i = 0; i < plan.agg.size(); ++i) {
                if (plan.agg[i].op == AggOp::ArgMax)
                    row.texts.push_back(a.argmax[sch.spec_argmax[i]].repr);
                else if (plan.agg[i].op == AggOp::Hist)
                    row.hists.push_back(finalize_hist(a, plan, i));
                else
                    row.values.push_back(finalize_value(a, plan, i));
            }
        }
        out_table.rows.push_back(std::move(row));
    }
    if (progress && *progress) (*progress)(total_shards, total_shards);
    return true;
}

bool system_collect(const ViewPlan& plan, ResultTable& out_table,
                    int shard_begin, int shard_end, const ProgressFn*) {
    // The plan's event-side time_bucket/group_by do not apply here - system
    // rows carry their own (host,name,time_bucket) grain. The CF now has the
    // same shard prefix as the AGG CF, so it range-splits identically.
    const auto tc = covering_tier(plan);
    if (!tc) return false;
    if (shard_end <= 0) shard_end = agg::AGG_KEY_NUM_SHARDS;

    const char seek[2] = {static_cast<char>((shard_begin >> 8) & 0xFF),
                          static_cast<char>(shard_begin & 0xFF)};
    auto for_each =
        [&](const std::function<void(std::string_view, std::string_view)>& cb) {
            auto it = tc->handle->db->new_iterator(rocksdb::cf::SYSTEM_METRICS);
            for (it->Seek(std::string_view(seek, 2)); it->Valid(); it->Next()) {
                const std::string_view kd(it->key().data(), it->key().size());
                if (kd.size() >= 2) {
                    const int shard = (static_cast<unsigned char>(kd[0]) << 8) |
                                      static_cast<unsigned char>(kd[1]);
                    if (shard >= shard_end) break;
                }
                cb(kd,
                   std::string_view(it->value().data(), it->value().size()));
            }
        };

    // First pass: the sorted union of metric names, so the schema is stable.
    std::set<std::string> name_set;
    for_each([&](std::string_view, std::string_view val) {
        auto m = agg::deserialize_system_value(val);
        if (m.metrics)
            for (const auto& [n, _] : *m.metrics) name_set.insert(n);
    });
    const std::vector<std::string> metric_names(name_set.begin(),
                                                name_set.end());

    out_table = ResultTable{};
    out_table.group_columns = {"host_hash", "name"};
    out_table.value_columns = {"time_bucket", "ts", "te", "count"};
    for (const auto& n : metric_names) out_table.value_columns.push_back(n);

    for_each([&](std::string_view key, std::string_view val) {
        auto k = agg::deserialize_system_key(key);
        auto m = agg::deserialize_system_value(val);
        ResultRow row;
        row.keys = {k.key.hhash, k.key.name};
        row.values.push_back(static_cast<double>(k.key.time_bucket));
        row.values.push_back(static_cast<double>(m.ts));
        row.values.push_back(static_cast<double>(m.te));
        row.values.push_back(static_cast<double>(m.count));
        for (const auto& n : metric_names) {
            double v = std::numeric_limits<double>::quiet_NaN();
            if (m.metrics) {
                auto it2 = m.metrics->find(n);
                if (it2 != m.metrics->end()) v = it2->second.mean();
            }
            row.values.push_back(v);
        }
        out_table.rows.push_back(std::move(row));
    });
    return true;
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#else  // !DFTRACER_UTILS_ENABLE_ARROW

namespace dftracer::utils::utilities::composites::dft::views::detail {
bool agg_tier_collect(const ViewPlan&, GroupMap&) { return false; }
bool events_profiles_collect(const ViewPlan&, ResultTable&) { return false; }
}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif
