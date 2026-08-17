#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/common/to_chars.h>
#include <dftracer/utils/core/common/transparent_string_hash.h>
#include <dftracer/utils/core/coro/channel.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/server/http_request.h>
#include <dftracer/utils/server/http_response.h>
#include <dftracer/utils/server/json_builder.h>
#include <dftracer/utils/server/router.h>
#include <dftracer/utils/server/signal_handler.h>
#include <dftracer/utils/server/trace_index.h>
#include <dftracer/utils/server/viz_api.h>
#include <dftracer/utils/server/viz_calltree.h>
#include <dftracer/utils/server/viz_density.h>
#include <dftracer/utils/server/viz_internal.h>
#include <dftracer/utils/server/viz_scan.h>
#include <dftracer/utils/server/viz_summary_build.h>
#include <dftracer/utils/utilities/common/json/json_doc_guard.h>
#include <dftracer/utils/utilities/common/json/json_value.h>
#include <dftracer/utils/utilities/common/query/query.h>
#include <dftracer/utils/utilities/composites/dft/views/view.h>
#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>
#include <dftracer/utils/utilities/composites/dft/views/view_definition.h>
#include <dftracer/utils/utilities/composites/dft/views/view_planner_utility.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scanner_utility.h>
#include <dftracer/utils/utilities/fileio/lines/sources/async_streaming_gz_line_generator.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <simdjson.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <vector>

namespace dftracer::utils::server {

using namespace dftracer::utils::utilities::composites::dft;
using namespace dftracer::utils::utilities::composites::dft::views;
using dftracer::utils::utilities::common::json::json_number;

static const std::unordered_set<std::string> HASH_METADATA_NAMES = {"FH", "HH",
                                                                    "SH"};

static constexpr int DEFAULT_VIEWPORT_WIDTH = 1920;
static constexpr int MIN_VIEWPORT_WIDTH = 320;
static constexpr int MAX_VIEWPORT_WIDTH = 8192;

// Rewrite the unsigned integer value of `key` (e.g. "\"dur\":") in `json` to

// Normalize event timestamps (when global_min > 0) and serialize the collected
// events plus metadata into the Chrome Trace Event Format body. `global_min` is
// the de-normalization base (already 0 unless normalization is active);
// `display_global_min` is the value reported in the metadata. Pure/synchronous.
static std::string build_viz_events_body(std::vector<std::string>& events,
                                         std::uint64_t global_min,
                                         double meta_begin, double meta_end,
                                         int limit, bool truncated,
                                         std::uint64_t display_global_min,
                                         TraceIndex::TimeMetric metric) {
    // Even without an offset, a non-US trace still needs ts/dur scaled to us.
    if (global_min > 0 || metric != TraceIndex::TimeMetric::US) {
        for (auto& event : events) {
            event = normalize_event_ts(event, global_min, metric);
        }
    }

    auto& b = scratch_json_builder();
    b.start_object();
    b.escape_and_append_with_quotes("events");
    b.append_colon();
    b.start_array();
    for (std::size_t i = 0; i < events.size(); ++i) {
        if (i > 0) b.append_comma();
        b.append_raw(events[i]);  // Already JSON
    }
    b.end_array();
    b.append_comma();
    b.escape_and_append_with_quotes("metadata");
    b.append_colon();
    b.start_object();
    b.append_key_value("begin", meta_begin);
    b.append_comma();
    b.append_key_value("end", meta_end);
    b.append_comma();
    b.append_key_value("count", events.size());
    b.append_comma();
    b.append_key_value("limit", limit);
    b.append_comma();
    b.append_key_value("truncated", truncated);
    b.append_comma();
    b.append_key_value("ts_normalized", global_min > 0);
    b.append_comma();
    b.append_key_value("global_min_timestamp_us", display_global_min);
    b.end_object();
    b.end_object();
    return std::string(b);
}

static coro::CoroTask<void> append_app_spans(std::vector<std::string>& out,
                                             TraceIndex& index, double begin,
                                             double end,
                                             const QueryParams& params);
static bool viz_summary_eligible(const QueryParams& params);

static coro::CoroTask<HttpResponse> handle_viz_events(const HttpRequest& req,
                                                      const QueryParams& params,
                                                      TraceIndex& index) {
    // Required: begin, end, summary
    if (!params.has("begin") || !params.has("end") || !params.has("summary")) {
        co_return HttpResponse::bad_request(
            "Missing required parameters: begin, end, summary");
    }

    double begin = params.get_double("begin", 0);
    double end = params.get_double("end", 0);
    int summary = params.get_int("summary", 1);
    if (summary < 1) summary = 1;

    // Validate the optional raw DSL query before it is spliced into the view.
    auto query = params.get("query");
    if (!query.empty() &&
        !utilities::common::query::try_parse(query).has_value()) {
        co_return HttpResponse::bad_request("Invalid query: " +
                                            std::string(query));
    }

    // Timestamp normalization: default ON, opt-out with ?ts_normalize=0
    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";

    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max()) {
            global_min = 0;  // No valid bounds, skip normalization
        }
    }

    // When normalization is active the user sends normalized
    // begin/end values (relative to global_min).  De-normalize them
    // so the predicate filters against absolute timestamps.
    double original_begin = begin;
    double original_end = end;
    // Client sends us; the scan matches the native index. Convert first, then
    // de-normalize against the native base. Identity for US traces.
    if (index.time_metric() != TraceIndex::TimeMetric::US) {
        begin = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(begin)));
        end = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(end)));
    }
    if (normalize && global_min > 0) {
        begin += static_cast<double>(global_min);
        end += static_cast<double>(global_min);
    }

    double min_dur =
        duration_threshold(begin, end, static_cast<unsigned>(summary));
    // Full detail (summary=1) has no duration floor, so a wide window would
    // return every sub-pixel event (millions on a large trace, hanging the
    // client). Bound it to ~1px at the client width; sub-pixel events cannot
    // be drawn anyway, and zooming in shrinks the window so detail returns.
    if (min_dur <= 0 && end > begin) {
        int px_width = params.get_int("width", DEFAULT_VIEWPORT_WIDTH);
        px_width = std::clamp(px_width, MIN_VIEWPORT_WIDTH, MAX_VIEWPORT_WIDTH);
        min_dur = (end - begin) / static_cast<double>(px_width);
    }

    // Overlap, bounded: look back at most `lookback` (the longest event's
    // duration, supplied by the client) so events that started before the
    // window but extend into it are included, without scanning to time 0.
    // lookback is in us; convert to native so scan_begin (native) is right.
    double lookback = params.get_double("lookback", 0);
    if (lookback < 0) lookback = 0;
    if (lookback > 0 && index.time_metric() != TraceIndex::TimeMetric::US)
        lookback = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(lookback)));
    double scan_begin = begin - lookback;
    if (scan_begin < 0) scan_begin = 0;

    ViewDefinition view = build_viz_view(params, scan_begin, end, min_dur);

    // Optional limit: 0 (default) means no limit.
    int limit = params.get_int("limit", 0);
    if (limit < 0) limit = 0;

    // Zoom-out fast path: when nothing is filtered and the duration threshold
    // is at least the summary's long-event threshold, the events that survive
    // are exactly a subset of the summary's cached long_events. Serve them from
    // memory instead of scanning every file (minutes for a large trace).
    if (min_dur > 0 && viz_summary_eligible(params)) {
        const VizSummary* s = co_await ensure_viz_summary(index);
        if (s != nullptr && s->long_threshold_us > 0 &&
            min_dur >= s->long_threshold_us) {
            auto pid_s = params.get("pid");
            auto tid_s = params.get("tid");
            bool has_pid = !pid_s.empty();
            bool has_tid = !tid_s.empty();
            std::int64_t want_pid =
                has_pid ? std::strtoll(pid_s.data(), nullptr, 10) : 0;
            std::int64_t want_tid =
                has_tid ? std::strtoll(tid_s.data(), nullptr, 10) : 0;
            std::vector<const VizSummary::AppSpan*> hits;
            for (const auto& ev : s->long_events) {
                if (static_cast<double>(ev.end - ev.begin) < min_dur) continue;
                if (static_cast<double>(ev.end) <= begin ||
                    static_cast<double>(ev.begin) >= end)
                    continue;
                if (has_pid && ev.pid != want_pid) continue;
                if (has_tid && ev.tid != want_tid) continue;
                hits.push_back(&ev);
            }
            // Keep the widest: a shallow zoom can match more of the list than
            // any response should carry.
            if (hits.size() > VizSummary::MAX_RESPONSE_SPANS) {
                std::nth_element(
                    hits.begin(), hits.begin() + VizSummary::MAX_RESPONSE_SPANS,
                    hits.end(),
                    [](const VizSummary::AppSpan* a,
                       const VizSummary::AppSpan* x) {
                        return a->end - a->begin > x->end - x->begin;
                    });
                hits.resize(VizSummary::MAX_RESPONSE_SPANS);
            }
            std::vector<std::string> collected;
            collected.reserve(hits.size());
            for (const auto* ev : hits) collected.push_back(ev->json);
            co_await append_app_spans(collected, index, begin, end, params);
            std::string body = build_viz_events_body(
                collected, global_min, original_begin, original_end, limit,
                false, index.native_to_us(index.global_min_timestamp_us()),
                index.time_metric());
            co_return HttpResponse::ok(body);
        }
    }

    std::vector<const TraceIndex::FileInfo*> target_files =
        select_viz_target_files(index, params, scan_begin, end);
    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    bool single_file = !params.get("file").empty();

    struct EvAcc {
        std::vector<std::string> events;
    };
    views::View v =
        views::View::from_files(to_view_files(target_files),
                                &index.bloom_cache())
            .phase(views::Phase::Events)
            .cancel_when([&req]() { return req.cancel_token.cancelled(); });
    if (view.query) v = v.filter(*view.query);
    if (!single_file) v = v.time_range(scan_begin, end);
    auto scan = co_await v.map_batches<EvAcc>(
        [](EvAcc& a, const std::vector<std::string_view>& events) {
            a.events.reserve(a.events.size() + events.size());
            for (auto ev : events) a.events.emplace_back(ev);
        },
        [](EvAcc&& x, EvAcc&& y) {
            x.events.reserve(x.events.size() + y.events.size());
            for (auto& s : y.events) x.events.emplace_back(std::move(s));
            return std::move(x);
        },
        slots, limit > 0 ? static_cast<std::uint64_t>(limit) : 0);

    bool truncated = scan.stats.truncated;
    std::vector<std::string> collected_events = std::move(scan.value.events);
    if (limit > 0 && static_cast<int>(collected_events.size()) > limit) {
        collected_events.resize(static_cast<std::size_t>(limit));
        truncated = true;
    }

    co_await append_app_spans(collected_events, index, begin, end, params);

    std::string body = build_viz_events_body(
        collected_events, global_min, original_begin, original_end, limit,
        truncated, index.native_to_us(index.global_min_timestamp_us()),
        index.time_metric());
    co_return HttpResponse::ok(body);
}

namespace {

enum class GroupBy { Name, Cat, Pid, Fhash };

static GroupBy parse_group_by(std::string_view g) {
    if (g == "cat") return GroupBy::Cat;
    if (g == "pid") return GroupBy::Pid;
    if (g == "fhash" || g == "file") return GroupBy::Fhash;
    return GroupBy::Name;
}

}  // namespace

// One aggregate row, uniform over the live-scan and summary paths.
struct StatRow {
    const std::string* key;
    std::uint64_t count;
    double total;
    double min;
    double max;
    double coverage;  // wall time the group was active (union of intervals)
};

// Coverage resolution: intervals are bucketed at (window / this) so concurrent
// events collapse to their union instead of summing. Sparse, so memory tracks
// distinct occupied buckets, not this count.
static constexpr double COVERAGE_BUCKETS = 8192.0;

// Per-group accumulator that merges real events (from the View) with the
// prorated contribution of ph=3 aggregates overlapping the window.
struct StatAgg {
    double count = 0;  // fractional: an aggregate contributes dft_cnt * overlap
    double total = 0;
    double min = std::numeric_limits<double>::infinity();
    double max = 0;
    // Union of active intervals as a difference map: bucket -> net (starts -
    // ends). Coverage is the length of buckets whose running sum stays > 0.
    ankerl::unordered_dense::map<std::int32_t, std::int32_t> cov;
    void add(double cnt, double sum, double dmin, double dmax) {
        count += cnt;
        total += sum;
        if (dmin < min) min = dmin;
        if (dmax > max) max = dmax;
    }
    void mark(double lo, double hi, double begin, double bw) {
        if (bw <= 0) return;
        auto blo = static_cast<std::int32_t>((lo - begin) / bw);
        auto bhi = static_cast<std::int32_t>((hi - begin) / bw);
        if (bhi <= blo) bhi = blo + 1;  // an event occupies at least its bucket
        cov[blo] += 1;
        cov[bhi] -= 1;
    }
    double coverage(double bw) const {
        if (cov.empty()) return 0;
        std::vector<std::pair<std::int32_t, std::int32_t>> pts(cov.begin(),
                                                               cov.end());
        std::sort(pts.begin(), pts.end());
        double buckets = 0;
        int running = 0;
        for (std::size_t i = 0; i + 1 < pts.size(); ++i) {
            running += pts[i].second;
            if (running > 0) buckets += pts[i + 1].first - pts[i].first;
        }
        return buckets * bw;
    }
};

template <typename builder_type>
void tag_invoke(simdjson::serialize_tag, builder_type& b, const StatRow& r) {
    b.start_object();
    b.append_key_value("name", *r.key);
    b.append_comma();
    b.append_key_value("count", r.count);
    b.append_comma();
    b.append_key_value("total", r.total);
    b.append_comma();
    b.append_key_value("avg",
                       r.count ? r.total / static_cast<double>(r.count) : 0.0);
    b.append_comma();
    b.append_key_value("min", r.min);
    b.append_comma();
    b.append_key_value("max", r.max);
    b.append_comma();
    b.append_key_value("coverage", r.coverage);
    b.end_object();
}

static std::string serialize_stats_body(std::uint64_t total_count,
                                        double total_dur, double wall,
                                        bool truncated,
                                        const std::vector<StatRow>& rows) {
    auto& b = scratch_json_builder();
    b.start_object();
    b.append_key_value("count", total_count);
    b.append_comma();
    b.append_key_value("total_dur", total_dur);
    b.append_comma();
    b.append_key_value("wall", wall);
    b.append_comma();
    b.append_key_value("truncated", truncated);
    b.append_comma();
    b.append_key_value("names", rows);
    b.end_object();
    return std::string(b);
}

// Scale duration fields (native trace unit -> us) before serialization. The
// `wall` value is derived from client-us begin/end and is already in us.
static void scale_stat_durations(std::vector<StatRow>& rows, double& total_dur,
                                 TraceIndex::TimeMetric metric) {
    const double us =
        dftracer::utils::utilities::composites::dft::time_metric_us_scale(
            metric);
    if (us == 1.0) return;
    for (auto& r : rows) {
        r.total *= us;
        r.min *= us;
        r.max *= us;
        r.coverage *= us;
    }
    total_dur *= us;
}

static const std::vector<VizSummary::GroupRow>& summary_group_rows(
    const VizSummary& s, GroupBy g) {
    switch (g) {
        case GroupBy::Cat:
            return s.by_cat;
        case GroupBy::Pid:
            return s.by_pid;
        case GroupBy::Fhash:
            return s.by_fhash;
        case GroupBy::Name:
        default:
            return s.by_name;
    }
}

// Whole-trace, unfiltered Analyze answers come from the prebuilt summary, so no
// live scan is needed. Any predicate or sub-range forces the live path.
static bool viz_stats_summary_eligible(const QueryParams& p, double begin_abs,
                                       double end_abs, TraceIndex& index) {
    if (!p.get("query").empty() || !p.get("cat").empty() ||
        !p.get("lanes").empty() || !p.get("filters").empty() ||
        !p.get("file").empty() || !p.get("pid").empty() ||
        !p.get("tid").empty())
        return false;
    std::uint64_t gmin = index.global_min_timestamp_us();
    std::uint64_t gmax = index.global_max_timestamp_us();
    if (gmin == std::numeric_limits<std::uint64_t>::max() || gmax <= gmin)
        return false;
    return begin_abs <= static_cast<double>(gmin) + 1.0 &&
           end_abs >= static_cast<double>(gmax) - 1.0;
}

// Stored per-key duration {sketches, sums} for a fast-path group, or
// {nullptr, nullptr} for groups without per-key aggregates (fhash).
static std::pair<const dftracer::utils::StringViewMap<
                     utilities::common::statistics::DDSketch>*,
                 const dftracer::utils::StringViewMap<double>*>
group_duration_maps(const indexing::ChunkStatistics& s, GroupBy group) {
    switch (group) {
        case GroupBy::Name:
            return {&s.name_duration_sketches, &s.name_duration_sums};
        case GroupBy::Cat:
            return {&s.cat_duration_sketches, &s.cat_duration_sums};
        case GroupBy::Pid:
            return {&s.pid_duration_sketches, &s.pid_duration_sums};
        default:
            return {nullptr, nullptr};  // fhash: too high-cardinality to store
    }
}

// Answers name/cat/pid duration aggregations from the cached per-chunk
// DDSketches (chunk_meta) - the fast path handle_viz_stats used inline, now
// behind the View agg_source interface so a chunk fully inside the window
// contributes its stored count/min/max/sum with no decode.
class ServerStatsSource : public views::detail::PartialSource {
   public:
    ServerStatsSource(TraceIndex& index,
                      const std::vector<const TraceIndex::FileInfo*>& files)
        : index_(index) {
        for (auto* fi : files) by_path_.emplace(fi->path, fi);
    }

    views::detail::PartialSource::Result lookup(
        const views::detail::PartialRequest& req,
        const std::function<void(views::detail::AggAccum&&)>& emit)
        const override {
        views::detail::PartialSource::Result res;
        if (req.needs_argmax || req.schema == nullptr) return res;
        if (req.time_bucket_us != 0 || req.group_by.size() != 1) return res;
        GroupBy group;
        switch (req.group_by[0].kind) {
            case views::GroupKey::Kind::Name:
                group = GroupBy::Name;
                break;
            case views::GroupKey::Kind::Cat:
                group = GroupBy::Cat;
                break;
            case views::GroupKey::Kind::Pid:
                group = GroupBy::Pid;
                break;
            default:
                return res;
        }
        if (!req.agg_field.empty() && req.agg_field != "dur") return res;
        const int fi = views::detail::schema_field_index(*req.schema, "dur");

        res.handled = true;
        const double begin = req.has_window
                                 ? req.begin
                                 : -std::numeric_limits<double>::infinity();
        const double end =
            req.has_window ? req.end : std::numeric_limits<double>::infinity();

        for (const auto& f : req.files) {
            auto it = by_path_.find(f.file_path);
            if (it == by_path_.end()) continue;
            auto meta = index_.chunk_meta(*it->second);
            if (!meta) continue;
            for (const auto& r : meta->stats) {
                const double c_min =
                    static_cast<double>(r.stats.min_timestamp_us);
                const double c_max =
                    static_cast<double>(r.stats.max_timestamp_us);
                if (c_min > c_max) continue;
                if (c_min < begin || c_max > end) continue;
                auto [sketches, sums] = group_duration_maps(r.stats, group);
                if (!sketches) continue;
                if (sketches->empty() && r.stats.duration_count > 0) continue;
                res.covered_chunks.emplace_back(f.file_path, r.checkpoint_idx);
                for (const auto& [key, sk] : *sketches) {
                    if (sk.empty()) continue;
                    views::detail::AggAccum a;
                    std::string kv(key);
                    if (req.group_by[0].kind == views::GroupKey::Kind::Cat)
                        for (char& c : kv)
                            c = static_cast<char>(
                                std::tolower(static_cast<unsigned char>(c)));
                    a.keys.push_back(std::move(kv));
                    a.count = sk.count();
                    a.fields.resize(req.schema->fields.size());
                    a.argmax.resize(req.schema->argmax_count);
                    if (fi >= 0) {
                        views::detail::FieldStat& fs = a.fields[fi];
                        fs.n = sk.count();
                        auto sit = sums->find(key);
                        fs.sum = sit != sums->end() ? sit->second : 0.0;
                        fs.min = sk.min();
                        fs.max = sk.max();
                    }
                    emit(std::move(a));
                }
            }
        }
        return res;
    }

   private:
    TraceIndex& index_;
    std::unordered_map<std::string, const TraceIndex::FileInfo*> by_path_;
};

// GET /api/viz/stats: server-side per-name aggregation over a time range.
// Scans in parallel worker coroutines, each folding into its own map, then
// merges single-threaded (no lock). Returns only the small aggregate table.
static coro::CoroTask<HttpResponse> handle_viz_stats(const HttpRequest& req,
                                                     const QueryParams& params,
                                                     TraceIndex& index) {
    if (!params.has("begin") || !params.has("end")) {
        co_return HttpResponse::bad_request(
            "Missing required parameters: begin, end");
    }

    double begin = params.get_double("begin", 0);
    double end = params.get_double("end", 0);

    auto query = params.get("query");
    if (!query.empty() &&
        !utilities::common::query::try_parse(query).has_value()) {
        co_return HttpResponse::bad_request("Invalid query: " +
                                            std::string(query));
    }

    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max())
            global_min = 0;
    }
    double original_begin = begin;
    double original_end = end;
    if (index.time_metric() != TraceIndex::TimeMetric::US) {
        begin = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(begin)));
        end = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(end)));
    }
    if (normalize && global_min > 0) {
        begin += static_cast<double>(global_min);
        end += static_cast<double>(global_min);
    }

    GroupBy group = parse_group_by(params.get("group"));

    const std::string cache_key =
        std::string(req.path) + "?" + params.canonical_key();
    if (auto hit = index.viz_cache().get(cache_key))
        co_return HttpResponse::ok(std::move(*hit));

    // Whole-trace, unfiltered Analyze: answer from the prebuilt summary (built
    // lazily here). Concurrent builds fall through to the live scan below.
    // Aggregated traces skip it so the live fold below prorates their ph=3
    // records (the summary rows omit them), keeping counts consistent with any
    // sub-window selection.
    if (viz_stats_summary_eligible(params, begin, end, index)) {
        const VizSummary* s = co_await ensure_viz_summary(index);
        if (s && !s->has_aggregated) {
            const auto& gr = summary_group_rows(*s, group);
            std::vector<StatRow> rows;
            rows.reserve(gr.size());
            std::uint64_t total_count = 0;
            double total_dur = 0;
            for (const auto& r : gr) {
                // Summary rows are pre-aggregated scalars with no intervals, so
                // coverage is unavailable (0); the live-scan path fills it in.
                rows.push_back({&r.key, r.count, r.total, r.min, r.max, 0});
                total_count += r.count;
                total_dur += r.total;
            }
            scale_stat_durations(rows, total_dur, index.time_metric());
            co_return HttpResponse::ok(serialize_stats_body(
                total_count, total_dur, original_end - original_begin, false,
                rows));
        }
    }

    // "What was active in this window", so include every event that OVERLAPS
    // [begin, end] with its duration clamped to the window (enclosing events
    // that started earlier still count), and prorate ph=3 aggregates by their
    // overlap. A plain ts-in-window sum leaves a small selection empty because
    // nothing starts inside it.
    //
    // The ts window is applied via time_range/clamping, not the query, so the
    // query carries only the user's field filters (a wide-open ts range keeps
    // build_viz_view's own ts clause a no-op) - otherwise it would reject the
    // enclosers, whose ts precedes `begin`.
    ViewDefinition view = build_viz_view(params, 0.0, 4e18, 0);
    std::vector<const TraceIndex::FileInfo*> target_files =
        select_viz_target_files(index, params, begin, end);

    // Whole-run enclosers (started long before the window) are taken from the
    // summary long-event list; shorter ones from a scan back bounded by the
    // same threshold. ph=3 aggregates prorate over the trace's aggregation
    // window.
    const VizSummary* s = co_await ensure_viz_summary(index);
    auto to_native = [&](double us) {
        return index.time_metric() == TraceIndex::TimeMetric::US
                   ? us
                   : static_cast<double>(
                         index.us_to_native(static_cast<std::uint64_t>(us)));
    };
    double enc_threshold_native =
        s && s->long_threshold_us > 0 ? to_native(s->long_threshold_us) : 0;
    double interval_native =
        s && s->agg_interval_us > 0 ? to_native(s->agg_interval_us) : 0;
    double scan_begin = begin - enc_threshold_native;
    if (scan_begin < 0) scan_begin = 0;

    const char* gcol = group == GroupBy::Cat     ? "cat"
                       : group == GroupBy::Pid   ? "pid"
                       : group == GroupBy::Fhash ? "fhash"
                                                 : "name";

    using StatMap = ankerl::unordered_dense::map<std::string, StatAgg>;
    auto merge_stats = [](StatMap&& a, StatMap&& b) {
        for (auto& [k, v] : b) {
            auto& x = a[k];
            x.count += v.count;
            x.total += v.total;
            if (v.min < x.min) x.min = v.min;
            if (v.max > x.max) x.max = v.max;
            for (auto& [bk, d] : v.cov) x.cov[bk] += d;
        }
        return std::move(a);
    };

    const double cov_bw = end > begin ? (end - begin) / COVERAGE_BUCKETS : 0;
    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    auto sv = views::View::from_files(to_view_files(target_files),
                                      &index.bloom_cache())
                  .phase(views::Phase::Any)
                  .metadata(false)
                  .time_range(scan_begin, end);
    if (view.query) sv = sv.filter(*view.query);

    auto scan = co_await sv.map_batches<StatMap>(
        [begin, end, scan_begin, interval_native, gcol, cov_bw](
            StatMap& acc, const std::vector<std::string_view>& events) {
            thread_local simdjson::dom::parser parser;
            thread_local std::string buf;
            for (auto ev : events) {
                buf.assign(ev);
                auto res = parser.parse(buf);
                if (res.error()) continue;
                auto root = res.value_unsafe();
                if (!root.is_object()) continue;
                auto tr = root["ts"];
                if (tr.error()) continue;
                double ts = json_number(tr.value_unsafe());
                // Events opening before the scan-back window are the whole-run
                // enclosers served from long_events below; skip here to not
                // double-count (chunk pruning still reads their chunk).
                if (ts < scan_begin) continue;
                bool is_agg = false;
                auto phr = root["ph"];
                if (!phr.error()) {
                    if (phr.is_int64())
                        is_agg = phr.get_int64().value_unsafe() == 3;
                    else if (phr.is_uint64())
                        is_agg = phr.get_uint64().value_unsafe() == 3;
                    else if (phr.is_string())
                        is_agg = phr.get_string().value_unsafe() == "A";
                }
                std::string key = extract_group_value(root, gcol);
                if (is_agg) {
                    if (interval_native <= 0) continue;
                    double lo = std::max(begin, ts);
                    double hi = std::min(end, ts + interval_native);
                    if (hi <= lo) continue;
                    double f = (hi - lo) / interval_native;
                    double cnt = 1, sum = 0, dmin = 0, dmax = 0;
                    auto args = root["args"];
                    if (!args.error() && args.is_object()) {
                        auto c = args["dft_cnt"];
                        if (!c.error()) cnt = json_number(c.value_unsafe());
                        auto sm = args["dur_sum"];
                        if (!sm.error())
                            sum = json_number(sm.value_unsafe());
                        else {
                            auto d = args["dur"];
                            if (!d.error()) sum = json_number(d.value_unsafe());
                        }
                        auto mn = args["dur_min"];
                        if (!mn.error()) dmin = json_number(mn.value_unsafe());
                        auto mx = args["dur_max"];
                        if (!mx.error()) dmax = json_number(mx.value_unsafe());
                    }
                    auto& a = acc[key];
                    a.add(cnt * f, sum * f, dmin, dmax);
                    a.mark(lo, hi, begin, cov_bw);
                } else {
                    auto dr = root["dur"];
                    if (dr.error()) continue;
                    double dur = json_number(dr.value_unsafe());
                    double lo = std::max(begin, ts);
                    double hi = std::min(end, ts + dur);
                    if (hi <= lo) continue;
                    auto& a = acc[key];
                    a.add(1.0, hi - lo, dur, dur);
                    a.mark(lo, hi, begin, cov_bw);
                }
            }
        },
        merge_stats, slots, 0);
    StatMap byKey = std::move(scan.value);

    // Whole-run enclosers that opened before the scan-back window. The scan's
    // query filter did not see these, so re-apply it per event.
    if (s) {
        simdjson::dom::parser lp;
        for (const auto& sp : s->long_events) {
            double sb = static_cast<double>(sp.begin);
            if (sb >= scan_begin) continue;  // already covered by the scan
            double lo = std::max(begin, sb);
            double hi = std::min(end, static_cast<double>(sp.end));
            if (hi <= lo) continue;
            auto pr = lp.parse(sp.json);
            if (pr.error() || !pr.value_unsafe().is_object()) continue;
            auto root = pr.value_unsafe();
            if (view.query &&
                !view.query->evaluate(utilities::common::json::JsonValue(root)))
                continue;
            double dur = static_cast<double>(sp.end - sp.begin);
            auto& a = byKey[extract_group_value(root, gcol)];
            a.add(1.0, hi - lo, dur, dur);
            a.mark(lo, hi, begin, cov_bw);
        }
    }

    std::vector<StatRow> rows;
    rows.reserve(byKey.size());
    std::uint64_t total_count = 0;
    double total_dur = 0;
    for (auto& kv : byKey) {
        auto cnt = static_cast<std::uint64_t>(kv.second.count + 0.5);
        double mn = kv.second.min == std::numeric_limits<double>::infinity()
                        ? 0
                        : kv.second.min;
        rows.push_back({&kv.first, cnt, kv.second.total, mn, kv.second.max,
                        kv.second.coverage(cov_bw)});
        total_count += cnt;
        total_dur += kv.second.total;
    }
    std::sort(rows.begin(), rows.end(), [](const StatRow& a, const StatRow& b) {
        return a.total > b.total;
    });

    scale_stat_durations(rows, total_dur, index.time_metric());
    std::string body = serialize_stats_body(
        total_count, total_dur, original_end - original_begin, false, rows);
    if (!req.cancel_token.cancelled()) index.viz_cache().put(cache_key, body);
    co_return HttpResponse::ok(std::move(body));
}

// GET /api/viz/layers: whole-trace reference data - the operation-name ->
// category map (a property of the name, not the view, so fetched once) plus the
// FH file counts: total declared vs. those an I/O event actually touched.
static coro::CoroTask<HttpResponse> handle_viz_layers(
    const HttpRequest& /*req*/, const QueryParams& /*p*/, TraceIndex& index) {
    const VizSummary* s = co_await ensure_viz_summary(index);
    auto& b = scratch_json_builder();
    b.start_object();
    b.escape_and_append_with_quotes("layers");
    b.append_colon();
    b.start_object();
    if (s) {
        bool first = true;
        for (const auto& kv : s->name_cats) {
            if (!first) b.append_comma();
            first = false;
            b.escape_and_append_with_quotes(kv.first);
            b.append_colon();
            b.escape_and_append_with_quotes(kv.second);
        }
    }
    b.end_object();
    b.append_comma();
    b.append_key_value("total_files",
                       static_cast<std::int64_t>(s ? s->total_files : 0));
    b.append_comma();
    b.append_key_value("io_files",
                       static_cast<std::int64_t>(s ? s->io_files : 0));
    b.end_object();
    co_return HttpResponse::ok(std::string(b));
}

// Streaming call-tree worker: claims whole files (work-stealing via
// `next_file`) and folds each file's events into `out` as a partial tree,
// discarding events per file so peak memory is one file's events, not the whole
// trace. Heavy locals are heap-allocated to keep the coroutine frame small.
static coro::CoroTask<void> calltree_stream_worker(
    const std::vector<const TraceIndex::FileInfo*>* files,
    std::atomic<std::size_t>* next_file, TraceIndex* index,
    const ViewDefinition* view, double begin, double end, bool scan_all_chunks,
    bool by_process, std::int64_t cap, std::atomic<std::int64_t>* produced,
    CancelToken cancel, std::vector<FlameNode>* out) {
    auto& arena = *out;
    arena.emplace_back();  // partial root (index 0)
    arena[0].name = "all";
    auto file_buf = std::make_unique<std::vector<FlameEv>>();
    auto open =
        std::make_unique<std::vector<std::pair<double, std::uint32_t>>>();
    auto proc_of = std::make_unique<
        ankerl::unordered_dense::map<std::int64_t, std::uint32_t>>();

    while (true) {
        if (cancel.cancelled()) co_return;
        if (produced->load(std::memory_order_relaxed) >= cap) co_return;
        std::size_t fi = next_file->fetch_add(1, std::memory_order_relaxed);
        if (fi >= files->size()) co_return;
        auto* file_info = (*files)[fi];
        if (file_info->uncompressed_size == 0 &&
            file_info->num_checkpoints == 0)
            continue;

        ViewPlannerInput builder_input;
        builder_input.with_view(*view)
            .with_file_path(file_info->path)
            .with_index_path(file_info->has_bloom_data ? file_info->index_path
                                                       : "")
            .with_uncompressed_size(file_info->uncompressed_size)
            .with_num_checkpoints(file_info->num_checkpoints)
            .with_bloom_cache(&index->bloom_cache())
            .with_time_range(begin, end)
            .with_scan_all_chunks(scan_all_chunks);
        ViewPlannerUtility builder;
        auto build_output = co_await builder.process(builder_input);
        if (!build_output || !build_output->file_may_match) continue;

        file_buf->clear();
        for (const auto& c : build_output->candidates) {
            if (cancel.cancelled()) co_return;
            ViewScannerInput reader_input;
            reader_input.with_file_path(file_info->path)
                .with_index_path(file_info->index_path)
                .with_byte_range(c.start_byte, c.end_byte)
                .with_checkpoint_idx(c.checkpoint_idx)
                .with_view(*view);
            ViewScannerUtility reader;
            auto gen = reader.process(reader_input);
            while (auto batch = co_await gen.next()) {
                FlameEv ev;
                for (auto e : batch->events)
                    if (parse_flame_ev(e, ev)) file_buf->push_back(ev);
            }
        }

        fold_file_events(arena, *proc_of, *open, *file_buf, by_process);
        produced->fetch_add(static_cast<std::int64_t>(file_buf->size()),
                            std::memory_order_relaxed);
        file_buf->clear();
    }
}

// GET /api/viz/calltree: merge events into a flamegraph tree. The hierarchy
// per pid/tid lane comes from ts/dur containment (same nesting the timeline
// draws); identical name-paths fold together across the whole trace.
static coro::CoroTask<HttpResponse> handle_viz_calltree(
    const HttpRequest& req, const QueryParams& params, TraceIndex& index) {
    if (!params.has("begin") || !params.has("end"))
        co_return HttpResponse::bad_request(
            "Missing required parameters: begin, end");

    double begin = params.get_double("begin", 0);
    double end = params.get_double("end", 0);

    auto query = params.get("query");
    if (!query.empty() &&
        !utilities::common::query::try_parse(query).has_value())
        co_return HttpResponse::bad_request("Invalid query: " +
                                            std::string(query));

    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max())
            global_min = 0;
    }
    if (index.time_metric() != TraceIndex::TimeMetric::US) {
        begin = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(begin)));
        end = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(end)));
    }
    if (normalize && global_min > 0) {
        begin += static_cast<double>(global_min);
        end += static_cast<double>(global_min);
    }

    ViewDefinition view = build_viz_view(params, begin, end, 0);
    // The flame tree keys on ts/dur containment and ignores ph=M metadata, so
    // drop it at the reader to engage the no-metadata fast path.
    view.with_include_metadata(false);
    int limit = params.get_int("limit", 0);
    if (limit < 0) limit = 0;

    std::vector<const TraceIndex::FileInfo*> target_files =
        select_viz_target_files(index, params, begin, end);

    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    bool by_process = params.get("group") == "pid";
    bool scan_all_chunks = !params.get("file").empty();
    const std::int64_t cap =
        limit > 0 ? limit : std::numeric_limits<std::int64_t>::max();

    const std::string cache_key =
        std::string(req.path) + "?" + params.canonical_key();
    if (auto hit = index.viz_cache().get(cache_key))
        co_return HttpResponse::ok(std::move(*hit));

    static constexpr const char* CANCELLED_TREE =
        R"({"truncated":true,"tree":{"name":"all","total":0,"self":0,"count":0,"children":[]}})";

    // Stream the tree: each worker claims whole files and folds them into its
    // own partial tree, discarding events per file (bounded memory), then the
    // partials merge. Lanes never cross files, so each partial is complete.
    std::size_t nworkers = std::max<std::size_t>(
        1, std::min(slots, target_files.empty() ? std::size_t{1}
                                                : target_files.size()));
    std::vector<std::vector<FlameNode>> arenas(nworkers);
    std::atomic<std::size_t> next_file{0};
    std::atomic<std::int64_t> produced{0};
    CancelToken cancel = req.cancel_token;

    {
        CoroScope scope;
        auto* files_ptr = &target_files;
        auto* index_ptr = &index;
        auto* view_ptr = &view;
        auto* next_ptr = &next_file;
        auto* produced_ptr = &produced;
        for (std::size_t w = 0; w < nworkers; ++w) {
            auto* out = &arenas[w];
            scope.spawn([files_ptr, next_ptr, index_ptr, view_ptr, begin, end,
                         scan_all_chunks, by_process, cap, produced_ptr, cancel,
                         out](CoroScope&) -> coro::CoroTask<void> {
                co_await calltree_stream_worker(files_ptr, next_ptr, index_ptr,
                                                view_ptr, begin, end,
                                                scan_all_chunks, by_process,
                                                cap, produced_ptr, cancel, out);
            });
        }
        co_await scope.join();
    }

    if (cancel.cancelled()) co_return HttpResponse::ok(CANCELLED_TREE);
    bool truncated = limit > 0 && produced.load() >= cap;

    for (std::size_t t = 1; t < arenas.size(); ++t)
        merge_flame_arena(arenas[0], 0, arenas[t], 0);
    std::vector<FlameNode> arena = std::move(arenas[0]);

    // Process frames are synthetic containers: total/count roll up from their
    // children, self is 0.
    if (by_process) {
        for (std::uint32_t pnode : arena[0].children) {
            double t = 0;
            std::uint64_t c = 0;
            for (std::uint32_t ch : arena[pnode].children) {
                t += arena[ch].total;
                c += arena[ch].count;
            }
            arena[pnode].total = t;
            arena[pnode].count = c;
            arena[pnode].self = 0;
        }
    }

    double root_total = 0;
    std::uint64_t root_count = 0;
    for (std::uint32_t c : arena[0].children) {
        root_total += arena[c].total;
        root_count += arena[c].count;
    }
    arena[0].total = root_total;
    arena[0].count = root_count;
    arena[0].self = 0;

    // Node total/self are summed native durations; scale to us for display.
    const double dur_us =
        dftracer::utils::utilities::composites::dft::time_metric_us_scale(
            index.time_metric());
    if (dur_us != 1.0) {
        for (auto& n : arena) {
            n.total *= dur_us;
            if (n.self > 0) n.self *= dur_us;
        }
    }

    auto& b = scratch_json_builder();
    b.start_object();
    b.append_key_value("truncated", truncated);
    b.append_comma();
    b.escape_and_append_with_quotes("tree");
    b.append_colon();
    serialize_flame_node(b, arena, 0);
    b.end_object();
    std::string body(b);
    index.viz_cache().put(cache_key, body);
    co_return HttpResponse::ok(std::move(body));
}

// One log-spaced duration bucket of the histogram response.
struct HistBucket {
    double lo;
    double hi;
    std::uint64_t count;
};

template <typename builder_type>
void tag_invoke(simdjson::serialize_tag, builder_type& b, const HistBucket& h) {
    b.start_object();
    b.append_key_value("lo", h.lo);
    b.append_comma();
    b.append_key_value("hi", h.hi);
    b.append_comma();
    b.append_key_value("count", h.count);
    b.end_object();
}

// GET /api/viz/histogram: the distribution of event durations matching the
// query in [begin, end]. Collects each matching dur, then reports exact
// percentiles and a log-spaced histogram of the shape. The caller narrows to
// one operation by folding its predicate (name == "...") into the query.
static coro::CoroTask<HttpResponse> handle_viz_histogram(
    const HttpRequest& req, const QueryParams& params, TraceIndex& index) {
    if (!params.has("begin") || !params.has("end"))
        co_return HttpResponse::bad_request(
            "Missing required parameters: begin, end");

    double begin = params.get_double("begin", 0);
    double end = params.get_double("end", 0);

    auto query = params.get("query");
    if (!query.empty() &&
        !utilities::common::query::try_parse(query).has_value())
        co_return HttpResponse::bad_request("Invalid query: " +
                                            std::string(query));

    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max())
            global_min = 0;
    }
    if (index.time_metric() != TraceIndex::TimeMetric::US) {
        begin = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(begin)));
        end = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(end)));
    }
    if (normalize && global_min > 0) {
        begin += static_cast<double>(global_min);
        end += static_cast<double>(global_min);
    }

    ViewDefinition view = build_viz_view(params, begin, end, 0);
    view.with_include_metadata(false);  // aggregate only; skip ph=M records
    int limit = params.get_int("limit", 0);
    if (limit < 0) limit = 0;
    int nbuckets = params.get_int("buckets", 40);
    nbuckets = std::clamp(nbuckets, 4, 200);

    const std::string cache_key =
        std::string(req.path) + "?" + params.canonical_key();
    if (auto hit = index.viz_cache().get(cache_key))
        co_return HttpResponse::ok(std::move(*hit));

    std::vector<const TraceIndex::FileInfo*> target_files =
        select_viz_target_files(index, params, begin, end);
    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    bool single_file = !params.get("file").empty();

    struct DurAcc {
        std::vector<double> durs;
    };
    views::View dv =
        views::View::from_files(to_view_files(target_files),
                                &index.bloom_cache())
            .phase(views::Phase::Events)
            .metadata(false)
            .cancel_when([&req]() { return req.cancel_token.cancelled(); });
    if (view.query) dv = dv.filter(*view.query);
    if (!single_file) dv = dv.time_range(begin, end);
    auto scan = co_await dv.map_batches<DurAcc>(
        [](DurAcc& a, const std::vector<std::string_view>& events) {
            for (auto e : events) {
                EventScalars s;
                if (parse_event_scalars(e, s) && s.has_dur)
                    a.durs.push_back(s.dur);
            }
        },
        [](DurAcc&& x, DurAcc&& y) {
            x.durs.reserve(x.durs.size() + y.durs.size());
            for (double d : y.durs) x.durs.push_back(d);
            return std::move(x);
        },
        slots, limit > 0 ? static_cast<std::uint64_t>(limit) : 0);

    bool truncated = scan.stats.truncated;
    std::vector<double> all = std::move(scan.value.durs);
    std::sort(all.begin(), all.end());

    // Durations are in the trace's native unit; scale to us for display.
    const double dur_us =
        dftracer::utils::utilities::composites::dft::time_metric_us_scale(
            index.time_metric());
    if (dur_us != 1.0)
        for (double& d : all) d *= dur_us;

    auto& sb = scratch_json_builder();
    if (all.empty()) {
        sb.start_object();
        sb.append_key_value("count", 0);
        sb.append_comma();
        sb.escape_and_append_with_quotes("buckets");
        sb.append_colon();
        sb.start_array();
        sb.end_array();
        sb.append_comma();
        sb.append_key_value("truncated", truncated);
        sb.end_object();
        std::string body(sb);
        if (!req.cancel_token.cancelled())
            index.viz_cache().put(cache_key, body);
        co_return HttpResponse::ok(std::move(body));
    }

    std::size_t n = all.size();
    double vmin = all.front();
    double vmax = all.back();
    double sum = 0;
    for (double d : all) sum += d;
    double mean = sum / static_cast<double>(n);
    auto pct = [&all, n](double p) {
        std::size_t idx =
            static_cast<std::size_t>(p * static_cast<double>(n - 1));
        return all[idx];
    };

    // Log-spaced buckets over [max(vmin,1), vmax]; sub-microsecond durs land in
    // the first bucket.
    double lo = std::max(vmin, 1.0);
    double hi = std::max(vmax, lo * 1.0000001);
    double lr = std::log(hi / lo);
    std::vector<std::uint64_t> counts(static_cast<std::size_t>(nbuckets), 0);
    for (double d : all) {
        double v = d < lo ? lo : d;
        std::size_t bi =
            lr > 0 ? static_cast<std::size_t>(static_cast<double>(nbuckets) *
                                              std::log(v / lo) / lr)
                   : 0;
        if (bi >= static_cast<std::size_t>(nbuckets)) bi = nbuckets - 1;
        counts[bi]++;
    }

    sb.start_object();
    sb.append_key_value("count", n);
    sb.append_comma();
    sb.append_key_value("min", vmin);
    sb.append_comma();
    sb.append_key_value("max", vmax);
    sb.append_comma();
    sb.append_key_value("mean", mean);
    sb.append_comma();
    sb.append_key_value("p50", pct(0.50));
    sb.append_comma();
    sb.append_key_value("p90", pct(0.90));
    sb.append_comma();
    sb.append_key_value("p95", pct(0.95));
    sb.append_comma();
    sb.append_key_value("p99", pct(0.99));
    sb.append_comma();
    sb.append_key_value("truncated", truncated);
    sb.append_comma();
    std::vector<HistBucket> buckets;
    buckets.reserve(static_cast<std::size_t>(nbuckets));
    for (int i = 0; i < nbuckets; ++i) {
        buckets.push_back(
            {lo * std::exp(lr * static_cast<double>(i) / nbuckets),
             lo * std::exp(lr * static_cast<double>(i + 1) / nbuckets),
             counts[static_cast<std::size_t>(i)]});
    }
    sb.append_key_value("buckets", buckets);
    sb.end_object();
    std::string body(sb);
    if (!req.cancel_token.cancelled()) index.viz_cache().put(cache_key, body);
    co_return HttpResponse::ok(std::move(body));
}

// A density block as it appears in the response: the map key/aggregate pair
// flattened with `ts` resolved from the column index. `name` borrows the
// aggregate's storage.
struct DensityBlock {
    std::string_view name;
    std::int64_t pid;
    std::int64_t tid;
    double ts;
    double dur;
    std::uint32_t count;
    double total;
    std::uint32_t depth;
    std::string_view group;  // group_by value; omitted from JSON when empty
    bool counter = false;    // ph="C" block: carries a mean counter value
    double value = 0;
};

template <typename builder_type>
void tag_invoke(simdjson::serialize_tag, builder_type& b,
                const DensityBlock& d) {
    b.start_object();
    if (!d.group.empty()) {
        b.append_key_value("group", d.group);
        b.append_comma();
    }
    b.append_key_value("name", d.name);
    b.append_comma();
    b.append_key_value("pid", d.pid);
    b.append_comma();
    b.append_key_value("tid", d.tid);
    b.append_comma();
    b.append_key_value("ts", d.ts);
    b.append_comma();
    b.append_key_value("dur", d.dur);
    b.append_comma();
    b.append_key_value("count", d.count);
    b.append_comma();
    b.append_key_value("total", d.total);
    b.append_comma();
    b.append_key_value("depth", d.depth);
    if (d.counter) {
        b.append_comma();
        b.append_key_value("counter", true);
        b.append_comma();
        b.append_key_value("value", d.value);
    }
    b.end_object();
}

// Serialize collected density blocks (+ optional individual events) into the
// /viz/density response body. Shared by the live-scan and summary paths.
static std::string serialize_density_body(
    const std::vector<std::string>& big, const DensityMap& dens,
    double original_begin, double original_end, double threshold, int limit,
    bool truncated, bool ts_normalized, std::uint64_t display_global_min,
    double max_dur, TraceIndex::TimeMetric metric,
    const std::vector<std::uint32_t>* big_depth = nullptr,
    const ankerl::unordered_dense::map<std::string, std::string>* group_names =
        nullptr) {
    // Blocks are positioned/sized in native threshold units; scale the visible
    // time fields (block ts/dur, duration sums, max_dur) to microseconds.
    const double us =
        dftracer::utils::utilities::composites::dft::time_metric_us_scale(
            metric);
    const double threshold_us = threshold * us;
    max_dur *= us;
    auto& b = scratch_json_builder();
    b.start_object();
    b.escape_and_append_with_quotes("events");
    b.append_colon();
    b.start_array();
    for (std::size_t i = 0; i < big.size(); ++i) {
        if (i > 0) b.append_comma();
        // Inject the server-computed depth as a sibling field (events end in
        // }).
        if (big_depth && i < big_depth->size() && !big[i].empty() &&
            big[i].back() == '}') {
            b.append_raw(std::string_view(big[i]).substr(0, big[i].size() - 1));
            b.append_raw(",\"depth\":");
            b.append(static_cast<std::uint64_t>((*big_depth)[i]));
            b.append_raw("}");
        } else {
            b.append_raw(big[i]);
        }
    }
    b.end_array();
    b.append_comma();
    std::vector<DensityBlock> blocks;
    blocks.reserve(dens.size());
    for (const auto& [k, a] : dens) {
        DensityBlock blk{
            a.name, k.pid, k.tid,
            original_begin + static_cast<double>(k.col) * threshold_us,
            threshold_us, a.count,
            // Counter blocks are "busy" for the whole bucket (uniform width);
            // event blocks carry their summed duration.
            a.counter ? threshold_us : a.total * us, a.depth, k.group};
        if (a.counter) {
            blk.counter = true;
            blk.value =
                a.count ? a.value_sum / static_cast<double>(a.count) : 0;
        }
        blocks.push_back(blk);
    }
    b.append_key_value("density", blocks);
    b.append_comma();
    b.escape_and_append_with_quotes("metadata");
    b.append_colon();
    b.start_object();
    b.append_key_value("begin", original_begin);
    b.append_comma();
    b.append_key_value("end", original_end);
    b.append_comma();
    b.append_key_value("count", big.size());
    b.append_comma();
    b.append_key_value("limit", limit);
    b.append_comma();
    b.append_key_value("density_count", dens.size());
    b.append_comma();
    b.append_key_value("truncated", truncated);
    b.append_comma();
    b.append_key_value("ts_normalized", ts_normalized);
    b.append_comma();
    b.append_key_value("global_min_timestamp_us", display_global_min);
    b.append_comma();
    b.append_key_value("max_dur", max_dur);
    if (group_names && !group_names->empty()) {
        b.append_comma();
        b.escape_and_append_with_quotes("group_names");
        b.append_colon();
        b.start_object();
        bool first = true;
        for (const auto& [hash, name] : *group_names) {
            if (!first) b.append_comma();
            first = false;
            b.escape_and_append_with_quotes(hash);
            b.append_colon();
            b.escape_and_append_with_quotes(name);
        }
        b.end_object();
    }
    b.end_object();
    b.end_object();
    return std::string(b);
}

// The summary is unfiltered, so any server-side predicate forces a live scan.
// pid/tid are exempt: they select whole lanes, which the summary can still do.
static bool viz_summary_eligible(const QueryParams& params) {
    return params.get("query").empty() && params.get("cat").empty() &&
           params.get("lanes").empty() && params.get("filters").empty() &&
           params.get("file").empty() && params.get("group_by").empty();
}

static coro::CoroTask<void> append_app_spans(std::vector<std::string>& out,
                                             TraceIndex& index, double begin,
                                             double end,
                                             const QueryParams& params) {
    if (!viz_summary_eligible(params)) co_return;
    const VizSummary* s = co_await ensure_viz_summary(index);
    if (!s) co_return;
    auto pid_s = params.get("pid");
    auto tid_s = params.get("tid");
    bool has_pid = !pid_s.empty();
    bool has_tid = !tid_s.empty();
    std::int64_t want_pid =
        has_pid ? std::strtoll(pid_s.data(), nullptr, 10) : 0;
    std::int64_t want_tid =
        has_tid ? std::strtoll(tid_s.data(), nullptr, 10) : 0;
    for (const auto& sp : s->app_spans) {
        if (has_pid && sp.pid != want_pid) continue;
        if (has_tid && sp.tid != want_tid) continue;
        if (static_cast<double>(sp.end) > begin &&
            static_cast<double>(sp.begin) < end)
            out.push_back(sp.json);
    }
}

// Re-aggregate one pyramid level's buckets over [begin_abs, end_abs] into
// pixel-column density blocks. `begin_abs`/`end_abs` are absolute us.
static std::string serve_density_from_summary(
    const VizSummary& s, const VizSummary::Level& level,
    const QueryParams& params, double begin_abs, double end_abs,
    double original_begin, double original_end, double threshold,
    bool ts_normalized, std::uint64_t display_global_min,
    TraceIndex::TimeMetric metric) {
    // normalize_event_ts subtracts a NATIVE-unit offset from the (native) event
    // ts, but display_global_min is in us; recover the native value so long
    // events aren't shifted off-screen on non-us traces (the us round-trip
    // loses at most sub-us, invisible on the timeline).
    const std::uint64_t native_global_min =
        dftracer::utils::utilities::composites::dft::scale_between(
            TraceIndex::TimeMetric::US, metric, display_global_min);
    auto pid_s = params.get("pid");
    auto tid_s = params.get("tid");
    bool has_pid = !pid_s.empty();
    bool has_tid = !tid_s.empty();
    std::int64_t want_pid =
        has_pid ? std::strtoll(pid_s.data(), nullptr, 10) : 0;
    std::int64_t want_tid =
        has_tid ? std::strtoll(tid_s.data(), nullptr, 10) : 0;

    std::int64_t b0 = s.bucket_of(begin_abs, level.bucket_us, level.nbuckets);
    std::int64_t b1 = s.bucket_of(end_abs, level.bucket_us, level.nbuckets);
    if (b0 < 0) b0 = 0;
    if (b1 < 0) b1 = static_cast<std::int64_t>(level.nbuckets) - 1;

    DensityMap dens;
    auto add_cell = [&](std::int64_t pid, std::int64_t tid, std::int64_t col,
                        std::uint32_t count, double total,
                        const VizSummary::Cell& cell) {
        if (count == 0 && total <= 0) return;
        auto& a = dens[DensityKey{pid, tid, col, {}}];
        a.count += count;
        a.total += total;
        if (static_cast<double>(cell.max_dur) > a.max_dur) {
            a.max_dur = static_cast<double>(cell.max_dur);
            if (cell.name_id < s.names.size()) a.name = s.names[cell.name_id];
        }
    };
    for (const auto& lane : level.lanes) {
        if (has_pid && lane.pid != want_pid) continue;
        if (has_tid && lane.tid != want_tid) continue;
        auto lo = std::lower_bound(lane.buckets.begin(), lane.buckets.end(),
                                   static_cast<std::uint32_t>(b0));
        for (auto it = lo;
             it != lane.buckets.end() && *it <= static_cast<std::uint32_t>(b1);
             ++it) {
            const auto& cell =
                lane.cells[static_cast<std::size_t>(it - lane.buckets.begin())];
            // A bucket never exceeds one column, so it falls in one column or
            // straddles two. Split it by overlap rather than dumping it whole
            // into the column holding its center, which would misplace a third
            // of the events once buckets and columns are of similar size.
            double bstart = static_cast<double>(s.t_begin) +
                            static_cast<double>(*it) * level.bucket_us;
            std::int64_t col =
                static_cast<std::int64_t>((bstart - begin_abs) / threshold);
            double edge = begin_abs + static_cast<double>(col + 1) * threshold;
            double head = edge - bstart;
            if (head >= level.bucket_us) {
                add_cell(lane.pid, lane.tid, col, cell.count,
                         static_cast<double>(cell.total), cell);
                continue;
            }
            double f = head / level.bucket_us;
            auto c0 = static_cast<std::uint32_t>(
                std::llround(static_cast<double>(cell.count) * f));
            if (c0 > cell.count) c0 = cell.count;
            double t0 = static_cast<double>(cell.total) * f;
            add_cell(lane.pid, lane.tid, col, c0, t0, cell);
            add_cell(lane.pid, lane.tid, col + 1, cell.count - c0,
                     static_cast<double>(cell.total) - t0, cell);
        }
    }

    // Long events at least one pixel wide are drawn as real spans; the rest
    // fold into density blocks, which is the split a live scan makes. The
    // widest win when there are more than a response can carry.
    std::vector<const VizSummary::AppSpan*> wide;
    auto fold_span = [&](const VizSummary::AppSpan& sp) {
        auto dur = static_cast<double>(sp.end - sp.begin);
        std::int64_t col = static_cast<std::int64_t>(
            (static_cast<double>(sp.begin) - begin_abs) / threshold);
        auto& a = dens[DensityKey{sp.pid, sp.tid, col, {}}];
        a.count += 1;
        a.total += dur;
        if (dur > a.max_dur) {
            a.max_dur = dur;
            if (sp.name_id < s.names.size()) a.name = s.names[sp.name_id];
        }
    };
    for (const auto& sp : s.long_events) {
        if (has_pid && sp.pid != want_pid) continue;
        if (has_tid && sp.tid != want_tid) continue;
        if (static_cast<double>(sp.end) <= begin_abs ||
            static_cast<double>(sp.begin) >= end_abs)
            continue;
        if (static_cast<double>(sp.end - sp.begin) >= threshold)
            wide.push_back(&sp);
        else
            fold_span(sp);
    }
    if (wide.size() > VizSummary::MAX_RESPONSE_SPANS) {
        std::nth_element(
            wide.begin(), wide.begin() + VizSummary::MAX_RESPONSE_SPANS,
            wide.end(),
            [](const VizSummary::AppSpan* a, const VizSummary::AppSpan* b) {
                return a->end - a->begin > b->end - b->begin;
            });
        for (std::size_t i = VizSummary::MAX_RESPONSE_SPANS; i < wide.size();
             ++i)
            fold_span(*wide[i]);
        wide.resize(VizSummary::MAX_RESPONSE_SPANS);
    }

    std::vector<std::string> spans;
    ankerl::unordered_dense::set<std::int64_t> span_lanes;
    for (const auto& sp : s.app_spans) {
        if (has_pid && sp.pid != want_pid) continue;
        if (has_tid && sp.tid != want_tid) continue;
        if (static_cast<double>(sp.end) <= begin_abs ||
            static_cast<double>(sp.begin) >= end_abs)
            continue;
        span_lanes.insert((sp.pid << 20) ^ sp.tid);
        spans.push_back(
            ts_normalized && display_global_min > 0
                ? normalize_event_ts(sp.json, native_global_min, metric)
                : sp.json);
    }
    // App spans get an injected depth 0; long events after them do not, so the
    // client stacks overlapping wide events instead of piling them on one row.
    const std::size_t napp_spans = spans.size();
    spans.reserve(spans.size() + wide.size());
    for (const auto* sp : wide) {
        span_lanes.insert((sp->pid << 20) ^ sp->tid);
        spans.push_back(
            ts_normalized && display_global_min > 0
                ? normalize_event_ts(sp->json, native_global_min, metric)
                : sp->json);
    }
    // Folded child activity sits one row below its app span, matching the
    // containment nesting the live path computes.
    if (!span_lanes.empty())
        for (auto& [k, a] : dens)
            if (span_lanes.count((k.pid << 20) ^ k.tid)) a.depth = 1;

    // Re-bucket the summary's ph="C" series onto this window's density columns
    // and fold them into the same density map, so the unfiltered/summary path
    // renders counters as ordinary event blocks like the live path. The summary
    // carries each series' emitting pid/tid (0 = node-level).
    if (!s.counter_series.empty() && threshold > 0 && s.fine_bucket_us > 0) {
        const std::size_t ncols =
            static_cast<std::size_t>((end_abs - begin_abs) / threshold) + 1;
        for (const auto& cd : s.counter_series) {
            std::string series = cd.name + "." + cd.key;
            for (std::size_t i = 0; i < cd.buckets.size(); ++i) {
                double ts_mid = static_cast<double>(s.t_begin) +
                                (static_cast<double>(cd.buckets[i]) + 0.5) *
                                    s.fine_bucket_us;
                long col = static_cast<long>((ts_mid - begin_abs) / threshold);
                if (col < 0 || col >= static_cast<long>(ncols)) continue;
                add_counter_bucket(dens, cd.pid, cd.tid, col, series, cd.sum[i],
                                   cd.cnt[i]);
            }
        }
    }

    std::vector<std::uint32_t> span_depth(napp_spans, 0);
    // Stack counter series above their lane's spans (the summary path skips
    // assign_view_depths, so counters would otherwise pile on depth 0).
    assign_counter_depths(spans, span_depth, dens);
    return serialize_density_body(
        spans, dens, original_begin, original_end, threshold, 0, false,
        ts_normalized, display_global_min, static_cast<double>(s.max_dur),
        metric, &span_depth, nullptr);
}

// Canonicalize one group column: resolved.*/r.* aliases map to their hash
// column and "args.x" to "x". Sets `rt` to the hash type when the canonical
// column is a hash (so its values can be resolved to names for display).
static std::string canonicalize_group_col(
    std::string col,
    std::optional<utilities::indexer::IndexDatabase::HashType>& rt) {
    using HashType = utilities::indexer::IndexDatabase::HashType;
    if (col.rfind("args.", 0) == 0 && col.find('.', 5) == std::string::npos)
        col = col.substr(5);
    auto is = [&](std::string_view a) { return col == a; };
    if (is("resolved.fpath") || is("r.fpath"))
        col = "fhash";
    else if (is("resolved.cwd") || is("r.cwd"))
        col = "cwd";
    else if (is("resolved.hostname") || is("r.hostname") ||
             is("resolved.host") || is("r.host"))
        col = "hhash";
    else if (is("resolved.exec") || is("r.exec"))
        col = "exec_hash";
    else if (is("resolved.cmd") || is("r.cmd"))
        col = "cmd_hash";
    rt = std::nullopt;
    if (col == "fhash" || col == "cwd")
        rt = HashType::FILE;
    else if (col == "hhash")
        rt = HashType::HOST;
    else if (col == "shash" || col == "exec_hash" || col == "cmd_hash")
        rt = HashType::STRING;
    return col;
}

// GET /api/viz/density: like /viz/events, but instead of dropping sub-pixel
// events it buckets them per (pid, tid, pixel-column) into density blocks so
// zoomed-out views still show where activity is. Returns full-size events (with
// args, for the detail panel) plus the aggregated density blocks.
static coro::CoroTask<HttpResponse> handle_viz_density(
    const HttpRequest& req, const QueryParams& params, TraceIndex& index) {
    if (!params.has("begin") || !params.has("end")) {
        co_return HttpResponse::bad_request(
            "Missing required parameters: begin, end");
    }

    double begin = params.get_double("begin", 0);
    double end = params.get_double("end", 0);
    int summary = params.get_int("summary", 2);
    if (summary < 1) summary = 1;

    auto query = params.get("query");
    if (!query.empty() &&
        !utilities::common::query::try_parse(query).has_value()) {
        co_return HttpResponse::bad_request("Invalid query: " +
                                            std::string(query));
    }

    // Optional density grouping column. A well-formed name that matches no
    // event field is fine (all blocks land in the client's "(none)" group);
    // only malformed names are rejected.
    std::string group_col(params.get("group_by"));
    if (group_col.size() > 128)
        co_return HttpResponse::bad_request("group_by too long");
    for (char c : group_col) {
        if (!std::isalnum(static_cast<unsigned char>(c)) && c != '_' &&
            c != '.' && c != '-' && c != ',')
            co_return HttpResponse::bad_request("Invalid group_by column");
    }

    // Hash columns auto-resolve for display: group values stay raw hashes, and
    // a group_names map (hash -> resolved name) rides along in the metadata.
    // resolved.*/r.* aliases map to their hash columns. group_by may list
    // several comma-separated columns; canonicalize each and record its hash
    // type so the composite value's components resolve independently.
    using HashType = utilities::indexer::IndexDatabase::HashType;
    std::vector<std::optional<HashType>> resolve_types;
    {
        std::string rebuilt;
        std::size_t start = 0;
        while (start <= group_col.size()) {
            auto comma = group_col.find(',', start);
            std::string part(group_col.substr(
                start, comma == std::string::npos ? group_col.size() - start
                                                  : comma - start));
            std::optional<HashType> rt;
            std::string canon = canonicalize_group_col(std::move(part), rt);
            if (!rebuilt.empty()) rebuilt.push_back(',');
            rebuilt += canon;
            resolve_types.push_back(rt);
            if (comma == std::string::npos) break;
            start = comma + 1;
        }
        group_col = std::move(rebuilt);
    }
    bool any_resolve = false;
    for (const auto& rt : resolve_types)
        if (rt) any_resolve = true;

    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max())
            global_min = 0;
    }
    double original_begin = begin;
    double original_end = end;
    if (index.time_metric() != TraceIndex::TimeMetric::US) {
        begin = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(begin)));
        end = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(end)));
    }
    if (normalize && global_min > 0) {
        begin += static_cast<double>(global_min);
        end += static_cast<double>(global_min);
    }

    // Bucket width (== the ~1px cutoff), in the client's actual pixels.
    int width = params.get_int("width", DEFAULT_VIEWPORT_WIDTH);
    width = std::clamp(width, MIN_VIEWPORT_WIDTH, MAX_VIEWPORT_WIDTH);
    double threshold =
        duration_threshold(begin, end, static_cast<unsigned>(summary),
                           static_cast<unsigned>(width));

    const std::string cache_key =
        std::string(req.path) + "?" + params.canonical_key();
    if (auto hit = index.viz_cache().get(cache_key))
        co_return HttpResponse::ok(std::move(*hit));

    // Unfiltered views come from the prebuilt summary pyramid (no event cap),
    // built lazily here; concurrent requests fall through to a live scan, as do
    // zooms finer than the pyramid's finest level.
    double cfg_agg_interval = 0;  // trace-declared aggregation window, us
    bool trace_has_agg = false;
    if (threshold > 0 && viz_summary_eligible(params)) {
        const VizSummary* s = co_await ensure_viz_summary(index);
        if (s && s->t_end > s->t_begin) {
            cfg_agg_interval = s->agg_interval_us;
            trace_has_agg = s->has_aggregated;
            // ph=3 aggregates are absent from the duration-built pyramid, so an
            // aggregated trace falls through to the (cheap) live scan.
            if (!s->has_aggregated) {
                if (const VizSummary::Level* level = s->level_for(threshold)) {
                    DFTRACER_UTILS_LOG_DEBUG(
                        "viz: density threshold %.0f us served from level "
                        "%.0f us",
                        threshold, level->bucket_us);
                    co_return HttpResponse::ok(serve_density_from_summary(
                        *s, *level, params, begin, end, original_begin,
                        original_end, threshold, global_min > 0,
                        index.native_to_us(index.global_min_timestamp_us()),
                        index.time_metric()));
                }
            }
        }
    }

    // summary=1 means "no aggregation", but the viewport is still finite.
    // Fold sub-pixel events into density blocks instead of dropping them.
    if (threshold <= 0 && end > begin)
        threshold = (end - begin) / static_cast<double>(width);

    // Enclosing events are fetched by a separate pass (below) so a large
    // `lookback` can never starve the in-window scan's budget. lookback arrives
    // in us; convert to the trace's native unit so scan_begin (native) is right
    // - otherwise on non-us traces the scan-back is off by the unit scale and
    // long events that enclose the window are never read.
    double lookback = params.get_double("lookback", 0);
    if (lookback < 0) lookback = 0;
    if (lookback > 0 && index.time_metric() != TraceIndex::TimeMetric::US)
        lookback = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(lookback)));

    // The client feeds back the longest event in the trace as `lookback`, so
    // honoring it literally rescans everything before the window. Anything that
    // wide is already in the summary's long-event list, so take enclosers from
    // there and scan back only far enough to catch the ones too short to list.
    // Only whole-run enclosers (>= half the span) are guaranteed complete in
    // the summary's long-event list (compact_long exempts them from its
    // per-bucket quota). Shorter enclosers the quota may have dropped, so the
    // live scan back must reach them instead of trusting the list. All three
    // uses below share this one threshold so there is no gap and no
    // double-serve.
    const VizSummary* enc_summary = nullptr;
    double enc_threshold_us = 0;
    double enc_threshold_native = 0;
    if (lookback > 0 && viz_summary_eligible(params)) {
        const VizSummary* s = co_await ensure_viz_summary(index);
        if (s != nullptr && s->long_threshold_us > 0) {
            enc_summary = s;
            enc_threshold_us = s->long_threshold_us;
            enc_threshold_native =
                index.time_metric() == TraceIndex::TimeMetric::US
                    ? enc_threshold_us
                    : static_cast<double>(index.us_to_native(
                          static_cast<std::uint64_t>(enc_threshold_us)));
            lookback = std::min(lookback, enc_threshold_native);
        }
    }

    double scan_begin = begin - lookback;
    if (scan_begin < 0) scan_begin = 0;

    int limit = params.get_int("limit", 0);
    if (limit < 0) limit = 0;

    // Per-scan partial: sub-pixel events fold into `dens`, the rest are kept
    // raw in `big` (with their durations). `max_dur` tracks the longest event
    // seen (folded ones included).
    struct Acc {
        std::vector<std::string> big;
        std::vector<double> big_dur;  // parallel to `big`
        DensityMap dens;
        double max_dur = 0;
        std::vector<AggRec> agg;      // ph=3 records, kept whole
    };
    auto merge_acc = [](Acc&& x, Acc&& y) {
        if (y.max_dur > x.max_dur) x.max_dur = y.max_dur;
        for (auto& d : y.big_dur) x.big_dur.push_back(d);
        for (auto& s : y.big) x.big.emplace_back(std::move(s));
        for (auto& r : y.agg) x.agg.emplace_back(std::move(r));
        for (auto& kv : y.dens) {
            auto it = x.dens.find(kv.first);
            if (it == x.dens.end())
                x.dens.emplace(kv.first, std::move(kv.second));
            else
                it->second.merge_from(kv.second);
        }
        return std::move(x);
    };
    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    const std::uint64_t scan_cap =
        limit > 0 ? static_cast<std::uint64_t>(limit) : 0;
    bool single_file = !params.get("file").empty();
    auto cancel_pred = [&req]() { return req.cancel_token.cancelled(); };
    // Build a window View over the target files. build_viz_view puts the ts
    // window into the per-event query; an explicit ?file= scans every chunk
    // (cached per-chunk bounds can be wrong under clock skew), so skip the
    // time_range chunk pruning then - the per-event filter still applies.
    auto make_window_view = [&](double lo, double hi) {
        ViewDefinition vd = build_viz_view(params, lo, hi, 0);
        views::View v =
            views::View::from_files(
                to_view_files(select_viz_target_files(index, params, lo, hi)),
                &index.bloom_cache())
                .phase(views::Phase::Events)
                .metadata(false)
                .cancel_when(cancel_pred);
        if (vd.query) v = v.filter(*vd.query);
        if (!single_file) v = v.time_range(lo, hi);
        return v;
    };

    // Number of density columns spanning [begin, end]; counter series share
    // this column grid.
    std::size_t ncols =
        threshold > 0 ? static_cast<std::size_t>((end - begin) / threshold) + 1
                      : 0;

    // Pass 1 (in-window): one scan over ph="X" and ph="C". ph="X" events fold
    // into density blocks (small) or stay raw (big); ph="C" events aggregate
    // into per-(name.arg) counter blocks on the same column grid, so counters
    // land in the same density map and render as ordinary events. The two
    // phases split as fused partition branches so a single decode feeds both.
    auto p1run = make_window_view(begin, end)
                     .phase(views::Phase::Any)
                     .partition(slots, scan_cap);
    auto ev_out = p1run.fold<Acc>(
        utilities::common::query::parse_or_throw("ph == 1 or ph == \"X\""),
        [&group_col, threshold, begin](Acc& acc, const auto& jv,
                                       std::string_view raw) {
            double dur = 0;
            if (!fold_density(jv.element(), threshold, begin, acc.dens, &dur,
                              group_col)) {
                acc.big.emplace_back(raw);
                acc.big_dur.push_back(dur);
            }
            if (dur > acc.max_dur) acc.max_dur = dur;
        },
        merge_acc);
    auto cs_out = p1run.fold<Acc>(
        utilities::common::query::parse_or_throw("ph == 2 or ph == \"C\""),
        [begin, threshold, ncols](Acc& acc, const auto& jv, std::string_view) {
            fold_counter_density(jv.element(), begin, threshold, ncols,
                                 acc.dens);
        },
        merge_acc);
    // ph=3 SELECTIVE-aggregation records: the individual events were dropped at
    // capture time, so keep each whole (all args intact for selection) and give
    // it a renderable span below once the aggregation window is known.
    auto ag_out = p1run.fold<Acc>(
        utilities::common::query::parse_or_throw("ph == 3 or ph == \"A\""),
        [](Acc& acc, const auto& jv, std::string_view raw) {
            collect_aggregated(jv.element(), raw, acc.agg);
        },
        merge_acc);
    auto p1 = co_await p1run.execute();
    bool truncated = p1.truncated;

    DensityMap dens = std::move(ev_out->dens);
    // Fold the counter blocks into the same density map (disjoint keys:
    // counters carry the series identity in the density key's group).
    for (auto& [k, a] : cs_out->dens) {
        auto it = dens.find(k);
        if (it == dens.end())
            dens.emplace(k, std::move(a));
        else
            it->second.merge_from(a);
    }
    std::vector<std::string> big = std::move(ev_out->big);
    std::vector<double> big_dur = std::move(ev_out->big_dur);
    // Longest event scanned (folded ones included); the client feeds it back as
    // `lookback` so deep zooms still catch long enclosing events.
    double max_dur = ev_out->max_dur;

    // Aggregation window, from the ts spacing of the in-window aggregates, with
    // the trace-declared cfg as cross-check and single-window fallback.
    double agg_interval = 0;
    if (trace_has_agg || !ag_out->agg.empty()) {
        std::vector<double> ats;
        ats.reserve(ag_out->agg.size());
        for (const auto& r : ag_out->agg) ats.push_back(r.ts);
        agg_interval = infer_agg_interval(std::move(ats));
        double cfg_native = cfg_agg_interval;
        if (cfg_agg_interval > 0 &&
            index.time_metric() != TraceIndex::TimeMetric::US)
            cfg_native = static_cast<double>(index.us_to_native(
                static_cast<std::uint64_t>(cfg_agg_interval)));
        if (agg_interval <= 0) {
            agg_interval = cfg_native;
        } else if (cfg_native > 0) {
            double diff = agg_interval > cfg_native ? agg_interval - cfg_native
                                                    : cfg_native - agg_interval;
            if (diff > 0.5 * cfg_native)
                DFTRACER_UTILS_LOG_WARN(
                    "viz: aggregated interval inferred %.0f differs from cfg "
                    "%.0f",
                    agg_interval, cfg_native);
        }
    }

    // An aggregate's ts is its window start, so one that opened before `begin`
    // still covers the view but the in-window pass skips it (ts < begin). Scan
    // back one window to recover these enclosing aggregates.
    if (agg_interval > 0 && begin > 0) {
        double asb = begin - agg_interval;
        if (asb < 0) asb = 0;
        if (asb < begin) {
            auto er = make_window_view(asb, begin)
                          .phase(views::Phase::Any)
                          .partition(slots, scan_cap);
            auto eo = er.fold<Acc>(
                utilities::common::query::parse_or_throw(
                    "ph == 3 or ph == \"A\""),
                [](Acc& acc, const auto& jv, std::string_view raw) {
                    collect_aggregated(jv.element(), raw, acc.agg);
                },
                merge_acc);
            co_await er.execute();
            for (auto& r : eo->agg) ag_out->agg.emplace_back(std::move(r));
        }
    }

    // Pass 2 (enclosers): keep only events still open at `begin`
    // (ts < begin <= ts + dur); these ancestor bars set containment depth.
    if (scan_begin < begin) {
        // Whole-run enclosers come from the summary below; excluding them here
        // keeps them from being served twice. Everything shorter is caught
        // here.
        double skip_from = enc_summary != nullptr
                               ? enc_threshold_native
                               : std::numeric_limits<double>::infinity();
        auto p2 =
            co_await make_window_view(scan_begin, begin)
                .map_batches<Acc>(
                    [begin, skip_from](
                        Acc& acc, const std::vector<std::string_view>& events) {
                        for (auto ev : events) {
                            double ts = 0, dur = 0;
                            if (!parse_ts_dur(ev, ts, dur)) continue;
                            if (dur > acc.max_dur) acc.max_dur = dur;
                            if (ts < begin && ts + dur > begin &&
                                dur < skip_from) {
                                acc.big.emplace_back(ev);
                                acc.big_dur.push_back(dur);
                            }
                        }
                    },
                    merge_acc, slots, scan_cap);
        truncated = truncated || p2.stats.truncated;
        if (p2.value.max_dur > max_dur) max_dur = p2.value.max_dur;
        for (auto d : p2.value.big_dur) big_dur.push_back(d);
        for (auto& s : p2.value.big) big.emplace_back(std::move(s));
    }
    // Enclosers wide enough to be listed in the summary, found without the
    // scan back toward the start of the trace that finding them live takes.
    if (enc_summary != nullptr) {
        auto pid_s = params.get("pid");
        auto tid_s = params.get("tid");
        bool has_pid = !pid_s.empty();
        bool has_tid = !tid_s.empty();
        std::int64_t want_pid =
            has_pid ? std::strtoll(pid_s.data(), nullptr, 10) : 0;
        std::int64_t want_tid =
            has_tid ? std::strtoll(tid_s.data(), nullptr, 10) : 0;
        for (const auto& sp : enc_summary->long_events) {
            auto dur = static_cast<double>(sp.end - sp.begin);
            // Shorter enclosers are caught by the live scan above; serving them
            // here too would double them (and the quota may have dropped some).
            if (dur < enc_threshold_us) continue;
            if (static_cast<double>(sp.begin) >= begin ||
                static_cast<double>(sp.end) <= begin)
                continue;
            if (has_pid && sp.pid != want_pid) continue;
            if (has_tid && sp.tid != want_tid) continue;
            if (dur > max_dur) max_dur = dur;
            big.push_back(sp.json);
            big_dur.push_back(dur);
        }
    }

    // Bound individual events by count. `threshold` is per-pixel time and is
    // blind to how many events share a pixel, so a narrow-but-dense window
    // (e.g. a whole trace compressed into a sliver by bad timestamps) would
    // otherwise return every event whole. Keep the longest - the slices wide
    // enough to see - and fold the rest into density blocks so none are lost.
    constexpr std::size_t MAX_DENSITY_EVENTS = 30000;
    const std::size_t cap =
        limit > 0 ? static_cast<std::size_t>(limit) : MAX_DENSITY_EVENTS;
    if (big.size() > cap) {
        std::vector<std::uint32_t> idx(big.size());
        std::iota(idx.begin(), idx.end(), std::uint32_t{0});
        std::nth_element(idx.begin(), idx.begin() + static_cast<long>(cap),
                         idx.end(),
                         [&big_dur](std::uint32_t a, std::uint32_t b) {
                             return big_dur[a] > big_dur[b];
                         });
        std::vector<std::uint32_t> overflow(
            idx.begin() + static_cast<long>(cap), idx.end());
        fold_overflow_events(big, big_dur, overflow, threshold, begin, dens,
                             group_col);
        idx.resize(cap);
        std::sort(idx.begin(), idx.end());
        std::vector<std::string> kept;
        std::vector<double> kept_dur;
        kept.reserve(cap);
        kept_dur.reserve(cap);
        for (std::uint32_t i : idx) {
            kept.emplace_back(std::move(big[i]));
            kept_dur.push_back(big_dur[i]);
        }
        big.swap(kept);
        big_dur.swap(kept_dur);
        truncated = true;
    }

    // Bound the block count too. A dense window occupies a block per pixel per
    // lane (threshold is per-pixel time, so many lanes x many columns), so
    // coarsen columns - merge pairs, doubling the effective bucket width -
    // until the map fits the budget. This is the trade the summary pyramid
    // makes, applied here for the live path; block ts/width use eff_threshold.
    constexpr std::size_t MAX_DENSITY_BLOCKS = 120000;
    double eff_threshold = threshold;
    while (dens.size() > MAX_DENSITY_BLOCKS && eff_threshold > 0) {
        DensityMap merged;
        merged.reserve(dens.size() / 2 + 1);
        for (auto& [k, a] : dens) {
            DensityKey nk{k.pid, k.tid, k.col >> 1, k.group};
            auto it = merged.find(nk);
            if (it == merged.end())
                merged.emplace(std::move(nk), std::move(a));
            else
                it->second.merge_from(a);
        }
        dens = std::move(merged);
        eff_threshold *= 2;
        truncated = true;
    }

    drop_unreferenced_hash_records(big);

    co_await append_app_spans(big, index, begin, end, params);

    // Aggregated (ph=3) records ride through as whole events so selection keeps
    // every arg (dur_sum, tag_min, ...); we only tag them and give them the
    // aggregation window as `dur` so they draw as a span. dur is left absent
    // when the window cannot be inferred (client falls back to args.dur_sum).
    // Held back from `big` here so they neither enter the greedy depth packer
    // An aggregate has no positions inside its window, but count + dur_sum let
    // us extrapolate uniformly-spaced synthetic events that position and nest
    // by containment like real events. This is the representation at EVERY zoom
    // - the view never re-skins between a merged block and individual events as
    // the user zooms. A per-request budget bounds the count: split it across
    // the aggregates, subsampling them when there are more than the budget so a
    // full-trace view never floods the response.
    if (agg_interval > 0 && !ag_out->agg.empty()) {
        constexpr std::size_t SYNTHETIC_BUDGET = 40000;
        std::size_t nagg = ag_out->agg.size();
        std::size_t per_agg_cap =
            std::min(static_cast<std::size_t>(std::max(1, width)),
                     std::max<std::size_t>(1, SYNTHETIC_BUDGET / nagg));
        std::size_t step =
            nagg > SYNTHETIC_BUDGET ? nagg / SYNTHETIC_BUDGET : 1;
        for (std::size_t i = 0; i < nagg; i += step)
            extrapolate_aggregate(ag_out->agg[i], agg_interval, begin, end,
                                  per_agg_cap, big, big_dur);
        if (agg_interval > max_dur) max_dur = agg_interval;
    }

    // Resolve hash group values (fhash -> path, hhash -> host, ...) for the
    // groups actually present; unresolvable hashes fall back to the raw value
    // on the client.
    ankerl::unordered_dense::map<std::string, std::string> group_names;
    if (any_resolve && !group_col.empty()) {
        // Split each composite group key into its per-column components and
        // collect the ones from hash columns; resolve each distinct hash once.
        ankerl::unordered_dense::map<std::string, HashType> to_resolve;
        auto collect = [&](std::string_view composite) {
            std::size_t start = 0, idx = 0;
            while (start <= composite.size() && idx < resolve_types.size()) {
                auto sep = composite.find(GROUP_SEP, start);
                auto part =
                    composite.substr(start, sep == std::string_view::npos
                                                ? composite.size() - start
                                                : sep - start);
                if (resolve_types[idx] && !part.empty())
                    to_resolve.emplace(std::string(part), *resolve_types[idx]);
                ++idx;
                if (sep == std::string_view::npos) break;
                start = sep + 1;
            }
        };
        for (const auto& [k, a] : dens)
            if (!k.group.empty()) collect(k.group);
        for (const auto& e : big) {
            auto g = extract_group_from_line(e, group_col);
            if (!g.empty()) collect(g);
        }
        for (const auto& [hash, type] : to_resolve) {
            auto name = index.resolve_hash(type, hash);
            if (!name.empty()) group_names.emplace(hash, std::move(name));
        }
    }

    // Stable per-event/-block depth (computed in absolute space, before ts
    // normalization rewrites the big strings; order is preserved in place).
    std::vector<std::uint32_t> big_depth =
        assign_view_depths(big, dens, begin, eff_threshold);
    // Counters get their own rows above the lane's events (assign_view_depths
    // leaves every counter block on depth 0).
    assign_counter_depths(big, big_depth, dens);
    if (global_min > 0 || index.time_metric() != TraceIndex::TimeMetric::US) {
        for (auto& e : big)
            e = normalize_event_ts(e, global_min, index.time_metric());
    }

    std::string body = serialize_density_body(
        big, dens, original_begin, original_end, eff_threshold, limit,
        truncated, global_min > 0,
        index.native_to_us(index.global_min_timestamp_us()), max_dur,
        index.time_metric(), &big_depth,
        group_names.empty() ? nullptr : &group_names);
    if (!req.cancel_token.cancelled()) index.viz_cache().put(cache_key, body);
    co_return HttpResponse::ok(std::move(body));
}

static std::string serialize_counters_body(const std::vector<double>& read,
                                           const std::vector<double>& write,
                                           const std::vector<double>& ops,
                                           double original_begin,
                                           double original_end, int buckets,
                                           double bucket_us, bool truncated) {
    auto& b = scratch_json_builder();
    b.start_object();
    b.append_key_value("begin", original_begin);
    b.append_comma();
    b.append_key_value("end", original_end);
    b.append_comma();
    b.append_key_value("buckets", static_cast<std::int64_t>(buckets));
    b.append_comma();
    b.append_key_value("bucket_us", bucket_us);
    b.append_comma();
    b.append_key_value("truncated", truncated);
    b.append_comma();
    b.append_key_value("read_bytes", read);
    b.append_comma();
    b.append_key_value("write_bytes", write);
    b.append_comma();
    b.append_key_value("ops", ops);
    b.end_object();
    return std::string(b);
}

// Re-aggregate the summary's finest counter buckets into `buckets` output
// buckets over [begin_abs, end_abs] (absolute us).
static std::string serve_counters_from_summary(
    const VizSummary& s, const VizSummary::Level& level, double begin_abs,
    double end_abs, double original_begin, double original_end, int buckets,
    double bucket_us_out, TraceIndex::TimeMetric metric) {
    std::vector<double> read(buckets, 0.0), write(buckets, 0.0),
        ops(buckets, 0.0);
    std::int64_t fb0 = s.bucket_of(begin_abs, level.bucket_us, level.nbuckets);
    std::int64_t fb1 = s.bucket_of(end_abs, level.bucket_us, level.nbuckets);
    if (fb0 < 0) fb0 = 0;
    if (fb1 < 0) fb1 = static_cast<std::int64_t>(level.nbuckets) - 1;
    for (std::int64_t fb = fb0; fb <= fb1; ++fb) {
        double center = static_cast<double>(s.t_begin) +
                        (static_cast<double>(fb) + 0.5) * level.bucket_us;
        long oi = static_cast<long>((center - begin_abs) / bucket_us_out);
        if (oi < 0 || oi >= buckets) continue;
        auto f = static_cast<std::size_t>(fb);
        read[oi] += level.read_bytes[f];
        write[oi] += level.write_bytes[f];
        ops[oi] += level.ops[f];
    }
    return serialize_counters_body(
        read, write, ops, original_begin, original_end, buckets,
        bucket_us_out *
            dftracer::utils::utilities::composites::dft::time_metric_us_scale(
                metric),
        false);
}

// GET /api/viz/counters: per-bucket read/write bytes and I/O op counts over
// a time range, for bandwidth/IOPS counter tracks. Aggregated server-side in
// parallel (per-worker arrays merged after join).
static coro::CoroTask<HttpResponse> handle_viz_counters(
    const HttpRequest& req, const QueryParams& params, TraceIndex& index) {
    if (!params.has("begin") || !params.has("end")) {
        co_return HttpResponse::bad_request(
            "Missing required parameters: begin, end");
    }

    double begin = params.get_double("begin", 0);
    double end = params.get_double("end", 0);
    int buckets = params.get_int("buckets", 800);
    if (buckets < 16) buckets = 16;
    if (buckets > 4000) buckets = 4000;

    auto query = params.get("query");
    if (!query.empty() &&
        !utilities::common::query::try_parse(query).has_value()) {
        co_return HttpResponse::bad_request("Invalid query: " +
                                            std::string(query));
    }

    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max())
            global_min = 0;
    }
    double original_begin = begin;
    double original_end = end;
    if (index.time_metric() != TraceIndex::TimeMetric::US) {
        begin = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(begin)));
        end = static_cast<double>(
            index.us_to_native(static_cast<std::uint64_t>(end)));
    }
    if (normalize && global_min > 0) {
        begin += static_cast<double>(global_min);
        end += static_cast<double>(global_min);
    }

    double bucket_us = (end - begin) / static_cast<double>(buckets);
    if (bucket_us <= 0) bucket_us = 1;

    const std::string cache_key =
        std::string(req.path) + "?" + params.canonical_key();
    if (auto hit = index.viz_cache().get(cache_key))
        co_return HttpResponse::ok(std::move(*hit));

    // Zoomed-out, unfiltered counter tracks come from the activity summary.
    if (viz_summary_eligible(params)) {
        const VizSummary* s = co_await ensure_viz_summary(index);
        if (s != nullptr && s->t_end > s->t_begin) {
            if (const VizSummary::Level* level = s->level_for(bucket_us)) {
                co_return HttpResponse::ok(serve_counters_from_summary(
                    *s, *level, begin, end, original_begin, original_end,
                    buckets, bucket_us, index.time_metric()));
            }
        }
    }

    ViewDefinition view = build_viz_view(params, begin, end, 0);
    // No cap: only zoomed-in or filtered counter queries reach the live path.
    int limit = params.get_int("limit", 0);
    if (limit < 0) limit = 0;

    const std::size_t nb = static_cast<std::size_t>(buckets);
    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    bool single_file = !params.get("file").empty();

    // Fold per-bucket read/write bytes and op counts over View's scan. Partials
    // init lazily (an unused slot stays empty, the identity for merge below).
    views::View v =
        views::View::from_files(
            to_view_files(select_viz_target_files(index, params, begin, end)),
            &index.bloom_cache())
            .phase(views::Phase::Events)
            .metadata(false)
            .cancel_when([&req]() { return req.cancel_token.cancelled(); });
    if (view.query) v = v.filter(*view.query);
    if (!single_file) v = v.time_range(begin, end);

    auto r = co_await v.map_batches<CounterAcc>(
        [begin, bucket_us, nb](CounterAcc& acc,
                               const std::vector<std::string_view>& events) {
            if (acc.ops.empty()) acc.init(nb);
            for (auto ev : events) fold_counter(ev, begin, bucket_us, nb, acc);
        },
        [](CounterAcc&& x, CounterAcc&& y) {
            if (y.ops.empty()) return std::move(x);
            if (x.ops.empty()) return std::move(y);
            x.merge_from(y);
            return std::move(x);
        },
        slots, limit > 0 ? static_cast<std::uint64_t>(limit) : 0);

    bool truncated = r.stats.truncated;
    CounterAcc total = std::move(r.value);
    if (total.ops.empty()) total.init(nb);

    std::string body = serialize_counters_body(
        total.read_bytes, total.write_bytes, total.ops, original_begin,
        original_end, buckets,
        bucket_us *
            dftracer::utils::utilities::composites::dft::time_metric_us_scale(
                index.time_metric()),
        truncated);
    if (!req.cancel_token.cancelled()) index.viz_cache().put(cache_key, body);
    co_return HttpResponse::ok(std::move(body));
}

// Process-spawning calls: dftracer POSIX (exact "fork"/"clone"/...) and kernel
// syscalls ("__arm64_sys_clone"). Excludes library helpers like ibv_*fork* and
// register_tm_clones, which contain "fork"/"clone" but don't spawn.

// One process in the proctree response. `host` borrows the hostname table;
// `rank` is null when the trace carries no "PR" metadata for the pid, and the
// key is then omitted.
struct ProcNode {
    std::int64_t pid;
    std::int64_t parent;
    std::uint64_t spawn_ts;
    std::uint64_t first_ts;
    std::string_view host;
    std::uint64_t bytes;
    std::uint64_t io_ops;
    double io_busy;
    const std::string* rank;
};

template <typename builder_type>
void tag_invoke(simdjson::serialize_tag, builder_type& b, const ProcNode& n) {
    b.start_object();
    b.append_key_value("pid", n.pid);
    b.append_comma();
    b.append_key_value("parent", n.parent);
    b.append_comma();
    b.append_key_value("spawn_ts", n.spawn_ts);
    b.append_comma();
    b.append_key_value("first_ts", n.first_ts);
    b.append_comma();
    b.append_key_value("host", n.host);
    b.append_comma();
    b.append_key_value("bytes", n.bytes);
    b.append_comma();
    b.append_key_value("io_ops", n.io_ops);
    b.append_comma();
    b.append_key_value("io_busy", n.io_busy);
    if (n.rank) {
        b.append_comma();
        b.append_key_value("rank", *n.rank);
    }
    b.end_object();
}

// GET /api/viz/proctree: infer the process fork hierarchy. The traces record
// the fork/clone in the parent but not the child pid, so link each process to
// the nearest preceding clone in another process (child start follows the clone
// by microseconds). Respects ?file= for per-node trees on multi-node traces.
static coro::CoroTask<HttpResponse> handle_viz_proctree(
    const HttpRequest& req, const QueryParams& params, TraceIndex& index) {
    std::uint64_t gmin = index.global_min_timestamp_us();
    std::uint64_t gmax = index.global_max_timestamp_us();
    if (gmin == std::numeric_limits<std::uint64_t>::max() || gmax <= gmin)
        co_return HttpResponse::ok("{\"nodes\":[]}");

    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t base = normalize ? gmin : 0;

    struct Acc {
        ankerl::unordered_dense::map<std::int64_t, std::uint64_t> first_ts;
        // Explicit parent from the process metadata's args.ppid.
        ankerl::unordered_dense::map<std::int64_t, std::int64_t> ppid;
        // (ts, parent_pid, child_pid): child_pid is args.ret when the fork
        // event records it (dftracer POSIX), else -1 to fall back to inference.
        std::vector<std::tuple<std::uint64_t, std::int64_t, std::int64_t>>
            forks;
        ankerl::unordered_dense::map<std::int64_t, std::uint64_t> bytes;
        // I/O operation count and busy time (us) per process, over I/O-category
        // (POSIX/STDIO/IO) events only.
        ankerl::unordered_dense::map<std::int64_t, std::uint64_t> io_ops;
        ankerl::unordered_dense::map<std::int64_t, double> io_busy;
        ankerl::unordered_dense::map<std::int64_t, std::string> pid_hhash;
        // pid -> rank, from "PR" metadata (args.name == "rank").
        ankerl::unordered_dense::map<std::int64_t, std::string> rank;
        dftracer::utils::StringViewMap<std::string> hh;
    };
    std::size_t slots = std::max<std::size_t>(1, index.max_concurrent());
    std::vector<Acc> accs(slots);
    auto on_batch = [&accs](std::size_t w,
                            const std::vector<std::string_view>& events) {
        thread_local simdjson::dom::parser parser;
        thread_local std::string buf;
        Acc& acc = accs[w];
        for (auto ev : events) {
            buf.assign(ev);
            auto res = parser.parse(buf);
            if (res.error()) continue;
            auto root = res.value_unsafe();
            EventScalars s;
            if (!parse_event_scalars(root, s)) continue;

            // HH metadata (hhash -> hostname) carries no ts.
            if (s.name == "HH") {
                auto a = root["args"];
                if (!a.error() && a.is_object()) {
                    auto v = a["value"];
                    auto n = a["name"];
                    if (!v.error() && v.is_string() && !n.error() &&
                        n.is_string())
                        acc.hh.emplace(
                            std::string(v.get_string().value_unsafe()),
                            std::string(n.get_string().value_unsafe()));
                }
                continue;
            }
            // PR metadata (pid -> rank) also carries no ts.
            if (s.name == "PR") {
                auto a = root["args"];
                if (!a.error() && a.is_object() && s.has_pid) {
                    auto an = a["name"];
                    auto av = a["value"];
                    if (!an.error() && an.is_string() &&
                        an.get_string().value_unsafe() == "rank" &&
                        !av.error() && av.is_string())
                        acc.rank.emplace(
                            s.pid, std::string(av.get_string().value_unsafe()));
                }
                continue;
            }
            if (!s.has_pid || !s.has_ts) continue;
            const std::int64_t pid = s.pid;
            const auto ts = static_cast<std::uint64_t>(s.ts);
            auto it = acc.first_ts.find(pid);
            if (it == acc.first_ts.end())
                acc.first_ts.emplace(pid, ts);
            else if (ts < it->second)
                it->second = ts;

            std::int64_t child = -1;
            double ret = 0;
            auto args = root["args"];
            if (!args.error() && args.is_object()) {
                auto ppr = args["ppid"];
                if (!ppr.error()) {
                    auto pp = static_cast<std::int64_t>(
                        json_number(ppr.value_unsafe()));
                    if (pp > 0 && pp != pid) acc.ppid[pid] = pp;
                }
                auto rr = args["ret"];
                if (!rr.error()) {
                    ret = json_number(rr.value_unsafe());
                    child = static_cast<std::int64_t>(ret);
                }
                auto hr = args["hhash"];
                if (!hr.error() && hr.is_string() &&
                    acc.pid_hhash.find(pid) == acc.pid_hhash.end())
                    acc.pid_hhash.emplace(
                        pid, std::string(hr.get_string().value_unsafe()));
            }
            // I/O bytes per process: args.ret on read/write ops.
            if (ret > 0 && (s.name.find("read") != std::string_view::npos ||
                            s.name.find("write") != std::string_view::npos))
                acc.bytes[pid] += static_cast<std::uint64_t>(ret);

            if (s.cat == "POSIX" || s.cat == "STDIO" || s.cat == "IO") {
                acc.io_ops[pid] += 1;
                if (s.has_dur) acc.io_busy[pid] += s.dur;
            }

            if (!s.name.empty() && is_fork_syscall(s.name))
                acc.forks.emplace_back(ts, pid, child > 0 ? child : -1);
        }
    };

    // The per-process totals are exact and whole-trace, so the summary answers
    // this identically to a scan; only a single-file view has to scan.
    bool single_file = !params.get("file").empty();
    const VizSummary* summary =
        single_file ? nullptr : co_await ensure_viz_summary(index);
    if (summary != nullptr) {
        Acc& acc = accs[0];
        for (const auto& p : summary->procs) {
            if (p.first_ts != 0) acc.first_ts.emplace(p.pid, p.first_ts);
            if (p.ppid > 0) acc.ppid.emplace(p.pid, p.ppid);
            if (p.bytes != 0) acc.bytes.emplace(p.pid, p.bytes);
            if (p.io_ops != 0) acc.io_ops.emplace(p.pid, p.io_ops);
            if (p.io_busy != 0) acc.io_busy.emplace(p.pid, p.io_busy);
            if (!p.hhash.empty()) acc.pid_hhash.emplace(p.pid, p.hhash);
            if (!p.rank.empty()) acc.rank.emplace(p.pid, p.rank);
        }
        for (const auto& f : summary->forks)
            acc.forks.emplace_back(f.ts, f.pid, f.child);
        for (const auto& h : summary->hosts) acc.hh.emplace(h.first, h.second);
    } else {
        auto files =
            select_viz_target_files(index, params, static_cast<double>(gmin),
                                    static_cast<double>(gmax));
        co_await views::View::from_files(to_view_files(files),
                                         &index.bloom_cache())
            .phase(views::Phase::Events)
            .cancel_when([&req]() { return req.cancel_token.cancelled(); })
            .for_each_batch(on_batch, slots);
    }

    ankerl::unordered_dense::map<std::int64_t, std::uint64_t> first_ts;
    ankerl::unordered_dense::map<std::int64_t, std::int64_t> parent_of;
    ankerl::unordered_dense::map<std::int64_t, std::uint64_t> spawn_of;
    std::vector<std::pair<std::uint64_t, std::int64_t>> inf_forks;  // (ts, pid)
    for (auto& a : accs) {
        for (auto& kv : a.first_ts) {
            auto it = first_ts.find(kv.first);
            if (it == first_ts.end())
                first_ts.emplace(kv.first, kv.second);
            else if (kv.second < it->second)
                it->second = kv.second;
        }
        // Explicit edges from the fork event's args.ret (child pid + spawn ts).
        for (auto& [ts, ppid, child] : a.forks) {
            if (child > 0) {
                parent_of[child] = ppid;
                spawn_of[child] = ts;
            } else {
                inf_forks.emplace_back(ts, ppid);
            }
        }
    }
    // args.ppid metadata fills in any process not linked by a fork event.
    for (auto& a : accs)
        for (auto& kv : a.ppid)
            if (parent_of.find(kv.first) == parent_of.end())
                parent_of.emplace(kv.first, kv.second);
    std::sort(inf_forks.begin(), inf_forks.end());

    // Merge host (hhash -> hostname resolved) and I/O bytes per process.
    dftracer::utils::StringViewMap<std::string> hh;
    ankerl::unordered_dense::map<std::int64_t, std::string> pid_hhash;
    ankerl::unordered_dense::map<std::int64_t, std::uint64_t> bytes;
    ankerl::unordered_dense::map<std::int64_t, std::uint64_t> io_ops;
    ankerl::unordered_dense::map<std::int64_t, double> io_busy;
    ankerl::unordered_dense::map<std::int64_t, std::string> rank;
    for (auto& a : accs) {
        for (auto& kv : a.hh) hh.emplace(kv.first, kv.second);
        for (auto& kv : a.pid_hhash) pid_hhash.emplace(kv.first, kv.second);
        for (auto& kv : a.bytes) bytes[kv.first] += kv.second;
        for (auto& kv : a.io_ops) io_ops[kv.first] += kv.second;
        for (auto& kv : a.io_busy) io_busy[kv.first] += kv.second;
        for (auto& kv : a.rank) rank.emplace(kv.first, kv.second);
    }

    std::vector<std::pair<std::uint64_t, std::int64_t>>
        procs;  // (first_ts, pid)
    procs.reserve(first_ts.size());
    for (auto& kv : first_ts) procs.emplace_back(kv.second, kv.first);
    std::sort(procs.begin(), procs.end());

    // Time-inference fallback (traces without args.ret/ppid): link a process to
    // the nearest preceding clone in another process.
    std::vector<bool> used(inf_forks.size(), false);
    std::vector<ProcNode> nodes;
    nodes.reserve(procs.size());
    const double proc_dur_us =
        dftracer::utils::utilities::composites::dft::time_metric_us_scale(
            index.time_metric());
    for (auto& [fts, pid] : procs) {
        std::int64_t parent = -1;
        std::uint64_t spawn_ts = 0;
        auto pit = parent_of.find(pid);
        auto sit = spawn_of.find(pid);
        // A fork cannot spawn a child that already existed before it: reject
        // such edges (pid reuse across runs) and fall back to time inference.
        bool valid = pit != parent_of.end() &&
                     (sit == spawn_of.end() || sit->second <= fts);
        if (valid) {
            parent = pit->second;
            if (sit != spawn_of.end()) spawn_ts = sit->second - base;
        } else {
            auto hi = std::upper_bound(
                inf_forks.begin(), inf_forks.end(),
                std::make_pair(fts, std::numeric_limits<std::int64_t>::max()));
            for (auto it = hi; it != inf_forks.begin();) {
                --it;
                auto idx = static_cast<std::size_t>(it - inf_forks.begin());
                if (!used[idx] && it->second != pid) {
                    used[idx] = true;
                    parent = it->second;
                    spawn_ts = it->first - base;
                    break;
                }
            }
        }
        std::string_view host;
        auto hp = pid_hhash.find(pid);
        if (hp != pid_hhash.end()) {
            auto hn = hh.find(hp->second);
            if (hn != hh.end()) host = hn->second;
        }
        auto bp = bytes.find(pid);
        auto op = io_ops.find(pid);
        auto ib = io_busy.find(pid);
        auto rk = rank.find(pid);
        nodes.push_back({pid, parent, index.native_to_us(spawn_ts),
                         index.native_to_us(fts > base ? fts - base : 0), host,
                         bp != bytes.end() ? bp->second : 0,
                         op != io_ops.end() ? op->second : 0,
                         (ib != io_busy.end() ? ib->second : 0.0) * proc_dur_us,
                         rk != rank.end() ? &rk->second : nullptr});
    }

    auto& sb = scratch_json_builder();
    sb.start_object();
    sb.append_key_value("nodes", nodes);
    sb.end_object();
    co_return HttpResponse::ok(std::string(sb));
}

// GET /api/viz/columns: the complete set of groupable columns in the trace
// (top-level scalar fields + args keys), harvested during the summary scan.
static coro::CoroTask<HttpResponse> handle_viz_columns(
    const HttpRequest& /*req*/, const QueryParams& /*params*/,
    TraceIndex& index) {
    // Prefer the durable set stored at index build (available immediately, no
    // scan). Fall back to the summary harvest for indexes built before column
    // discovery existed.
    std::vector<std::string> columns;
    bool ready = true;
    {
        ankerl::unordered_dense::set<std::string> roots;
        for (const auto& f : index.files())
            if (!f.index_path.empty()) roots.insert(f.index_path);
        ankerl::unordered_dense::set<std::string> merged;
        for (const auto& r : roots) {
            try {
                utilities::indexer::IndexDatabase db(
                    r, dftracer::utils::utilities::indexer::IndexOpenMode::
                           ReadOnly);
                for (auto& c : db.query_all_columns())
                    merged.emplace(std::move(c));
            } catch (...) {
            }
        }
        columns.assign(merged.begin(), merged.end());
        std::sort(columns.begin(), columns.end());
    }
    if (columns.empty()) {
        const VizSummary* s = co_await ensure_viz_summary(index);
        if (s) columns = s->columns;
        ready = s != nullptr;  // false => client retries after summary builds
    }

    auto& b = scratch_json_builder();
    b.start_object();
    b.escape_and_append_with_quotes("columns");
    b.append_colon();
    b.start_array();
    bool first = true;
    for (const auto& c : columns) {
        if (!first) b.append_comma();
        first = false;
        b.escape_and_append_with_quotes(c);
    }
    b.end_array();
    b.append_comma();
    b.append_key_value("ready", ready);
    b.end_object();
    co_return HttpResponse::ok(std::string(b));
}

static coro::CoroTask<HttpResponse> handle_viz_breaks(
    const HttpRequest& /*req*/, const QueryParams& params, TraceIndex& index) {
    auto ts_norm_param = params.get("ts_normalize");
    bool normalize = ts_norm_param.empty() || ts_norm_param != "0";
    std::uint64_t global_min = 0;
    if (normalize) {
        global_min = index.global_min_timestamp_us();
        if (global_min == std::numeric_limits<std::uint64_t>::max())
            global_min = 0;
    }
    const VizSummary* s = co_await ensure_viz_summary(index);
    auto& b = scratch_json_builder();
    b.start_object();
    b.escape_and_append_with_quotes("gaps");
    b.append_colon();
    b.start_array();
    if (s) {
        bool first = true;
        for (const auto& g : s->idle_gaps) {
            if (!first) b.append_comma();
            first = false;
            b.start_object();
            b.append_key_value("begin",
                               index.native_to_us(g.first - global_min));
            b.append_comma();
            b.append_key_value("end",
                               index.native_to_us(g.second - global_min));
            b.end_object();
        }
    }
    b.end_array();
    b.append_comma();
    b.append_key_value("multi_run", s && !s->idle_gaps.empty());
    b.end_object();
    co_return HttpResponse::ok(std::string(b));
}

void register_viz_api(Router& router, TraceIndex& index) {
    auto* index_ptr = &index;
    const RouteParam BEGIN{"begin", "Window start (us)", true, "0"};
    const RouteParam END{"end", "Window end (us)", true, "999999999"};
    const RouteParam SUMMARY{"summary", "LOD level (1=full detail)", true, "1"};

    router.get(
        "/api/viz/proctree",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_proctree(req, params, *index_ptr);
        },
        RouteDoc{
            "Inferred process/fork hierarchy with host, rank, and I/O.",
            "Visualization",
            {{"file", "Limit to one trace file", false, ""}},
            R"({"nodes":[{"pid":100,"parent":-1,"host":"node01","rank":"0",)"
            R"("bytes":16384,"io_ops":4,"io_busy":600.0}]})"});

    router.get(
        "/api/viz/counters",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_counters(req, params, *index_ptr);
        },
        RouteDoc{"Read/write bytes and I/O op counts per time bucket.",
                 "Visualization",
                 {BEGIN, END, SUMMARY},
                 R"({"buckets":[{"ts":0,"read_bytes":4096,"write_bytes":0,)"
                 R"("read_ops":1,"write_ops":0}]})"});

    router.get(
        "/api/viz/breaks",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_breaks(req, params, *index_ptr);
        },
        RouteDoc{
            "Globally-idle time gaps and multi-run detection.",
            "Visualization",
            {{"ts_normalize", "Normalize to global min (default on)", false,
              "1"}},
            R"({"gaps":[{"begin":50000,"end":900000}],"multi_run":true})"});

    router.get(
        "/api/viz/columns",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_columns(req, params, *index_ptr);
        },
        RouteDoc{"Groupable columns present in the trace.",
                 "Visualization",
                 {},
                 R"({"columns":["cat","name","mhost","fhash"],"ready":true})"});

    router.get(
        "/api/viz/events",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_events(req, params, *index_ptr);
        },
        RouteDoc{"Events for rendering: time-windowed, LOD-aggregated.",
                 "Visualization",
                 {BEGIN,
                  END,
                  SUMMARY,
                  {"pid", "Filter by process id", false, ""},
                  {"cat", "Filter by category", false, ""},
                  {"query", "DSL predicate, e.g. dur >= 1000", false, ""}},
                 R"({"events":[],"metadata":{"begin":0,"end":1000000,)"
                 R"("count":42,"truncated":false,"ts_normalized":true}})"});

    router.get(
        "/api/viz/density",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_density(req, params, *index_ptr);
        },
        RouteDoc{"Sub-pixel events bucketed into density blocks.",
                 "Visualization",
                 {BEGIN,
                  END,
                  {"summary", "LOD level", true, "2"},
                  {"width", "Canvas width in px (sets the 1px fold cutoff)",
                   false, "1920"}},
                 R"({"events":[],"density":[{"pid":100,"tid":100,"ts":0,)"
                 R"("dur":24457,"count":910,"total":22044,"depth":0}]})"});

    router.get(
        "/api/viz/stats",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_stats(req, params, *index_ptr);
        },
        RouteDoc{"Per-name aggregation over a time range (Analyze).",
                 "Visualization",
                 {BEGIN, END, SUMMARY},
                 R"({"count":100,"total_dur":5000,"names":[{"name":"read",)"
                 R"("count":50,"total":2500,"avg":50,"min":10,"max":90}]})"});

    router.get(
        "/api/viz/calltree",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_calltree(req, params, *index_ptr);
        },
        RouteDoc{
            "Merged flamegraph tree from ts/dur containment.",
            "Visualization",
            {BEGIN,
             END,
             SUMMARY,
             {"group", "Set to 'pid' to keep processes separate", false, ""}},
            R"({"name":"root","total":5000,"self":0,"count":0,)"
            R"("children":[{"name":"read","total":2500,"self":2500,)"
            R"("count":50}]})"});

    router.get(
        "/api/viz/histogram",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_histogram(req, params, *index_ptr);
        },
        RouteDoc{"Duration distribution: percentiles + log-spaced buckets.",
                 "Visualization",
                 {BEGIN,
                  END,
                  SUMMARY,
                  {"query", "DSL predicate to narrow to one op", false, ""}},
                 R"({"min":10,"max":900,"p50":150,"p99":880,"buckets":[]})"});

    router.get(
        "/api/viz/layers",
        [index_ptr](const HttpRequest& req,
                    const QueryParams& params) -> coro::CoroTask<HttpResponse> {
            co_return co_await handle_viz_layers(req, params, *index_ptr);
        },
        RouteDoc{"Operation-name to category map; declared vs I/O files.",
                 "Visualization",
                 {},
                 R"({"layers":{"read":"POSIX","write":"POSIX"},)"
                 R"("total_files":2,"io_files":2})"});
}

}  // namespace dftracer::utils::server
