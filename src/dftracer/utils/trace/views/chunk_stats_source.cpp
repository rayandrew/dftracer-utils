#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/trace/indexing/chunk_statistics.h>
#include <dftracer/utils/trace/views/chunk_stats_source.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>

#include <cctype>
#include <limits>

namespace dftracer::utils::trace::views {

namespace {
using dftracer::utils::utilities::indexer::IndexDatabase;
using dftracer::utils::utilities::indexer::internal::get_logical_path;
using indexing::ChunkStatistics;

// The per-key duration sketch + sum maps for a group column, or {null,null}.
std::pair<const StringViewMap<utilities::common::statistics::DDSketch>*,
          const StringViewMap<double>*>
group_maps(const ChunkStatistics& s, GroupKey::Kind kind) {
    switch (kind) {
        case GroupKey::Kind::Name:
            return {&s.name_duration_sketches, &s.name_duration_sums};
        case GroupKey::Kind::Cat:
            return {&s.cat_duration_sketches, &s.cat_duration_sums};
        case GroupKey::Kind::Pid:
            return {&s.pid_duration_sketches, &s.pid_duration_sums};
        default:
            return {nullptr, nullptr};
    }
}
}  // namespace

detail::PartialSource::Result ChunkStatsSource::lookup(
    const detail::PartialRequest& req,
    const std::function<void(detail::AggAccum&&)>& emit) const {
    detail::PartialSource::Result res;

    // Only single-column name/cat/pid duration aggregates map to the stored
    // per-key sketches; anything else scans.
    // Grouped by name/cat/pid, this agg_source has no representative event to
    // offer, so it cannot answer an ArgMax; let the scan handle those.
    if (req.needs_argmax || req.schema == nullptr) return res;
    if (req.time_bucket_us != 0 || req.group_by.size() != 1) return res;
    const GroupKey::Kind kind = req.group_by[0].kind;
    if (kind != GroupKey::Kind::Name && kind != GroupKey::Kind::Cat &&
        kind != GroupKey::Kind::Pid)
        return res;
    if (!req.agg_field.empty() && req.agg_field != "dur") return res;
    const int fi = detail::schema_field_index(*req.schema, "dur");

    res.handled = true;
    const double begin =
        req.has_window ? req.begin : -std::numeric_limits<double>::infinity();
    const double end =
        req.has_window ? req.end : std::numeric_limits<double>::infinity();

    for (const auto& f : req.files) {
        if (f.index_path.empty() || !fs::exists(f.index_path)) continue;
        try {
            IndexDatabase db(
                f.index_path,
                dftracer::utils::utilities::indexer::IndexOpenMode::ReadOnly);
            const int fid = db.get_file_info_id(get_logical_path(f.file_path));
            if (fid < 0) continue;

            for (const auto& r : db.query_chunk_statistics(fid)) {
                const double c_min =
                    static_cast<double>(r.stats.min_timestamp_us);
                const double c_max =
                    static_cast<double>(r.stats.max_timestamp_us);
                if (c_min > c_max) continue;  // corrupt bounds -> scan
                if (c_min < begin || c_max > end) continue;  // not covered

                auto [sketches, sums] = group_maps(r.stats, kind);
                if (!sketches) continue;
                if (sketches->empty()) continue;

                res.covered_chunks.emplace_back(f.file_path, r.checkpoint_idx);
                for (const auto& [key, sketch] : *sketches) {
                    if (sketch.empty()) continue;
                    // A sketch carries no sum-of-squares, so the dur stat's
                    // sumsq stays 0 (Var/Std/SumSq are kept on the scan path).
                    detail::AggAccum a;
                    std::string kv(key);
                    if (kind == GroupKey::Kind::Cat)
                        for (char& c : kv)
                            c = static_cast<char>(
                                std::tolower(static_cast<unsigned char>(c)));
                    a.keys.push_back(std::move(kv));
                    a.count = sketch.count();
                    a.fields.resize(req.schema->fields.size());
                    a.argmax.resize(req.schema->argmax_count);
                    if (fi >= 0) {
                        detail::FieldStat& fs = a.fields[fi];
                        fs.n = sketch.count();
                        auto it = sums->find(key);
                        fs.sum = it != sums->end() ? it->second : 0.0;
                        fs.min = sketch.min();
                        fs.max = sketch.max();
                    }
                    emit(std::move(a));
                }
            }
        } catch (const std::exception&) {
            // Unreadable index: leave uncovered so the scan handles it.
        }
    }
    return res;
}

}  // namespace dftracer::utils::trace::views
