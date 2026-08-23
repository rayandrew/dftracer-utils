#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/trace/views/bloom_fold.h>
#include <dftracer/utils/trace/views/index_write_lock.h>
#include <dftracer/utils/utilities/indexer/index_batch_sink.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/index_database_writer_context.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>

#include <exception>
#include <mutex>
#include <string_view>
#include <variant>
#include <vector>

namespace dftracer::utils::trace::views::detail {

namespace {

using BV = visitors::BloomCore;
constexpr std::uint32_t NO_ID = dftracer::utils::StringIntern::NO_ID;

std::string_view resolve_or_empty(const dftracer::utils::StringIntern& intern,
                                  std::uint32_t id) {
    return id == NO_ID ? std::string_view{} : intern.resolve(id);
}

}  // namespace

void BloomFold::step(const FoldBatch& batch) {
    FileState& fs = files_[batch.unit.file_path];
    if (fs.index_path.empty()) fs.index_path = batch.unit.index_path;

    auto it = fs.chunks.find(batch.unit.checkpoint_idx);
    if (it == fs.chunks.end()) {
        it = fs.chunks.emplace(batch.unit.checkpoint_idx, ChunkState{}).first;
        BV::init_chunk_state(it->second, config_, /*extra_dims=*/{});
    }
    ChunkState& chunk = it->second;

    for (const auto& e : batch.events) {
        if (e.phase == RecordPhase::METADATA) {
            std::string_view name, value;
            for (const auto& [k, v] : e.args) {
                const auto* id = std::get_if<std::uint32_t>(&v);
                if (!id) continue;
                const auto key = intern_->resolve(k);
                if (key == "name")
                    name = intern_->resolve(*id);
                else if (key == "value")
                    value = intern_->resolve(*id);
            }
            BV::observe_metadata(chunk, resolve_or_empty(*intern_, e.name_id),
                                 value, name);
            continue;
        }

        const std::string_view name = resolve_or_empty(*intern_, e.name_id);
        const std::string_view cat = resolve_or_empty(*intern_, e.cat_id);
        const std::string_view hhash = resolve_or_empty(*intern_, e.hhash_id);
        const std::string_view fhash = resolve_or_empty(*intern_, e.fhash_id);

        std::string_view shash;
        for (const auto& [k, v] : e.args) {
            const auto* id = std::get_if<std::uint32_t>(&v);
            if (!id) continue;
            const auto key = intern_->resolve(k);
            if (key == "cmd_hash") {
                shash = intern_->resolve(*id);
                break;
            }
            if (key == "exec_hash" && shash.empty())
                shash = intern_->resolve(*id);
        }

        // Groupable columns: name/cat, args keys, and fhash/hhash (lifted out
        // of args). pid/tid/ts/dur are axis fields, not offered as columns.
        if (fs.col_seen_names.find(name) == fs.col_seen_names.end()) {
            fs.col_seen_names.emplace(name);
            if (!name.empty()) fs.columns.emplace("name");
            if (!cat.empty()) fs.columns.emplace("cat");
            if (!fhash.empty()) fs.columns.emplace("fhash");
            if (!hhash.empty()) fs.columns.emplace("hhash");
            for (const auto& [k, v] : e.args)
                fs.columns.emplace(intern_->resolve(k));
        }

        BV::observe_data(chunk, fs.pidtid, config_, name, cat, e.pid, e.tid,
                         e.ts, e.dur, e.has_dur, hhash, fhash, shash);
    }
}

void BloomFold::drop_unit(const ScanUnit& unit) {
    auto it = files_.find(unit.file_path);
    if (it != files_.end()) it->second.chunks.erase(unit.checkpoint_idx);
}

void BloomFold::merge(Fold& slice) {
    auto& other = static_cast<BloomFold&>(slice);
    for (auto& [file, ofs] : other.files_) {
        FileState& fs = files_[file];
        if (fs.index_path.empty()) fs.index_path = ofs.index_path;
        for (auto& [cp, ochunk] : ofs.chunks) {
            auto it = fs.chunks.find(cp);
            if (it == fs.chunks.end()) {
                fs.chunks.emplace(cp, std::move(ochunk));
            } else {
                BV::merge_chunk_state(it->second, ochunk);
            }
        }
        for (const auto& c : ofs.columns) fs.columns.emplace(c);
    }
}

coro::CoroTask<bool> BloomFold::finalize(const CoverageSet& covered) {
    namespace idx = utilities::indexer;
    bool wrote = false;

    for (auto& [file, fs] : files_) {
        if (fs.index_path.empty() || fs.chunks.empty()) continue;
        // A chunk missing from a partial read would persist as an empty pruner
        // entry that later reads as "no events".
        if (!covered.covers_file(file)) continue;

        int file_id = -1;
        try {
            idx::IndexDatabase ro(fs.index_path, idx::IndexOpenMode::ReadOnly);
            file_id =
                ro.get_file_info_id(idx::internal::get_logical_path(file));
            if (file_id < 0 || ro.has_bloom_data(file_id)) continue;
        } catch (const std::exception& e) {
            DFTRACER_UTILS_LOG_WARN("bloom delta check failed for %s: %s",
                                    fs.index_path.c_str(), e.what());
            continue;
        }

        // Sparse map -> dense checkpoint vector; a whole-file read fills every
        // checkpoint, so any gap is an empty chunk for an unobserved member.
        const std::uint64_t max_cp = fs.chunks.rbegin()->first;
        std::vector<ChunkState> chunks(static_cast<std::size_t>(max_cp) + 1);
        for (auto& c : chunks) BV::init_chunk_state(c, config_, {});
        for (auto& [cp, chunk] : fs.chunks)
            chunks[static_cast<std::size_t>(cp)] = std::move(chunk);

        std::unique_lock<std::mutex> lk(index_write_mutex(fs.index_path));
        try {
            idx::IndexDatabase db(fs.index_path);
            auto writer = db.begin_write();
            BV::persist_bloom(*writer, file_id, chunks, config_,
                              /*extra_dims=*/{}, fs.columns);
            writer->add_file_capability(file_id,
                                        idx::IndexFileEntryCapability::BLOOM);
            writer->commit();
            wrote = true;
        } catch (const std::exception& e) {
            // A locked or read-only index persists nothing rather than failing
            // the query.
            DFTRACER_UTILS_LOG_WARN("bloom materialization skipped for %s: %s",
                                    fs.index_path.c_str(), e.what());
        }
    }

    co_return wrote;
}

void BloomFold::write_to_sink(
    dftracer::utils::utilities::indexer::IndexBatchSink& sink, int file_id) {
    if (files_.empty()) return;
    FileState& fs = files_.begin()->second;
    if (fs.chunks.empty()) return;
    const std::uint64_t max_cp = fs.chunks.rbegin()->first;
    std::vector<ChunkState> chunks(static_cast<std::size_t>(max_cp) + 1);
    for (auto& c : chunks) BV::init_chunk_state(c, config_, {});
    for (auto& [cp, chunk] : fs.chunks)
        chunks[static_cast<std::size_t>(cp)] = std::move(chunk);
    BV::persist_bloom_sink(sink, file_id, chunks, config_, /*extra_dims=*/{},
                           fs.columns);
}

}  // namespace dftracer::utils::trace::views::detail
