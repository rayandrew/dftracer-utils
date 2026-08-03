#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_BLOOM_FOLD_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_BLOOM_FOLD_H

#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/common/transparent_string_hash.h>
#include <dftracer/utils/utilities/composites/dft/views/fold.h>
#include <dftracer/utils/utilities/composites/dft/visitors/bloom_core.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace dftracer::utils::utilities::composites::dft::views::detail {

/// Builds the per-checkpoint pruner index (bloom filters, chunk statistics,
/// dimension stats, columns) from the events a scan already delivers and writes
/// it to each file's index, as a byproduct of the pass the query runs anyway.
///
/// Chunks are held in a sparse map keyed by checkpoint, so fuse's round-robin
/// unit distribution (each worker sees a non-contiguous set) never balloons a
/// dense array; a whole-file read fills every checkpoint before finalize.
class BloomFold : public Fold {
   public:
    explicit BloomFold(dftracer::utils::StringIntern& intern,
                       visitors::BloomCore::ChunkIndexerConfig config = {})
        : intern_(&intern), config_(std::move(config)) {}

    /// A filtered read would build a pruner index that later reads cannot tell
    /// from a complete one.
    bool accepts(const ScanShape& shape) const override {
        return !shape.filtered;
    }
    bool needs_args() const override { return true; }

    std::unique_ptr<Fold> slice() const override {
        return std::make_unique<BloomFold>(*intern_, config_);
    }

    void step(const FoldBatch& batch) override;
    void seal_unit(const ScanUnit& /*unit*/) override {}
    void drop_unit(const ScanUnit& unit) override;
    void merge(Fold& slice) override;
    coro::CoroTask<bool> finalize(const CoverageSet& covered) override;

    /// Write the single harvested file's bloom/stats/dimension records to a
    /// caller-owned sink (RocksDB or SST), for the streaming index build which
    /// owns the write transaction. Assumes one file; no coverage gate. Consumes
    /// the chunks.
    void write_to_sink(
        dftracer::utils::utilities::indexer::IndexBatchSink& sink, int file_id);

    /// Total data events harvested across this fold's files. Call before
    /// write_to_sink, which consumes the chunks.
    std::uint64_t total_events() const {
        std::uint64_t total = 0;
        for (const auto& [file, fs] : files_)
            for (const auto& [cp, chunk] : fs.chunks)
                total += chunk.statistics.total_events;
        return total;
    }

    /// Harvested per-checkpoint chunks for a file (nullptr if unseen), for
    /// tests. Moved-from once finalize or write_to_sink has run.
    const std::map<std::uint64_t, visitors::BloomCore::ChunkState>* file_chunks(
        const std::string& file) const {
        auto it = files_.find(file);
        return it == files_.end() ? nullptr : &it->second.chunks;
    }

   private:
    using ChunkState = visitors::BloomCore::ChunkState;

    struct FileState {
        std::string index_path;
        std::map<std::uint64_t, ChunkState> chunks;
        dftracer::utils::StringViewSet columns;
        dftracer::utils::StringViewSet col_seen_names;
        visitors::BloomCore::PidTidCache pidtid;
    };

    dftracer::utils::StringIntern* intern_;
    visitors::BloomCore::ChunkIndexerConfig config_;
    std::unordered_map<std::string, FileState> files_;
};

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_BLOOM_FOLD_H
