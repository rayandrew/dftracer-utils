#ifndef DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_INDEX_BATCH_WRITER_H
#define DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_INDEX_BATCH_WRITER_H

#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/coro/channel.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/trace/views/aggregation_fold.h>
#include <dftracer/utils/trace/views/bloom_fold.h>
#include <dftracer/utils/trace/views/dict_fold.h>
#include <dftracer/utils/utilities/indexer/index_batch_sink.h>
#include <dftracer/utils/utilities/indexer/internal/gzip/gzip_indexer.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace dftracer::utils::utilities::indexer::internal {

using trace::views::detail::AggregationFold;
using trace::views::detail::BloomFold;
using trace::views::detail::DictFold;

struct ParsedIndexJob {
    int file_id = 0;
    std::string file_path;
    gzip::GzipBuildArtifacts artifacts;
    // Owns the folds' intern so it outlives them through the channel; declared
    // first so it is destroyed last.
    std::unique_ptr<dftracer::utils::StringIntern> intern;
    std::unique_ptr<BloomFold> bloom_fold;
    std::unique_ptr<DictFold> dict_fold;
    std::unique_ptr<AggregationFold> agg_fold;
    bool success = true;
    std::string error_message;
};

struct BatchWriterMetrics {
    std::atomic<std::uint64_t> write_ns{0};
    std::atomic<std::size_t> files_written{0};
    std::atomic<std::size_t> batches_committed{0};
};

/// Drain `channel`, group `ParsedIndexJob`s into batches of `batch_size`,
/// and commit each batch through a fresh `IndexBatchSink` produced by
/// `make_sink()`. The caller-provided `commit_sink(sink)` finalises the
/// batch: for RocksDB-backed sinks it calls `.commit()`; for SST-backed
/// sinks it flushes to disk and routes `Artifacts` to a registry.
///
/// `MakeSink` must be invocable as `() -> std::unique_ptr<IndexBatchSink>`
/// (or any subclass thereof). `CommitSink` must be invocable as
/// `(IndexBatchSink&) -> void`.
template <typename MakeSink, typename CommitSink>
inline coro::CoroTask<void> index_batch_write_worker(
    coro::Channel<ParsedIndexJob>* channel, std::size_t batch_size,
    BatchWriterMetrics* metrics, MakeSink make_sink, CommitSink commit_sink) {
    std::vector<ParsedIndexJob> batch;
    batch.reserve(batch_size);

    auto flush = [&]() {
        if (batch.empty()) return;
        auto start = std::chrono::steady_clock::now();

        auto sink_owned = make_sink();
        IndexBatchSink& sink = *sink_owned;
        for (auto& job : batch) {
            if (!job.success) continue;
            try {
                for (const auto& member : job.artifacts.members) {
                    sink.insert_gzip_member(job.file_id, member);
                }
                sink.insert_file_metadata(
                    job.file_id, job.artifacts.checkpoint_size,
                    job.artifacts.total_lines, job.artifacts.total_uc_size);
                if (job.bloom_fold) {
                    job.bloom_fold->write_to_sink(sink, job.file_id);
                }
                if (job.dict_fold) {
                    job.dict_fold->write_to_sink(sink);
                }
                if (job.agg_fold) {
                    job.agg_fold->write_to_sink(sink, job.file_id);
                }
            } catch (const std::exception& e) {
                job.success = false;
                job.error_message = e.what();
                DFTRACER_UTILS_LOG_ERROR(
                    "Failed to write index for %s: %s; file dropped from index",
                    job.file_path.c_str(), e.what());
            }
        }
        commit_sink(sink);

        auto end = std::chrono::steady_clock::now();
        if (metrics) {
            metrics->write_ns.fetch_add(
                static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(end -
                                                                         start)
                        .count()),
                std::memory_order_relaxed);
            std::size_t written = 0;
            for (const auto& job : batch) {
                if (job.success) ++written;
            }
            metrics->files_written.fetch_add(written,
                                             std::memory_order_relaxed);
            metrics->batches_committed.fetch_add(1, std::memory_order_relaxed);
        }
        batch.clear();
    };

    // Each job carries its file's hash table, so a batch of the nominal size
    // holds every entry of `batch_size` high-cardinality files at once. Flush
    // early once the accumulated entries cross a budget to bound peak heap.
    static constexpr std::size_t MAX_BATCH_HASH_ENTRIES = 2u * 1024 * 1024;
    std::size_t batch_hash_entries = 0;

    while (auto item = co_await channel->receive()) {
        std::size_t entries =
            item->dict_fold ? item->dict_fold->entry_count() : 0;
        batch.push_back(std::move(*item));
        batch_hash_entries += entries;
        if (batch.size() >= batch_size ||
            batch_hash_entries >= MAX_BATCH_HASH_ENTRIES) {
            flush();
            batch_hash_entries = 0;
        }
    }
    flush();
    co_return;
}

}  // namespace dftracer::utils::utilities::indexer::internal

#endif  // DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_INDEX_BATCH_WRITER_H
