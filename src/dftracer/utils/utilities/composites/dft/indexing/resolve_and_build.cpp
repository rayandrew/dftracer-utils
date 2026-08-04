#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/common/platform_compat.h>
#include <dftracer/utils/core/common/scratch.h>
#include <dftracer/utils/core/env.h>
#include <dftracer/utils/core/rocksdb/column_families.h>
#include <dftracer/utils/core/rocksdb/db_manager.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_drain.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_serialization.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/event_aggregator.h>
#include <dftracer/utils/utilities/composites/dft/indexing/resolve_and_build.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/aggregation_fold.h>
#include <dftracer/utils/utilities/fileio/compress/gzip_rechunker.h>
#include <dftracer/utils/utilities/indexer/index_builder_utility.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>

#include <optional>
#include <set>
#include <stdexcept>
#include <system_error>

namespace dftracer::utils::utilities::composites::dft::indexing {

using aggregators::AGG_GLOBAL_CONFIG_KEY;
using aggregators::AggGlobalConfig;
using aggregators::EventAggregator;
using aggregators::serialize_agg_global_config;
using indexer::internal::get_logical_path;

coro::CoroTask<ResolverResult> resolve_and_build_index(
    CoroScope* scope, ResolveAndBuildInput input) {
    // Determine parallelism
    std::size_t parallelism = input.parallelism;
    if (parallelism == 0) {
        parallelism = hardware_concurrency();
    }

    // Initial resolve
    IndexResolverUtility resolver;
    ResolverInput resolve_input;
    resolve_input.directory = std::move(input.directory);
    resolve_input.files = std::move(input.files);
    resolve_input.index_dir = input.index_dir;
    resolve_input.require_checkpoints = input.require_checkpoints;
    resolve_input.require_bloom = input.require_bloom;
    resolve_input.require_aggregation = input.require_aggregation;
    resolve_input.aggregation_config = input.aggregation_config;

    auto result = co_await resolver.process(resolve_input);

    if (result.all_files.empty()) {
        co_return result;
    }

    // Neither a cached aggregation tier at a different interval nor a merged
    // aggregation polluted by a changed source can be refined in place; discard
    // every affected index root and rebuild from scratch.
    if ((result.needs_augmentation || result.stale_detected) &&
        !input.force_rebuild) {
        std::set<std::string> index_roots;
        for (const auto& file : result.all_files) {
            index_roots.insert(
                internal::determine_index_path(file, input.index_dir));
        }
        for (const auto& root : index_roots) {
            DFTRACER_UTILS_LOG_WARN("%s; rebuilding %s",
                                    result.stale_detected
                                        ? "Source changed since indexing"
                                        : "Aggregation interval changed",
                                    root.c_str());
            // Close any cached open handle to this index (the resolve above
            // retains one via RocksDBManager, and a prior read may hold the
            // agg tier) before removing the directory, or the removal fails
            // with EBUSY while a RocksDB file is still open.
            dftracer::utils::rocksdb::RocksDBManager::instance().reset(root);
            std::error_code ec;
            fs::remove_all(root, ec);
            if (ec) {
                throw DFTUtilsException(ErrorCode::IO,
                                        "failed to remove stale index " + root +
                                            ": " + ec.message());
            }
        }
        result = co_await resolver.process(resolve_input);
        if (result.all_files.empty()) {
            co_return result;
        }
    }

    // Collect files that need work (checkpoint or aggregation)
    // When force_rebuild is set, process all files
    std::vector<std::string> files_needing_work;
    if (input.force_rebuild) {
        // all_files is already a vector of strings
        files_needing_work = result.all_files;
    } else {
        std::set<std::string> files_needing_work_set;
        for (const auto& item : result.needs_checkpoint) {
            files_needing_work_set.insert(item.file_path);
        }
        for (const auto& item : result.needs_aggregation) {
            files_needing_work_set.insert(item.file_path);
        }
        files_needing_work.assign(files_needing_work_set.begin(),
                                  files_needing_work_set.end());
    }

    if (!files_needing_work.empty()) {
        DFTRACER_UTILS_LOG_INFO(
            "Building index for %zu files (checkpoint: %zu, aggregation: %zu)",
            files_needing_work.size(), result.needs_checkpoint.size(),
            result.needs_aggregation.size());

        // Build a fresh single-root index on node-local scratch when the
        // destination is a slow filesystem, then publish it back. Incremental
        // and multi-root builds run in place.
        std::string build_index_dir = input.index_dir;
        std::string build_index_path = result.index_path;
        std::string staged_final_path;
        std::optional<ScratchSession> scratch;
        // should_stage is cheap for the common local destination and short
        // circuits before any mount probing; only a network destination reaches
        // the single-root and size checks below.
        if (!fs::exists(result.index_path) && should_stage(result.index_path)) {
            std::set<std::string> roots;
            for (const auto& f : files_needing_work)
                roots.insert(
                    internal::determine_index_path(f, input.index_dir));
            if (roots.size() == 1) {
                std::uint64_t input_bytes = 0;
                for (const auto& f : files_needing_work) {
                    std::error_code ec;
                    input_bytes += fs::file_size(f, ec);
                }
                const int factor =
                    dftracer::utils::Env::get<int>("DFTRACER_INDEX_SIZE_FACTOR")
                        .value_or(3);
                scratch.emplace(input_bytes * static_cast<std::uint64_t>(
                                                  factor > 0 ? factor : 1));
                if (scratch->valid()) {
                    staged_final_path = *roots.begin();
                    build_index_dir = scratch->dir();
                    build_index_path = internal::determine_index_path(
                        files_needing_work.front(), build_index_dir);
                    DFTRACER_UTILS_LOG_INFO("Staging index build at %s -> %s",
                                            build_index_path.c_str(),
                                            staged_final_path.c_str());
                } else {
                    scratch.reset();
                }
            }
        }
        const bool staging = scratch && scratch->valid();

        // Set up aggregation components if needed
        std::shared_ptr<dftracer::utils::rocksdb::RocksDatabase> agg_db;
        std::unique_ptr<EventAggregator> merger;
        std::shared_ptr<aggregators::AggregationConfig> agg_config_ptr;

        if (input.require_aggregation && input.aggregation_config) {
            agg_db =
                EventAggregator::open_with_merge_operator(build_index_path);
            merger = std::make_unique<EventAggregator>(agg_db, 0);
            agg_config_ptr = std::make_shared<aggregators::AggregationConfig>(
                *input.aggregation_config);
        }

        auto batch_config = std::make_shared<indexer::IndexBuildBatchConfig>();
        batch_config->file_paths = files_needing_work;
        batch_config->index_dir = build_index_dir;
        batch_config->checkpoint_size = input.checkpoint_size;
        batch_config->parallelism = parallelism;
        batch_config->force_rebuild = input.force_rebuild;
        batch_config->build_bloom = input.build_bloom;
        batch_config->rebuild_root_summaries = true;

        // Aggregate each file via an AggregationFold when required
        if (agg_db && agg_config_ptr) {
            batch_config->agg_fold_factory =
                [agg_config_ptr, intern = merger->intern_table()](
                    dftracer::utils::StringIntern& build_intern)
                -> std::unique_ptr<
                    composites::dft::views::detail::AggregationFold> {
                return std::make_unique<
                    composites::dft::views::detail::AggregationFold>(
                    build_intern, intern, *agg_config_ptr, 0);
            };
        }

        auto batch_result = co_await indexer::IndexBatchBuilderUtility::process(
            scope, std::move(batch_config));

        // Drain the folds' out-of-band outputs and merge aggregation results
        std::vector<std::string> processed_files;
        if (merger) {
            processed_files =
                merge_aggregation_folds(batch_result.agg_outputs, merger.get());

            // Persist accumulated min/max time bucket so a later read-only
            // reopen recovers the trace origin (otherwise time_range is
            // emitted as an absolute bucket index).
            merger->persist_time_bounds();

            // Write global config and per-file markers
            if (!processed_files.empty()) {
                namespace rcf = dftracer::utils::rocksdb::cf;
                indexer::IndexDatabase idx_db(build_index_path,
                                              indexer::IndexOpenMode::ReadOnly);

                auto batch = agg_db->begin_batch();

                // Write global config (0xFFFE key)
                AggGlobalConfig global_cfg;
                global_cfg.time_interval_us =
                    input.aggregation_config->time_interval_us;
                global_cfg.config_hash = 0;
                agg_db->put(batch, rcf::AGGREGATION,
                            std::string_view(AGG_GLOBAL_CONFIG_KEY, 2),
                            serialize_agg_global_config(global_cfg));

                // Write per-file markers
                for (const auto& file_path : processed_files) {
                    int file_id =
                        idx_db.get_file_info_id(get_logical_path(file_path));
                    if (file_id >= 0) {
                        agg_db->put(batch, rcf::AGGREGATION,
                                    aggregators::make_agg_file_key(file_id),
                                    std::string_view());
                    }
                }

                agg_db->commit_batch(batch);

                // Compact aggregation CFs so all Merge entries become Puts.
                // This allows concurrent ReadOnly access without merge
                // operators.
                agg_db->compact(rcf::AGGREGATION);
                agg_db->compact(rcf::SYSTEM_METRICS);
            }
        }

        // Publish once every handle to the staged store is closed, so the
        // re-resolve below reads the destination.
        if (staging) {
            merger.reset();
            agg_db.reset();
            dftracer::utils::rocksdb::RocksDBManager::instance().reset(
                build_index_path);
            publish_path(build_index_path, staged_final_path);
        }

        // Re-resolve newly built files to get file_ids
        ResolverInput refresh_input;
        refresh_input.files.reserve(result.needs_checkpoint.size());
        for (const auto& item : result.needs_checkpoint) {
            refresh_input.files.push_back(item.file_path);
        }
        refresh_input.index_dir = input.index_dir;
        refresh_input.require_checkpoints = true;

        if (!refresh_input.files.empty()) {
            auto refresh_result = co_await resolver.process(refresh_input);

            // Merge newly indexed into cached
            for (auto& entry : refresh_result.cached) {
                result.cached.push_back(std::move(entry));
            }

            // Update needs_checkpoint with any that still failed
            result.needs_checkpoint =
                std::move(refresh_result.needs_checkpoint);
        }

        // Clear needs_aggregation since we processed them
        result.needs_aggregation.clear();
    }

    DFTRACER_UTILS_LOG_INFO(
        "Resolve complete: %zu total, %zu cached, %zu failed checkpoint",
        result.all_files.size(), result.cached.size(),
        result.needs_checkpoint.size());

    co_return result;
}

coro::CoroTask<void> ensure_indexes_fresh(CoroScope* scope,
                                          std::string directory,
                                          std::vector<std::string> files,
                                          std::string index_dir,
                                          bool force_rebuild) {
    ResolveAndBuildInput in;
    in.directory = std::move(directory);
    in.files = std::move(files);
    in.index_dir = std::move(index_dir);
    in.require_checkpoints = true;
    in.force_rebuild = force_rebuild;
    co_await resolve_and_build_index(scope, std::move(in));
}

coro::CoroTask<void> ensure_index_fresh(CoroScope* scope, std::string directory,
                                        std::string file, std::string index_dir,
                                        bool force_rebuild) {
    std::vector<std::string> files;
    if (!file.empty()) files.push_back(std::move(file));
    co_await ensure_indexes_fresh(scope, std::move(directory), std::move(files),
                                  std::move(index_dir), force_rebuild);
}

coro::CoroTask<MemberNormalizeResult> normalize_members_for_ingest(
    std::vector<std::string> files, std::uint64_t member_size) {
    namespace gzc = fileio::compress;
    MemberNormalizeResult result;
    result.files.reserve(files.size());
    for (auto& f : files) {
        const std::string parent = fs::path(f).parent_path().string();
        const std::string split_dir =
            (parent.empty() ? std::string(".") : parent) + "/split";
        bool did = false;
        std::string nf = co_await gzc::rechunk_to_dir_if_needed(
            f, split_dir, member_size, did);
        if (did) result.split.emplace_back(f, nf);
        result.files.push_back(std::move(nf));
    }
    co_return result;
}

}  // namespace dftracer::utils::utilities::composites::dft::indexing
