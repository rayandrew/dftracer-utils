#include <dftracer/utils/core/common/byte_view.h>
#include <dftracer/utils/core/common/constants.h>
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/string_intern.h>
#include <dftracer/utils/core/coro/channel.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/bloom_fold.h>
#include <dftracer/utils/utilities/composites/dft/views/coverage.h>
#include <dftracer/utils/utilities/composites/dft/views/dict_fold.h>
#include <dftracer/utils/utilities/composites/dft/views/fold.h>
#include <dftracer/utils/utilities/composites/dft/views/fold_event.h>
#include <dftracer/utils/utilities/composites/dft/views/view_definition.h>
#include <dftracer/utils/utilities/composites/dft/views/view_executor.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scan.h>
#include <dftracer/utils/utilities/composites/dft/views/view_scanner_utility.h>
#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>
#include <dftracer/utils/utilities/fileio/parallel/merge.h>
#include <dftracer/utils/utilities/fileio/parallel/parallel_writer.h>
#include <dftracer/utils/utilities/indexer/index_builder_utility.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/index_database_writer_context.h>
#include <dftracer/utils/utilities/indexer/internal/gzip_member_record.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>
#include <simdjson.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

// Fused export + index: write the View's events as a new multi-member trace AND
// build its member+bloom+stats index in one pass (no re-inflate of the output).
// Scan stays parallel (producer coroutines over a channel); a single in-order
// consumer frames + writes members and feeds the index visitors, so
// member_idx == checkpoint_idx and no post-hoc offset remap is needed.
namespace dftracer::utils::utilities::composites::dft::views::detail {

namespace {

// Per-part index data, accumulated while writing and persisted together at the
// end so all parts land in one index transaction sharing one index_path.
struct PartData {
    std::string path;
    std::vector<utilities::indexer::internal::GzipMemberRecord> members;
    std::uint64_t total_uc = 0;
    std::uint64_t total_lines = 0;
};

// `<stem>-<idx>.pfw[.gz]` from a base like `<stem>.pfw[.gz]`; the base itself
// when there is no rollover, so single-file callers keep their exact path.
std::string part_path(const std::string& base, int idx, bool multi) {
    if (!multi) return base;
    const std::size_t pos = base.rfind(".pfw");
    if (pos == std::string::npos) return base + "-" + std::to_string(idx);
    return base.substr(0, pos) + "-" + std::to_string(idx) + base.substr(pos);
}

}  // namespace

coro::CoroTask<ExportStats> run_export_trace_indexed(
    const ViewPlan& plan, const TraceWriteOptions& opts,
    const ProgressFn* progress) {
    namespace pfw = utilities::fileio::parallel;
    namespace cmp = utilities::fileio::compress;
    namespace idx = utilities::indexer;
    // Member size defaults to the checkpoint granularity (a member == a chunk).
    constexpr std::size_t DEFAULT_FLUSH_BYTES =
        constants::indexer::DEFAULT_CHECKPOINT_SIZE;
    constexpr std::size_t BUFFER_HEADROOM_BYTES = 1 * 1024 * 1024;
    constexpr std::size_t MAX_PRODUCERS = 16;

    ViewDefinition vdef = make_vdef(plan, /*for_aggregation=*/false);
    std::uint64_t skipped = 0;
    auto units = co_await gather_units(plan, vdef, skipped);

    const std::size_t member_size =
        opts.member_size ? opts.member_size : DEFAULT_FLUSH_BYTES;
    const std::size_t part_size = opts.part_size;  // 0 = single file
    const bool multi = part_size > 0;

    const std::size_t num_producers = std::max<std::size_t>(
        1, std::min<std::size_t>(std::max<std::size_t>(units.size(), 1),
                                 MAX_PRODUCERS));

    // Index folds key their state per file_path, so one fold set spans all
    // parts; finalize (after the members are committed) writes bloom + dict.
    const std::string index_path =
        composites::dft::internal::determine_index_path(opts.output_path,
                                                        opts.index_path);
    dftracer::utils::StringIntern intern;
    DictFold dict(intern);
    BloomFold bloom(intern);
    std::array<Fold*, 2> folds{&dict, &bloom};

    std::vector<PartData> parts;
    std::atomic<std::uint64_t> matched{0}, scanned{0}, units_done{0};
    bool ok = true;

    auto ch = coro::make_channel<std::string>(2 * num_producers + 1);

    co_await run_coro_scope([&](CoroScope& scope) -> coro::CoroTask<void> {
        co_await scope.scope([&](CoroScope& child) -> coro::CoroTask<void> {
            // Consumer: assemble members in arrival order, write + index each,
            // rolling to a new part file once the current part fills up.
            // Members are numbered from 0 within each part, so member_idx
            // doubles as the checkpoint idx per part with no remap.
            child.spawn([&](CoroScope& cscope) -> coro::CoroTask<void> {
                cmp::GzipMemberCompressor comp(opts.level);
                std::string buf;
                std::vector<std::uint8_t> scratch;
                simdjson::dom::parser parser;
                std::string parse_buf;
                std::vector<FoldEvent> fold_events;

                std::unique_ptr<pfw::ParallelWriter> writer;
                pfw::FileLayout layout = pfw::FileLayout::STRIPED;
                std::string cur_path;
                PartData part;

                auto open_part = [&]() -> coro::CoroTask<bool> {
                    cur_path = part_path(opts.output_path,
                                         static_cast<int>(parts.size()), multi);
                    auto cw = pfw::make_writer_for_path(
                        {cur_path, /*baseline_workers=*/1, member_size,
                         BUFFER_HEADROOM_BYTES, opts.compress});
                    writer = std::move(cw.writer);
                    layout = cw.layout.layout;
                    part = PartData{};
                    part.path = cur_path;
                    if (co_await writer->open(cur_path, /*num_workers=*/1,
                                              opts.compress, &cscope) != 0)
                        co_return false;
                    co_return true;
                };

                auto close_part = [&]() -> coro::CoroTask<bool> {
                    if (!writer) co_return true;
                    if (co_await writer->close() != 0) co_return false;
                    if (layout == pfw::FileLayout::SHARDED) {
                        auto shards = writer->output_paths();
                        if (co_await pfw::merge_shards(cur_path, shards) != 0)
                            co_return false;
                    }
                    parts.push_back(std::move(part));
                    writer.reset();
                    co_return true;
                };

                auto emit = [&](std::size_t cut) -> coro::CoroTask<bool> {
                    // Harvest this member's plaintext (buf[0, cut)) into the
                    // index folds before it is compressed away; the scan buffer
                    // has no simdjson padding, so parse each line into a reused
                    // padded buffer and keep owned FoldEvents.
                    fold_events.clear();
                    for (std::size_t ls = 0, i = 0; i < cut; ++i) {
                        if (buf[i] != '\n') continue;
                        std::string_view line(buf.data() + ls, i - ls);
                        ls = i + 1;
                        if (line.empty()) continue;
                        parse_buf.assign(line);
                        parse_buf.resize(
                            line.size() + simdjson::SIMDJSON_PADDING, '\0');
                        auto doc =
                            parser.parse(parse_buf.data(), line.size(), false);
                        if (doc.error()) continue;
                        fold_events.push_back(extract_fold_event(
                            doc.value_unsafe(), intern, /*needs_args=*/true));
                    }
                    ScanUnit unit;
                    unit.file_path = cur_path;
                    unit.index_path = index_path;
                    unit.checkpoint_idx = part.members.size();
                    FoldBatch fb{std::span<const FoldEvent>(fold_events), unit};
                    for (auto* f : folds) f->step(fb);
                    for (auto* f : folds) f->seal_unit(unit);

                    std::uint64_t lines = 0;
                    for (std::size_t i = 0; i < cut; ++i)
                        if (buf[i] == '\n') ++lines;

                    ByteView chunk;
                    if (opts.compress) {
                        if (!comp.compress_member_into(scratch, buf.data(),
                                                       cut))
                            co_return false;
                        chunk = ByteView(
                            reinterpret_cast<const char*>(scratch.data()),
                            scratch.size());
                    } else {
                        chunk = ByteView(buf.data(), cut);
                    }
                    if (co_await writer->write_chunk(0, chunk) != 0)
                        co_return false;
                    auto span = writer->last_member(0);

                    idx::internal::GzipMemberRecord rec;
                    rec.member_idx = part.members.size();
                    rec.c_offset = span ? span->offset : 0;
                    rec.c_size = span ? span->length : chunk.size();
                    rec.uc_offset = part.total_uc;
                    rec.uc_size = cut;
                    rec.first_line_num = part.total_lines;
                    rec.last_line_num =
                        part.total_lines + (lines ? lines - 1 : 0);
                    part.members.push_back(rec);

                    part.total_uc += cut;
                    part.total_lines += lines;
                    buf.erase(0, cut);

                    // Roll at this member boundary once the part is full.
                    if (multi && part.total_uc >= part_size) {
                        if (!co_await close_part()) co_return false;
                        if (!co_await open_part()) co_return false;
                    }
                    co_return true;
                };

                if (!co_await open_part()) {
                    ok = false;
                    co_return;
                }
                auto cons = ch->consumer();
                while (auto item = co_await cons.receive()) {
                    buf.append(*item);
                    while (buf.size() >= member_size) {
                        std::size_t cut = member_size;
                        while (cut < buf.size() && buf[cut] != '\n') ++cut;
                        if (cut >= buf.size()) break;  // no newline yet
                        ++cut;  // include it: member holds whole lines
                        if (!co_await emit(cut)) {
                            ok = false;
                            co_return;
                        }
                    }
                }
                if (!buf.empty() && !co_await emit(buf.size())) {
                    ok = false;
                    co_return;
                }
                if (!co_await close_part()) ok = false;
                co_return;
            });

            // One shared producer, captured by reference (GCC rejects a
            // move-only init-capture in a coroutine lambda) and released when
            // this sub-scope joins, so the channel closes and the consumer
            // drains before close().
            {
                auto prod = ch->producer();
                co_await child.scope(
                    [&](CoroScope& workers) -> coro::CoroTask<void> {
                        for (std::size_t w = 0; w < num_producers; ++w) {
                            workers.spawn([&, w](CoroScope&)
                                              -> coro::CoroTask<void> {
                                std::string batch;
                                for (std::size_t i = w; i < units.size();
                                     i += num_producers) {
                                    if (is_cancelled(plan)) co_return;
                                    ViewScannerInput sin = make_scanner_input(
                                        units[i], vdef, vdef.query);
                                    ViewScannerUtility scanner;
                                    auto gen = scanner.process(sin);
                                    while (auto b = co_await gen.next()) {
                                        if (is_cancelled(plan)) co_return;
                                        matched.fetch_add(
                                            b->events_matched,
                                            std::memory_order_relaxed);
                                        scanned.fetch_add(
                                            b->events_scanned,
                                            std::memory_order_relaxed);
                                        if (b->events.empty()) continue;
                                        batch.clear();
                                        for (const auto& ev : b->events) {
                                            batch.append(ev);
                                            batch.push_back('\n');
                                        }
                                        co_await prod.send(batch);
                                    }
                                    if (progress)
                                        (*progress)(
                                            units_done.fetch_add(
                                                1, std::memory_order_relaxed) +
                                                1,
                                            units.size());
                                }
                                co_return;
                            });
                        }
                        co_return;
                    });
            }
            co_return;
        });
    });

    if (!ok || parts.empty()) {
        throw DFTUtilsException(
            ErrorCode::IO, "export_trace failed writing " + opts.output_path);
    }

    // Persist the index for every part in one transaction. Registering each
    // with its exact mtime/size makes the resolver accept it as fresh; any
    // later edit trips the stale check and rebuilds.
    try {
        idx::IndexDatabase db(index_path);
        auto w = db.begin_write();
        // BLOOM is claimed by BloomFold::finalize below; a crash between the
        // two writes leaves a valid bloom-less index that a later scan
        // rebuilds.
        const auto caps = idx::IndexFileEntryCapability::INDEXING_COMPLETE |
                          idx::IndexFileEntryCapability::MEMBERS |
                          idx::IndexFileEntryCapability::FILE_SUMMARY;
        for (const auto& part : parts) {
            auto logical = idx::internal::get_logical_path(part.path);
            const auto hash = idx::internal::calculate_file_hash(part.path);
            const auto mtime = static_cast<std::uint64_t>(
                idx::internal::get_file_modification_time(part.path));
            const auto bytes = idx::internal::file_size_bytes(part.path);
            int fid =
                w->get_or_create_file_info(logical, hash, caps, mtime, bytes);
            w->delete_chunk_statistics(fid);
            for (const auto& m : part.members) w->insert_gzip_member(fid, m);
            w->insert_file_metadata(fid, member_size, part.total_lines,
                                    part.total_uc);
        }
        w->commit();
    } catch (const std::exception& e) {
        throw DFTUtilsException(ErrorCode::IO,
                                std::string("fused index build failed for ") +
                                    opts.output_path + ": " + e.what());
    }

    // Members are committed, so the folds' finalize can open the index, resolve
    // each file, and write the bloom + hash dictionary as a second transaction.
    CoverageSet whole_file;
    for (const auto& part : parts) whole_file.add_file(part.path);
    for (auto* f : folds) co_await f->finalize(whole_file);

    ExportStats st;
    st.chunks_skipped = skipped;
    st.chunks_scanned = units.size();
    st.events_matched = matched.load(std::memory_order_relaxed);
    st.events_scanned = scanned.load(std::memory_order_relaxed);
    co_return st;
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
