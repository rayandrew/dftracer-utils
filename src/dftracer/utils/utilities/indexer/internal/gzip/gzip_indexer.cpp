#include <dftracer/utils/core/common/constants.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/coro/channel.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/core/pipeline/executor.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>
#include <dftracer/utils/utilities/indexer/error.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/index_database_writer_context.h>
#include <dftracer/utils/utilities/indexer/index_visitor.h>
#include <dftracer/utils/utilities/indexer/internal/checkpoint_size.h>
#include <dftracer/utils/utilities/indexer/internal/common/gzip_member_scanner.h>
#include <dftracer/utils/utilities/indexer/internal/gzip/gzip_indexer.h>
#include <dftracer/utils/utilities/indexer/internal/helpers.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dftracer::utils::utilities::indexer::internal::gzip {

using dftracer::utils::utilities::indexer::IndexDatabase;

namespace {

// -- Parallel path ---------------------------------------------------------
//
// When the file is multi-member gzip (the dftracer runtime format), divide
// the members across N worker coroutines and stream inflated chunks through
// per-worker channels to a single dispatcher.
//
// Workers report the absolute end of every member they finish; the
// dispatcher turns those into records with global uc_offsets and line
// numbers and appends them to the shared `members` vector in order.

struct ParallelInflateMsg {
    std::unique_ptr<std::vector<unsigned char>> data;
    // Per-worker monotonic sequence. Load-bearing: moodycamel (our channel
    // backend) does not guarantee strict FIFO without producer tokens, so
    // the dispatcher reorders by this before handing chunks to visitors.
    std::uint64_t seq = 0;
    std::uint64_t lines = 0;
    // Absolute compressed offset just past the member this chunk ended.
    bool member_ended = false;
    std::uint64_t member_c_end = 0;
};

using ParallelChan = dftracer::utils::coro::Channel<ParallelInflateMsg>;

namespace compress = dftracer::utils::utilities::fileio::compress;

// Decode the gzip member starting at compressed offset `c_off` into `out`
// (grown as needed). Returns the number of compressed bytes the member
// consumed, or nullopt on read/decode failure. Peak memory is one member.
static dftracer::utils::coro::CoroTask<std::optional<std::uint64_t>>
decode_member_at(int fd, std::uint64_t c_off, std::uint64_t file_size,
                 const compress::GzipMemberDecompressor& dec,
                 std::vector<unsigned char>& out, std::size_t& out_len) {
    constexpr std::size_t READ_CHUNK = 1u << 20;
    if (out.size() < (1u << 20)) out.resize(1u << 20);
    std::vector<unsigned char> comp;
    std::uint64_t next_read = c_off;
    compress::DecompressResult res{};
    while (true) {
        if (!comp.empty()) {
            const compress::GzipDecode status = dec.decompress_status(
                comp.data(), comp.size(), out.data(), out.size(), res);
            if (status == compress::GzipDecode::Ok) break;
            if (status == compress::GzipDecode::InsufficientSpace) {
                out.resize(out.size() * 2);
                continue;
            }
            // BadData may just mean the member is not fully buffered yet.
        }
        if (next_read >= file_size) co_return std::nullopt;
        const std::size_t want = static_cast<std::size_t>(
            std::min<std::uint64_t>(READ_CHUNK, file_size - next_read));
        const std::size_t old = comp.size();
        comp.resize(old + want);
        const ssize_t n = co_await dftracer::utils::io::pread(
            fd, comp.data() + old, want, static_cast<off_t>(next_read));
        if (n <= 0) co_return std::nullopt;
        comp.resize(old + static_cast<std::size_t>(n));
        next_read += static_cast<std::uint64_t>(n);
    }
    out_len = res.out_bytes;
    co_return res.in_bytes;
}

static std::uint64_t count_newlines(const unsigned char* p, std::size_t n) {
    std::uint64_t lines = 0;
    for (std::size_t i = 0; i < n; ++i)
        if (p[i] == '\n') ++lines;
    return lines;
}

// Each worker owns a member-aligned compressed range and decodes its members
// one at a time with libdeflate, emitting one message per member. strip and
// extend handle lines that straddle a slice boundary in the distributed
// (member_begin > 0) case; single-process workers use neither and rely on the
// dispatcher accumulator to reassemble lines across worker boundaries.
static dftracer::utils::coro::CoroTask<bool> parallel_worker(
    int fd, std::uint64_t range_c_start, std::uint64_t range_c_end,
    bool strip_leading_partial, bool extend_to_newline_past_end,
    dftracer::utils::coro::ChannelProducer<ParallelInflateMsg> producer) {
    auto guard = producer.guard();

    struct stat st;
    if (::fstat(fd, &st) != 0) co_return false;
    const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);

    compress::GzipMemberDecompressor dec;
    if (!dec.valid()) co_return false;

    std::vector<unsigned char> out;
    std::uint64_t c_off = range_c_start;
    std::uint64_t seq = 0;
    bool first = true;

    while (c_off < range_c_end) {
        std::size_t out_len = 0;
        auto in_bytes =
            co_await decode_member_at(fd, c_off, file_size, dec, out, out_len);
        if (!in_bytes) co_return false;

        std::size_t emit_start = 0;
        // The first member of a mid-file slice opens mid-line (the tail of a
        // line that began in the previous slice); drop up to its first \n so
        // the dispatcher sees a clean start.
        if (strip_leading_partial && first) {
            emit_start = out_len;
            for (std::size_t i = 0; i < out_len; ++i) {
                if (out[i] == '\n') {
                    emit_start = i + 1;
                    break;
                }
            }
        }
        first = false;

        const std::size_t emit_len = out_len - emit_start;
        ParallelInflateMsg msg;
        msg.seq = seq++;
        msg.lines = count_newlines(out.data() + emit_start, emit_len);
        msg.data = std::make_unique<std::vector<unsigned char>>(
            out.data() + emit_start, out.data() + emit_start + emit_len);
        msg.member_ended = true;
        msg.member_c_end = c_off + *in_bytes;
        if (!(co_await producer.send(std::move(msg)))) co_return true;

        c_off += *in_bytes;
    }

    // Non-last slice: capture the line straddling this slice's end by decoding
    // the next slice's first member and emitting up to and including its first
    // \n. No member record for it (member_ended stays false).
    if (extend_to_newline_past_end && range_c_end < file_size) {
        std::size_t out_len = 0;
        auto in_bytes = co_await decode_member_at(fd, range_c_end, file_size,
                                                  dec, out, out_len);
        if (!in_bytes) co_return false;
        std::size_t take = out_len;
        for (std::size_t i = 0; i < out_len; ++i) {
            if (out[i] == '\n') {
                take = i + 1;
                break;
            }
        }
        ParallelInflateMsg msg;
        msg.seq = seq++;
        msg.lines = count_newlines(out.data(), take);
        msg.data = std::make_unique<std::vector<unsigned char>>(
            out.data(), out.data() + take);
        msg.member_ended = false;
        co_await producer.send(std::move(msg));
    }

    co_return true;
}
static dftracer::utils::coro::CoroTask<bool> parallel_dispatcher(
    const std::vector<std::shared_ptr<ParallelChan>>& chans,
    std::uint64_t member_idx_base, std::uint64_t member_c_base,
    std::uint64_t& total_lines, std::uint64_t& total_uc_size,
    std::vector<GzipMemberRecord>& members,
    const Indexer::VisitorList& visitors) {
    const bool has_visitors = !visitors.empty();
    std::uint64_t global_uc = 0;

    std::uint64_t member_idx = member_idx_base;
    std::uint64_t member_c_start = member_c_base;
    std::uint64_t member_uc_start = 0;
    std::uint64_t member_first_line = total_lines + 1;

    std::uint64_t total_chunks_received = 0;

    auto process_msg =
        [&](ParallelInflateMsg& msg) -> dftracer::utils::coro::CoroTask<void> {
        const std::size_t data_len = msg.data ? msg.data->size() : 0;
        ++total_chunks_received;

        if (has_visitors && data_len > 0) {
            const char* data = reinterpret_cast<const char*>(msg.data->data());
            for (auto& v : visitors) {
                co_await v.get().on_chunk(data, data_len, member_idx);
            }
        }

        global_uc += data_len;
        total_lines += msg.lines;

        if (msg.member_ended) {
            members.push_back(GzipMemberRecord{
                .member_idx = member_idx,
                .c_offset = member_c_start,
                .c_size = msg.member_c_end - member_c_start,
                .uc_offset = member_uc_start,
                .uc_size = global_uc - member_uc_start,
                .first_line_num = member_first_line,
                .last_line_num = total_lines,
            });
            member_c_start = msg.member_c_end;
            member_uc_start = global_uc;
            member_first_line = total_lines + 1;

            if (has_visitors) {
                for (auto& v : visitors) {
                    co_await v.get().on_checkpoint(member_idx);
                }
            }
            ++member_idx;
        }

        co_return;
    };

    // Per-worker reorder buffer: moodycamel::ConcurrentQueue (backing our
    // coro::Channel) does not guarantee strict FIFO without explicit
    // producer tokens, so we re-sort by msg.seq here. Channel capacity is
    // bounded so the buffer is also bounded (~channel capacity entries).
    auto drain_visitors = [&]() -> dftracer::utils::coro::CoroTask<void> {
        for (auto& v : visitors) {
            if (v.get().wants_drain()) {
                co_await v.get().drain_pending();
            }
        }
    };

    for (auto& chan : chans) {
        std::uint64_t expected_seq = 0;
        std::map<std::uint64_t, ParallelInflateMsg> pending;
        while (auto msg_opt = co_await chan->receive()) {
            auto& incoming = *msg_opt;
            if (incoming.seq == expected_seq) {
                co_await process_msg(incoming);
                co_await drain_visitors();
                ++expected_seq;
                auto it = pending.find(expected_seq);
                while (it != pending.end()) {
                    co_await process_msg(it->second);
                    co_await drain_visitors();
                    pending.erase(it);
                    ++expected_seq;
                    it = pending.find(expected_seq);
                }
            } else {
                pending.emplace(incoming.seq, std::move(incoming));
            }
        }
        while (!pending.empty()) {
            auto it = pending.begin();
            if (it->first != expected_seq) break;
            co_await process_msg(it->second);
            co_await drain_visitors();
            pending.erase(it);
            ++expected_seq;
        }
    }

    if (has_visitors) {
        for (auto& v : visitors) co_await v.get().flush();
    }
    total_uc_size = global_uc;
    co_return true;
}

static dftracer::utils::coro::CoroTask<bool> process_chunks_parallel(
    CoroScope* scope, int fd, std::uint64_t slice_c_end,
    std::uint64_t file_size, std::vector<GzipMember> scanned_members,
    std::uint64_t member_idx_base, bool strip_slice_leading_partial,
    std::uint64_t& total_lines, std::uint64_t& total_uc_size,
    std::vector<GzipMemberRecord>& members,
    const Indexer::VisitorList& visitors) {
    // Cap worker count at member count, a reasonable default, and the actual
    // parallelism available. Fanning out past the thread count only adds
    // channel/scheduling overhead; on a single-threaded drive (RunLoop, the
    // no-executor .get() path) 16 workers ping-ponging bounded channels
    // cooperatively on one thread is pathologically slow, so collapse to one.
    constexpr std::size_t DEFAULT_MAX_WORKERS = 16;
    constexpr std::size_t CHAN_CAP = 4;
    const std::size_t hw = std::max<std::size_t>(1, available_parallelism());
    const std::size_t num_workers = std::min<std::size_t>(
        {DEFAULT_MAX_WORKERS, scanned_members.size(), hw});
    const std::uint64_t member_c_base = scanned_members.front().c_offset;

    std::vector<std::shared_ptr<ParallelChan>> chans;
    chans.reserve(num_workers);
    for (std::size_t i = 0; i < num_workers; ++i) {
        chans.push_back(
            dftracer::utils::coro::make_channel<ParallelInflateMsg>(CHAN_CAP));
    }

    // Partition members contiguously, remainder spread over the first few
    // workers so range counts differ by at most 1.
    std::vector<std::pair<std::size_t, std::size_t>> ranges(num_workers);
    {
        const std::size_t per = scanned_members.size() / num_workers;
        const std::size_t rem = scanned_members.size() % num_workers;
        std::size_t cursor = 0;
        for (std::size_t w = 0; w < num_workers; ++w) {
            const std::size_t count = per + (w < rem ? 1 : 0);
            ranges[w] = {cursor, cursor + count};
            cursor += count;
        }
    }

    bool dispatcher_ok = true;
    std::shared_ptr<std::vector<GzipMember>> members_shared =
        std::make_shared<std::vector<GzipMember>>(std::move(scanned_members));

    co_await scope->scope(
        [&](CoroScope& child) -> dftracer::utils::coro::CoroTask<void> {
            for (std::size_t w = 0; w < num_workers; ++w) {
                const auto [rs, re] = ranges[w];
                const std::uint64_t c_start = (*members_shared)[rs].c_offset;
                const std::uint64_t c_end = (re < members_shared->size())
                                                ? (*members_shared)[re].c_offset
                                                : slice_c_end;
                auto producer = chans[w]->producer();
                // Only the very first worker of a mid-file slice needs
                // to strip the leading partial line; subsequent workers
                // see contiguous (whole-line-aligned) data from their
                // predecessor's stream.
                const bool strip_this = strip_slice_leading_partial && (w == 0);
                // The LAST worker of a NON-LAST slice extends past
                // `slice_c_end` to capture the line that straddles the
                // slice boundary. If this slice's end is file end, the
                // slice IS the last one -- no extension needed.
                const bool extend_this =
                    (w + 1 == num_workers) && (slice_c_end < file_size);
                child.spawn([fd, c_start, c_end, strip_this, extend_this,
                             producer = std::move(producer)](CoroScope&) mutable
                                -> dftracer::utils::coro::CoroTask<void> {
                    co_await parallel_worker(fd, c_start, c_end, strip_this,
                                             extend_this, std::move(producer));
                });
            }

            child.spawn(
                [&chans, member_idx_base, member_c_base, &total_lines,
                 &total_uc_size, &members, &visitors, &dispatcher_ok](
                    CoroScope&) -> dftracer::utils::coro::CoroTask<void> {
                    dispatcher_ok = co_await parallel_dispatcher(
                        chans, member_idx_base, member_c_base, total_lines,
                        total_uc_size, members, visitors);
                });

            co_return;
        });

    co_return dispatcher_ok;
}

static dftracer::utils::coro::CoroTask<bool> process_chunks(
    CoroScope* scope, int fd, const GzipMemberSlice* slice,
    std::uint64_t& total_lines, std::uint64_t& total_uc_size,
    std::vector<GzipMemberRecord>& members,
    const Indexer::VisitorList& visitors) {
    struct stat st;
    if (::fstat(fd, &st) != 0) co_return false;
    const std::uint64_t file_size = static_cast<std::uint64_t>(st.st_size);

    // Pre-scanned slice path: caller supplied the member map and a range.
    // Used by the MPI/distributed indexer to split one file across ranks
    // without re-scanning. Member indices start at `member_begin` so
    // multiple slices of the same file_id produce disjoint SST entries.
    if (slice != nullptr && slice->members != nullptr &&
        slice->member_end > slice->member_begin) {
        const auto& all = *slice->members;
        const std::size_t mb = slice->member_begin;
        const std::size_t me = slice->member_end;
        if (me > all.size() || mb >= me) co_return false;
        // Slice end: next-member offset if this isn't the last slice of
        // the file, else EOF. Crucial: a non-last slice's workers must
        // not inflate past this boundary into another slice's bytes.
        const std::uint64_t slice_c_end =
            (me < all.size()) ? all[me].c_offset : file_size;
        std::vector<GzipMember> sliced(all.begin() + mb, all.begin() + me);
        co_return co_await process_chunks_parallel(
            scope, fd, slice_c_end, file_size, std::move(sliced),
            /*member_idx_base=*/mb, /*strip_slice_leading_partial=*/mb > 0,
            total_lines, total_uc_size, members, visitors);
    }

    // Discover member boundaries so the inflate pass can fan out. The scan
    // is zero-copy, sequential, and fast relative to inflate. A file whose
    // members cannot be enumerated is processed as one member spanning the
    // whole file, which the same path handles with a single worker.
    std::vector<GzipMember> scanned;
    if (file_size >= 18) {
        co_await enumerate_gzip_member_candidates(fd, file_size, scanned);
    }
    if (scanned.empty()) scanned.push_back(GzipMember{0, file_size});

    co_return co_await process_chunks_parallel(
        scope, fd, file_size, file_size, std::move(scanned),
        /*member_idx_base=*/0, /*strip_slice_leading_partial=*/false,
        total_lines, total_uc_size, members, visitors);
}

}  // namespace

dftracer::utils::coro::CoroTask<std::optional<GzipBuildArtifacts>>
build_gzip_index_artifacts(const std::string& gz_path, std::uint64_t ckpt_size,
                           const Indexer::VisitorList& visitors,
                           CoroScope* scope, const GzipMemberSlice* slice) {
    int fd = ::open(gz_path.c_str(), O_RDONLY);
    if (fd < 0) {
        co_return std::nullopt;
    }

    if (!visitors.empty()) {
        const std::uint64_t compressed_bytes = file_size_bytes(gz_path);
        const std::size_t estimated = static_cast<std::size_t>(
            compressed_bytes / (ckpt_size > 0 ? ckpt_size : 1));
        for (auto& visitor : visitors) {
            visitor.get().begin(estimated);
        }
    }

    std::uint64_t total_lines = 0;
    std::uint64_t total_uc_size = 0;
    std::vector<GzipMemberRecord> members;

    const bool success = co_await process_chunks(
        scope, fd, slice, total_lines, total_uc_size, members, visitors);
    ::close(fd);

    if (!success) {
        co_return std::nullopt;
    }

    // A non-gzip input (e.g. an uncompressed .pfw) scans to zero members, which
    // would otherwise silently produce an empty index. dftracer emits .pfw.gz.
    if (slice == nullptr && members.empty() && file_size_bytes(gz_path) > 0) {
        DFTRACER_UTILS_LOG_WARN(
            "Indexer: no gzip members found in %s; it is not a gzip stream, so "
            "the index will be empty. dftracer emits .pfw.gz - gzip the trace.",
            gz_path.c_str());
    }

    GzipBuildArtifacts artifacts;
    artifacts.checkpoint_size = ckpt_size;
    artifacts.total_lines = total_lines;
    artifacts.total_uc_size = total_uc_size;
    artifacts.members = std::move(members);
    co_return artifacts;
}

void persist_gzip_index_artifacts(IndexDatabaseWriterContext& db, int file_id,
                                  const GzipBuildArtifacts& artifacts) {
    for (const auto& member : artifacts.members) {
        db.insert_gzip_member(file_id, member);
    }
    db.insert_file_metadata(file_id, artifacts.checkpoint_size,
                            artifacts.total_lines, artifacts.total_uc_size);
}

GzipIndexer::GzipIndexer(const std::string& gz_path_,
                         const std::string& idx_path_, std::uint64_t ckpt_size_,
                         bool force_rebuild_)
    : gz_path(gz_path_),
      gz_path_logical_path(get_logical_path(gz_path_)),
      index_path(normalize_index_root(idx_path_)),
      ckpt_size(ckpt_size_),
      force_rebuild(force_rebuild_),
      cached_is_valid(false),
      cached_file_id(-1),
      cached_max_bytes(0),
      cached_max_bytes_ready(false),
      cached_num_lines(0),
      cached_num_lines_ready(false),
      cached_checkpoint_size(0),
      cached_checkpoint_size_ready(false) {
    if (gz_path.empty()) {
        throw IndexerError(IndexerError::Type::INVALID_ARGUMENT,
                           "gz_path must not be empty");
    }

    if (!fs::exists(gz_path)) {
        throw IndexerError(IndexerError::Type::FILE_ERROR,
                           "gz_path does not exist: " + gz_path);
    }

    if (ckpt_size == 0) {
        throw IndexerError(IndexerError::Type::INVALID_ARGUMENT,
                           "ckpt_size must be greater than 0");
    }

    open();
}

GzipIndexer::~GzipIndexer() {
    DFTRACER_UTILS_LOG_DEBUG("Destroying GZIP indexer for %s", gz_path.c_str());
    close();
}

GzipIndexer::GzipIndexer(GzipIndexer&& other) noexcept
    : gz_path(std::move(other.gz_path)),
      gz_path_logical_path(std::move(other.gz_path_logical_path)),
      index_path(std::move(other.index_path)),
      ckpt_size(other.ckpt_size),
      force_rebuild(other.force_rebuild),
      visitors_(std::move(other.visitors_)),
      cached_is_valid(other.cached_is_valid.load()),
      cached_file_id(other.cached_file_id.load()),
      cached_max_bytes(other.cached_max_bytes.load()),
      cached_max_bytes_ready(other.cached_max_bytes_ready.load()),
      cached_num_lines(other.cached_num_lines.load()),
      cached_num_lines_ready(other.cached_num_lines_ready.load()),
      cached_checkpoint_size(other.cached_checkpoint_size.load()),
      cached_checkpoint_size_ready(other.cached_checkpoint_size_ready.load()),
      cached_members(std::move(other.cached_members)),
      cached_loaded(other.cached_loaded.load()) {}

GzipIndexer& GzipIndexer::operator=(GzipIndexer&& other) noexcept {
    if (this != &other) {
        gz_path = std::move(other.gz_path);
        gz_path_logical_path = std::move(other.gz_path_logical_path);
        index_path = std::move(other.index_path);
        ckpt_size = other.ckpt_size;
        force_rebuild = other.force_rebuild;
        visitors_ = std::move(other.visitors_);
        cached_is_valid.store(other.cached_is_valid.load());
        cached_file_id.store(other.cached_file_id.load());
        cached_max_bytes.store(other.cached_max_bytes.load());
        cached_max_bytes_ready.store(other.cached_max_bytes_ready.load());
        cached_num_lines.store(other.cached_num_lines.load());
        cached_num_lines_ready.store(other.cached_num_lines_ready.load());
        cached_checkpoint_size.store(other.cached_checkpoint_size.load());
        cached_checkpoint_size_ready.store(
            other.cached_checkpoint_size_ready.load());
        cached_loaded.store(other.cached_loaded.load());
        std::lock_guard<std::mutex> lock(cached_members_mutex);
        cached_members = std::move(other.cached_members);
    }
    return *this;
}

void GzipIndexer::open() {}

void GzipIndexer::close() {}

dftracer::utils::coro::CoroTask<void> GzipIndexer::build_async() const {
    if (!force_rebuild && !need_rebuild()) {
        co_return;
    }

    IndexDatabase db(index_path);
    const std::time_t mtime = get_file_modification_time(gz_path);
    const auto hash = calculate_file_hash(gz_path);
    const std::uint64_t bytes = file_size_bytes(gz_path);
    const std::uint64_t final_ckpt_size =
        determine_checkpoint_size(ckpt_size, gz_path);
    const std::string logical = gz_path_logical_path;
    auto writer = db.begin_write();
    const int file_id = writer->get_or_create_file_info(
        logical, hash, IndexFileEntryCapability::NONE,
        static_cast<std::uint64_t>(mtime), bytes);
    writer->commit();

    std::optional<GzipBuildArtifacts> artifacts;
    co_await run_coro_scope(
        [&](CoroScope& scope) -> dftracer::utils::coro::CoroTask<void> {
            artifacts = co_await build_gzip_index_artifacts(
                gz_path, final_ckpt_size, visitors_, &scope);
        });
    if (!artifacts) {
        throw IndexerError(IndexerError::Type::BUILD_ERROR,
                           "Failed to build index for " + gz_path);
    }

    {
        auto w = db.begin_write();
        persist_gzip_index_artifacts(*w, file_id, *artifacts);
        w->commit();
    }

    (void)mtime;
    (void)bytes;

    cached_is_valid = true;
    cached_file_id = file_id;
    cached_checkpoint_size = final_ckpt_size;
    cached_checkpoint_size_ready = true;
    cached_num_lines = db.get_num_lines(file_id);
    cached_num_lines_ready = true;
    cached_max_bytes = db.get_max_bytes(file_id);
    cached_max_bytes_ready = true;
    std::lock_guard<std::mutex> lock(cached_members_mutex);
    cached_members = db.query_gzip_members(file_id);
    cached_loaded.store(true, std::memory_order_release);
    co_return;
}

bool GzipIndexer::is_valid() const { return cached_is_valid; }

bool GzipIndexer::exists() const {
    return fs::exists(index_path) && fs::is_directory(index_path);
}

bool GzipIndexer::need_rebuild() const {
    if (is_valid()) {
        return false;
    }
    if (!exists()) {
        return true;
    }

    try {
        IndexDatabase db(
            index_path,
            dftracer::utils::utilities::indexer::IndexOpenMode::ReadOnly);
        const auto stored_hash = db.get_file_hash(gz_path_logical_path);
        const int file_id = db.get_file_info_id(gz_path_logical_path);
        if (!stored_hash || file_id < 0) {
            return true;
        }

        const auto current_hash = calculate_file_hash(gz_path);
        const auto current_ckpt_size = db.get_checkpoint_size(file_id);
        return current_hash != *stored_hash || current_ckpt_size == 0;
    } catch (...) {
        return true;
    }
}

const std::string& GzipIndexer::get_index_path() const { return index_path; }

const std::string& GzipIndexer::get_archive_path() const { return gz_path; }

const std::string& GzipIndexer::get_gz_path() const { return gz_path; }

void GzipIndexer::ensure_loaded() const {
    if (cached_loaded.load(std::memory_order_acquire)) {
        return;
    }
    std::lock_guard<std::mutex> lock(cached_members_mutex);
    if (cached_loaded.load(std::memory_order_relaxed)) {
        return;
    }
    IndexDatabase db(
        index_path,
        dftracer::utils::utilities::indexer::IndexOpenMode::ReadOnly);
    const int file_id = db.get_file_info_id(gz_path_logical_path);
    cached_file_id.store(file_id, std::memory_order_relaxed);
    if (file_id != -1) {
        cached_num_lines.store(db.get_num_lines(file_id),
                               std::memory_order_relaxed);
        cached_num_lines_ready.store(true, std::memory_order_relaxed);
        cached_max_bytes.store(db.get_max_bytes(file_id),
                               std::memory_order_relaxed);
        cached_max_bytes_ready.store(true, std::memory_order_relaxed);
        cached_checkpoint_size.store(db.get_checkpoint_size(file_id),
                                     std::memory_order_relaxed);
        cached_checkpoint_size_ready.store(true, std::memory_order_relaxed);
        cached_members = db.query_gzip_members(file_id);
    }
    cached_loaded.store(true, std::memory_order_release);
}

std::uint64_t GzipIndexer::get_max_bytes() const {
    ensure_loaded();
    return cached_max_bytes.load(std::memory_order_relaxed);
}

std::uint64_t GzipIndexer::get_checkpoint_size() const {
    ensure_loaded();
    return cached_checkpoint_size.load(std::memory_order_relaxed);
}

std::uint64_t GzipIndexer::get_num_lines() const {
    ensure_loaded();
    return cached_num_lines.load(std::memory_order_relaxed);
}

int GzipIndexer::get_file_id() const {
    ensure_loaded();
    return cached_file_id.load(std::memory_order_relaxed);
}

int GzipIndexer::find_file_id(const std::string& path) const {
    IndexDatabase db(
        index_path,
        dftracer::utils::utilities::indexer::IndexOpenMode::ReadOnly);
    return db.get_file_info_id(get_logical_path(path));
}

bool GzipIndexer::find_member(std::size_t target_offset,
                              GzipMemberRecord& member) const {
    ensure_loaded();
    std::lock_guard<std::mutex> lock(cached_members_mutex);
    for (const auto& candidate : cached_members) {
        if (target_offset < candidate.uc_offset + candidate.uc_size) {
            member = candidate;
            return true;
        }
    }
    return false;
}

std::vector<GzipMemberRecord> GzipIndexer::get_members() const {
    ensure_loaded();
    std::lock_guard<std::mutex> lock(cached_members_mutex);
    return cached_members;
}

}  // namespace dftracer::utils::utilities::indexer::internal::gzip
