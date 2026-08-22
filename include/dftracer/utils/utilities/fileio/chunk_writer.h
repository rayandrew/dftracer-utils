#ifndef DFTRACER_UTILS_UTILITIES_FILEIO_CHUNK_WRITER_H
#define DFTRACER_UTILS_UTILITIES_FILEIO_CHUNK_WRITER_H

#include <dftracer/utils/core/common/byte_view.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

namespace dftracer::utils::utilities::fileio {

struct ChunkWriterConfig {
    std::string output_dir;
    std::string base_name;
    std::size_t chunk_size_bytes = 256 * 1024 * 1024;
    /// Uncompressed bytes per gzip member within a compressed chunk. A positive
    /// value emits multi-member gzip (closed at line boundaries once the member
    /// exceeds this size) so readers can inflate/index members in parallel, and
    /// bounds peak memory to one member. 0 keeps a single member per chunk,
    /// which buffers the whole chunk (up to chunk_size_bytes) before
    /// compressing.
    std::size_t member_size_bytes = 0;
    /// Coalesce output into writes of this many bytes so I/O granularity suits
    /// a parallel filesystem (Lustre/GPFS) instead of many tiny writes.
    std::size_t io_flush_bytes = 16 * 1024 * 1024;
    bool compress = true;
    int compression_level = 6;
    bool json_array_wrapper = true;

    using ChunkRotationCallback = std::function<void(
        std::size_t chunk_index, const std::string& chunk_path,
        std::size_t event_count, std::size_t byte_count)>;
    ChunkRotationCallback on_chunk_complete;

    ChunkWriterConfig& with_output_dir(std::string dir) {
        output_dir = std::move(dir);
        return *this;
    }
    ChunkWriterConfig& with_member_size(std::size_t bytes) {
        member_size_bytes = bytes;
        return *this;
    }
    ChunkWriterConfig& with_compression(bool enabled) {
        compress = enabled;
        return *this;
    }
    ChunkWriterConfig& with_compression_level(int level) {
        compression_level = level;
        return *this;
    }
};

struct ChunkInfo {
    std::string path;
    std::size_t bytes_written = 0;
    std::size_t events_written = 0;
    int chunk_index = 0;
};

class ChunkWriter {
   public:
    explicit ChunkWriter(ChunkWriterConfig config);
    ~ChunkWriter();

    ChunkWriter(const ChunkWriter&) = delete;
    ChunkWriter& operator=(const ChunkWriter&) = delete;

    coro::CoroTask<void> open();
    coro::CoroTask<void> write_line(ByteView line);
    coro::CoroTask<void> write_bytes(ByteView data);
    coro::CoroTask<void> close();

    std::size_t total_bytes_written() const { return total_bytes_; }
    std::size_t total_events_written() const { return total_events_; }
    const std::vector<ChunkInfo>& chunks() const { return chunks_; }
    bool is_open() const { return open_; }

   private:
    /// Emits the accumulated buffer as one gzip member, or raw when compression
    /// is off.
    coro::CoroTask<void> flush_member();
    void append_member(const char* data, std::size_t len);
    /// Coalesce writes to `io_flush_bytes` granularity for the PFS.
    coro::CoroTask<void> write_out(const char* data, std::size_t size);
    coro::CoroTask<void> flush_io();
    coro::CoroTask<void> finalize_current_chunk();
    coro::CoroTask<void> open_next_chunk();
    std::string chunk_path(int index) const;

    ChunkWriterConfig config_;
    int fd_ = -1;
    bool open_ = false;
    int chunk_index_ = 0;
    std::size_t current_chunk_bytes_ = 0;
    std::size_t current_chunk_events_ = 0;
    std::size_t current_member_bytes_ = 0;
    std::size_t total_bytes_ = 0;
    std::size_t total_events_ = 0;

    static constexpr std::size_t WRITE_BUFFER_SIZE = 256 * 1024;
    /// Uncompressed bytes of the current gzip member (or the coalescing buffer
    /// for the uncompressed path).
    std::vector<char> member_buffer_;
    std::vector<char> io_buffer_;
    std::vector<std::uint8_t> compressed_scratch_;

    std::optional<compress::GzipMemberCompressor> compressor_;

    std::vector<ChunkInfo> chunks_;
};

}  // namespace dftracer::utils::utilities::fileio

#endif  // DFTRACER_UTILS_UTILITIES_FILEIO_CHUNK_WRITER_H
