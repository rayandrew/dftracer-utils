#include <dftracer/utils/core/common/byte_view.h>
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/core/coro/yield.h>
#include <dftracer/utils/core/io/io.h>
#include <dftracer/utils/utilities/fileio/chunk_writer.h>
#include <fcntl.h>

namespace dftracer::utils::utilities::fileio {

ChunkWriter::ChunkWriter(ChunkWriterConfig config)
    : config_(std::move(config)) {
    member_buffer_.reserve(WRITE_BUFFER_SIZE);
    if (config_.compress) {
        compressor_.emplace(config_.compression_level);
    }
}

ChunkWriter::~ChunkWriter() {
    if (open_) {
        DFTRACER_UTILS_LOG_WARN("ChunkWriter destroyed while open: %s",
                                chunk_path(chunk_index_).c_str());
    }
}

std::string ChunkWriter::chunk_path(int index) const {
    std::string name = config_.base_name + "_chunk" + std::to_string(index) +
                       ".pfw" + (config_.compress ? ".gz" : "");
    return config_.output_dir + "/" + name;
}

coro::CoroTask<void> ChunkWriter::open() {
    if (!fs::exists(config_.output_dir)) {
        fs::create_directories(config_.output_dir);
    }
    co_await open_next_chunk();
}

coro::CoroTask<void> ChunkWriter::open_next_chunk() {
    std::string path = chunk_path(chunk_index_);

    ssize_t result =
        co_await io::open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (result < 0) {
        throw DFTUtilsException(ErrorCode::IO,
                                "Cannot open chunk file: " + path);
    }
    fd_ = static_cast<int>(result);
    open_ = true;
    current_chunk_bytes_ = 0;
    current_chunk_events_ = 0;
    current_member_bytes_ = 0;
    member_buffer_.clear();
    io_buffer_.clear();

    if (config_.json_array_wrapper) {
        append_member("[\n", 2);
    }
}

void ChunkWriter::append_member(const char* data, std::size_t len) {
    member_buffer_.insert(member_buffer_.end(), data, data + len);
}

coro::CoroTask<void> ChunkWriter::write_line(ByteView line) {
    if (current_chunk_events_ > 0) {
        member_buffer_.push_back('\n');
    }
    append_member(line.as<char>(), line.size());
    current_chunk_bytes_ += line.size() + 1;
    current_member_bytes_ += line.size() + 1;
    current_chunk_events_++;
    total_events_++;

    if (config_.compress) {
        // Close the gzip member at this line boundary once it is large enough,
        // so the chunk is multi-member (parallel-inflatable) and peak memory
        // stays at one member.
        if (config_.member_size_bytes > 0 &&
            current_member_bytes_ >= config_.member_size_bytes) {
            co_await flush_member();
        }
    } else if (member_buffer_.size() >= WRITE_BUFFER_SIZE) {
        co_await flush_member();
    }

    if (current_chunk_bytes_ >= config_.chunk_size_bytes) {
        co_await finalize_current_chunk();
        chunk_index_++;
        co_await open_next_chunk();
    }

    // Yield every 256 events to prevent stack overflow from synchronous
    // coroutine completion chains. This is internal to ChunkWriter so
    // callers don't need to manage yielding.
    if ((total_events_ & 0xff) == 0) {
        co_await coro::yield();
    }
}

coro::CoroTask<void> ChunkWriter::write_bytes(ByteView data) {
    append_member(data.as<char>(), data.size());
    current_chunk_bytes_ += data.size();
    current_member_bytes_ += data.size();

    if (!config_.compress && member_buffer_.size() >= WRITE_BUFFER_SIZE) {
        co_await flush_member();
    }
}

coro::CoroTask<void> ChunkWriter::flush_member() {
    if (member_buffer_.empty()) co_return;

    if (compressor_) {
        if (!compressor_->compress_member_into(compressed_scratch_,
                                               member_buffer_.data(),
                                               member_buffer_.size())) {
            throw DFTUtilsException(ErrorCode::COMPRESSION,
                                    "gzip member compression failed");
        }
        co_await write_out(
            reinterpret_cast<const char*>(compressed_scratch_.data()),
            compressed_scratch_.size());
    } else {
        co_await write_out(member_buffer_.data(), member_buffer_.size());
    }

    member_buffer_.clear();
    current_member_bytes_ = 0;
}

coro::CoroTask<void> ChunkWriter::write_out(const char* data,
                                            std::size_t size) {
    if (size == 0) co_return;
    io_buffer_.insert(io_buffer_.end(), data, data + size);
    total_bytes_ += size;
    if (io_buffer_.size() >= config_.io_flush_bytes) {
        co_await io::write(fd_, io_buffer_.data(), io_buffer_.size());
        io_buffer_.clear();
    }
}

coro::CoroTask<void> ChunkWriter::flush_io() {
    if (!io_buffer_.empty()) {
        co_await io::write(fd_, io_buffer_.data(), io_buffer_.size());
        io_buffer_.clear();
    }
}

coro::CoroTask<void> ChunkWriter::finalize_current_chunk() {
    if (config_.json_array_wrapper) {
        if (current_chunk_events_ > 0) {
            append_member("\n]\n", 3);
        } else {
            append_member("]\n", 2);
        }
    }

    co_await flush_member();
    co_await flush_io();
    co_await io::close(fd_);
    fd_ = -1;

    auto path = chunk_path(chunk_index_);
    chunks_.push_back(ChunkInfo{
        .path = path,
        .bytes_written = current_chunk_bytes_,
        .events_written = current_chunk_events_,
        .chunk_index = chunk_index_,
    });

    if (config_.on_chunk_complete) {
        config_.on_chunk_complete(static_cast<std::size_t>(chunk_index_), path,
                                  current_chunk_events_, current_chunk_bytes_);
    }
}

coro::CoroTask<void> ChunkWriter::close() {
    if (!open_) co_return;
    co_await finalize_current_chunk();
    open_ = false;
}

}  // namespace dftracer::utils::utilities::fileio
