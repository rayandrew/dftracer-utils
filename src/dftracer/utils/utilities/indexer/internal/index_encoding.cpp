#include <dftracer/utils/core/rocksdb/key_codec.h>
#include <dftracer/utils/utilities/indexer/internal/index_encoding.h>
#include <dftracer/utils/utilities/indexer/internal/payload_codec.h>

#include <algorithm>
#include <vector>

namespace dftracer::utils::utilities::indexer::internal::encoding {

namespace {
namespace rocks = dftracer::utils::rocksdb;
}  // namespace

std::string prefix_for_file(int file_id) {
    return rocks::KeyCodec::encode_be32(static_cast<std::uint32_t>(file_id));
}

std::string metadata_key(int file_id) { return prefix_for_file(file_id); }

std::string gzip_member_prefix(int file_id) {
    std::string key("m|");
    key.reserve(2 + sizeof(std::uint32_t));
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    return key;
}

std::string gzip_member_key(int file_id, std::uint64_t member_idx) {
    std::string key = gzip_member_prefix(file_id);
    key.reserve(2 + sizeof(std::uint32_t) + sizeof(std::uint64_t));
    append_u64(key, member_idx);
    return key;
}

std::string encode_gzip_member_value(const GzipMemberRecord& member) {
    std::string value;
    append_u64(value, member.c_offset);
    append_u64(value, member.c_size);
    append_u64(value, member.uc_offset);
    append_u64(value, member.uc_size);
    append_u64(value, member.first_line_num);
    append_u64(value, member.last_line_num);
    return value;
}

std::string encode_metadata_record(std::uint64_t checkpoint_size,
                                   std::uint64_t total_lines,
                                   std::uint64_t total_uc_size) {
    std::string value;
    append_u64(value, checkpoint_size);
    append_u64(value, total_lines);
    append_u64(value, total_uc_size);
    return value;
}

std::string make_dimension_key(int file_id, std::string_view dimension) {
    std::string key("d|");
    key.reserve(2 + sizeof(std::uint32_t) + dimension.size());
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    key.append(dimension);
    return key;
}

std::string make_column_key(int file_id, std::string_view column) {
    std::string key("c|");
    key.reserve(2 + sizeof(std::uint32_t) + column.size());
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    key.append(column);
    return key;
}

std::string chunk_bloom_key(int file_id, std::string_view dimension,
                            std::uint64_t checkpoint_idx) {
    std::string key = prefix_for_file(file_id);
    key.reserve(sizeof(std::uint32_t) + dimension.size() + 1 +
                sizeof(std::uint64_t));
    key.append(dimension);
    key.push_back('\0');
    append_u64(key, checkpoint_idx);
    return key;
}

std::string file_bloom_key(int file_id, std::string_view dimension) {
    std::string key = prefix_for_file(file_id);
    key.reserve(sizeof(std::uint32_t) + dimension.size());
    key.append(dimension);
    return key;
}

std::string chunk_stats_key(int file_id, std::uint64_t checkpoint_idx) {
    std::string key = prefix_for_file(file_id);
    key.reserve(sizeof(std::uint32_t) + sizeof(std::uint64_t));
    append_u64(key, checkpoint_idx);
    return key;
}

std::string file_scalar_stats_key(int file_id) {
    return prefix_for_file(file_id);
}

std::string file_category_counts_key(int file_id) {
    return prefix_for_file(file_id);
}

std::string file_pid_tid_counts_key(int file_id) {
    return prefix_for_file(file_id);
}

std::string file_name_counts_key(int file_id) {
    return prefix_for_file(file_id);
}

std::string chunk_dim_stats_key(int file_id, std::uint64_t checkpoint_idx,
                                std::string_view dimension) {
    std::string key = prefix_for_file(file_id);
    key.reserve(sizeof(std::uint32_t) + sizeof(std::uint64_t) +
                dimension.size());
    append_u64(key, checkpoint_idx);
    key.append(dimension);
    return key;
}

std::string encode_bloom_value(std::span<const unsigned char> blob,
                               std::uint64_t num_entries) {
    std::string value;
    append_u64(value, num_entries);
    value.append(reinterpret_cast<const char*>(blob.data()), blob.size());
    return value;
}

std::string encode_chunk_statistics_value(
    const composites::dft::indexing::ChunkStatistics& stats) {
    std::string value;
    append_u64(value, stats.total_events);
    append_u64(value, stats.min_timestamp_us);
    append_u64(value, stats.max_timestamp_us);
    append_i64(value, stats.duration_sum_us);
    append_u64(value, stats.duration_min_us);
    append_u64(value, stats.duration_max_us);
    append_u64(value, stats.duration_count);
    append_double(value, stats.duration_m2);

    auto duration_sketch = stats.duration_sketch.serialize();
    append_blob(value, duration_sketch);

    auto duration_histogram = stats.duration_histogram.to_json();
    append_string(value, duration_histogram);

    auto name_sketches = stats.serialize_name_duration_sketches();
    append_blob(value, name_sketches);
    append_string(value, stats.name_duration_histograms_json());
    append_string(value, stats.name_duration_sums_json());
    append_string(value, stats.name_duration_sum_sqs_json());
    append_string(value, stats.name_category_json());

    auto ts_hist = stats.timestamp_histogram.serialize();
    append_blob(value, ts_hist);

    append_u32(value, static_cast<std::uint32_t>(stats.sub_zonemaps.size()));
    for (const auto& z : stats.sub_zonemaps) {
        append_u64(value, z.event_count);
        append_u64(value, z.min_timestamp_us);
        append_u64(value, z.max_timestamp_us);
        append_u64(value, z.min_duration_us);
        append_u64(value, z.max_duration_us);
    }

    // Per-cat and per-pid duration aggregates (tail: decoders built before this
    // stop after sub_zonemaps and leave these maps empty).
    append_blob(
        value, composites::dft::indexing::ChunkStatistics::serialize_sketch_map(
                   stats.cat_duration_sketches));
    append_string(value, stats.cat_duration_sums_json());
    append_blob(
        value, composites::dft::indexing::ChunkStatistics::serialize_sketch_map(
                   stats.pid_duration_sketches));
    append_string(value, stats.pid_duration_sums_json());

    return value;
}

std::string encode_chunk_dimension_stats_value(
    const composites::dft::indexing::ChunkDimensionStats& stats,
    std::size_t value_counts_cap) {
    std::string value;
    append_u64(value, stats.distinct_count);
    append_string(value, stats.min_value);
    append_string(value, stats.max_value);
    append_string(value, stats.value_type);
    auto compressed = stats.compress_value_counts(value_counts_cap);
    append_u8(value, compressed.has_value() ? 1 : 0);
    if (compressed) {
        append_blob(value, *compressed);
    }
    return value;
}

std::string name_lookup_key(std::string_view name) {
    std::string key("s|");
    key.reserve(2 + name.size());
    key.append(name);
    return key;
}

std::string name_reverse_key(std::uint64_t name_id) {
    std::string key("i|");
    key.reserve(2 + sizeof(std::uint64_t));
    append_u64(key, name_id);
    return key;
}

std::string name_file_posting_key(std::uint64_t name_id, int file_id) {
    std::string key("n|");
    key.reserve(2 + sizeof(std::uint64_t) + sizeof(std::uint32_t));
    append_u64(key, name_id);
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    return key;
}

std::string name_file_owner_key(int file_id, std::uint64_t name_id) {
    std::string key("o|");
    key.reserve(2 + sizeof(std::uint32_t) + sizeof(std::uint64_t));
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    append_u64(key, name_id);
    return key;
}

std::string name_file_owner_prefix(int file_id) {
    std::string key("o|");
    key.reserve(2 + sizeof(std::uint32_t));
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    return key;
}

std::string name_chunk_posting_key(std::uint64_t name_id, int file_id,
                                   std::uint64_t checkpoint_idx) {
    std::string key("n|");
    key.reserve(2 + 2 * sizeof(std::uint64_t) + sizeof(std::uint32_t));
    append_u64(key, name_id);
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    append_u64(key, checkpoint_idx);
    return key;
}

std::string name_chunk_owner_key(int file_id, std::uint64_t name_id,
                                 std::uint64_t checkpoint_idx) {
    std::string key("o|");
    key.reserve(2 + sizeof(std::uint32_t) + 2 * sizeof(std::uint64_t));
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    append_u64(key, name_id);
    append_u64(key, checkpoint_idx);
    return key;
}

std::string name_chunk_owner_prefix(int file_id) {
    std::string key("o|");
    key.reserve(2 + sizeof(std::uint32_t));
    rocks::KeyCodec::append_be32(key, static_cast<std::uint32_t>(file_id));
    return key;
}

std::string hash_table_forward_key(std::uint8_t type, std::string_view hash) {
    std::string key;
    key.reserve(1 + hash.size());
    key.push_back(static_cast<char>(type));
    key.append(hash);
    return key;
}

std::string hash_table_reverse_key(std::uint8_t type, std::string_view name) {
    std::string key;
    key.reserve(1 + name.size());
    key.push_back(static_cast<char>(type + 4));
    key.append(name);
    return key;
}

}  // namespace dftracer::utils::utilities::indexer::internal::encoding
