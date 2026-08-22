#ifndef DFTRACER_UTILS_PLUGINS_PLUGIN_EXPORTS_H
#define DFTRACER_UTILS_PLUGINS_PLUGIN_EXPORTS_H

// Single source exposing every host utility to the codegen and the registry;
// both gen_plugin_abi and plugin_exports.cpp include it. Names the native I/O
// types (for reflection) and the stateless scalar functors.

#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/core/common/hash/hex64.h>
#include <dftracer/utils/core/coro/async_generator.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/plugins/reflect.h>
#include <dftracer/utils/trace/event_collector_utility.h>
#include <dftracer/utils/trace/event_id.h>
#include <dftracer/utils/trace/metadata_collector_utility.h>
#include <dftracer/utils/trace/statistics/trace_statistics.h>
#include <dftracer/utils/trace/views/view_scanner_utility.h>
#include <dftracer/utils/utilities/fileio/file_compressor_utility.h>
#include <dftracer/utils/utilities/fileio/file_decompressor_utility.h>
#include <dftracer/utils/utilities/fileio/file_reader_utility.h>
#include <dftracer/utils/utilities/filesystem/directory_scanner_utility.h>
#include <dftracer/utils/utilities/filesystem/pattern_directory_scanner_utility.h>
#include <dftracer/utils/utilities/text/line_filter.h>

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

// Stateless scalar functors exported as utilities; each crosses the ABI via its
// scalar IN/OUT rather than a reflected struct.
namespace dftracer::utils::plugins::scalar {

struct Fnv1a {
    std::uint64_t operator()(std::string_view s) const {
        return hash::fnv1a_hash(s);
    }
};

struct Hex64Parse {
    std::optional<std::uint64_t> operator()(std::string_view s) const {
        return hash::parse_hex64(s);
    }
};

struct Hex64Format {
    dftu_hex16 operator()(std::uint64_t v) const {
        dftu_hex16 o{};
        hash::format_hex64(v, o.c);
        return o;
    }
};

}  // namespace dftracer::utils::plugins::scalar

// EventId is non-aggregate but has public fields; reflect it here rather than
// in its library header.
DFTU_REFLECT(::dftracer::utils::trace::EventId, id, pid, tid);

// Callback + data: predicate crosses as a C function pointer, line as data.
DFTU_REFLECT(::dftracer::utils::utilities::text::FilterableLine, line,
             predicate);
DFTU_REFLECT(::dftracer::utils::utilities::fileio::lines::Line, content,
             line_number);

// Only the data path fields are marshalled; view/query/fold_intern keep their
// defaults host-side (scan every event, no context).
DFTU_REFLECT(::dftracer::utils::trace::views::ViewScannerInput, file_path,
             index_path, checkpoint_size, event_batch_size, start_byte,
             end_byte);
DFTU_REFLECT(::dftracer::utils::trace::views::ViewScannerBatch, events_matched,
             events_scanned);

// Sketch projection: duration_sketch (DDSketch) crosses as dftu_quantiles.
DFTU_REFLECT(::dftracer::utils::trace::statistics::TraceStatistics, success,
             num_chunks, merged);
DFTU_REFLECT(::dftracer::utils::trace::indexing::ChunkStatistics, total_events,
             duration_count, duration_min_us, duration_max_us, duration_sketch);

// File-metadata table row; the full 18-field struct is reflected.
DFTU_REFLECT(::dftracer::utils::trace::MetadataCollectorUtilityInput, file_path,
             index_path, checkpoint_size, force_rebuild, compute_hash);
DFTU_REFLECT(::dftracer::utils::trace::MetadataCollectorUtilityOutput,
             file_path, index_path, size_mb, start_line, end_line, valid_events,
             size_per_line, success, has_index, index_valid, compressed_size,
             uncompressed_size, num_lines, checkpoint_size, num_checkpoints,
             format, error_message, event_hash);

// fs::path crosses as its string form (dftu_bytes) via the is_bytes_like route.
DFTU_REFLECT(::dftracer::utils::utilities::filesystem::FileEntry, path, size,
             mtime, is_directory, is_regular_file);
DFTU_REFLECT(::dftracer::utils::utilities::text::Text, content);
DFTU_REFLECT(
    ::dftracer::utils::utilities::filesystem::DirectoryScannerUtilityInput,
    path, recursive, populate_size);
DFTU_REFLECT(::dftracer::utils::utilities::filesystem::
                 PatternDirectoryScannerUtilityInput,
             path, recursive, populate_size, patterns);

DFTU_REFLECT(
    ::dftracer::utils::utilities::fileio::FileDecompressionUtilityOutput,
    input_path, output_path, compressed_size, decompressed_size);

#endif  // DFTRACER_UTILS_PLUGINS_PLUGIN_EXPORTS_H
