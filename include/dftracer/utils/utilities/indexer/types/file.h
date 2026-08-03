#ifndef DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_FILE_H
#define DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_FILE_H

#include <dftracer/utils/utilities/indexer/index_file_entry_capability.h>

#include <cstdint>

namespace dftracer::utils::utilities::indexer {

struct FileMetadataResult {
    std::uint64_t checkpoint_size = 0;
    std::uint64_t num_lines = 0;
    std::uint64_t max_bytes = 0;
};

struct FileRegistryEntry {
    int file_id = -1;
    IndexFileEntryCapability capabilities = IndexFileEntryCapability::NONE;
};

}  // namespace dftracer::utils::utilities::indexer

#endif  // DFTRACER_UTILS_UTILITIES_INDEXER_TYPES_FILE_H
