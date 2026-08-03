#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/utilities/indexer/internal/indexer.h>
#include <dftracer/utils/utilities/indexer/internal/indexer_factory.h>

#include <cstring>

using namespace dftracer::utils::utilities::indexer::internal;

extern "C" {

static int validate_handle(dft_indexer_handle_t indexer) {
    return indexer ? 0 : -1;
}

static std::shared_ptr<Indexer> *cast_indexer(dft_indexer_handle_t indexer) {
    return static_cast<std::shared_ptr<Indexer> *>(indexer);
}

dft_indexer_handle_t dft_indexer_create(const char *gz_path,
                                        const char *index_path,
                                        uint64_t checkpoint_size,
                                        int force_rebuild) {
    if (!gz_path || !index_path || checkpoint_size == 0) {
        DFTRACER_UTILS_LOG_ERROR("%s",
                                 "Invalid parameters for indexer creation");
        return nullptr;
    }

    try {
        auto indexer = IndexerFactory::create(
            gz_path, index_path, checkpoint_size, force_rebuild != 0);
        if (indexer) {
            return static_cast<dft_indexer_handle_t>(
                new std::shared_ptr<Indexer>(indexer));
        }
        return nullptr;
    } catch (const std::exception &e) {
        DFTRACER_UTILS_LOG_ERROR("Failed to create DFT indexer: %s", e.what());
        return nullptr;
    }
}

int dft_indexer_build(dft_indexer_handle_t indexer) {
    if (validate_handle(indexer) < 0) {
        return -1;
    }

    try {
        (*cast_indexer(indexer))->build();
        return 0;
    } catch (const std::exception &e) {
        DFTRACER_UTILS_LOG_ERROR("Failed to build index: %s", e.what());
        return -1;
    }
}

int dft_indexer_need_rebuild(dft_indexer_handle_t indexer) {
    if (validate_handle(indexer)) {
        return -1;
    }

    try {
        return (*cast_indexer(indexer))->need_rebuild() ? 1 : 0;
    } catch (const std::exception &e) {
        DFTRACER_UTILS_LOG_ERROR("Failed to check if rebuild is needed: %s",
                                 e.what());
        return -1;
    }
}

int dft_indexer_exists(dft_indexer_handle_t indexer) {
    if (validate_handle(indexer) < 0) {
        return -1;
    }

    try {
        return (*cast_indexer(indexer))->exists() ? 1 : 0;
    } catch (const std::exception &e) {
        DFTRACER_UTILS_LOG_ERROR("Failed to check if index exists: %s",
                                 e.what());
        return -1;
    }
}

uint64_t dft_indexer_get_max_bytes(dft_indexer_handle_t indexer) {
    if (validate_handle(indexer) < 0) {
        return 0;
    }

    try {
        return (*cast_indexer(indexer))->get_max_bytes();
    } catch (const std::exception &e) {
        DFTRACER_UTILS_LOG_ERROR("Failed to get max bytes: %s", e.what());
        return 0;
    }
}

uint64_t dft_indexer_get_num_lines(dft_indexer_handle_t indexer) {
    if (validate_handle(indexer) < 0) {
        return 0;
    }

    try {
        return (*cast_indexer(indexer))->get_num_lines();
    } catch (const std::exception &e) {
        DFTRACER_UTILS_LOG_ERROR("Failed to get number of lines: %s", e.what());
        return 0;
    }
}

void dft_indexer_destroy(dft_indexer_handle_t indexer) {
    if (indexer) {
        delete cast_indexer(indexer);
    }
}

}  // extern "C"
