#ifndef DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_INDEXER_H
#define DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_INDEXER_H

#include <dftracer/utils/core/common/constants.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void *dftu_indexer_handle_t;

// C API function declarations
dftu_indexer_handle_t dftu_indexer_create(const char *gz_path,
                                          const char *index_path,
                                          uint64_t checkpoint_size,
                                          int force_rebuild);
int dftu_indexer_build(dftu_indexer_handle_t indexer);
int dftu_indexer_need_rebuild(dftu_indexer_handle_t indexer);
int dftu_indexer_exists(dftu_indexer_handle_t indexer);
uint64_t dftu_indexer_get_max_bytes(dftu_indexer_handle_t indexer);
uint64_t dftu_indexer_get_num_lines(dftu_indexer_handle_t indexer);
void dftu_indexer_destroy(dftu_indexer_handle_t indexer);

#ifdef __cplusplus
}

#include <dftracer/utils/core/common/archive_format.h>
#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/utilities/indexer/index_visitor.h>
#include <dftracer/utils/utilities/indexer/internal/gzip_member_record.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace dftracer::utils::utilities::indexer::internal {

/**
 * Abstract base interface for all indexer implementations.
 * This provides a common API for GZIP, TAR.GZ, and other archive indexers.
 */
class Indexer {
   public:
    static constexpr std::uint64_t DEFAULT_CHECKPOINT_SIZE =
        constants::indexer::DEFAULT_CHECKPOINT_SIZE;

    virtual ~Indexer() = default;

    // Core indexer operations
    virtual coro::CoroTask<void> build_async() const = 0;

    void build() const { build_async().get(); }
    virtual bool need_rebuild() const = 0;
    virtual bool exists() const = 0;

    using VisitorList = std::vector<std::reference_wrapper<IndexVisitor>>;

    virtual void set_visitors(VisitorList visitors) { (void)visitors; }

    // Metadata accessors
    virtual const std::string &get_index_path() const = 0;
    virtual const std::string &get_archive_path() const = 0;
    virtual std::uint64_t get_checkpoint_size() const = 0;
    virtual std::uint64_t get_max_bytes() const = 0;
    virtual std::uint64_t get_num_lines() const = 0;

    /// Member containing `target_offset` (uncompressed). A gzip member is a
    /// self-contained stream, so seeking to one needs neither an inflate
    /// dictionary nor bit priming.
    virtual bool find_member(std::size_t target_offset,
                             GzipMemberRecord &member) const = 0;

    /// Every member in file order. Empty when the file was never indexed.
    virtual std::vector<GzipMemberRecord> get_members() const = 0;

    // Archive format identification
    virtual ArchiveFormat get_format_type() const = 0;
    virtual const char *get_format_name() const = 0;

   protected:
    Indexer() = default;
    Indexer(const Indexer &) = delete;
    Indexer &operator=(const Indexer &) = delete;
};

}  // namespace dftracer::utils::utilities::indexer::internal

#endif  // __cplusplus

#endif  // DFTRACER_UTILS_UTILITIES_INDEXER_INTERNAL_INDEXER_H
