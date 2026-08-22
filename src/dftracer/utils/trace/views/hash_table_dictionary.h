#ifndef DFTRACER_UTILS_TRACE_VIEWS_HASH_TABLE_DICTIONARY_H
#define DFTRACER_UTILS_TRACE_VIEWS_HASH_TABLE_DICTIONARY_H

#include <dftracer/utils/core/coro/task.h>
#include <dftracer/utils/trace/views/coverage.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <unordered_map>

namespace dftracer::utils::utilities::indexer {
class IndexBatchSink;
}  // namespace dftracer::utils::utilities::indexer

namespace dftracer::utils::trace::views::detail {

/// Accumulates a hash -> name dictionary (FH/HH/SH/PR metadata) and writes it
/// to an index's HASH_TABLES. Shared by every path that harvests the dictionary
/// from a scan it is riding along.
///
/// Coverage unit is the whole FILE: a hash is defined at its first use, so a
/// partially read file may reference a hash whose definition lives in a member
/// never visited. Entries land in a pending set and promote to sealed only when
/// their unit seals (publish-on-completion), so an aborted unit contributes
/// nothing. Over the byte budget it abandons rather than grow unbounded; the
/// scan it rides along is unaffected.
class HashTableDictionary {
   public:
    explicit HashTableDictionary(std::uint64_t budget_bytes = 64ull << 20)
        : budget_(budget_bytes) {}

    /// FH/HH/SH/PR name -> type code; false for any other record.
    static bool metadata_type(std::string_view name, std::uint8_t& out);

    /// First non-empty path wins; every entry is written to this index.
    void set_index_path(std::string_view path) {
        if (index_path_.empty()) index_path_.assign(path);
    }

    void add(std::uint8_t type, std::string_view hash, std::string_view name);

    void seal_unit(std::string_view file_path);
    void drop_unit() { pending_.clear(); }
    void merge(HashTableDictionary& other);

    /// Writes entries whose source file `covered` read whole, skipping any the
    /// index already resolves. Returns false when nothing was persisted.
    coro::CoroTask<bool> commit(const CoverageSet& covered);

    /// Write the sealed entries to a caller-owned sink (RocksDB or SST), for a
    /// fresh index build that owns the write transaction - no delta check or
    /// coverage gate, since the caller has established the whole file.
    void write_to_sink(utilities::indexer::IndexBatchSink& sink) const;

    std::size_t entry_count() const noexcept { return sealed_.size(); }
    bool abandoned() const noexcept { return abandoned_; }
    std::uint64_t budget() const noexcept { return budget_; }

   private:
    // [type][hash] -> name, flat so one map serves all four hash types.
    using Dict = std::unordered_map<std::string, std::string>;

    static std::string entry_key(std::uint8_t type, std::string_view hash);

    Dict pending_;  // current unit; published only when it seals
    Dict sealed_;
    // Source file per entry, so commit writes only entries whose file the scan
    // read whole.
    std::unordered_map<std::string, std::string> entry_file_;
    std::string index_path_;
    std::uint64_t budget_;
    std::uint64_t bytes_ = 0;
    bool abandoned_ = false;
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_HASH_TABLE_DICTIONARY_H
