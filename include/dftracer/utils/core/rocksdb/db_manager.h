#ifndef DFTRACER_UTILS_CORE_ROCKSDB_DB_MANAGER_H
#define DFTRACER_UTILS_CORE_ROCKSDB_DB_MANAGER_H

#include <dftracer/utils/core/rocksdb/database.h>

#include <condition_variable>
#include <cstddef>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace dftracer::utils::rocksdb {

// Process-wide registry of open RocksDB instances keyed by their normalized
// .dftindex root path. Besides the weak dedup map, it retains a bounded LRU of
// ReadOnly handles so sequential reads (plan phases, later queries, Dask tasks)
// reuse one open handle instead of reopening per operation. Only ReadOnly
// handles are retained, so a builder's exclusive lock releases with its ref;
// the set is per process, so multiprocess ReadOnly fan-out stays safe. Cap via
// DFTRACER_UTILS_ROCKSDB_CACHE.
class RocksDBManager {
   public:
    static RocksDBManager& instance();

    std::shared_ptr<RocksDatabase> get_or_open(
        const std::string& db_path,
        RocksDatabase::OpenMode open_mode = RocksDatabase::OpenMode::ReadWrite,
        RocksDatabase::CfOptionsOverride cf_override = nullptr);
    void reset(const std::string& db_path);
    void shutdown();

   private:
    RocksDBManager() = default;

    using RetainedList =
        std::list<std::pair<std::string, std::shared_ptr<RocksDatabase>>>;

    // All three helpers require mutex_ held.
    void retain_locked(const std::string& db_path,
                       const std::shared_ptr<RocksDatabase>& db);
    void drop_retained_locked(const std::string& db_path);
    std::size_t retain_cap();

    std::mutex mutex_;
    std::condition_variable cv_;
    std::unordered_map<std::string, std::weak_ptr<RocksDatabase>> databases_;
    std::unordered_set<std::string> opening_;
    RetainedList retained_;
    std::unordered_map<std::string, RetainedList::iterator> retained_index_;
    std::size_t retain_cap_ = 0;  // 0 until first read from env
};

}  // namespace dftracer::utils::rocksdb

#endif  // DFTRACER_UTILS_CORE_ROCKSDB_DB_MANAGER_H
