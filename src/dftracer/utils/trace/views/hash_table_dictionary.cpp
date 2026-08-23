#include <dftracer/utils/core/common/logging.h>
#include <dftracer/utils/trace/views/hash_table_dictionary.h>
#include <dftracer/utils/trace/views/index_write_lock.h>
#include <dftracer/utils/utilities/indexer/index_batch_sink.h>
#include <dftracer/utils/utilities/indexer/index_database.h>
#include <dftracer/utils/utilities/indexer/index_database_writer_context.h>

#include <exception>
#include <mutex>
#include <utility>
#include <vector>

namespace dftracer::utils::trace::views::detail {

namespace {

// Matches IndexDatabase::HashType.
constexpr std::uint8_t TYPE_FILE = 0;
constexpr std::uint8_t TYPE_HOST = 1;
constexpr std::uint8_t TYPE_STRING = 2;
constexpr std::uint8_t TYPE_PROC = 3;

}  // namespace

bool HashTableDictionary::metadata_type(std::string_view name,
                                        std::uint8_t& out) {
    if (name == "FH") {
        out = TYPE_FILE;
    } else if (name == "HH") {
        out = TYPE_HOST;
    } else if (name == "SH") {
        out = TYPE_STRING;
    } else if (name == "PR") {
        out = TYPE_PROC;
    } else {
        return false;
    }
    return true;
}

std::string HashTableDictionary::entry_key(std::uint8_t type,
                                           std::string_view hash) {
    std::string k(1, static_cast<char>('0' + type));
    k.append(hash);
    return k;
}

void HashTableDictionary::add(std::uint8_t type, std::string_view hash,
                              std::string_view name) {
    if (abandoned_) return;
    auto key = entry_key(type, hash);
    bytes_ += key.size() + name.size();
    pending_.insert_or_assign(std::move(key), std::string(name));

    // Over budget: stop building rather than slow the scan or grow unbounded.
    // The query is unaffected.
    if (bytes_ > budget_) {
        abandoned_ = true;
        pending_.clear();
        sealed_.clear();
        entry_file_.clear();
    }
}

void HashTableDictionary::seal_unit(std::string_view file_path) {
    if (abandoned_) return;
    for (auto& [k, v] : pending_) {
        entry_file_.insert_or_assign(k, std::string(file_path));
        sealed_.insert_or_assign(k, std::move(v));
    }
    pending_.clear();
}

void HashTableDictionary::merge(HashTableDictionary& other) {
    if (other.abandoned_) {
        abandoned_ = true;
        sealed_.clear();
        entry_file_.clear();
        return;
    }
    if (abandoned_) return;
    if (index_path_.empty()) index_path_ = other.index_path_;
    for (auto& [k, v] : other.sealed_) {
        auto it = other.entry_file_.find(k);
        if (it != other.entry_file_.end())
            entry_file_.insert_or_assign(k, it->second);
        sealed_.insert_or_assign(k, std::move(v));
    }
    other.sealed_.clear();
}

coro::CoroTask<bool> HashTableDictionary::commit(const CoverageSet& covered) {
    if (abandoned_ || sealed_.empty() || index_path_.empty()) co_return false;

    // Only files the scan read whole: a partially read file may define a hash
    // in a member never visited, and a missing entry resolves to an empty name
    // rather than failing.
    std::vector<std::pair<std::string, std::string>> writable;
    writable.reserve(sealed_.size());
    for (const auto& [k, v] : sealed_) {
        auto it = entry_file_.find(k);
        if (it == entry_file_.end()) continue;
        if (!covered.covers_file(it->second)) continue;
        writable.emplace_back(k, v);
    }
    if (writable.empty()) co_return false;

    // Drop what the index already resolves, so the cost scales with new hashes
    // rather than total ones. A read open takes no exclusive lock, so this
    // avoids the write open entirely when nothing is new.
    try {
        utilities::indexer::IndexDatabase ro(
            index_path_, utilities::indexer::IndexOpenMode::ReadOnly);
        std::unordered_map<std::uint8_t,
                           std::unordered_map<std::string, std::string>>
            existing;
        std::vector<std::pair<std::string, std::string>> fresh;
        fresh.reserve(writable.size());
        for (auto& [k, v] : writable) {
            const auto type = static_cast<std::uint8_t>(k[0] - '0');
            auto it = existing.find(type);
            if (it == existing.end())
                it = existing
                         .emplace(
                             type,
                             ro.query_hash_table(
                                 static_cast<utilities::indexer::IndexDatabase::
                                                 HashType>(type)))
                         .first;
            if (it->second.count(std::string(std::string_view(k).substr(1))))
                continue;
            fresh.emplace_back(std::move(k), std::move(v));
        }
        writable.swap(fresh);
    } catch (const std::exception& e) {
        // Unreadable index: it is not known what is already there, so writing
        // blind would be unbounded waste. Declining is the conservative side,
        // and must not be silent - this is a skip, not an empty dictionary.
        DFTRACER_UTILS_LOG_WARN("dictionary delta check failed for %s: %s",
                                index_path_.c_str(), e.what());
        co_return false;
    }
    if (writable.empty()) co_return false;

    // A write open takes RocksDB's exclusive directory lock, so concurrent
    // committers would fight over it. Serialize them per index instead of
    // caching a handle: a cached writer would hold the lock for the process
    // lifetime and block the indexer, which runs in the same process as the
    // query in dfanalyzer.
    std::unique_lock<std::mutex> lk(index_write_mutex(index_path_));

    try {
        utilities::indexer::IndexDatabase db(index_path_);
        auto writer = db.begin_write();
        for (const auto& [k, v] : writable) {
            const auto type = static_cast<std::uint8_t>(k[0] - '0');
            writer->insert_hash_table_entry(type, std::string_view(k).substr(1),
                                            v);
        }
        writer->commit();
    } catch (const std::exception& e) {
        // A read-only or already-locked index must degrade to "answered the
        // query, persisted nothing", never to a failed query - but it must not
        // look like "there was nothing to persist".
        DFTRACER_UTILS_LOG_WARN("dictionary materialization skipped for %s: %s",
                                index_path_.c_str(), e.what());
        co_return false;
    }
    co_return true;
}

void HashTableDictionary::write_to_sink(
    utilities::indexer::IndexBatchSink& sink) const {
    for (const auto& [k, v] : sealed_) {
        const auto type = static_cast<std::uint8_t>(k[0] - '0');
        sink.insert_hash_table_entry(type, std::string_view(k).substr(1), v);
    }
}

}  // namespace dftracer::utils::trace::views::detail
