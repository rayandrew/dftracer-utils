#include <dftracer/utils/core/rocksdb/db_manager.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/agg_db_open.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_intern.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/aggregation_serialization.h>

#include <string>
#include <string_view>

namespace dftracer::utils::utilities::composites::dft::aggregators {

std::unique_ptr<AggDbHandle> open_agg_db(const std::string& index_path,
                                         std::string& error_msg) {
    std::shared_ptr<dftracer::utils::rocksdb::RocksDatabase> db;
    try {
        db = EventAggregator::open_read_only_with_merge_operator(index_path);
    } catch (...) {
        auto& mgr = dftracer::utils::rocksdb::RocksDBManager::instance();
        mgr.reset(index_path);
        db = mgr.get_or_open(
            index_path,
            dftracer::utils::rocksdb::RocksDatabase::OpenMode::ReadOnly);
        if (db && db->is_open()) {
            load_intern_dictionary(*db, *intern_for_index(index_path));
        }
    }
    if (!db || !db->is_open()) {
        error_msg = "Failed to open aggregation database";
        return nullptr;
    }
    std::string config_val;
    auto key = std::string_view(AGG_GLOBAL_CONFIG_KEY,
                                sizeof(AGG_GLOBAL_CONFIG_KEY) - 1);
    if (!db->get(key, &config_val, dftracer::utils::rocksdb::cf::AGGREGATION)
             .ok()) {
        error_msg = "No aggregation config found - was aggregation enabled?";
        return nullptr;
    }
    auto cfg = deserialize_agg_global_config(config_val);
    auto handle = std::make_unique<AggDbHandle>();
    handle->db = db;
    handle->agg = std::make_unique<EventAggregator>(db, cfg.config_hash);
    return handle;
}

}  // namespace dftracer::utils::utilities::composites::dft::aggregators
