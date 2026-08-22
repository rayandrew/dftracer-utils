#include <dftracer/utils/trace/views/view_resolver.h>
#include <dftracer/utils/utilities/indexer/index_database.h>

namespace dftracer::utils::trace::views::detail {

const std::string GroupResolver::EMPTY;

GroupResolver::GroupResolver(const std::vector<std::string>& index_paths) {
    using utilities::indexer::IndexDatabase;
    for (const auto& path : index_paths) {
        if (path.empty()) continue;
        try {
            IndexDatabase db(path, utilities::indexer::IndexOpenMode::ReadOnly);
            for (auto& [h, n] :
                 db.query_hash_table(IndexDatabase::HashType::FILE))
                file_.emplace(h, n);
            for (auto& [h, n] :
                 db.query_hash_table(IndexDatabase::HashType::HOST))
                host_.emplace(h, n);
        } catch (const std::exception&) {
            // Unreadable index: those hashes just resolve to "".
        }
    }
}

const std::string& GroupResolver::file_path(const std::string& fhash) const {
    auto it = file_.find(fhash);
    return it != file_.end() ? it->second : EMPTY;
}

const std::string& GroupResolver::host_name(const std::string& hhash) const {
    auto it = host_.find(hhash);
    return it != host_.end() ? it->second : EMPTY;
}

}  // namespace dftracer::utils::trace::views::detail
