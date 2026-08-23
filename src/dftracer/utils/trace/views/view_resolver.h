#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_RESOLVER_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_RESOLVER_H

#include <string>
#include <unordered_map>
#include <vector>

namespace dftracer::utils::trace::views::detail {

/// Index-backed name maps for resolved-name group dimensions. Loads the FILE
/// and HOST hash tables from the given indexes once; resolution is an in-memory
/// lookup used by the post-aggregation re-key pass (per distinct group, never
/// per event). Unknown hashes resolve to "" (dfanalyzer treats it as missing).
class GroupResolver {
   public:
    explicit GroupResolver(const std::vector<std::string>& index_paths);

    const std::string& file_path(const std::string& fhash) const;
    const std::string& host_name(const std::string& hhash) const;

   private:
    static const std::string EMPTY;
    std::unordered_map<std::string, std::string> file_;
    std::unordered_map<std::string, std::string> host_;
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_RESOLVER_H
