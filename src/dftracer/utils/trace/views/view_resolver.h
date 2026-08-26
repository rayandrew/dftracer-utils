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
    /// pid (as text) -> rank, harvested from the trace's PR metadata at query
    /// time (not indexed), so it is populated separately from the constructor.
    const std::string& rank(const std::string& pid) const;
    void set_rank(const std::string& pid, std::string rank) const {
        rank_[pid] = std::move(rank);
    }

   private:
    static const std::string EMPTY;
    std::unordered_map<std::string, std::string> file_;
    std::unordered_map<std::string, std::string> host_;
    // Harvested from PR metadata at query time (not from the index), after the
    // resolver is constructed, so it is filled through a const handle.
    mutable std::unordered_map<std::string, std::string> rank_;
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_RESOLVER_H
