#include <dftracer/utils/trace/views/index_write_lock.h>

#include <memory>
#include <unordered_map>

namespace dftracer::utils::trace::views::detail {

std::mutex& index_write_mutex(const std::string& index_path) {
    static std::mutex registry_mtx;
    static std::unordered_map<std::string, std::unique_ptr<std::mutex>> mutexes;
    std::lock_guard<std::mutex> lk(registry_mtx);
    auto& slot = mutexes[index_path];
    if (!slot) slot = std::make_unique<std::mutex>();
    return *slot;
}

}  // namespace dftracer::utils::trace::views::detail
