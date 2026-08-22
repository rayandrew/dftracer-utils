#include <dftracer/utils/trace/views/dict_fold.h>

#include <cstdint>
#include <string_view>
#include <variant>

namespace dftracer::utils::trace::views::detail {

void DictFold::step(const FoldBatch& batch) {
    dict_.set_index_path(batch.unit.index_path);
    for (const auto& e : batch.events) {
        if (e.phase != RecordPhase::METADATA) continue;
        std::uint8_t type = 0;
        if (!HashTableDictionary::metadata_type(intern_->resolve(e.name_id),
                                                type))
            continue;

        // A metadata record registers one hash: args["name"] is the name and
        // args["value"] is its hash, both interned strings.
        std::string_view name, value;
        for (const auto& [k, v] : e.args) {
            const auto* id = std::get_if<std::uint32_t>(&v);
            if (!id) continue;
            const auto key = intern_->resolve(k);
            if (key == "name")
                name = intern_->resolve(*id);
            else if (key == "value")
                value = intern_->resolve(*id);
        }
        if (name.empty() || value.empty()) continue;
        dict_.add(type, value, name);
    }
}

}  // namespace dftracer::utils::trace::views::detail
