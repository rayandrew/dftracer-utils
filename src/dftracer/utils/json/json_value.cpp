#include <dftracer/utils/json/json_value.h>

#include <cstring>

namespace dftracer::utils::json {

JsonValue JsonValue::at(const char* path) const {
    if (!valid_ || !path) return JsonValue();

    JsonValue current = *this;
    const char* start = path;

    while (*start) {
        const char* end = start;
        while (*end && *end != '.') end++;

        size_t key_len = end - start;
        if (key_len == 0) {
            start = (*end == '.') ? end + 1 : end;
            continue;
        }

        std::string_view key_sv(start, key_len);
        JsonValue next = current[key_sv];

        // A numeric segment addresses an array element when the object-key
        // lookup found nothing and the current node is an array (e.g.
        // "tags.0.name").
        if (!next.exists() && current.is_array()) {
            bool all_digits = true;
            std::size_t idx = 0;
            for (char c : key_sv) {
                if (c < '0' || c > '9') {
                    all_digits = false;
                    break;
                }
                idx = idx * 10 + static_cast<std::size_t>(c - '0');
            }
            if (all_digits) next = current[idx];
        }
        current = next;

        if (!current.exists()) {
            return JsonValue();
        }

        start = (*end == '.') ? end + 1 : end;
    }

    return current;
}

JsonValue JsonValue::at(const std::string& path) const {
    return at(path.c_str());
}

JsonValue JsonValue::at(std::string_view path) const {
    std::string path_str(path);
    return at(path_str.c_str());
}

}  // namespace dftracer::utils::json
