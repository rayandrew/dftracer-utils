#include <dftracer/utils/json/json_value.h>

#include <cstring>

namespace dftracer::utils::json {

JsonValue JsonValue::at(const char* path) const {
    if (!valid_ || !path) return JsonValue();

    JsonValue current = *this;
    const char* p = path;

    while (*p) {
        if (*p == '.') {
            ++p;
            continue;
        }
        // Bracket index: `a[0]`.
        if (*p == '[') {
            ++p;
            std::size_t idx = 0;
            bool any = false;
            while (*p >= '0' && *p <= '9') {
                idx = idx * 10 + static_cast<std::size_t>(*p - '0');
                ++p;
                any = true;
            }
            if (*p == ']') ++p;
            if (!any) return JsonValue();
            current = current[idx];
            if (!current.exists()) return JsonValue();
            continue;
        }

        const char* start = p;
        while (*p && *p != '.' && *p != '[') ++p;
        std::string_view key_sv(start, static_cast<std::size_t>(p - start));

        JsonValue next = current[key_sv];
        // A numeric segment addresses an array element when the object-key
        // lookup found nothing and the current node is an array (e.g.
        // "tags.0.name").
        if (!next.exists() && current.is_array()) {
            bool all_digits = !key_sv.empty();
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
        if (!current.exists()) return JsonValue();
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
