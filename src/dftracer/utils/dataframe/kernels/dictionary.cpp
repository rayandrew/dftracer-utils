#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/kernels/dictionary.h>

#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace dftracer::utils::dataframe {

Series dictionary_encode(const Series& v) {
    return Series{dftu_series_dictionary_encode(v.handle())};
}

}  // namespace dftracer::utils::dataframe

dftu_series* dftu_series_dictionary_encode(const dftu_series* v) {
    using dftracer::utils::dataframe::Buffer;
    using dftracer::utils::dataframe::Encoding;
    using dftracer::utils::dataframe::TypeId;
    if (v->encoding != Encoding::Flat) return nullptr;
    if (v->type != TypeId::String && v->type != TypeId::Binary) return nullptr;
    if (!v->offsets || !v->data) return nullptr;

    const std::int32_t* off =
        reinterpret_cast<const std::int32_t*>(v->offsets->data());
    const char* data = reinterpret_cast<const char*>(v->data->data());

    std::unordered_map<std::string_view, std::int32_t> codes_of;
    std::vector<std::int32_t> codes(static_cast<std::size_t>(v->length));
    std::vector<std::int32_t> dict_offsets{0};
    std::string dict_data;
    for (std::int64_t i = 0; i < v->length; ++i) {
        std::string_view s(data + off[i],
                           static_cast<std::size_t>(off[i + 1] - off[i]));
        auto it = codes_of.find(s);
        std::int32_t code;
        if (it == codes_of.end()) {
            code = static_cast<std::int32_t>(codes_of.size());
            codes_of.emplace(s, code);
            dict_data.append(s);
            dict_offsets.push_back(static_cast<std::int32_t>(dict_data.size()));
        } else {
            code = it->second;
        }
        codes[static_cast<std::size_t>(i)] = code;
    }

    dftu_series* dict = dftu_series_new_string(
        static_cast<dftu_dtype>(v->type), dict_offsets.data(), dict_data.data(),
        static_cast<std::int64_t>(codes_of.size()), nullptr);
    if (dict == nullptr) return nullptr;

    auto* out = new dftu_series();
    out->type = v->type;
    out->encoding = Encoding::Dictionary;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;
    out->data = Buffer::allocate(codes.size() * sizeof(std::int32_t));
    std::memcpy(out->data->data(), codes.data(),
                codes.size() * sizeof(std::int32_t));
    out->child = std::shared_ptr<dftu_series>(dict);
    return out;
}
