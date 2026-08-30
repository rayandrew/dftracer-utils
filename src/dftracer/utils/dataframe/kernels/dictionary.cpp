#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/kernels/dictionary.h>
#include <dftracer/utils/dataframe/parallel.h>

#include <algorithm>
#include <cstring>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace dftracer::utils::dataframe {

Series dictionary_encode(const Series& v) {
    return Series{dftu_series_dictionary_encode(v.handle())};
}

namespace {

// Rows per parallel_encode chunk; also the threshold below which the plain
// single-pass build runs (fan-out plus a merge pass is pure overhead there).
constexpr std::int64_t DICT_ENCODE_GRAIN = 1 << 18;

// Same output as the single-pass build (stable first-seen dict order, same
// codes), computed with two parallel passes: each chunk builds a local
// first-seen map, then the chunks are merged by picking, for each distinct
// value, the smallest global row index it first appeared at - that ordering
// is exactly what the serial single pass would have assigned.
std::vector<std::int32_t> dictionary_encode_parallel(
    const std::int32_t* off, const char* data, std::int64_t n,
    std::vector<std::int32_t>& dict_offsets, std::string& dict_data) {
    const std::int64_t nchunks =
        (n + DICT_ENCODE_GRAIN - 1) / DICT_ENCODE_GRAIN;
    std::vector<std::unordered_map<std::string_view, std::int64_t>> local(
        static_cast<std::size_t>(nchunks));
    parallel_for(n, DICT_ENCODE_GRAIN, [&](std::int64_t b, std::int64_t e) {
        auto& m = local[static_cast<std::size_t>(b / DICT_ENCODE_GRAIN)];
        for (std::int64_t i = b; i < e; ++i) {
            std::string_view s(data + off[i],
                               static_cast<std::size_t>(off[i + 1] - off[i]));
            m.try_emplace(s, i);
        }
    });

    std::unordered_map<std::string_view, std::int64_t> first_seen;
    for (auto& m : local)
        for (auto& [s, idx] : m) {
            auto it = first_seen.find(s);
            if (it == first_seen.end() || idx < it->second) first_seen[s] = idx;
        }

    std::vector<std::pair<std::int64_t, std::string_view>> ordered;
    ordered.reserve(first_seen.size());
    for (auto& [s, idx] : first_seen) ordered.emplace_back(idx, s);
    std::sort(ordered.begin(), ordered.end());

    std::unordered_map<std::string_view, std::int32_t> code_of;
    code_of.reserve(ordered.size());
    dict_offsets.assign(1, 0);
    for (std::size_t c = 0; c < ordered.size(); ++c) {
        const std::string_view s = ordered[c].second;
        code_of.emplace(s, static_cast<std::int32_t>(c));
        dict_data.append(s);
        dict_offsets.push_back(static_cast<std::int32_t>(dict_data.size()));
    }

    std::vector<std::int32_t> codes(static_cast<std::size_t>(n));
    parallel_for(n, DICT_ENCODE_GRAIN, [&](std::int64_t b, std::int64_t e) {
        for (std::int64_t i = b; i < e; ++i) {
            std::string_view s(data + off[i],
                               static_cast<std::size_t>(off[i + 1] - off[i]));
            codes[static_cast<std::size_t>(i)] = code_of.at(s);
        }
    });
    return codes;
}

}  // namespace

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

    std::vector<std::int32_t> codes;
    std::vector<std::int32_t> dict_offsets{0};
    std::string dict_data;
    if (dftracer::utils::dataframe::parallel_backend_installed() &&
        v->length > dftracer::utils::dataframe::DICT_ENCODE_GRAIN) {
        codes = dftracer::utils::dataframe::dictionary_encode_parallel(
            off, data, v->length, dict_offsets, dict_data);
    } else {
        codes.resize(static_cast<std::size_t>(v->length));
        std::unordered_map<std::string_view, std::int32_t> codes_of;
        for (std::int64_t i = 0; i < v->length; ++i) {
            std::string_view s(data + off[i],
                               static_cast<std::size_t>(off[i + 1] - off[i]));
            auto it = codes_of.find(s);
            std::int32_t code;
            if (it == codes_of.end()) {
                code = static_cast<std::int32_t>(codes_of.size());
                codes_of.emplace(s, code);
                dict_data.append(s);
                dict_offsets.push_back(
                    static_cast<std::int32_t>(dict_data.size()));
            } else {
                code = it->second;
            }
            codes[static_cast<std::size_t>(i)] = code;
        }
    }

    dftu_series* dict = dftu_series_new_string(
        static_cast<dftu_dtype>(v->type), dict_offsets.data(), dict_data.data(),
        static_cast<std::int64_t>(dict_offsets.size() - 1), nullptr);
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
