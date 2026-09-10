#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/varwidth_offsets.h>
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
// is exactly what the serial single pass would have assigned. `Off` is the
// source column's offset width (int32 for String/Binary, int64 for
// LargeString/LargeBinary); the dictionary's own offsets are built at the
// same width, so a DICTIONARY-encoded Large column's child is still a
// Large-typed, wide-offset column downstream ops can read.
template <class Off>
std::vector<std::int32_t> dictionary_encode_parallel(
    const Off* off, const char* data, std::int64_t n,
    std::vector<Off>& dict_offsets, std::string& dict_data) {
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
        dict_offsets.push_back(static_cast<Off>(dict_data.size()));
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

template <class Off>
dftu_series* dictionary_encode_w(const dftu_series* v) {
    const Off* off = reinterpret_cast<const Off*>(offsets_of<Off>(*v)->data());
    const char* data = reinterpret_cast<const char*>(v->data->data());

    std::vector<std::int32_t> codes;
    std::vector<Off> dict_offsets{0};
    std::string dict_data;
    if (parallel_backend_installed() && v->length > DICT_ENCODE_GRAIN) {
        codes = dictionary_encode_parallel<Off>(off, data, v->length,
                                                dict_offsets, dict_data);
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
                dict_offsets.push_back(static_cast<Off>(dict_data.size()));
            } else {
                code = it->second;
            }
            codes[static_cast<std::size_t>(i)] = code;
        }
    }

    auto* dict = new dftu_series();
    dict->type = v->type;
    dict->encoding = Encoding::Flat;
    dict->length = static_cast<std::int64_t>(dict_offsets.size() - 1);
    std::size_t off_bytes = dict_offsets.size() * sizeof(Off);
    offsets_of<Off>(*dict) = Buffer::allocate(off_bytes);
    std::memcpy(offsets_of<Off>(*dict)->data(), dict_offsets.data(), off_bytes);
    dict->data = Buffer::allocate(dict_data.size());
    if (!dict_data.empty())
        std::memcpy(dict->data->data(), dict_data.data(), dict_data.size());

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

}  // namespace

}  // namespace dftracer::utils::dataframe

dftu_series* dftu_series_dictionary_encode(const dftu_series* v) {
    using namespace dftracer::utils::dataframe;
    if (v->encoding != Encoding::Flat) return nullptr;
    const TypeId base = narrow_varwidth_type(v->type);
    if (base != TypeId::String && base != TypeId::Binary) return nullptr;
    const bool wide = is_wide_offset_type(v->type);
    if ((wide && !v->offsets64) || (!wide && !v->offsets) || !v->data)
        return nullptr;
    return wide ? dictionary_encode_w<std::int64_t>(v)
                : dictionary_encode_w<std::int32_t>(v);
}
