#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/internal/column_data.h>
#include <dftracer/utils/dataframe/internal/substr_simd.h>
#include <dftracer/utils/dataframe/kernels/string_ops.h>

#include <cstring>
#include <regex>
#include <string>
#include <string_view>
#include <vector>

// Highway static dispatch (baseline target): only the two
// naturally-vectorizable string ops use it - str_len_bytes (adjacent int32
// offset difference) and the ASCII case folds (byte compare + conditional add).
// Every other op is variable-length per row and is legitimately scalar.
#include <hwy/highway.h>
namespace hn = hwy::HWY_NAMESPACE;

using dftracer::utils::dataframe::substr_find;

namespace dftracer::utils::dataframe {

Series str_eq(const Series& v, std::string_view rhs) {
    return Series{dftu_series_str_eq(v.handle(), rhs.data(),
                                     static_cast<std::int32_t>(rhs.size()))};
}
Series str_contains(const Series& v, std::string_view needle) {
    return Series{dftu_series_str_contains(
        v.handle(), needle.data(), static_cast<std::int32_t>(needle.size()))};
}
Series str_starts_with(const Series& v, std::string_view prefix) {
    return Series{dftu_series_str_starts_with(
        v.handle(), prefix.data(), static_cast<std::int32_t>(prefix.size()))};
}

}  // namespace dftracer::utils::dataframe

namespace {

using dftracer::utils::dataframe::Buffer;
using dftracer::utils::dataframe::buffer_bytes;
using dftracer::utils::dataframe::Encoding;
using dftracer::utils::dataframe::TypeId;

std::string_view value_at(const dftu_series& c, std::int64_t i) {
    const std::int32_t* off =
        reinterpret_cast<const std::int32_t*>(c.offsets->data());
    const char* data = reinterpret_cast<const char*>(c.data->data());
    return std::string_view(data + off[i],
                            static_cast<std::size_t>(off[i + 1] - off[i]));
}

// Apply a string predicate, returning a Bool column. On a DICTIONARY input the
// predicate is evaluated once per dictionary entry, then codes are mapped.
template <class Pred>
dftu_series* string_predicate(const dftu_series* v, Pred pred) {
    if (v->type != TypeId::String && v->type != TypeId::Binary) return nullptr;

    auto* out = new dftu_series();
    out->type = TypeId::Bool;
    out->encoding = Encoding::Flat;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;
    std::size_t bytes = buffer_bytes(TypeId::Bool, v->length);
    out->data = Buffer::allocate(bytes);
    std::memset(out->data->data(), 0, bytes);
    std::uint8_t* bits = out->data->data();
    auto set = [&](std::int64_t i) {
        bits[i >> 3] |= static_cast<std::uint8_t>(1u << (i & 7));
    };

    if (v->encoding == Encoding::Flat) {
        for (std::int64_t i = 0; i < v->length; ++i)
            if (pred(value_at(*v, i))) set(i);
    } else if (v->encoding == Encoding::Dictionary && v->child) {
        const dftu_series& dict = *v->child;
        std::vector<char> hit(static_cast<std::size_t>(dict.length));
        for (std::int64_t k = 0; k < dict.length; ++k)
            hit[static_cast<std::size_t>(k)] = pred(value_at(dict, k)) ? 1 : 0;
        const std::int32_t* codes =
            reinterpret_cast<const std::int32_t*>(v->data->data());
        for (std::int64_t i = 0; i < v->length; ++i)
            if (hit[static_cast<std::size_t>(codes[i])]) set(i);
    } else {
        delete out;
        return nullptr;
    }
    return out;
}

// Per-row byte reader over a String/Binary column that transparently resolves a
// DICTIONARY input (row i -> dictionary entry codes[i]). Nullness comes from
// the top-level validity bitmap in both encodings.
class RowReader {
   public:
    explicit RowReader(const dftu_series* v) : v_(v) {
        if (v->type != TypeId::String && v->type != TypeId::Binary) return;
        if (v->encoding == Encoding::Flat && v->offsets && v->data) {
            off_ = reinterpret_cast<const std::int32_t*>(v->offsets->data());
            data_ = reinterpret_cast<const char*>(v->data->data());
            ok_ = true;
        } else if (v->encoding == Encoding::Dictionary && v->child &&
                   v->child->offsets && v->child->data && v->data) {
            off_ = reinterpret_cast<const std::int32_t*>(
                v->child->offsets->data());
            data_ = reinterpret_cast<const char*>(v->child->data->data());
            codes_ = reinterpret_cast<const std::int32_t*>(v->data->data());
            ok_ = true;
        }
    }
    bool ok() const { return ok_; }
    bool is_null(std::int64_t i) const {
        if (!v_->validity) return false;
        const std::uint8_t* bm = v_->validity->data();
        return ((bm[i >> 3] >> (i & 7)) & 1) == 0;
    }
    std::string_view at(std::int64_t i) const {
        std::int64_t k = codes_ ? codes_[i] : i;
        return std::string_view(
            data_ + off_[k], static_cast<std::size_t>(off_[k + 1] - off_[k]));
    }

   private:
    const dftu_series* v_;
    const std::int32_t* off_ = nullptr;
    const char* data_ = nullptr;
    const std::int32_t* codes_ = nullptr;
    bool ok_ = false;
};

// Int64 output column with the same length/validity as `v` (nulls preserved).
dftu_series* make_i64(const dftu_series* v,
                      const std::vector<std::int64_t>& vals) {
    auto* out = new dftu_series();
    out->type = TypeId::Int64;
    out->encoding = Encoding::Flat;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;
    std::size_t bytes = buffer_bytes(TypeId::Int64, v->length);
    out->data = Buffer::allocate(bytes);
    if (bytes != 0) std::memcpy(out->data->data(), vals.data(), bytes);
    return out;
}

// Flat String output column from one string per row (size == v->length); shares
// v's validity so null rows stay null (their bytes are ignored).
dftu_series* make_string(const dftu_series* v,
                         const std::vector<std::string>& parts) {
    auto* out = new dftu_series();
    out->type = TypeId::String;
    out->encoding = Encoding::Flat;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;
    std::size_t off_bytes =
        static_cast<std::size_t>(v->length + 1) * sizeof(std::int32_t);
    out->offsets = Buffer::allocate(off_bytes);
    std::size_t total = 0;
    for (const std::string& p : parts) total += p.size();
    out->data = Buffer::allocate(total);
    std::int32_t* od = reinterpret_cast<std::int32_t*>(out->offsets->data());
    char* bd =
        total != 0 ? reinterpret_cast<char*>(out->data->data()) : nullptr;
    std::int32_t pos = 0;
    od[0] = 0;
    for (std::int64_t i = 0; i < v->length; ++i) {
        const std::string& p = parts[static_cast<std::size_t>(i)];
        if (!p.empty()) std::memcpy(bd + pos, p.data(), p.size());
        pos += static_cast<std::int32_t>(p.size());
        od[i + 1] = pos;
    }
    return out;
}

// ---- SIMD helpers --------------------------------------------------------

// out[i] = off[i+1] - off[i], widened to int64. Highway over the int32 offsets.
void len_bytes_simd(const std::int32_t* off, std::int64_t* out, std::size_t n) {
    const hn::ScalableTag<std::int64_t> d64;
    const hn::Rebind<std::int32_t, decltype(d64)> d32;
    const std::size_t lanes = hn::Lanes(d64);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto a = hn::LoadU(d32, off + i + 1);
        const auto b = hn::LoadU(d32, off + i);
        hn::StoreU(hn::PromoteTo(d64, hn::Sub(a, b)), d64, out + i);
    }
    for (; i < n; ++i) out[i] = static_cast<std::int64_t>(off[i + 1] - off[i]);
}

// UTF-8 character count of a byte range: every byte that is not a continuation
// byte ((c & 0xC0) != 0x80) starts a character. Highway compares a block and
// folds the mask with CountTrue.
std::int64_t count_char_starts(const std::uint8_t* p, std::size_t len) {
    const hn::ScalableTag<std::uint8_t> d;
    const auto vc0 = hn::Set(d, 0xC0);
    const auto v80 = hn::Set(d, 0x80);
    const std::size_t lanes = hn::Lanes(d);
    std::int64_t count = 0;
    std::size_t i = 0;
    for (; i + lanes <= len; i += lanes) {
        const auto m = hn::Ne(hn::And(hn::LoadU(d, p + i), vc0), v80);
        count += static_cast<std::int64_t>(hn::CountTrue(d, m));
    }
    for (; i < len; ++i)
        if ((p[i] & 0xC0) != 0x80) ++count;
    return count;
}

// ASCII case fold of a byte buffer: `add` is +32 (upper->lower) or -32
// (lower->upper); `lo`/`hi` bound the source case. Non-matching bytes pass
// through unchanged (ASCII-only fold).
void ascii_fold_simd(const std::uint8_t* in, std::uint8_t* out, std::size_t n,
                     std::uint8_t lo, std::uint8_t hi, int add) {
    const hn::ScalableTag<std::uint8_t> d;
    const auto vlo = hn::Set(d, lo);
    const auto vhi = hn::Set(d, hi);
    const auto vadd = hn::Set(d, static_cast<std::uint8_t>(add));
    const std::size_t lanes = hn::Lanes(d);
    std::size_t i = 0;
    for (; i + lanes <= n; i += lanes) {
        const auto v = hn::LoadU(d, in + i);
        const auto m = hn::And(hn::Ge(v, vlo), hn::Le(v, vhi));
        hn::StoreU(hn::IfThenElse(m, hn::Add(v, vadd), v), d, out + i);
    }
    for (; i < n; ++i) {
        const std::uint8_t c = in[i];
        out[i] = (c >= lo && c <= hi) ? static_cast<std::uint8_t>(c + add) : c;
    }
}

// Fold that keeps offsets/validity and rewrites only the (byte-length
// preserving) data buffer. Fast path SIMD for a FLAT input; a DICTIONARY input
// falls back through make_string per row.
dftu_series* ascii_fold(const dftu_series* v, std::uint8_t lo, std::uint8_t hi,
                        int add) {
    RowReader r(v);
    if (!r.ok()) return nullptr;
    if (v->encoding == Encoding::Flat) {
        auto* out = new dftu_series();
        out->type = v->type;
        out->encoding = Encoding::Flat;
        out->length = v->length;
        out->null_count = v->null_count;
        out->validity = v->validity;
        out->offsets = v->offsets;  // byte lengths unchanged
        const std::size_t data_len = v->data ? v->data->size() : 0;
        out->data = Buffer::allocate(data_len);
        if (data_len != 0)
            ascii_fold_simd(v->data->data(), out->data->data(), data_len, lo,
                            hi, add);
        return out;
    }
    std::vector<std::string> parts(static_cast<std::size_t>(v->length));
    for (std::int64_t i = 0; i < v->length; ++i) {
        if (r.is_null(i)) continue;
        std::string s(r.at(i));
        for (char& c : s)
            if (static_cast<std::uint8_t>(c) >= lo &&
                static_cast<std::uint8_t>(c) <= hi)
                c = static_cast<char>(c + add);
        parts[static_cast<std::size_t>(i)] = std::move(s);
    }
    return make_string(v, parts);
}

// Per-row string transform (scalar); null rows pass through as null.
template <class Fn>
dftu_series* string_transform(const dftu_series* v, Fn fn) {
    RowReader r(v);
    if (!r.ok()) return nullptr;
    std::vector<std::string> parts(static_cast<std::size_t>(v->length));
    for (std::int64_t i = 0; i < v->length; ++i) {
        if (r.is_null(i)) continue;
        parts[static_cast<std::size_t>(i)] = fn(r.at(i));
    }
    return make_string(v, parts);
}

// Per-row int64 transform (scalar); null rows keep the shared validity.
template <class Fn>
dftu_series* int_transform(const dftu_series* v, Fn fn) {
    RowReader r(v);
    if (!r.ok()) return nullptr;
    std::vector<std::int64_t> vals(static_cast<std::size_t>(v->length), 0);
    for (std::int64_t i = 0; i < v->length; ++i) {
        if (r.is_null(i)) continue;
        vals[static_cast<std::size_t>(i)] = fn(r.at(i));
    }
    return make_i64(v, vals);
}

bool is_ascii_ws(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' ||
           c == '\v';
}

std::string_view lstrip_ws(std::string_view s) {
    std::size_t b = 0;
    while (b < s.size() && is_ascii_ws(s[b])) ++b;
    return s.substr(b);
}
std::string_view rstrip_ws(std::string_view s) {
    std::size_t e = s.size();
    while (e > 0 && is_ascii_ws(s[e - 1])) --e;
    return s.substr(0, e);
}

std::string replace_literal(std::string_view s, std::string_view pat,
                            std::string_view repl, bool all) {
    // An empty pattern has no well-defined occurrence; leave the row unchanged
    // (also avoids a zero-advance loop in the all-occurrences case).
    if (pat.empty()) return std::string(s);
    std::string out;
    std::size_t pos = 0;
    while (true) {
        std::size_t hit = s.find(pat, pos);
        if (hit == std::string_view::npos) break;
        out.append(s.substr(pos, hit - pos));
        out.append(repl);
        pos = hit + pat.size();
        if (!all) break;
    }
    out.append(s.substr(pos));
    return out;
}

std::string slice_bytes(std::string_view s, std::int64_t start,
                        std::int64_t length) {
    const std::int64_t len = static_cast<std::int64_t>(s.size());
    std::int64_t begin = start;
    if (begin < 0) begin += len;
    if (begin < 0) begin = 0;
    if (begin > len) begin = len;
    // A negative length means "to the end" (polars slice with no length).
    std::int64_t end = length < 0 ? len : begin + length;
    if (end < begin) end = begin;
    if (end > len) end = len;
    return std::string(s.substr(static_cast<std::size_t>(begin),
                                static_cast<std::size_t>(end - begin)));
}

std::string pad(std::string_view s, std::int64_t width, char fill, bool left) {
    const std::int64_t len = static_cast<std::int64_t>(s.size());
    if (len >= width) return std::string(s);
    std::string fillstr(static_cast<std::size_t>(width - len), fill);
    if (left) return fillstr + std::string(s);
    return std::string(s) + fillstr;
}

std::string zfill(std::string_view s, std::int64_t width) {
    const std::int64_t len = static_cast<std::int64_t>(s.size());
    if (len >= width) return std::string(s);
    const std::size_t pad_n = static_cast<std::size_t>(width - len);
    // A leading sign stays first; zeros are inserted after it (Python zfill).
    if (!s.empty() && (s[0] == '+' || s[0] == '-'))
        return std::string(1, s[0]) + std::string(pad_n, '0') +
               std::string(s.substr(1));
    return std::string(pad_n, '0') + std::string(s);
}

// ---- LIKE / glob matching ------------------------------------------------

// One classified pattern token: a literal byte, `%` (any run), or `_` (any one
// character). `\` escapes the following byte to a literal in the source
// pattern; the classification is done once, then reused for every row.
enum class GlobKind : std::uint8_t { Literal, AnyRun, AnyOne };
struct GlobToken {
    GlobKind kind;
    char ch;  // valid when kind == Literal
};

std::vector<GlobToken> classify_like(std::string_view pattern) {
    std::vector<GlobToken> toks;
    toks.reserve(pattern.size());
    for (std::size_t i = 0; i < pattern.size(); ++i) {
        char c = pattern[i];
        if (c == '\\' && i + 1 < pattern.size()) {
            toks.push_back({GlobKind::Literal, pattern[++i]});
        } else if (c == '%') {
            toks.push_back({GlobKind::AnyRun, 0});
        } else if (c == '_') {
            toks.push_back({GlobKind::AnyOne, 0});
        } else {
            toks.push_back({GlobKind::Literal, c});
        }
    }
    return toks;
}

// Classic two-pointer wildcard match (no backtracking blowup): `%` == `*`
// (matches any run, including empty), `_` == `?` (matches exactly one byte).
bool glob_match(const std::vector<GlobToken>& toks, std::string_view s) {
    std::size_t si = 0, pi = 0;
    std::size_t star = std::string_view::npos;  // last AnyRun token index
    std::size_t star_si = 0;                    // s position when star seen
    const std::size_t n = s.size();
    const std::size_t m = toks.size();
    while (si < n) {
        if (pi < m && toks[pi].kind == GlobKind::AnyOne) {
            ++si;
            ++pi;
        } else if (pi < m && toks[pi].kind == GlobKind::Literal &&
                   toks[pi].ch == s[si]) {
            ++si;
            ++pi;
        } else if (pi < m && toks[pi].kind == GlobKind::AnyRun) {
            star = pi;
            star_si = si;
            ++pi;
        } else if (star != std::string_view::npos) {
            pi = star + 1;
            ++star_si;
            si = star_si;
        } else {
            return false;
        }
    }
    while (pi < m && toks[pi].kind == GlobKind::AnyRun) ++pi;
    return pi == m;
}

}  // namespace

dftu_series* dftu_series_str_eq(const dftu_series* v, const char* rhs,
                                int32_t rhs_len) {
    std::string_view r(rhs, static_cast<std::size_t>(rhs_len));
    return string_predicate(v, [r](std::string_view s) { return s == r; });
}

dftu_series* dftu_series_str_contains(const dftu_series* v, const char* needle,
                                      int32_t needle_len) {
    std::string_view n(needle, static_cast<std::size_t>(needle_len));
    return string_predicate(v, [n](std::string_view s) {
        return substr_find(s.data(), static_cast<std::int64_t>(s.size()),
                           n.data(), static_cast<std::int64_t>(n.size())) >= 0;
    });
}

dftu_series* dftu_series_str_starts_with(const dftu_series* v,
                                         const char* prefix,
                                         int32_t prefix_len) {
    std::string_view p(prefix, static_cast<std::size_t>(prefix_len));
    return string_predicate(
        v, [p](std::string_view s) { return s.starts_with(p); });
}

dftu_series* dftu_series_str_ends_with(const dftu_series* v, const char* suffix,
                                       int32_t suffix_len) {
    std::string_view p(suffix, static_cast<std::size_t>(suffix_len));
    return string_predicate(v,
                            [p](std::string_view s) { return s.ends_with(p); });
}

dftu_series* dftu_series_str_matches(const dftu_series* v, const char* pattern,
                                     int32_t pattern_len) {
    std::regex re;
    try {
        re.assign(std::string(pattern, static_cast<std::size_t>(pattern_len)),
                  std::regex::ECMAScript);
    } catch (const std::regex_error&) {
        return nullptr;
    }
    return string_predicate(v, [&re](std::string_view s) {
        return std::regex_match(s.begin(), s.end(), re);
    });
}

dftu_series* dftu_series_str_like(const dftu_series* v, const char* pattern,
                                  int32_t pattern_len) {
    std::string_view pat(pattern, static_cast<std::size_t>(pattern_len));
    std::vector<GlobToken> toks = classify_like(pat);

    // Classify once, then dispatch to the cheapest matcher. A run of AnyRun/
    // AnyOne only through the affixes hits equality / affix / substring; any
    // interior wildcard falls to the general two-pointer glob.
    std::size_t n_run = 0, n_one = 0;
    for (const GlobToken& t : toks) {
        if (t.kind == GlobKind::AnyRun) ++n_run;
        if (t.kind == GlobKind::AnyOne) ++n_one;
    }

    auto literal_of = [&](std::size_t begin, std::size_t end) {
        std::string out;
        for (std::size_t i = begin; i < end; ++i) out.push_back(toks[i].ch);
        return out;
    };

    if (n_one == 0 && n_run == 0) {
        std::string lit = literal_of(0, toks.size());
        return string_predicate(v,
                                [lit](std::string_view s) { return s == lit; });
    }
    if (n_one == 0 && n_run == 1) {
        const bool lead =
            !toks.empty() && toks.front().kind == GlobKind::AnyRun;
        const bool trail =
            !toks.empty() && toks.back().kind == GlobKind::AnyRun;
        if (trail && !lead) {  // "text%" -> starts_with
            std::string pre = literal_of(0, toks.size() - 1);
            return string_predicate(
                v, [pre](std::string_view s) { return s.starts_with(pre); });
        }
        if (lead && !trail) {  // "%text" -> ends_with
            std::string suf = literal_of(1, toks.size());
            return string_predicate(
                v, [suf](std::string_view s) { return s.ends_with(suf); });
        }
    }
    if (n_one == 0 && n_run == 2 && toks.size() >= 2 &&
        toks.front().kind == GlobKind::AnyRun &&
        toks.back().kind == GlobKind::AnyRun) {  // "%text%" -> contains
        std::string mid = literal_of(1, toks.size() - 1);
        return string_predicate(v, [mid](std::string_view s) {
            return substr_find(s.data(), static_cast<std::int64_t>(s.size()),
                               mid.data(),
                               static_cast<std::int64_t>(mid.size())) >= 0;
        });
    }
    return string_predicate(v, [toks = std::move(toks)](std::string_view s) {
        return glob_match(toks, s);
    });
}

dftu_series* dftu_series_str_len_bytes(const dftu_series* v) {
    RowReader r(v);
    if (!r.ok()) return nullptr;
    // FLAT: adjacent int32 offset difference, vectorized; DICTIONARY: scalar.
    if (v->encoding == Encoding::Flat && v->offsets) {
        std::vector<std::int64_t> vals(static_cast<std::size_t>(v->length), 0);
        if (v->length > 0)
            len_bytes_simd(
                reinterpret_cast<const std::int32_t*>(v->offsets->data()),
                vals.data(), static_cast<std::size_t>(v->length));
        // Null rows keep 0 length regardless; validity marks them null.
        return make_i64(v, vals);
    }
    return int_transform(v, [](std::string_view s) {
        return static_cast<std::int64_t>(s.size());
    });
}

dftu_series* dftu_series_str_len_chars(const dftu_series* v) {
    // FLAT: count non-continuation bytes per row with the vectorized scan;
    // DICTIONARY and other encodings take the scalar per-row path.
    if (v->encoding == Encoding::Flat && v->offsets && v->data) {
        const std::int32_t* off =
            reinterpret_cast<const std::int32_t*>(v->offsets->data());
        const std::uint8_t* data = v->data->data();
        std::vector<std::int64_t> vals(static_cast<std::size_t>(v->length), 0);
        for (std::int64_t i = 0; i < v->length; ++i)
            vals[static_cast<std::size_t>(i)] = count_char_starts(
                data + off[i], static_cast<std::size_t>(off[i + 1] - off[i]));
        return make_i64(v, vals);
    }
    return int_transform(v, [](std::string_view s) {
        std::int64_t count = 0;
        for (unsigned char c : s)
            if ((c & 0xC0) != 0x80) ++count;  // non-continuation byte
        return count;
    });
}

dftu_series* dftu_series_str_find(const dftu_series* v, const char* needle,
                                  int32_t needle_len) {
    std::string_view n(needle, static_cast<std::size_t>(needle_len));
    return int_transform(v, [n](std::string_view s) {
        return substr_find(s.data(), static_cast<std::int64_t>(s.size()),
                           n.data(), static_cast<std::int64_t>(n.size()));
    });
}

dftu_series* dftu_series_to_lowercase(const dftu_series* v) {
    return ascii_fold(v, 'A', 'Z', 32);
}

dftu_series* dftu_series_to_uppercase(const dftu_series* v) {
    return ascii_fold(v, 'a', 'z', -32);
}

dftu_series* dftu_series_str_strip(const dftu_series* v) {
    return string_transform(v, [](std::string_view s) {
        return std::string(rstrip_ws(lstrip_ws(s)));
    });
}
dftu_series* dftu_series_str_lstrip(const dftu_series* v) {
    return string_transform(
        v, [](std::string_view s) { return std::string(lstrip_ws(s)); });
}
dftu_series* dftu_series_str_rstrip(const dftu_series* v) {
    return string_transform(
        v, [](std::string_view s) { return std::string(rstrip_ws(s)); });
}

dftu_series* dftu_series_str_replace(const dftu_series* v, const char* pat,
                                     int32_t pat_len, const char* repl,
                                     int32_t repl_len) {
    std::string_view p(pat, static_cast<std::size_t>(pat_len));
    std::string_view r(repl, static_cast<std::size_t>(repl_len));
    return string_transform(v, [p, r](std::string_view s) {
        return replace_literal(s, p, r, false);
    });
}
dftu_series* dftu_series_str_replace_all(const dftu_series* v, const char* pat,
                                         int32_t pat_len, const char* repl,
                                         int32_t repl_len) {
    std::string_view p(pat, static_cast<std::size_t>(pat_len));
    std::string_view r(repl, static_cast<std::size_t>(repl_len));
    return string_transform(v, [p, r](std::string_view s) {
        return replace_literal(s, p, r, true);
    });
}

dftu_series* dftu_series_str_slice(const dftu_series* v, int64_t start,
                                   int64_t length) {
    return string_transform(v, [start, length](std::string_view s) {
        return slice_bytes(s, start, length);
    });
}

dftu_series* dftu_series_str_pad_start(const dftu_series* v, int64_t width,
                                       char fill) {
    return string_transform(v, [width, fill](std::string_view s) {
        return pad(s, width, fill, true);
    });
}
dftu_series* dftu_series_str_pad_end(const dftu_series* v, int64_t width,
                                     char fill) {
    return string_transform(v, [width, fill](std::string_view s) {
        return pad(s, width, fill, false);
    });
}
dftu_series* dftu_series_str_zfill(const dftu_series* v, int64_t width) {
    return string_transform(
        v, [width](std::string_view s) { return zfill(s, width); });
}

dftu_series* dftu_series_str_split(const dftu_series* v, const char* sep,
                                   int32_t sep_len) {
    RowReader r(v);
    if (!r.ok()) return nullptr;
    std::string_view s_sep(sep, static_cast<std::size_t>(sep_len));

    std::vector<std::string> all_parts;
    std::vector<std::int32_t> list_off(static_cast<std::size_t>(v->length + 1),
                                       0);
    for (std::int64_t i = 0; i < v->length; ++i) {
        if (!r.is_null(i)) {
            std::string_view row = r.at(i);
            if (s_sep.empty()) {
                // No well-defined split point: one part, the whole string.
                all_parts.emplace_back(row);
            } else {
                std::size_t pos = 0;
                while (true) {
                    std::size_t hit = row.find(s_sep, pos);
                    if (hit == std::string_view::npos) {
                        all_parts.emplace_back(row.substr(pos));
                        break;
                    }
                    all_parts.emplace_back(row.substr(pos, hit - pos));
                    pos = hit + s_sep.size();
                }
            }
        }
        list_off[static_cast<std::size_t>(i + 1)] =
            static_cast<std::int32_t>(all_parts.size());
    }

    // Flat String child of every part across rows.
    auto* child = new dftu_series();
    child->type = TypeId::String;
    child->encoding = Encoding::Flat;
    child->length = static_cast<std::int64_t>(all_parts.size());
    std::size_t coff_bytes =
        static_cast<std::size_t>(all_parts.size() + 1) * sizeof(std::int32_t);
    child->offsets = Buffer::allocate(coff_bytes);
    std::size_t ctotal = 0;
    for (const std::string& p : all_parts) ctotal += p.size();
    child->data = Buffer::allocate(ctotal);
    std::int32_t* cod = reinterpret_cast<std::int32_t*>(child->offsets->data());
    char* cbd =
        ctotal != 0 ? reinterpret_cast<char*>(child->data->data()) : nullptr;
    std::int32_t cpos = 0;
    cod[0] = 0;
    for (std::size_t k = 0; k < all_parts.size(); ++k) {
        const std::string& p = all_parts[k];
        if (!p.empty()) std::memcpy(cbd + cpos, p.data(), p.size());
        cpos += static_cast<std::int32_t>(p.size());
        cod[k + 1] = cpos;
    }

    auto* out = new dftu_series();
    out->type = TypeId::List;
    out->encoding = Encoding::Flat;
    out->length = v->length;
    out->null_count = v->null_count;
    out->validity = v->validity;  // a null input row stays a null list
    std::size_t loff_bytes =
        static_cast<std::size_t>(v->length + 1) * sizeof(std::int32_t);
    out->offsets = Buffer::allocate(loff_bytes);
    std::memcpy(out->offsets->data(), list_off.data(), loff_bytes);
    out->child = std::shared_ptr<dftu_series>(child);
    return out;
}
