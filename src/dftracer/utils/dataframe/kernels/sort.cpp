#include <dftracer/utils/dataframe/kernels/sort.h>
#include <dftracer/utils/dataframe/parallel.h>
#include <hwy/contrib/sort/vqsort.h>

#include <algorithm>
#include <bit>
#include <cstdint>
#include <limits>
#include <numeric>
#include <string_view>
#include <vector>

namespace dftracer::utils::dataframe {

namespace {

// Scalar fallback for strings, nested types, and columns with nulls.
template <class T>
int cmp_numeric(const Series& v, std::int64_t a, std::int64_t b) {
    const T* d = v.data<T>();
    if (d[a] < d[b]) return -1;
    if (d[a] > d[b]) return 1;
    return 0;
}

int cmp_bool(const Series& v, std::int64_t a, std::int64_t b) {
    const std::uint8_t* bits = v.data<std::uint8_t>();
    int xa = (bits[a >> 3] >> (a & 7)) & 1;
    int xb = (bits[b >> 3] >> (b & 7)) & 1;
    return xa - xb;
}

int cmp_row(const Series& v, std::int64_t a, std::int64_t b) {
    switch (v.type()) {
        case TypeId::Bool:
            return cmp_bool(v, a, b);
        case TypeId::Int8:
            return cmp_numeric<std::int8_t>(v, a, b);
        case TypeId::Int16:
            return cmp_numeric<std::int16_t>(v, a, b);
        case TypeId::Int32:
            return cmp_numeric<std::int32_t>(v, a, b);
        case TypeId::Int64:
            return cmp_numeric<std::int64_t>(v, a, b);
        case TypeId::Uint8:
            return cmp_numeric<std::uint8_t>(v, a, b);
        case TypeId::Uint16:
            return cmp_numeric<std::uint16_t>(v, a, b);
        case TypeId::Uint32:
            return cmp_numeric<std::uint32_t>(v, a, b);
        case TypeId::Uint64:
            return cmp_numeric<std::uint64_t>(v, a, b);
        case TypeId::Float32:
            return cmp_numeric<float>(v, a, b);
        case TypeId::Float64:
            return cmp_numeric<double>(v, a, b);
        case TypeId::String: {
            std::string_view sa = v.string_at(a);
            std::string_view sb = v.string_at(b);
            int c = sa.compare(sb);
            return c < 0 ? -1 : (c > 0 ? 1 : 0);
        }
        default:
            return 0;
    }
}

Series argsort_scalar(const Series& v, bool descending) {
    const std::int64_t n = v.length();
    std::vector<std::int64_t> idx(static_cast<std::size_t>(n));
    std::iota(idx.begin(), idx.end(), 0);
    const bool has_nulls = v.null_count() > 0;
    std::stable_sort(idx.begin(), idx.end(),
                     [&](std::int64_t a, std::int64_t b) {
                         if (has_nulls) {
                             bool na = v.is_null(a), nb = v.is_null(b);
                             if (na || nb) return nb && !na;  // nulls last
                         }
                         int c = cmp_row(v, a, b);
                         return descending ? c > 0 : c < 0;
                     });
    return Series::flat_i64(idx.data(), n);
}

// SIMD fast path. Map each value to an order-preserving unsigned key (signed
// ints flip the sign bit; floats flip the sign bit, or all bits when negative),
// pack (key << width | row-index) into a wider int, and VQSort: sorting by the
// full word orders by key with the index breaking ties, so it is stable.
// Descending inverts the key bits and keeps the index ascending, stable both
// ways.

std::uint32_t sortable_f32(float f) {
    std::uint32_t u = std::bit_cast<std::uint32_t>(f);
    return u ^ ((u >> 31) ? 0xFFFFFFFFu : 0x80000000u);
}
std::uint64_t sortable_f64(double f) {
    std::uint64_t u = std::bit_cast<std::uint64_t>(f);
    return u ^ ((u >> 63) ? ~0ULL : 0x8000000000000000ULL);
}

// Parallel full sort of `n` packed keys: SIMD-sort runs in parallel, then a
// bottom-up parallel merge. Falls back to a single VQSort when no backend is
// installed (the merge would be pure overhead) or n is small. `less` must match
// VQSort's ascending order so the merge is consistent.
template <class T, class Less>
void parallel_packed_sort(T* data, std::size_t n, Less less) {
    constexpr std::size_t RUN = std::size_t{1} << 18;
    if (!parallel_backend_installed() || n <= RUN) {
        hwy::VQSort(data, n, hwy::SortAscending());
        return;
    }
    const std::size_t nruns = (n + RUN - 1) / RUN;
    parallel_for(static_cast<std::int64_t>(nruns), 1,
                 [&](std::int64_t c0, std::int64_t c1) {
                     for (std::int64_t c = c0; c < c1; ++c) {
                         const std::size_t lo =
                             static_cast<std::size_t>(c) * RUN;
                         const std::size_t hi = std::min(n, lo + RUN);
                         hwy::VQSort(data + lo, hi - lo, hwy::SortAscending());
                     }
                 });
    std::vector<T> scratch(n);
    T* src = data;
    T* dst = scratch.data();
    for (std::size_t width = RUN; width < n; width *= 2) {
        const std::size_t step = width * 2;
        const std::size_t npairs = (n + step - 1) / step;
        parallel_for(static_cast<std::int64_t>(npairs), 1,
                     [&](std::int64_t p0, std::int64_t p1) {
                         for (std::int64_t p = p0; p < p1; ++p) {
                             const std::size_t lo =
                                 static_cast<std::size_t>(p) * step;
                             const std::size_t mid = std::min(n, lo + width);
                             const std::size_t hi = std::min(n, lo + step);
                             std::merge(src + lo, src + mid, src + mid,
                                        src + hi, dst + lo, less);
                         }
                     });
        std::swap(src, dst);
    }
    if (src != data) std::copy(src, src + n, data);
}

// Pack 32-bit keys with the index into uint64 and (partial-)sort. `k < 0` sorts
// fully; otherwise only the first k indices are produced (VQPartialSort).
template <class KeyFn>
Series argsort_pack32(std::int64_t n, bool descending, std::int64_t k,
                      KeyFn key_of) {
    std::vector<std::uint64_t> packed(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint32_t key = key_of(i);
        if (descending) key = ~key;
        packed[static_cast<std::size_t>(i)] =
            (static_cast<std::uint64_t>(key) << 32) |
            static_cast<std::uint32_t>(i);
    }
    std::int64_t out_n = n;
    if (k >= 0 && k < n) {
        hwy::VQPartialSort(packed.data(), static_cast<std::size_t>(n),
                           static_cast<std::size_t>(k), hwy::SortAscending());
        out_n = k;
    } else {
        parallel_packed_sort(
            packed.data(), static_cast<std::size_t>(n),
            [](std::uint64_t a, std::uint64_t b) { return a < b; });
    }
    std::vector<std::int64_t> idx(static_cast<std::size_t>(out_n));
    for (std::int64_t i = 0; i < out_n; ++i)
        idx[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(
            static_cast<std::uint32_t>(packed[static_cast<std::size_t>(i)]));
    return Series::flat_i64(idx.data(), out_n);
}

// Pack 64-bit keys with the index into uint128 (hi = key, lo = index).
template <class KeyFn>
Series argsort_pack64(std::int64_t n, bool descending, std::int64_t k,
                      KeyFn key_of) {
    std::vector<hwy::uint128_t> packed(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i) {
        std::uint64_t key = key_of(i);
        if (descending) key = ~key;
        packed[static_cast<std::size_t>(i)].hi = key;
        packed[static_cast<std::size_t>(i)].lo = static_cast<std::uint64_t>(i);
    }
    std::int64_t out_n = n;
    if (k >= 0 && k < n) {
        hwy::VQPartialSort(packed.data(), static_cast<std::size_t>(n),
                           static_cast<std::size_t>(k), hwy::SortAscending());
        out_n = k;
    } else {
        parallel_packed_sort(
            packed.data(), static_cast<std::size_t>(n),
            [](const hwy::uint128_t& a, const hwy::uint128_t& b) {
                return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo);
            });
    }
    std::vector<std::int64_t> idx(static_cast<std::size_t>(out_n));
    for (std::int64_t i = 0; i < out_n; ++i)
        idx[static_cast<std::size_t>(i)] =
            static_cast<std::int64_t>(packed[static_cast<std::size_t>(i)].lo);
    return Series::flat_i64(idx.data(), out_n);
}

// Try the SIMD path; returns an invalid Series when the type/shape is not
// eligible (variable-width, nested, has nulls, or too many rows to pack). `k`
// is passed through: `k < 0` = full sort, else a partial sort of the first k.
Series argsort_simd(const Series& v, bool descending, std::int64_t k) {
    const std::int64_t n = v.length();
    if (v.null_count() > 0) return Series{};
    const bool wide_index = n > std::numeric_limits<std::uint32_t>::max();

    switch (v.type()) {
        case TypeId::Bool: {
            const std::uint8_t* b = v.data<std::uint8_t>();
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint32_t>((b[i >> 3] >> (i & 7)) & 1);
            });
        }
        case TypeId::Int8:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint32_t>(
                    static_cast<std::uint8_t>(v.data<std::int8_t>()[i]) ^
                    0x80u);
            });
        case TypeId::Int16:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint32_t>(
                    static_cast<std::uint16_t>(v.data<std::int16_t>()[i]) ^
                    0x8000u);
            });
        case TypeId::Int32:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint32_t>(v.data<std::int32_t>()[i]) ^
                       0x80000000u;
            });
        case TypeId::Uint8:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint32_t>(v.data<std::uint8_t>()[i]);
            });
        case TypeId::Uint16:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint32_t>(v.data<std::uint16_t>()[i]);
            });
        case TypeId::Uint32:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return v.data<std::uint32_t>()[i];
            });
        case TypeId::Float32:
            if (wide_index) return Series{};
            return argsort_pack32(n, descending, k, [&](std::int64_t i) {
                return sortable_f32(v.data<float>()[i]);
            });
        case TypeId::Int64:
            return argsort_pack64(n, descending, k, [&](std::int64_t i) {
                return static_cast<std::uint64_t>(v.data<std::int64_t>()[i]) ^
                       0x8000000000000000ULL;
            });
        case TypeId::Uint64:
            return argsort_pack64(n, descending, k, [&](std::int64_t i) {
                return v.data<std::uint64_t>()[i];
            });
        case TypeId::Float64:
            return argsort_pack64(n, descending, k, [&](std::int64_t i) {
                return sortable_f64(v.data<double>()[i]);
            });
        default:
            return Series{};  // String / nested: scalar
    }
}

}  // namespace

Series argsort(const Series& v, bool descending) {
    Series simd = argsort_simd(v, descending, -1);
    if (simd.valid()) return simd;
    return argsort_scalar(v, descending);
}

Series topk_indices(const Series& v, std::int64_t k, bool largest) {
    const std::int64_t n = v.length();
    k = k < 0 ? 0 : (k > n ? n : k);
    // largest -> descending key order; the first k are the k largest.
    Series simd = argsort_simd(v, largest, k);
    if (simd.valid()) return simd;
    Series full = argsort_scalar(v, largest);  // strings / nulls / nested
    const std::int64_t* idx = full.data<std::int64_t>();
    std::vector<std::int64_t> head(idx, idx + k);
    return Series::flat_i64(head.data(), k);
}

}  // namespace dftracer::utils::dataframe

extern "C" dftu_series* dftu_series_argsort(const dftu_series* v,
                                            int32_t descending) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series col{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series out =
        dftracer::utils::dataframe::argsort(col, descending != 0);
    col.release();  // do not free the borrowed input
    return out.release();
}
