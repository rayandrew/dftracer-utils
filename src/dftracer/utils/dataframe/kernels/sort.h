#ifndef DFTRACER_UTILS_DATAFRAME_KERNELS_SORT_H
#define DFTRACER_UTILS_DATAFRAME_KERNELS_SORT_H

#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/parallel.h>
#include <hwy/contrib/sort/vqsort.h>

#include <algorithm>
#include <cstddef>
#include <vector>

namespace dftracer::utils::dataframe {

/// Three-way order of rows `a` and `b` of `v` (-1/0/1) on the column's own
/// domain. Only defined for an is_orderable_type() column; 0 for any other.
int compare_rows(const Series& v, std::int64_t a, std::int64_t b);

/// Stable argsort: an Int64 column of row indices ordering `v` ascending, or
/// descending when `descending`. Null rows sort last in both directions.
/// An invalid Series for a nested column, which has no order.
Series argsort(const Series& v, bool descending);

/// An Int64 column of the k row indices of the k largest (or smallest) values,
/// ordered best-first (partial sort). k is clamped to [0, length].
Series topk_indices(const Series& v, std::int64_t k, bool largest);

/// Parallel full sort of `n` values in place: SIMD-sort (VQSort) runs per chunk
/// in parallel, then a bottom-up parallel merge. Falls back to a single VQSort
/// when no backend is installed (the merge would be pure overhead) or `n` is
/// small. `less` must match VQSort's ascending order so the merge is
/// consistent, and `T` must be a type VQSort supports (built-in numeric types,
/// or a packed key/index type such as hwy::uint128_t).
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

/// Parallel partial sort: the `k` smallest of `n` values end up sorted in
/// `data[0, k)` (the rest is left unspecified), matching VQPartialSort's
/// contract. Each chunk produces its own top-k candidates in parallel (the
/// true top-k of the whole array must appear among some chunk's top-k), and a
/// final partial sort over the union of candidates picks the overall k. Falls
/// back to a single VQPartialSort when no backend is installed or `n` is
/// small.
template <class T>
void parallel_partial_sort(T* data, std::size_t n, std::size_t k) {
    constexpr std::size_t RUN = std::size_t{1} << 18;
    if (!parallel_backend_installed() || n <= RUN || k == 0) {
        hwy::VQPartialSort(data, n, k, hwy::SortAscending());
        return;
    }
    const std::size_t nruns = (n + RUN - 1) / RUN;
    std::vector<std::vector<T>> cand(nruns);
    parallel_for(static_cast<std::int64_t>(nruns), 1,
                 [&](std::int64_t c0, std::int64_t c1) {
                     for (std::int64_t c = c0; c < c1; ++c) {
                         const std::size_t lo =
                             static_cast<std::size_t>(c) * RUN;
                         const std::size_t hi = std::min(n, lo + RUN);
                         const std::size_t take = std::min(k, hi - lo);
                         hwy::VQPartialSort(data + lo, hi - lo, take,
                                            hwy::SortAscending());
                         cand[static_cast<std::size_t>(c)].assign(
                             data + lo, data + lo + take);
                     }
                 });
    std::vector<T> merged;
    merged.reserve(k * nruns);
    for (const std::vector<T>& part : cand)
        merged.insert(merged.end(), part.begin(), part.end());
    const std::size_t kk = std::min(k, merged.size());
    hwy::VQPartialSort(merged.data(), merged.size(), kk, hwy::SortAscending());
    std::copy(merged.begin(), merged.begin() + static_cast<std::ptrdiff_t>(kk),
              data);
}

}  // namespace dftracer::utils::dataframe

#endif  // DFTRACER_UTILS_DATAFRAME_KERNELS_SORT_H
