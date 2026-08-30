// Benchmark: parallel group-by via the dataframe parallel_for seam.
//
// group_agg already fans out per-chunk AggStates through parallel_for and merges
// the partials. With no backend installed it runs serial; this bench installs a
// std::thread backend and reports group_by wall-clock at 1/2/4/8 workers so the
// speedup is measured, not assumed.
//
// Built only with DFTRACER_UTILS_BUILD_BENCHMARKS=ON; run manually. Use an
// optimized build (the dev preset, -O2) - a Debug build's numbers are noise.
//
//   dataframe_parallel_bench [rows] [groups]

#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/field_stat.h>
#include <dftracer/utils/dataframe/kernels/field_stat.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/dataframe/parallel.h>
#include <dftracer/utils/dataframe/series.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>

using namespace dftracer::utils::dataframe;

namespace {

int g_threads = 1;

// Backend contract: call `body` once per grain-sized chunk (group_agg indexes
// its partials by begin/grain), distributing chunks across g_threads workers via
// an atomic counter (dynamic schedule).
void thread_backend(void*, std::int64_t n, std::int64_t grain,
                    void (*body)(void*, std::int64_t, std::int64_t),
                    void* bctx) {
    const std::int64_t nchunks = (n + grain - 1) / grain;
    std::atomic<std::int64_t> next{0};
    auto worker = [&]() {
        for (;;) {
            const std::int64_t k = next.fetch_add(1);
            if (k >= nchunks) break;
            const std::int64_t b = k * grain;
            const std::int64_t e = std::min(n, b + grain);
            body(bctx, b, e);
        }
    };
    std::vector<std::thread> pool;
    for (int t = 1; t < g_threads; ++t) pool.emplace_back(worker);
    worker();
    for (auto& th : pool) th.join();
}

template <class T>
void do_not_optimize(T&& v) {
    asm volatile("" : : "r,m"(v) : "memory");
}

double time_group_by(const DataFrame& df, const std::vector<GroupAgg>& aggs,
                     int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        DataFrame out = df.group_by("k", aggs);
        const auto t1 = std::chrono::steady_clock::now();
        do_not_optimize(out.num_rows());
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

}  // namespace

int main(int argc, char** argv) {
    const std::int64_t rows = argc > 1 ? std::atoll(argv[1]) : 20'000'000;
    const std::int64_t groups = argc > 2 ? std::atoll(argv[2]) : 1000;

    std::vector<std::int64_t> k(static_cast<std::size_t>(rows));
    std::vector<std::int64_t> v(static_cast<std::size_t>(rows));
    std::vector<std::int64_t> s(static_cast<std::size_t>(rows));
    for (std::int64_t i = 0; i < rows; ++i) {
        k[static_cast<std::size_t>(i)] = i % groups;
        v[static_cast<std::size_t>(i)] = i;
        s[static_cast<std::size_t>(i)] = static_cast<std::int64_t>(
            (static_cast<std::uint64_t>(i) * 2654435761ULL) % 1000003ULL);
    }
    DataFrame df;
    df.names = {"k", "v", "s"};
    df.columns.push_back(Series::flat_i64(k.data(), rows));
    df.columns.push_back(Series::flat_i64(v.data(), rows));
    df.columns.push_back(Series::flat_i64(s.data(), rows));
    std::vector<GroupAgg> aggs{{Agg::Sum, "v", "sum", 0.0},
                               {Agg::Mean, "v", "mean", 0.0},
                               {Agg::Count, "", "cnt", 0.0}};

    std::printf("group_by: %lld rows, %lld groups, sum+mean+count\n",
                static_cast<long long>(rows), static_cast<long long>(groups));

    // Baselines for ops that are already SIMD (sort = Highway VQSort) or
    // bandwidth-bound, to judge whether parallelizing them is worth it.
    {
        set_parallel_backend(nullptr, nullptr);
        double s_sort = 1e300, s_desc = 1e300;
        for (int r = 0; r < 3; ++r) {
            auto t0 = std::chrono::steady_clock::now();
            DataFrame o = df.sort_by("v");
            auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(o.num_rows());
            s_sort = std::min(
                s_sort,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
            t0 = std::chrono::steady_clock::now();
            DataFrame d = df.describe();
            t1 = std::chrono::steady_clock::now();
            do_not_optimize(d.num_rows());
            s_desc = std::min(
                s_desc,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        std::printf("baselines (serial): sort_by %.2f ms | describe %.2f ms\n",
                    s_sort, s_desc);
    }

    set_parallel_backend(nullptr, nullptr);  // serial baseline
    const double serial = time_group_by(df, aggs, 5);
    std::printf("  serial          : %8.2f ms  (1.00x)\n", serial);

    set_parallel_backend(thread_backend, nullptr);
    for (int t : {1, 2, 4, 8}) {
        g_threads = t;
        const double ms = time_group_by(df, aggs, 5);
        std::printf("  parallel %2d thr : %8.2f ms  (%.2fx)\n", t, ms,
                    serial / ms);
    }

    // The real installer: fan out through the core coroutine runtime.
    install_runtime_parallel_backend();
    const double rt = time_group_by(df, aggs, 5);
    std::printf("  runtime backend : %8.2f ms  (%.2fx)\n", rt, serial / rt);

    // Lazy group_by (GroupByCursor bounded parallel sink) over the same data.
    auto time_lazy = [&](int reps) {
        double best = 1e300;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            DataFrame out = df.lazy().group_by("k", aggs).collect();
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(out.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double lz_ser = time_lazy(5);
    install_runtime_parallel_backend();
    const double lz_par = time_lazy(5);
    std::printf("lazy group_by: serial %8.2f ms | runtime %8.2f ms (%.2fx)\n",
                lz_ser, lz_par, lz_ser / lz_par);

    // Consolidation overhead probe: route through lazy with ONE giant morsel
    // (no chunking/concat). Shows the pure cursor-wrapping cost vs eager.
    auto one_morsel = [&]() {
        double best = 1e300;
        for (int r = 0; r < 5; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            DataFrame out = df.lazy().group_by("k", aggs).collect(rows);
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(out.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    const double lz_one = one_morsel();
    std::printf("lazy group_by (1 morsel, runtime): %8.2f ms\n", lz_one);

    // sort_by with the parallel gather (take) enabled.
    auto time_sort = [&]() {
        double best = 1e300;
        for (int r = 0; r < 3; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            DataFrame out = df.sort_by("s");  // scattered key
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(out.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double sort_ser = time_sort();
    DataFrame sorted_serial = df.sort_by("s");
    install_runtime_parallel_backend();
    const double sort_par = time_sort();
    DataFrame sorted_par = df.sort_by("s");
    std::printf("sort_by: serial %8.2f ms | runtime %8.2f ms (%.2fx)\n",
                sort_ser, sort_par, sort_ser / sort_par);

    // Correctness: parallel sort must equal serial sort element-wise.
    const std::int64_t* a = sorted_serial.column("s").data<std::int64_t>();
    const std::int64_t* b = sorted_par.column("s").data<std::int64_t>();
    const std::int64_t* av = sorted_serial.column("v").data<std::int64_t>();
    const std::int64_t* bv = sorted_par.column("v").data<std::int64_t>();
    bool ok = sorted_serial.num_rows() == sorted_par.num_rows();
    for (std::int64_t i = 0; ok && i < sorted_par.num_rows(); ++i)
        ok = a[i] == b[i] && av[i] == bv[i];
    std::printf("sort_by correctness (parallel == serial): %s\n",
                ok ? "OK" : "MISMATCH");

    // Parallel reduction atom: field_stat_reduce over a 20M column.
    auto time_stat = [&]() {
        double best = 1e300;
        FieldStat fs;
        for (int r = 0; r < 5; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            fs = field_stat_reduce(df.column("v"), 0, rows);
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(fs.sum);
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double st_ser = time_stat();
    install_runtime_parallel_backend();
    const double st_par = time_stat();
    std::printf("field_stat_reduce: serial %6.2f ms | runtime %6.2f ms (%.2fx)\n",
                st_ser, st_par, st_ser / st_par);

    // unique: parallel row-key gen + serial dedup (all rows distinct here).
    auto time_unique = [&](int reps) {
        double best = 1e300;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            DataFrame out = df.lazy().unique().collect();
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(out.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double uq_ser = time_unique(3);
    install_runtime_parallel_backend();
    const double uq_par = time_unique(3);
    std::printf("unique: serial %8.2f ms | runtime %8.2f ms (%.2fx)\n", uq_ser,
                uq_par, uq_ser / uq_par);
    return ok ? 0 : 1;
}
