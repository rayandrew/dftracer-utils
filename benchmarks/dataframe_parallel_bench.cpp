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
    for (std::int64_t i = 0; i < rows; ++i) {
        k[static_cast<std::size_t>(i)] = i % groups;
        v[static_cast<std::size_t>(i)] = i;
    }
    DataFrame df;
    df.names = {"k", "v"};
    df.columns.push_back(Series::flat_i64(k.data(), rows));
    df.columns.push_back(Series::flat_i64(v.data(), rows));
    std::vector<GroupAgg> aggs{{Agg::Sum, "v", "sum", 0.0},
                               {Agg::Mean, "v", "mean", 0.0},
                               {Agg::Count, "", "cnt", 0.0}};

    std::printf("group_by: %lld rows, %lld groups, sum+mean+count\n",
                static_cast<long long>(rows), static_cast<long long>(groups));

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
    return 0;
}
