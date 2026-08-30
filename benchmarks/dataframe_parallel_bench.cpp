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
#include <dftracer/utils/dataframe/kernels/stats.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/dataframe/parallel.h>
#include <dftracer/utils/dataframe/series.h>

#include <dftracer/utils/dataframe/batch_ops.h>

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

    // unique: parallel row-key gen + radix-partitioned lock-free dedup.
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
    DataFrame uq_serial_out = df.lazy().unique().collect();
    install_runtime_parallel_backend();
    const double uq_par = time_unique(3);
    DataFrame uq_par_out = df.lazy().unique().collect();
    bool uq_ok = uq_serial_out.num_rows() == uq_par_out.num_rows();
    {
        const std::int64_t* uv = uq_serial_out.column("v").data<std::int64_t>();
        const std::int64_t* pv = uq_par_out.column("v").data<std::int64_t>();
        for (std::int64_t i = 0; uq_ok && i < uq_serial_out.num_rows(); ++i)
            uq_ok = uv[i] == pv[i];
    }
    std::printf("unique (all-distinct): serial %8.2f ms | runtime %8.2f ms "
                "(%.2fx) correctness: %s\n",
                uq_ser, uq_par, uq_ser / uq_par, uq_ok ? "OK" : "MISMATCH");
    ok = ok && uq_ok;

    // Duplicate-heavy scenario (single low-cardinality column): the case
    // is_duplicated / is_unique / unique are actually used for, and where the
    // radix-partitioned dedup / group_agg-count reuse pays off.
    DataFrame dup_df;
    dup_df.names = {"k"};
    dup_df.columns.push_back(Series::flat_i64(k.data(), rows));

    auto bit_at = [](const Series& s, std::int64_t idx) -> int {
        return (s.data<std::uint8_t>()[idx >> 3] >> (idx & 7)) & 1;
    };

    auto time_unique_dup = [&](int reps) {
        double best = 1e300;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            DataFrame out = dup_df.lazy().unique().collect();
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(out.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double uqd_ser = time_unique_dup(3);
    DataFrame uqd_serial_out = dup_df.lazy().unique().collect();
    install_runtime_parallel_backend();
    const double uqd_par = time_unique_dup(3);
    DataFrame uqd_par_out = dup_df.lazy().unique().collect();
    bool uqd_ok = uqd_serial_out.num_rows() == uqd_par_out.num_rows();
    {
        const std::int64_t* uv = uqd_serial_out.column("k").data<std::int64_t>();
        const std::int64_t* pv = uqd_par_out.column("k").data<std::int64_t>();
        for (std::int64_t i = 0; uqd_ok && i < uqd_serial_out.num_rows(); ++i)
            uqd_ok = uv[i] == pv[i];
    }
    std::printf("unique (dup-heavy, %lld groups): serial %8.2f ms | runtime "
                "%8.2f ms (%.2fx) correctness: %s\n",
                static_cast<long long>(groups), uqd_ser, uqd_par,
                uqd_ser / uqd_par, uqd_ok ? "OK" : "MISMATCH");
    ok = ok && uqd_ok;

    auto time_isdup = [&](bool want_unique, int reps) {
        double best = 1e300;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            DataFrame out = want_unique
                               ? dup_df.lazy().is_unique().collect()
                               : dup_df.lazy().is_duplicated().collect();
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(out.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };

    set_parallel_backend(nullptr, nullptr);
    const double dup_ser = time_isdup(false, 3);
    DataFrame dup_serial_out = dup_df.lazy().is_duplicated().collect();
    install_runtime_parallel_backend();
    const double dup_par = time_isdup(false, 3);
    DataFrame dup_par_out = dup_df.lazy().is_duplicated().collect();
    bool dup_ok = dup_serial_out.num_rows() == dup_par_out.num_rows();
    {
        const Series& sa = dup_serial_out.column("is_duplicated");
        const Series& sb = dup_par_out.column("is_duplicated");
        for (std::int64_t i = 0; dup_ok && i < dup_serial_out.num_rows(); ++i)
            dup_ok = bit_at(sa, i) == bit_at(sb, i);
    }
    std::printf("is_duplicated: serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                dup_ser, dup_par, dup_ser / dup_par, dup_ok ? "OK" : "MISMATCH");
    ok = ok && dup_ok;

    set_parallel_backend(nullptr, nullptr);
    const double uni_ser = time_isdup(true, 3);
    DataFrame uni_serial_out = dup_df.lazy().is_unique().collect();
    install_runtime_parallel_backend();
    const double uni_par = time_isdup(true, 3);
    DataFrame uni_par_out = dup_df.lazy().is_unique().collect();
    bool uni_ok = uni_serial_out.num_rows() == uni_par_out.num_rows();
    {
        const Series& sa = uni_serial_out.column("is_unique");
        const Series& sb = uni_par_out.column("is_unique");
        for (std::int64_t i = 0; uni_ok && i < uni_serial_out.num_rows(); ++i)
            uni_ok = bit_at(sa, i) == bit_at(sb, i);
    }
    std::printf("is_unique: serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                uni_ser, uni_par, uni_ser / uni_par, uni_ok ? "OK" : "MISMATCH");
    ok = ok && uni_ok;

    // String predicate row loop (regex / LIKE): the compute-bound per-row
    // scan parallelized on 8-row (byte) chunk boundaries in string_ops.cpp.
    const std::int64_t srows = 5'000'000;
    std::vector<std::string> str_vals(static_cast<std::size_t>(srows));
    for (std::int64_t i = 0; i < srows; ++i)
        str_vals[static_cast<std::size_t>(i)] =
            "event_" + std::to_string(i % 97) + "_tag" + std::to_string(i % 13);
    Series str_col = Series::strings(str_vals);

    auto time_regex = [&](int reps) {
        double best = 1e300;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            Series mask = str_col.str_matches("event_[0-9]+_tag1.*");
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(mask.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double re_ser = time_regex(3);
    Series re_serial_out = str_col.str_matches("event_[0-9]+_tag1.*");
    install_runtime_parallel_backend();
    const double re_par = time_regex(3);
    Series re_par_out = str_col.str_matches("event_[0-9]+_tag1.*");
    bool re_ok = re_serial_out.length() == re_par_out.length();
    for (std::int64_t i = 0; re_ok && i < re_serial_out.length(); ++i)
        re_ok = bit_at(re_serial_out, i) == bit_at(re_par_out, i);
    std::printf("str_matches (regex): serial %8.2f ms | runtime %8.2f ms "
                "(%.2fx) correctness: %s\n",
                re_ser, re_par, re_ser / re_par, re_ok ? "OK" : "MISMATCH");
    ok = ok && re_ok;

    auto time_like = [&](int reps) {
        double best = 1e300;
        for (int r = 0; r < reps; ++r) {
            const auto t0 = std::chrono::steady_clock::now();
            Series mask = str_col.str_like("event_%_tag1?");
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(mask.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double lk_ser = time_like(3);
    Series lk_serial_out = str_col.str_like("event_%_tag1?");
    install_runtime_parallel_backend();
    const double lk_par = time_like(3);
    Series lk_par_out = str_col.str_like("event_%_tag1?");
    bool lk_ok = lk_serial_out.length() == lk_par_out.length();
    for (std::int64_t i = 0; lk_ok && i < lk_serial_out.length(); ++i)
        lk_ok = bit_at(lk_serial_out, i) == bit_at(lk_par_out, i);
    std::printf("str_like (glob): serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                lk_ser, lk_par, lk_ser / lk_par, lk_ok ? "OK" : "MISMATCH");
    ok = ok && lk_ok;

    // quantile/median: the full VQSort over a wide numeric column.
    Series qcol = df.column("s");  // scattered i64 -> cast to f64 internally
    auto time_quantile = [&](int reps) {
        double best = 1e300;
        double r = 0.0;
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            r = quantile(qcol, 0.5);
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(r);
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return best;
    };
    set_parallel_backend(nullptr, nullptr);
    const double q_ser = time_quantile(5);
    const double q_ser_val = quantile(qcol, 0.5);
    install_runtime_parallel_backend();
    const double q_par = time_quantile(5);
    const double q_par_val = quantile(qcol, 0.5);
    bool q_ok = q_ser_val == q_par_val;
    std::printf("quantile(0.5): serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                q_ser, q_par, q_ser / q_par, q_ok ? "OK" : "MISMATCH");
    ok = ok && q_ok;

    // rolling_var: O(n*window) with each row's window independent.
    const std::int64_t rwin = 64;
    auto time_rolling_var = [&](int reps) {
        double best = 1e300;
        Series r;
        for (int i = 0; i < reps; ++i) {
            const auto t0 = std::chrono::steady_clock::now();
            r = qcol.rolling_var(rwin);
            const auto t1 = std::chrono::steady_clock::now();
            do_not_optimize(r.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(t1 - t0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    double rv_ser = 1e300;
    {
        const auto t0 = std::chrono::steady_clock::now();
        Series r = qcol.rolling_var(rwin);
        const auto t1 = std::chrono::steady_clock::now();
        rv_ser = std::chrono::duration<double, std::milli>(t1 - t0).count();
        do_not_optimize(r.length());
    }
    Series rv_serial_out = time_rolling_var(3);
    install_runtime_parallel_backend();
    Series rv_par_out = time_rolling_var(3);
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 3; ++i) {
        Series r = qcol.rolling_var(rwin);
        do_not_optimize(r.length());
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double rv_par =
        std::chrono::duration<double, std::milli>(t1 - t0).count() / 3.0;
    bool rv_ok = rv_serial_out.length() == rv_par_out.length();
    {
        const double* a2 = rv_serial_out.data<double>();
        const double* b2 = rv_par_out.data<double>();
        for (std::int64_t i = 0; rv_ok && i < rv_serial_out.length(); ++i) {
            const bool na = rv_serial_out.is_null(i);
            const bool nb = rv_par_out.is_null(i);
            if (na != nb) {
                rv_ok = false;
            } else if (!na) {
                rv_ok = std::abs(a2[i] - b2[i]) < 1e-6 * (1.0 + std::abs(a2[i]));
            }
        }
    }
    std::printf("rolling_var(w=%lld): serial %8.2f ms | runtime %8.2f ms "
                "(%.2fx) correctness: %s\n",
                static_cast<long long>(rwin), rv_ser, rv_par, rv_ser / rv_par,
                rv_ok ? "OK" : "MISMATCH");
    ok = ok && rv_ok;

    // topk_indices (via top_k): SIMD partial sort of a wide numeric column.
    const std::int64_t topk_k = 1000;
    auto time_topk = [&](int reps) {
        double best = 1e300;
        Series r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = qcol.top_k(topk_k);
            const auto tt1 = std::chrono::steady_clock::now();
            do_not_optimize(r.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    Series tk_serial_out = time_topk(3);
    double tk_ser = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = qcol.top_k(topk_k);
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        tk_ser = std::min(
            tk_ser, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    install_runtime_parallel_backend();
    Series tk_par_out = time_topk(3);
    double tk_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = qcol.top_k(topk_k);
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        tk_par = std::min(
            tk_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    bool tk_ok = tk_serial_out.length() == tk_par_out.length();
    {
        std::vector<std::int64_t> sv, pv;
        for (std::int64_t i = 0; i < tk_serial_out.length(); ++i)
            sv.push_back(tk_serial_out.data<std::int64_t>()[i]);
        for (std::int64_t i = 0; i < tk_par_out.length(); ++i)
            pv.push_back(tk_par_out.data<std::int64_t>()[i]);
        std::sort(sv.begin(), sv.end());
        std::sort(pv.begin(), pv.end());
        tk_ok = tk_ok && sv == pv;
    }
    std::printf("top_k(k=%lld): serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                static_cast<long long>(topk_k), tk_ser, tk_par, tk_ser / tk_par,
                tk_ok ? "OK" : "MISMATCH");
    ok = ok && tk_ok;

    // to_dummies: O(n*d) nested string-eq loop fixed to an O(n) bucket pass.
    // d dummy columns of n rows each, so keep both bounded (d * n int8 cells).
    const std::int64_t drows = std::min<std::int64_t>(rows, 4'000'000);
    const std::int64_t dgroups = std::min<std::int64_t>(groups, 64);
    DataFrame dummy_df;
    dummy_df.names = {"cat"};
    {
        std::vector<std::string> cats(static_cast<std::size_t>(drows));
        for (std::int64_t i = 0; i < drows; ++i)
            cats[static_cast<std::size_t>(i)] =
                "c" + std::to_string(i % dgroups);
        dummy_df.columns.push_back(Series::strings(cats));
    }
    auto time_dummies = [&](int reps) {
        double best = 1e300;
        DataFrame r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = dummy_df.to_dummies("cat");
            const auto tt1 = std::chrono::steady_clock::now();
            do_not_optimize(r.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    DataFrame td_serial_out = time_dummies(1);
    const double td_ser = [&] {
        const auto tt0 = std::chrono::steady_clock::now();
        DataFrame r = dummy_df.to_dummies("cat");
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.num_rows());
        return std::chrono::duration<double, std::milli>(tt1 - tt0).count();
    }();
    install_runtime_parallel_backend();
    DataFrame td_par_out = time_dummies(3);
    double td_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        DataFrame r = dummy_df.to_dummies("cat");
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.num_rows());
        td_par = std::min(
            td_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    bool td_ok = td_serial_out.num_rows() == td_par_out.num_rows() &&
                td_serial_out.num_columns() == td_par_out.num_columns();
    for (std::int64_t c = 1;
        td_ok && c < static_cast<std::int64_t>(td_serial_out.num_columns());
        ++c) {
        const Series& a2 = td_serial_out.columns[static_cast<std::size_t>(c)];
        const Series& b2 = td_par_out.columns[static_cast<std::size_t>(c)];
        for (std::int64_t i = 0; td_ok && i < a2.length(); ++i)
            td_ok = a2.data<std::int8_t>()[i] == b2.data<std::int8_t>()[i];
    }
    std::printf("to_dummies (%lld groups): serial %8.2f ms | runtime %8.2f ms "
                "(%.2fx) correctness: %s\n",
                static_cast<long long>(groups), td_ser, td_par, td_ser / td_par,
                td_ok ? "OK" : "MISMATCH");
    ok = ok && td_ok;

    // hash_partition: per-row multi-col hash into shared buckets.
    auto time_hp = [&](int reps) {
        double best = 1e300;
        std::vector<DataFrame> r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = hash_partition(dummy_df, {"cat"}, 64);
            const auto tt1 = std::chrono::steady_clock::now();
            std::int64_t total = 0;
            for (const auto& p : r) total += p.num_rows();
            do_not_optimize(total);
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    std::vector<DataFrame> hp_serial_out = time_hp(3);
    double hp_ser = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        std::vector<DataFrame> r = hash_partition(dummy_df, {"cat"}, 64);
        const auto tt1 = std::chrono::steady_clock::now();
        std::int64_t total = 0;
        for (const auto& p : r) total += p.num_rows();
        do_not_optimize(total);
        hp_ser = std::min(
            hp_ser, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    install_runtime_parallel_backend();
    std::vector<DataFrame> hp_par_out = time_hp(3);
    double hp_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        std::vector<DataFrame> r = hash_partition(dummy_df, {"cat"}, 64);
        const auto tt1 = std::chrono::steady_clock::now();
        std::int64_t total = 0;
        for (const auto& p : r) total += p.num_rows();
        do_not_optimize(total);
        hp_par = std::min(
            hp_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    std::int64_t hp_ser_total = 0, hp_par_total = 0;
    for (const auto& p : hp_serial_out) hp_ser_total += p.num_rows();
    for (const auto& p : hp_par_out) hp_par_total += p.num_rows();
    bool hp_ok = hp_ser_total == drows && hp_par_total == drows &&
                hp_serial_out.size() == hp_par_out.size();
    std::printf("hash_partition (64 parts): serial %8.2f ms | runtime %8.2f "
                "ms (%.2fx) correctness: %s\n",
                hp_ser, hp_par, hp_ser / hp_par, hp_ok ? "OK" : "MISMATCH");
    ok = ok && hp_ok;

    // is_in: general (string) path, no SIMD fast path.
    Series needles = Series::strings({"c1", "c5", "c9", "c13"});
    auto time_is_in = [&](int reps) {
        double best = 1e300;
        Series r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = dummy_df.column("cat").is_in(needles);
            const auto tt1 = std::chrono::steady_clock::now();
            do_not_optimize(r.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    Series iin_serial_out = time_is_in(3);
    double iin_ser = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = dummy_df.column("cat").is_in(needles);
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        iin_ser = std::min(
            iin_ser, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    install_runtime_parallel_backend();
    Series iin_par_out = time_is_in(3);
    double iin_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = dummy_df.column("cat").is_in(needles);
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        iin_par = std::min(
            iin_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    bool iin_ok = iin_serial_out.length() == iin_par_out.length();
    for (std::int64_t i = 0; iin_ok && i < iin_serial_out.length(); ++i)
        iin_ok = bit_at(iin_serial_out, i) == bit_at(iin_par_out, i);
    std::printf("is_in (string): serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                iin_ser, iin_par, iin_ser / iin_par, iin_ok ? "OK" : "MISMATCH");
    ok = ok && iin_ok;

    // sort_by_multi: std::stable_sort of indices with a multi-col comparator.
    DataFrame multi_df;
    multi_df.names = {"k", "v"};
    multi_df.columns.push_back(Series::flat_i64(k.data(), rows));
    multi_df.columns.push_back(Series::flat_i64(v.data(), rows));
    auto time_sbm = [&](int reps) {
        double best = 1e300;
        DataFrame r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = sort_by_multi(multi_df, {"k", "v"}, false);
            const auto tt1 = std::chrono::steady_clock::now();
            do_not_optimize(r.num_rows());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    DataFrame sbm_serial_out = time_sbm(3);
    double sbm_ser = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        DataFrame r = sort_by_multi(multi_df, {"k", "v"}, false);
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.num_rows());
        sbm_ser = std::min(
            sbm_ser, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    install_runtime_parallel_backend();
    DataFrame sbm_par_out = time_sbm(3);
    double sbm_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        DataFrame r = sort_by_multi(multi_df, {"k", "v"}, false);
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.num_rows());
        sbm_par = std::min(
            sbm_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    bool sbm_ok = sbm_serial_out.num_rows() == sbm_par_out.num_rows();
    {
        const std::int64_t* ka =
            sbm_serial_out.column("k").data<std::int64_t>();
        const std::int64_t* kb = sbm_par_out.column("k").data<std::int64_t>();
        const std::int64_t* va =
            sbm_serial_out.column("v").data<std::int64_t>();
        const std::int64_t* vb = sbm_par_out.column("v").data<std::int64_t>();
        for (std::int64_t i = 0; sbm_ok && i < sbm_serial_out.num_rows(); ++i)
            sbm_ok = ka[i] == kb[i] && va[i] == vb[i];
    }
    std::printf("sort_by_multi: serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                sbm_ser, sbm_par, sbm_ser / sbm_par, sbm_ok ? "OK" : "MISMATCH");
    ok = ok && sbm_ok;

    // cumsum: two-pass parallel prefix scan (dense, no nulls) over the wide
    // scattered i64 column.
    auto time_cumsum = [&](int reps) {
        double best = 1e300;
        Series r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = qcol.cumsum();
            const auto tt1 = std::chrono::steady_clock::now();
            do_not_optimize(r.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    Series cs_serial_out = time_cumsum(3);
    double cs_ser = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = qcol.cumsum();
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        cs_ser = std::min(
            cs_ser, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    install_runtime_parallel_backend();
    Series cs_par_out = time_cumsum(3);
    double cs_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = qcol.cumsum();
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        cs_par = std::min(
            cs_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    bool cs_ok = cs_serial_out.length() == cs_par_out.length();
    {
        const std::int64_t* a2 = cs_serial_out.data<std::int64_t>();
        const std::int64_t* b2 = cs_par_out.data<std::int64_t>();
        for (std::int64_t i = 0; cs_ok && i < cs_serial_out.length(); ++i)
            cs_ok = a2[i] == b2[i];
    }
    std::printf("cumsum: serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                cs_ser, cs_par, cs_ser / cs_par, cs_ok ? "OK" : "MISMATCH");
    ok = ok && cs_ok;

    // cummax: same scan, max-combine.
    auto time_cummax = [&](int reps) {
        double best = 1e300;
        Series r;
        for (int i = 0; i < reps; ++i) {
            const auto tt0 = std::chrono::steady_clock::now();
            r = qcol.cummax();
            const auto tt1 = std::chrono::steady_clock::now();
            do_not_optimize(r.length());
            best = std::min(
                best,
                std::chrono::duration<double, std::milli>(tt1 - tt0).count());
        }
        return r;
    };
    set_parallel_backend(nullptr, nullptr);
    Series cx_serial_out = time_cummax(3);
    double cx_ser = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = qcol.cummax();
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        cx_ser = std::min(
            cx_ser, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    install_runtime_parallel_backend();
    Series cx_par_out = time_cummax(3);
    double cx_par = 1e300;
    for (int i = 0; i < 3; ++i) {
        const auto tt0 = std::chrono::steady_clock::now();
        Series r = qcol.cummax();
        const auto tt1 = std::chrono::steady_clock::now();
        do_not_optimize(r.length());
        cx_par = std::min(
            cx_par, std::chrono::duration<double, std::milli>(tt1 - tt0).count());
    }
    bool cx_ok = cx_serial_out.length() == cx_par_out.length();
    {
        const std::int64_t* a2 = cx_serial_out.data<std::int64_t>();
        const std::int64_t* b2 = cx_par_out.data<std::int64_t>();
        for (std::int64_t i = 0; cx_ok && i < cx_serial_out.length(); ++i)
            cx_ok = a2[i] == b2[i];
    }
    std::printf("cummax: serial %8.2f ms | runtime %8.2f ms (%.2fx) "
                "correctness: %s\n",
                cx_ser, cx_par, cx_ser / cx_par, cx_ok ? "OK" : "MISMATCH");
    ok = ok && cx_ok;

    return ok ? 0 : 1;
}
