#include <dftracer/utils/dataframe/abi.h>                   // dftu_series_data
#include <dftracer/utils/dataframe/field_stat.h>            // FieldStat
#include <dftracer/utils/dataframe/internal/column_read.h>  // read_f64
#include <dftracer/utils/dataframe/kernels/cast.h>          // cast_simd
#include <dftracer/utils/dataframe/kernels/field_stat.h>    // field_stat_reduce
#include <dftracer/utils/dataframe/kernels/filter.h>        // take
#include <dftracer/utils/dataframe/kernels/sort.h>  // argsort, parallel_packed_sort
#include <dftracer/utils/dataframe/kernels/stats.h>
#include <dftracer/utils/dataframe/parallel.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <deque>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace dftracer::utils::dataframe {

namespace {

// Non-null values as doubles (moment/quantile stats are numeric). The dense
// (no-null, FLAT) case is a straight widen-to-double over a contiguous buffer,
// so it goes through the SIMD cast fast path (memcpy for Float64, one
// Convert/Promote for Float32/Int32/Int64); only the null-skipping compaction
// path stays scalar.
std::vector<double> nonnull_values(const Series& v) {
    const std::int64_t n = v.length();
    if (n > 0 && v.null_count() == 0 && v.encoding() == Encoding::Flat) {
        const void* src = dftu_series_data(v.handle());
        if (src) {
            std::vector<double> out(static_cast<std::size_t>(n));
            if (v.type() == TypeId::Float64) {
                std::memcpy(out.data(), src,
                            static_cast<std::size_t>(n) * sizeof(double));
            } else if (!cast_simd(static_cast<std::int32_t>(v.type()),
                                  static_cast<std::int32_t>(TypeId::Float64),
                                  src, out.data(),
                                  static_cast<std::size_t>(n))) {
                for (std::int64_t i = 0; i < n; ++i) out[i] = read_f64(v, i);
            }
            return out;
        }
    }
    const bool has_nulls = v.null_count() > 0;
    std::vector<double> out;
    out.reserve(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        if (!has_nulls || !v.is_null(i)) out.push_back(read_f64(v, i));
    return out;
}

bool equal_at(const Series& v, std::int64_t a, std::int64_t b) {
    switch (value_domain(v.type())) {
        case ValueDomain::Bytes:
            return read_bytes(v, a) == read_bytes(v, b);
        case ValueDomain::Numeric:
            return read_f64(v, a) == read_f64(v, b);
        case ValueDomain::None:
            throw std::invalid_argument(
                std::string("column type '") + type_name(v.type()) +
                "' has no comparable per-row value (nested type)");
    }
    return false;
}

// Decimal128/256 are ValueDomain::Bytes (exact compare needs the scaled
// integer) but read_f64 decodes them to a real number, so they window like
// Float16; every other Bytes/None type has no numeric magnitude to widen.
bool refuse_non_numeric(const char* op, TypeId t) {
    if (value_domain(t) == ValueDomain::Numeric || t == TypeId::Decimal128 ||
        t == TypeId::Decimal256)
        return false;
    DFTRACER_UTILS_LOG_ERROR("%s: no numeric value for type '%s'", op,
                             type_name(t));
    return true;
}

// field_stat_reduce's scalar fallback only takes the double branch for
// Float32/Float64; Float16 and Decimal128/256 fall to read_i64's default-0
// case. Fold those two through read_f64 by hand instead.
FieldStat moment_stat(const Series& v) {
    FieldStat fs;
    const std::int64_t n = v.length();
    const bool has_nulls = v.null_count() > 0;
    for (std::int64_t i = 0; i < n; ++i)
        if (!has_nulls || !v.is_null(i)) fs.add(read_f64(v, i));
    return fs;
}

FieldStat moment_stat_for(const Series& v) {
    const TypeId t = v.type();
    if (t == TypeId::Float16 || t == TypeId::Decimal128 ||
        t == TypeId::Decimal256)
        return moment_stat(v);
    return field_stat_reduce(v, 0, v.length());
}

}  // namespace

double variance(const Series& v, bool sample) {
    if (refuse_non_numeric("variance", v.type())) return 0.0;
    return moment_stat_for(v).variance(sample);
}

double stddev(const Series& v, bool sample) {
    if (refuse_non_numeric("stddev", v.type())) return 0.0;
    return moment_stat_for(v).stddev(sample);
}

double skewness(const Series& v) {
    if (refuse_non_numeric("skewness", v.type())) return 0.0;
    return moment_stat_for(v).skewness();
}

double kurtosis(const Series& v) {
    if (refuse_non_numeric("kurtosis", v.type())) return 0.0;
    return moment_stat_for(v).kurtosis();
}

double quantile(const Series& v, double q) {
    if (refuse_non_numeric("quantile", v.type())) return std::nan("");
    std::vector<double> x = nonnull_values(v);
    const std::size_t n = x.size();
    if (n == 0) return std::nan("");
    parallel_packed_sort(x.data(), n, [](double a, double b) { return a < b; });
    if (q <= 0.0) return x.front();
    if (q >= 1.0) return x.back();
    double pos = q * static_cast<double>(n - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    double frac = pos - static_cast<double>(lo);
    if (lo + 1 >= n) return x[lo];
    return x[lo] * (1.0 - frac) + x[lo + 1] * frac;
}

double median(const Series& v) { return quantile(v, 0.5); }

std::int64_t nunique(const Series& v) {
    const std::int64_t n = v.length();
    if (n == 0) return 0;
    if (refuse_nested_value("nunique", v.type())) return 0;
    Series order = argsort(v, false);  // nulls sort last
    const std::int64_t* idx = order.data<std::int64_t>();
    const bool has_nulls = v.null_count() > 0;
    std::int64_t cnt = 0, prev = -1;
    for (std::int64_t k = 0; k < n; ++k) {
        std::int64_t i = idx[k];
        if (has_nulls && v.is_null(i)) break;  // nulls are last
        if (prev < 0 || !equal_at(v, i, prev)) {
            ++cnt;
            prev = i;
        }
    }
    return cnt;
}

Series unique(const Series& v) {
    const std::int64_t n = v.length();
    if (refuse_nested_value("unique", v.type())) return Series{};
    Series order = argsort(v, false);
    const std::int64_t* idx = order.data<std::int64_t>();
    const bool has_nulls = v.null_count() > 0;
    std::vector<std::int64_t> keep;
    std::int64_t prev = -1;
    for (std::int64_t k = 0; k < n; ++k) {
        std::int64_t i = idx[k];
        if (has_nulls && v.is_null(i)) break;
        if (prev < 0 || !equal_at(v, i, prev)) {
            keep.push_back(i);
            prev = i;
        }
    }
    return take(v, keep);
}

Series rank(const Series& v, RankMethod method, bool descending) {
    const std::int64_t n = v.length();
    if (refuse_nested_value("rank", v.type())) return Series{};
    std::vector<double> ranks(static_cast<std::size_t>(n), std::nan(""));
    Series order = argsort(v, descending);  // nulls sort last
    const std::int64_t* idx = order.data<std::int64_t>();
    const bool has_nulls = v.null_count() > 0;
    std::int64_t k = 0, dense = 0;
    while (k < n) {
        std::int64_t i = idx[k];
        if (has_nulls && v.is_null(i)) break;  // nulls keep a NaN rank
        std::int64_t j = k + 1;
        while (j < n) {
            std::int64_t ij = idx[j];
            if (has_nulls && v.is_null(ij)) break;
            if (!equal_at(v, i, ij)) break;
            ++j;
        }
        ++dense;
        for (std::int64_t t = k; t < j; ++t) {
            double r = 0.0;
            switch (method) {
                case RankMethod::Ordinal:
                    r = static_cast<double>(t + 1);
                    break;
                case RankMethod::Min:
                    r = static_cast<double>(k + 1);
                    break;
                case RankMethod::Dense:
                    r = static_cast<double>(dense);
                    break;
                case RankMethod::Average:
                    r = static_cast<double>(k + 1 + j) / 2.0;
                    break;
            }
            ranks[static_cast<std::size_t>(idx[t])] = r;
        }
        k = j;
    }
    return Series::flat(TypeId::Float64, ranks.data(), n);
}

Series rolling(const Series& v, std::int64_t window, RollingOp op) {
    if (refuse_non_numeric("rolling", v.type())) return Series{};
    const std::int64_t n = v.length();
    if (window < 1) window = 1;
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>((n + 7) / 8), 0);
    // Every full window (rows [window-1, n)) is valid; set that contiguous run
    // in bulk (byte memset for the interior) instead of one bit per row.
    {
        std::int64_t a = window - 1;
        if (a < 0) a = 0;
        std::int64_t i = a;
        for (; i < n && (i & 7); ++i)
            valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
        const std::int64_t full_end = n & ~std::int64_t{7};
        if (i < full_end) {
            std::memset(&valid[static_cast<std::size_t>(i >> 3)], 0xFF,
                        static_cast<std::size_t>((full_end - i) / 8));
            i = full_end;
        }
        for (; i < n; ++i)
            valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    }

    // Widen the whole column to double once (SIMD cast for the dense case), so
    // the window loop indexes a plain array instead of paying a per-element
    // read_f64 type-dispatch twice per row.
    std::vector<double> vals;
    if (n > 0 && v.null_count() == 0 && v.encoding() == Encoding::Flat) {
        vals = nonnull_values(v);
    } else {
        vals.resize(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] = read_f64(v, i);
    }
    const double* p = vals.data();

    if (op == RollingOp::Sum || op == RollingOp::Mean) {
        double run = 0.0;
        for (std::int64_t i = 0; i < n; ++i) {
            run += p[i];
            if (i >= window) run -= p[i - window];
            if (i >= window - 1)
                out[static_cast<std::size_t>(i)] =
                    op == RollingOp::Mean ? run / static_cast<double>(window)
                                          : run;
        }
    } else {
        // Monotonic deque of indices: front holds the window extreme.
        const bool is_max = op == RollingOp::Max;
        std::deque<std::int64_t> dq;
        for (std::int64_t i = 0; i < n; ++i) {
            const double x = p[i];
            while (!dq.empty()) {
                const double b = p[dq.back()];
                if (is_max ? (b <= x) : (b >= x))
                    dq.pop_back();
                else
                    break;
            }
            dq.push_back(i);
            if (dq.front() <= i - window) dq.pop_front();
            if (i >= window - 1)
                out[static_cast<std::size_t>(i)] = p[dq.front()];
        }
    }
    return Series::flat(TypeId::Float64, out.data(), n, valid.data());
}

namespace {

// Quantile (linear interpolation between the two nearest ranks) of a scratch
// vector; the caller may leave it unsorted (this sorts in place). NaN if empty.
double quantile_sorted(std::vector<double>& x, double q) {
    const std::size_t n = x.size();
    if (n == 0) return std::nan("");
    std::sort(x.begin(), x.end());
    if (q <= 0.0) return x.front();
    if (q >= 1.0) return x.back();
    double pos = q * static_cast<double>(n - 1);
    std::size_t lo = static_cast<std::size_t>(pos);
    double frac = pos - static_cast<double>(lo);
    if (lo + 1 >= n) return x[lo];
    return x[lo] * (1.0 - frac) + x[lo + 1] * frac;
}

// Sample variance of a scratch window (divide by n-1); 0 for fewer than 2.
double window_sample_var(const std::vector<double>& x) {
    const std::size_t n = x.size();
    if (n < 2) return 0.0;
    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= static_cast<double>(n);
    double s2 = 0.0;
    for (double v : x) {
        double d = v - mean;
        s2 += d * d;
    }
    return s2 / static_cast<double>(n - 1);
}

// Materialize each trailing `window` as a scratch vector and reduce it with
// `agg`; the first window-1 rows are null. Mirrors rolling's windowing (no
// per-window null-skip: every cell is read as a double). Rows are independent
// (each reads only its own trailing window), so above a size threshold and
// with a backend installed the row range fans out through parallel_for; the
// valid mask is filled in bulk first so worker threads only ever touch their
// own disjoint `out` slots.
template <class Agg>
Series rolling_window(const Series& v, std::int64_t window, Agg agg,
                      const char* op_name) {
    if (refuse_non_numeric(op_name, v.type())) return Series{};
    const std::int64_t n = v.length();
    if (window < 1) window = 1;
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>((n + 7) / 8), 0);
    {
        std::int64_t a = std::max<std::int64_t>(window - 1, 0);
        std::int64_t i = a;
        for (; i < n && (i & 7); ++i)
            valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
        const std::int64_t full_end = n & ~std::int64_t{7};
        if (i < full_end) {
            std::memset(&valid[static_cast<std::size_t>(i >> 3)], 0xFF,
                        static_cast<std::size_t>((full_end - i) / 8));
            i = full_end;
        }
        for (; i < n; ++i)
            valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    }

    std::vector<double> vals;
    if (n > 0 && v.null_count() == 0 && v.encoding() == Encoding::Flat) {
        vals = nonnull_values(v);
    } else {
        vals.resize(static_cast<std::size_t>(n));
        for (std::int64_t i = 0; i < n; ++i)
            vals[static_cast<std::size_t>(i)] = read_f64(v, i);
    }
    const double* p = vals.data();

    const std::int64_t begin = std::min<std::int64_t>(window - 1, n);
    const std::int64_t rows = n - begin;
    constexpr std::int64_t GRAIN = 1 << 12;
    if (parallel_backend_installed() && rows > GRAIN) {
        parallel_for(rows, GRAIN, [&](std::int64_t b, std::int64_t e) {
            std::vector<double> buf;
            buf.reserve(static_cast<std::size_t>(window));
            for (std::int64_t i = begin + b; i < begin + e; ++i) {
                buf.clear();
                for (std::int64_t j = i - window + 1; j <= i; ++j)
                    buf.push_back(p[j]);
                out[static_cast<std::size_t>(i)] = agg(buf);
            }
        });
    } else {
        std::vector<double> buf;
        buf.reserve(static_cast<std::size_t>(window));
        for (std::int64_t i = begin; i < n; ++i) {
            buf.clear();
            for (std::int64_t j = i - window + 1; j <= i; ++j)
                buf.push_back(p[j]);
            out[static_cast<std::size_t>(i)] = agg(buf);
        }
    }
    return Series::flat(TypeId::Float64, out.data(), n, valid.data());
}

}  // namespace

Series rolling_var(const Series& v, std::int64_t window) {
    return rolling_window(
        v, window,
        [](const std::vector<double>& w) { return window_sample_var(w); },
        "rolling_var");
}

Series rolling_std(const Series& v, std::int64_t window) {
    return rolling_window(
        v, window,
        [](const std::vector<double>& w) {
            return std::sqrt(window_sample_var(w));
        },
        "rolling_std");
}

Series rolling_median(const Series& v, std::int64_t window) {
    return rolling_window(
        v, window,
        [](std::vector<double>& w) { return quantile_sorted(w, 0.5); },
        "rolling_median");
}

Series rolling_quantile(const Series& v, std::int64_t window, double q) {
    return rolling_window(
        v, window,
        [q](std::vector<double>& w) { return quantile_sorted(w, q); },
        "rolling_quantile");
}

// Two chunk-transform composers shared by the EWM family: `y -> a*y + c`
// applied first, then `y -> a*y + c` applied second, composes to
// `y -> (a1*a2)*y + (a2*c1 + c2)`.
using AffineT = std::pair<double, double>;

static AffineT affine_compose(AffineT lhs, AffineT rhs) {
    return AffineT{lhs.first * rhs.first, rhs.first * lhs.second + rhs.second};
}

// Parallel prefix scan for out[i] = a*out[i-1] + c(i), out[0] = c(0), where
// the coefficient `a` is the same at every step (ewm_mean's recurrence, and
// ewm_std's variance recurrence). Chunks scan locally from seed 0, then a
// single a^k correction folds in the true entering value - cheap because `a`
// never changes within a chunk. A chunk containing index 0 needs no
// correction: c(0) fixes the true value outright, independent of any seed.
template <class C>
void affine_scan_const_a(std::int64_t n, std::int64_t grain, double a, C&& c,
                         std::vector<double>& out) {
    if (n == 0) return;
    if (!parallel_backend_installed() || n <= grain) {
        double y = c(0);
        out[0] = y;
        for (std::int64_t i = 1; i < n; ++i) {
            y = a * y + c(i);
            out[static_cast<std::size_t>(i)] = y;
        }
        return;
    }
    parallel_prefix_scan<AffineT>(
        n, grain, AffineT{1.0, 0.0},
        [&](std::int64_t b, std::int64_t e) -> AffineT {
            double y = (b == 0) ? c(0) : 0.0;
            if (b == 0) out[0] = y;
            std::int64_t i0 = (b == 0) ? 1 : b;
            for (std::int64_t i = i0; i < e; ++i) {
                y = a * y + c(i);
                out[static_cast<std::size_t>(i)] = y;
            }
            if (b == 0) return AffineT{0.0, y};
            return AffineT{std::pow(a, e - b), y};
        },
        [&](std::int64_t b, std::int64_t e, AffineT offset) {
            if (b == 0) return;
            const double v_pre = offset.second;
            double p = a;
            for (std::int64_t i = b; i < e; ++i) {
                out[static_cast<std::size_t>(i)] += p * v_pre;
                p *= a;
            }
        },
        affine_compose);
}

// Parallel prefix scan for out[i] = a(i)*out[i-1] + c(i), out[0] = c(0),
// where the coefficient may vary by index but is known independent of any
// other row's value (ewm_std's running mean, whose coefficient depends only
// on the precomputed cumulative weight). Unlike the constant-a case, a
// single a^k correction does not exist, so each chunk records its per-index
// cumulative product of a(i) in `aprod` for the later correction pass to
// multiply against the true entering value.
template <class A, class C>
void affine_scan_var_a(std::int64_t n, std::int64_t grain, A&& a, C&& c,
                       std::vector<double>& out, std::vector<double>& aprod) {
    if (n == 0) return;
    if (!parallel_backend_installed() || n <= grain) {
        double y = c(0);
        out[0] = y;
        for (std::int64_t i = 1; i < n; ++i) {
            y = a(i) * y + c(i);
            out[static_cast<std::size_t>(i)] = y;
        }
        return;
    }
    parallel_prefix_scan<AffineT>(
        n, grain, AffineT{1.0, 0.0},
        [&](std::int64_t b, std::int64_t e) -> AffineT {
            double y = (b == 0) ? c(0) : 0.0;
            if (b == 0) out[0] = y;
            std::int64_t i0 = (b == 0) ? 1 : b;
            double aacc = 1.0;
            for (std::int64_t i = i0; i < e; ++i) {
                const double ai = a(i);
                y = ai * y + c(i);
                out[static_cast<std::size_t>(i)] = y;
                aacc *= ai;
                aprod[static_cast<std::size_t>(i)] = aacc;
            }
            if (b == 0) return AffineT{0.0, y};
            return AffineT{aacc, y};
        },
        [&](std::int64_t b, std::int64_t e, AffineT offset) {
            if (b == 0) return;
            const double v_pre = offset.second;
            for (std::int64_t i = b; i < e; ++i)
                out[static_cast<std::size_t>(i)] +=
                    aprod[static_cast<std::size_t>(i)] * v_pre;
        },
        affine_compose);
}

Series ewm_mean(const Series& v, double alpha) {
    if (refuse_non_numeric("ewm_mean", v.type())) return Series{};
    const std::int64_t n = v.length();
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    if (n == 0) return Series::flat(TypeId::Float64, out.data(), n);
    const double beta = 1.0 - alpha;
    affine_scan_const_a(
        n, std::int64_t{1} << 14, beta,
        [&](std::int64_t i) {
            return i == 0 ? read_f64(v, 0) : alpha * read_f64(v, i);
        },
        out);
    return Series::flat(TypeId::Float64, out.data(), n);
}

Series ewm_std(const Series& v, double alpha) {
    if (refuse_non_numeric("ewm_std", v.type())) return Series{};
    const std::int64_t n = v.length();
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>((n + 7) / 8), 0);
    if (n == 0) return Series::flat(TypeId::Float64, out.data(), n);
    // Incremental reliability-weight EW variance: at step i observation j
    // carries weight decay^(i-j). w_sum tracks the total weight, w2_sum the
    // sum of squared weights, and s the weighted sum of squared deviations;
    // the debiased sample variance is s / (w_sum - w2_sum / w_sum).
    const double decay = 1.0 - alpha;
    constexpr std::int64_t GRAIN = 1 << 14;
    // decay == 1 (alpha == 0) breaks the closed-form geometric sum below
    // (division by 1 - decay); that case is rare enough to just run serial.
    if (parallel_backend_installed() && n > GRAIN && alpha > 0.0) {
        // w_sum[i] and w2_sum[i] solve the same two recurrences as below but
        // depend only on i, not on any row's value, so each index is a
        // closed-form geometric sum computable independently in parallel:
        // w_sum[i] = sum_{k=0}^{i} decay^k, w2_sum[i] = sum_{k=0}^{i} decay^2k.
        std::vector<double> w_sum(static_cast<std::size_t>(n), 1.0);
        std::vector<double> w2_sum(static_cast<std::size_t>(n), 1.0);
        const double decay2 = decay * decay;
        parallel_for(n, GRAIN, [&](std::int64_t b, std::int64_t e) {
            for (std::int64_t i = std::max<std::int64_t>(b, 1); i < e; ++i) {
                w_sum[static_cast<std::size_t>(i)] =
                    (1.0 - std::pow(decay, i + 1)) / (1.0 - decay);
                w2_sum[static_cast<std::size_t>(i)] =
                    (1.0 - std::pow(decay2, i + 1)) / (1.0 - decay2);
            }
        });

        // mean[i] = mean[i-1]*(1 - 1/w_sum[i]) + x[i]/w_sum[i]: an affine
        // scan whose coefficient depends only on i (via w_sum), so it runs
        // through the variable-coefficient scan above.
        std::vector<double> mean(static_cast<std::size_t>(n), 0.0);
        std::vector<double> aprod(static_cast<std::size_t>(n), 0.0);
        affine_scan_var_a(
            n, GRAIN,
            [&](std::int64_t i) {
                return 1.0 - 1.0 / w_sum[static_cast<std::size_t>(i)];
            },
            [&](std::int64_t i) {
                return i == 0 ? read_f64(v, 0)
                              : read_f64(v, i) /
                                    w_sum[static_cast<std::size_t>(i)];
            },
            mean, aprod);

        // s[i] = decay*s[i-1] + delta*(x[i]-mean[i]), delta = x[i]-mean[i-1]:
        // constant coefficient `decay`, so it reuses the ewm_mean-style scan.
        affine_scan_const_a(
            n, GRAIN, decay,
            [&](std::int64_t i) {
                if (i == 0) return 0.0;
                const double x = read_f64(v, i);
                const double delta = x - mean[static_cast<std::size_t>(i - 1)];
                return delta * (x - mean[static_cast<std::size_t>(i)]);
            },
            out);

        parallel_for(n, GRAIN, [&](std::int64_t b, std::int64_t e) {
            for (std::int64_t i = std::max<std::int64_t>(b, 1); i < e; ++i) {
                const std::size_t ui = static_cast<std::size_t>(i);
                const double denom = w_sum[ui] - w2_sum[ui] / w_sum[ui];
                if (denom > 0.0) {
                    out[ui] = std::sqrt(out[ui] / denom);
                    valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
                } else {
                    out[ui] = 0.0;
                }
            }
        });
        return Series::flat(TypeId::Float64, out.data(), n, valid.data());
    }
    double w_sum = 1.0, w2_sum = 1.0, mean = read_f64(v, 0), s = 0.0;
    for (std::int64_t i = 1; i < n; ++i) {
        const double x = read_f64(v, i);
        const double w_prev = decay * w_sum;
        const double w_new = w_prev + 1.0;
        w2_sum = decay * decay * w2_sum + 1.0;
        w_sum = w_new;
        const double delta = x - mean;
        mean += delta / w_new;
        s = decay * s + 1.0 * delta * (x - mean);
        const double denom = w_sum - w2_sum / w_sum;
        if (denom > 0.0) {
            out[static_cast<std::size_t>(i)] = std::sqrt(s / denom);
            valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
        }
    }
    return Series::flat(TypeId::Float64, out.data(), n, valid.data());
}

Series cut(const Series& v, const Series& breaks) {
    if (refuse_non_numeric("cut", v.type())) return Series{};
    const std::int64_t n = v.length();
    const std::int64_t nb = breaks.length();
    std::vector<double> edges(static_cast<std::size_t>(nb));
    for (std::int64_t k = 0; k < nb; ++k)
        edges[static_cast<std::size_t>(k)] = read_f64(breaks, k);
    std::sort(edges.begin(), edges.end());
    std::vector<std::int32_t> out(static_cast<std::size_t>(n), -1);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>((n + 7) / 8), 0);
    const bool has_nulls = v.null_count() > 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if (has_nulls && v.is_null(i)) continue;
        const double x = read_f64(v, i);
        auto it = std::upper_bound(edges.begin(), edges.end(), x);
        out[static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(it - edges.begin());
        valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    }
    return Series::flat(TypeId::Int32, out.data(), n, valid.data());
}

Series qcut(const Series& v, std::int32_t q) {
    if (refuse_non_numeric("qcut", v.type())) return Series{};
    if (q < 1) q = 1;
    std::vector<double> vals = nonnull_values(v);
    std::vector<double> edges;
    edges.reserve(static_cast<std::size_t>(q - 1));
    for (std::int32_t k = 1; k < q; ++k)
        edges.push_back(quantile_sorted(
            vals, static_cast<double>(k) / static_cast<double>(q)));
    // nonnull_values order does not matter to quantile_sorted (it sorts a
    // copy's contents in place, so successive edges stay consistent).
    std::sort(edges.begin(), edges.end());
    const std::int64_t n = v.length();
    std::vector<std::int32_t> out(static_cast<std::size_t>(n), -1);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>((n + 7) / 8), 0);
    const bool has_nulls = v.null_count() > 0;
    for (std::int64_t i = 0; i < n; ++i) {
        if (has_nulls && v.is_null(i)) continue;
        const double x = read_f64(v, i);
        auto it = std::upper_bound(edges.begin(), edges.end(), x);
        out[static_cast<std::size_t>(i)] =
            static_cast<std::int32_t>(it - edges.begin());
        valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    }
    return Series::flat(TypeId::Int32, out.data(), n, valid.data());
}

Series search_sorted(const Series& v, const Series& values) {
    const std::int64_t n = v.length();
    std::vector<double> self(static_cast<std::size_t>(n));
    for (std::int64_t i = 0; i < n; ++i)
        self[static_cast<std::size_t>(i)] = read_f64(v, i);
    const std::int64_t m = values.length();
    std::vector<std::int64_t> out(static_cast<std::size_t>(m), 0);
    for (std::int64_t k = 0; k < m; ++k) {
        const double qv = read_f64(values, k);
        auto it = std::lower_bound(self.begin(), self.end(), qv);
        out[static_cast<std::size_t>(k)] =
            static_cast<std::int64_t>(it - self.begin());
    }
    return Series::flat(TypeId::Int64, out.data(), m);
}

Series interpolate(const Series& v) {
    const std::int64_t n = v.length();
    std::vector<double> out(static_cast<std::size_t>(n), 0.0);
    std::vector<std::uint8_t> valid(static_cast<std::size_t>((n + 7) / 8), 0);
    auto mark = [&](std::int64_t i) {
        valid[static_cast<std::size_t>(i >> 3)] |= (1u << (i & 7));
    };
    const bool has_nulls = v.null_count() > 0;
    // Fast path: no nulls -> a plain widening copy.
    if (!has_nulls) {
        for (std::int64_t i = 0; i < n; ++i) {
            out[static_cast<std::size_t>(i)] = read_f64(v, i);
            mark(i);
        }
        return Series::flat(TypeId::Float64, out.data(), n, valid.data());
    }
    // Collect anchors (non-null rows) first; the null runs between
    // consecutive anchors are then independent of each other, so they can be
    // filled in parallel once the anchor list (and its values) is known.
    std::vector<std::int64_t> anchors;
    for (std::int64_t i = 0; i < n; ++i) {
        if (v.is_null(i)) continue;
        out[static_cast<std::size_t>(i)] = read_f64(v, i);
        mark(i);
        anchors.push_back(i);
    }
    constexpr std::int64_t INTERPOLATE_GRAIN = 1 << 16;
    const std::int64_t npairs = static_cast<std::int64_t>(anchors.size()) - 1;
    if (parallel_backend_installed() && npairs > INTERPOLATE_GRAIN) {
        // Two threads filling adjacent runs can land bits in the same
        // validity byte, so bit-sets here must be atomic (a plain `|=` would
        // race); fold into the plain bitmap once every run is filled.
        std::vector<std::atomic<std::uint8_t>> avalid(valid.size());
        for (std::size_t i = 0; i < valid.size(); ++i) avalid[i] = valid[i];
        auto amark = [&](std::int64_t i) {
            avalid[static_cast<std::size_t>(i >> 3)].fetch_or(
                static_cast<std::uint8_t>(1u << (i & 7)),
                std::memory_order_relaxed);
        };
        parallel_for(
            npairs, INTERPOLATE_GRAIN, [&](std::int64_t b, std::int64_t e) {
                for (std::int64_t k = b; k < e; ++k) {
                    const std::int64_t i0 =
                        anchors[static_cast<std::size_t>(k)];
                    const std::int64_t i1 =
                        anchors[static_cast<std::size_t>(k) + 1];
                    if (i1 - i0 <= 1) continue;
                    const double a = out[static_cast<std::size_t>(i0)];
                    const double cur = out[static_cast<std::size_t>(i1)];
                    const double span = static_cast<double>(i1 - i0);
                    for (std::int64_t j = i0 + 1; j < i1; ++j) {
                        const double t = static_cast<double>(j - i0) / span;
                        out[static_cast<std::size_t>(j)] = a + (cur - a) * t;
                        amark(j);
                    }
                }
            });
        for (std::size_t i = 0; i < valid.size(); ++i)
            valid[i] = avalid[i].load(std::memory_order_relaxed);
    } else {
        for (std::int64_t k = 0; k < npairs; ++k) {
            const std::int64_t i0 = anchors[static_cast<std::size_t>(k)];
            const std::int64_t i1 = anchors[static_cast<std::size_t>(k) + 1];
            if (i1 - i0 <= 1) continue;
            const double a = out[static_cast<std::size_t>(i0)];
            const double cur = out[static_cast<std::size_t>(i1)];
            const double span = static_cast<double>(i1 - i0);
            for (std::int64_t j = i0 + 1; j < i1; ++j) {
                const double t = static_cast<double>(j - i0) / span;
                out[static_cast<std::size_t>(j)] = a + (cur - a) * t;
                mark(j);
            }
        }
    }
    return Series::flat(TypeId::Float64, out.data(), n, valid.data());
}

}  // namespace dftracer::utils::dataframe

extern "C" {

double dftu_series_variance(const dftu_series* v, int32_t sample) {
    if (!v) return 0.0;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    double r = dftracer::utils::dataframe::variance(c, sample != 0);
    c.release();
    return r;
}
double dftu_series_stddev(const dftu_series* v, int32_t sample) {
    if (!v) return 0.0;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    double r = dftracer::utils::dataframe::stddev(c, sample != 0);
    c.release();
    return r;
}
double dftu_series_skewness(const dftu_series* v) {
    if (!v) return 0.0;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    double r = dftracer::utils::dataframe::skewness(c);
    c.release();
    return r;
}
double dftu_series_kurtosis(const dftu_series* v) {
    if (!v) return 0.0;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    double r = dftracer::utils::dataframe::kurtosis(c);
    c.release();
    return r;
}
double dftu_series_quantile(const dftu_series* v, double q) {
    if (!v) return 0.0;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    double r = dftracer::utils::dataframe::quantile(c, q);
    c.release();
    return r;
}
int64_t dftu_series_nunique(const dftu_series* v) {
    if (!v) return 0;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    int64_t r = dftracer::utils::dataframe::nunique(c);
    c.release();
    return r;
}
dftu_series* dftu_series_unique(const dftu_series* v) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::unique(c);
    c.release();
    return r.release();
}
dftu_series* dftu_series_rank(const dftu_series* v, dftu_rank_method method,
                              int32_t descending) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r = dftracer::utils::dataframe::rank(
        c, static_cast<dftracer::utils::dataframe::RankMethod>(method),
        descending != 0);
    c.release();
    return r.release();
}
dftu_series* dftu_series_rolling(const dftu_series* v, int64_t window,
                                 dftu_rolling_op op) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r = dftracer::utils::dataframe::rolling(
        c, window, static_cast<dftracer::utils::dataframe::RollingOp>(op));
    c.release();
    return r.release();
}
dftu_series* dftu_series_rolling_var(const dftu_series* v, int64_t window) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::rolling_var(c, window);
    c.release();
    return r.release();
}
dftu_series* dftu_series_rolling_std(const dftu_series* v, int64_t window) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::rolling_std(c, window);
    c.release();
    return r.release();
}
dftu_series* dftu_series_rolling_median(const dftu_series* v, int64_t window) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::rolling_median(c, window);
    c.release();
    return r.release();
}
dftu_series* dftu_series_rolling_quantile(const dftu_series* v, int64_t window,
                                          double q) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::rolling_quantile(c, window, q);
    c.release();
    return r.release();
}
dftu_series* dftu_series_ewm_mean(const dftu_series* v, double alpha) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::ewm_mean(c, alpha);
    c.release();
    return r.release();
}
dftu_series* dftu_series_ewm_std(const dftu_series* v, double alpha) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::ewm_std(c, alpha);
    c.release();
    return r.release();
}
dftu_series* dftu_series_cut(const dftu_series* v, const dftu_series* breaks) {
    if (!v || !breaks) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series b{const_cast<dftu_series*>(breaks)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::cut(c, b);
    c.release();
    b.release();
    return r.release();
}
dftu_series* dftu_series_qcut(const dftu_series* v, int32_t q) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::qcut(c, q);
    c.release();
    return r.release();
}
dftu_series* dftu_series_search_sorted(const dftu_series* v,
                                       const dftu_series* values) {
    if (!v || !values) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series q{const_cast<dftu_series*>(values)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::search_sorted(c, q);
    c.release();
    q.release();
    return r.release();
}
dftu_series* dftu_series_interpolate(const dftu_series* v) {
    if (!v) return nullptr;
    dftracer::utils::dataframe::Series c{const_cast<dftu_series*>(v)};
    dftracer::utils::dataframe::Series r =
        dftracer::utils::dataframe::interpolate(c);
    c.release();
    return r.release();
}

}  // extern "C"
