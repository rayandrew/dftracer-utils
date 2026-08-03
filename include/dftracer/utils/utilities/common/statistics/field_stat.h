#ifndef DFTRACER_UTILS_UTILITIES_COMMON_STATISTICS_FIELD_STAT_H
#define DFTRACER_UTILS_UTILITIES_COMMON_STATISTICS_FIELD_STAT_H

#include <cstdint>

namespace dftracer::utils::utilities::common::statistics {

/// One field's running sufficient statistic as raw power sums (sumsq = sum x^2,
/// m3 = sum x^3, m4 = sum x^4) plus n/min/max: a complete mergeable summary for
/// count/sum/min/max/mean/var/std/skew/kurt. The shared aggregation atom for
/// both the View and the aggregation tier.
struct FieldStat {
    std::uint64_t n = 0;  // events where the field was present
    double sum = 0;
    double sumsq = 0;
    double m3 = 0;
    double m4 = 0;
    double min = 0;
    double max = 0;

    void add(double x) {
        if (n == 0) {
            min = max = x;
        } else {
            if (x < min) min = x;
            if (x > max) max = x;
        }
        const double x2 = x * x;
        sum += x;
        sumsq += x2;
        m3 += x2 * x;
        m4 += x2 * x2;
        ++n;
    }
    void merge(const FieldStat& o) {
        if (o.n == 0) return;
        if (n == 0) {
            min = o.min;
            max = o.max;
        } else {
            if (o.min < min) min = o.min;
            if (o.max > max) max = o.max;
        }
        sum += o.sum;
        sumsq += o.sumsq;
        m3 += o.m3;
        m4 += o.m4;
        n += o.n;
    }
};

}  // namespace dftracer::utils::utilities::common::statistics

#endif  // DFTRACER_UTILS_UTILITIES_COMMON_STATISTICS_FIELD_STAT_H
