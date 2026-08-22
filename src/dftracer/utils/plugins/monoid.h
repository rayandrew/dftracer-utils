#ifndef DFTRACER_UTILS_PLUGINS_MONOID_H
#define DFTRACER_UTILS_PLUGINS_MONOID_H

#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/dataframe/field_stat.h>
#include <dftracer/utils/plugins/abi.h>
#include <dftracer/utils/utilities/common/statistics/ddsketch.h>
#include <dftracer/utils/utilities/common/statistics/distinct_sketch.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dftracer::utils::plugins {

// Compare/storage family of a typed min/max monoid; NONE for every other kind.
enum class MinMaxFamily { NONE, SIGNED, UNSIGNED, FLOAT };

struct MinMaxKind {
    bool is_min = false;
    bool is_max = false;
    MinMaxFamily family = MinMaxFamily::NONE;
};

inline MinMaxKind minmax_kind(dftu_monoid_kind k) {
    switch (k) {
        case DFTU_MONOID_MIN_I8:
        case DFTU_MONOID_MIN_I16:
        case DFTU_MONOID_MIN_I32:
        case DFTU_MONOID_MIN_I64:
            return {true, false, MinMaxFamily::SIGNED};
        case DFTU_MONOID_MIN_U8:
        case DFTU_MONOID_MIN_U16:
        case DFTU_MONOID_MIN_U32:
        case DFTU_MONOID_MIN_U64:
            return {true, false, MinMaxFamily::UNSIGNED};
        case DFTU_MONOID_MIN_F32:
        case DFTU_MONOID_MIN_F64:
            return {true, false, MinMaxFamily::FLOAT};
        case DFTU_MONOID_MAX_I8:
        case DFTU_MONOID_MAX_I16:
        case DFTU_MONOID_MAX_I32:
        case DFTU_MONOID_MAX_I64:
            return {false, true, MinMaxFamily::SIGNED};
        case DFTU_MONOID_MAX_U8:
        case DFTU_MONOID_MAX_U16:
        case DFTU_MONOID_MAX_U32:
        case DFTU_MONOID_MAX_U64:
            return {false, true, MinMaxFamily::UNSIGNED};
        case DFTU_MONOID_MAX_F32:
        case DFTU_MONOID_MAX_F64:
            return {false, true, MinMaxFamily::FLOAT};
        default:
            return {};
    }
}

// Argmin/argmax family: keep a payload at the extreme of a f64 `by`.
inline bool monoid_is_argby(dftu_monoid_kind k) {
    return k == DFTU_MONOID_ARGMIN_I64 || k == DFTU_MONOID_ARGMAX_I64 ||
           k == DFTU_MONOID_ARGMIN_STR || k == DFTU_MONOID_ARGMAX_STR;
}

inline bool monoid_argby_is_max(dftu_monoid_kind k) {
    return k == DFTU_MONOID_ARGMAX_I64 || k == DFTU_MONOID_ARGMAX_STR;
}

// _STR payloads are interned ids resolved to labels at materialize.
inline bool monoid_argby_is_str(dftu_monoid_kind k) {
    return k == DFTU_MONOID_ARGMIN_STR || k == DFTU_MONOID_ARGMAX_STR;
}

// Full-row min-by/max-by (DISTINCT ON): keep a whole payload row of int64 slots
// at the extreme `by`. Created only via map_new_argrow, never as a component.
inline bool monoid_is_argrow(dftu_monoid_kind k) {
    return k == DFTU_MONOID_ARGMIN_ROW || k == DFTU_MONOID_ARGMAX_ROW;
}

inline bool monoid_argrow_is_max(dftu_monoid_kind k) {
    return k == DFTU_MONOID_ARGMAX_ROW;
}

// Bounded top-k/bottom-k: keep the k payloads at the k extreme `by` keys.
inline bool monoid_is_topk(dftu_monoid_kind k) {
    return k == DFTU_MONOID_TOPK_I64 || k == DFTU_MONOID_TOPK_STR ||
           k == DFTU_MONOID_BOTTOMK_I64 || k == DFTU_MONOID_BOTTOMK_STR;
}

// BOTTOMK keeps the smallest `by` keys; TOPK the largest.
inline bool monoid_topk_is_bottom(dftu_monoid_kind k) {
    return k == DFTU_MONOID_BOTTOMK_I64 || k == DFTU_MONOID_BOTTOMK_STR;
}

// _STR payloads are interned ids resolved to labels at materialize.
inline bool monoid_topk_is_str(dftu_monoid_kind k) {
    return k == DFTU_MONOID_TOPK_STR || k == DFTU_MONOID_BOTTOMK_STR;
}

// Approximate heavy-hitters (Space-Saving): the k most frequent values with
// approximate counts, in bounded memory. Metwally, Agrawal, El Abbadi,
// "Efficient Computation of Frequent and Top-k Elements in Data Streams"
// (2005).
inline bool monoid_is_approx_topk(dftu_monoid_kind k) {
    return k == DFTU_MONOID_APPROX_TOPK_I64 || k == DFTU_MONOID_APPROX_TOPK_STR;
}

// _STR values are interned ids resolved to labels at materialize.
inline bool monoid_approx_topk_is_str(dftu_monoid_kind k) {
    return k == DFTU_MONOID_APPROX_TOPK_STR;
}

// Bottom-k-by-hash uniform sample of DISTINCT items (KMV / bottom-k min-hash;
// Bar-Yossef et al., "Counting Distinct Elements in a Data Stream", 2002): keep
// the k items with the smallest hash(item). Bottom-k not reservoir because a
// stable hash makes the kept set deterministic and mergeable (union, keep k
// smallest).
inline bool monoid_is_sample(dftu_monoid_kind k) {
    return k == DFTU_MONOID_SAMPLE_I64 || k == DFTU_MONOID_SAMPLE_STR;
}

// _STR items are interned ids resolved to labels at materialize.
inline bool monoid_sample_is_str(dftu_monoid_kind k) {
    return k == DFTU_MONOID_SAMPLE_STR;
}

// Moment stats accumulated in a FieldStat power-sum; all materialize to double.
inline bool monoid_is_moment(dftu_monoid_kind k) {
    return k == DFTU_MONOID_MEAN || k == DFTU_MONOID_VARIANCE ||
           k == DFTU_MONOID_STDDEV || k == DFTU_MONOID_SKEWNESS ||
           k == DFTU_MONOID_KURTOSIS;
}

// Two-variable co-moment stats accumulated in raw power sums (n, sx, sy, sxx,
// syy, sxy) and fed via add_xy; all materialize to double.
inline bool monoid_is_comoment(dftu_monoid_kind k) {
    return k == DFTU_MONOID_CORR || k == DFTU_MONOID_COVAR_POP ||
           k == DFTU_MONOID_COVAR_SAMP || k == DFTU_MONOID_REGR_SLOPE ||
           k == DFTU_MONOID_REGR_INTERCEPT || k == DFTU_MONOID_REGR_R2;
}

namespace detail {

inline void put_u64(std::string& out, std::uint64_t v) {
    char b[sizeof(v)];
    std::memcpy(b, &v, sizeof(v));
    out.append(b, sizeof(b));
}

inline void put_f64(std::string& out, double v) {
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    put_u64(out, bits);
}

inline bool take_u64(const std::byte*& p, std::size_t& n, std::uint64_t& v) {
    if (n < sizeof(v)) return false;
    std::memcpy(&v, p, sizeof(v));
    p += sizeof(v);
    n -= sizeof(v);
    return true;
}

inline bool take_f64(const std::byte*& p, std::size_t& n, double& v) {
    std::uint64_t bits;
    if (!take_u64(p, n, bits)) return false;
    std::memcpy(&v, &bits, sizeof(v));
    return true;
}

}  // namespace detail

// A named mergeable accumulator over the fixed monoid vocabulary. merge is
// associative and commutative so worker-slice contribution order is irrelevant.
class MonoidAccumulator {
   public:
    explicit MonoidAccumulator(dftu_monoid_kind kind) : kind_(kind) {
        MinMaxKind mm = minmax_kind(kind_);
        if (mm.family == MinMaxFamily::FLOAT)
            f64_ = mm.is_min ? std::numeric_limits<double>::infinity()
                             : -std::numeric_limits<double>::infinity();
        else if (mm.family == MinMaxFamily::SIGNED)
            u64_ = static_cast<std::uint64_t>(
                mm.is_min ? std::numeric_limits<std::int64_t>::max()
                          : std::numeric_limits<std::int64_t>::min());
        else if (mm.family == MinMaxFamily::UNSIGNED)
            u64_ = mm.is_min ? std::numeric_limits<std::uint64_t>::max() : 0;
        else if (kind_ == DFTU_MONOID_BOOL_AND)
            u64_ = 1;
    }

    dftu_monoid_kind kind() const { return kind_; }

    void add_u64(std::uint64_t v) {
        MinMaxKind mm = minmax_kind(kind_);
        if (mm.family == MinMaxFamily::SIGNED) {
            std::int64_t sv = static_cast<std::int64_t>(v);
            std::int64_t cur = static_cast<std::int64_t>(u64_);
            if (mm.is_min ? sv < cur : sv > cur) u64_ = v;
            return;
        }
        if (mm.family == MinMaxFamily::UNSIGNED) {
            if (mm.is_min ? v < u64_ : v > u64_) u64_ = v;
            return;
        }
        switch (kind_) {
            case DFTU_MONOID_COUNTER:
                u64_ += v;
                break;
            case DFTU_MONOID_BOOL_AND:
                u64_ = (u64_ != 0 && v != 0) ? 1 : 0;
                break;
            case DFTU_MONOID_BOOL_OR:
                u64_ = (u64_ != 0 || v != 0) ? 1 : 0;
                break;
            case DFTU_MONOID_BITSET_OR:
                u64_ |= v;
                break;
            case DFTU_MONOID_DISTINCT:
                distinct_.add_hash(dftracer::utils::hash::fnv1a_mix(v));
                break;
            case DFTU_MONOID_SET_STR:
            case DFTU_MONOID_SET_I64:
                set_.insert(static_cast<std::int64_t>(v));
                break;
            default:
                break;
        }
    }

    // Append one (order_key, element) pair to a LIST_STR/LIST_I64 value; no
    // dedup.
    void add_ordered(std::int64_t order, std::int64_t element) {
        if (kind_ == DFTU_MONOID_LIST_STR || kind_ == DFTU_MONOID_LIST_I64)
            list_.emplace_back(order, element);
    }

    void add_f64(double v, double w) {
        MinMaxKind mm = minmax_kind(kind_);
        if (mm.family == MinMaxFamily::FLOAT) {
            if (mm.is_min ? v < f64_ : v > f64_) f64_ = v;
            return;
        }
        switch (kind_) {
            case DFTU_MONOID_SUM_F64:
                f64_ += v;
                break;
            case DFTU_MONOID_SKETCH:
                sketch_.add(v, w);
                sum_ += v * w;
                break;
            case DFTU_MONOID_MEAN:
            case DFTU_MONOID_VARIANCE:
            case DFTU_MONOID_STDDEV:
            case DFTU_MONOID_SKEWNESS:
            case DFTU_MONOID_KURTOSIS:
                fstat_.add(v);
                break;
            default:
                break;
        }
    }

    // Accumulate one (x, y) pair into the two-variable co-moment power sums; x
    // is the independent variable, y the dependent.
    void add_xy(double x, double y) {
        if (!monoid_is_comoment(kind_)) return;
        ++co_n_;
        co_sx_ += x;
        co_sy_ += y;
        co_sxx_ += x * x;
        co_syy_ += y * y;
        co_sxy_ += x * y;
    }

    // Ties on `by` keep the smaller payload id, so the result is independent of
    // contribution and merge order.
    void add_argby(double by, std::int64_t payload) {
        if (!monoid_is_argby(kind_)) return;
        bool take;
        if (!argby_has_) {
            take = true;
        } else if (by == argby_by_) {
            take = payload < argby_payload_;
        } else {
            take = monoid_argby_is_max(kind_) ? by > argby_by_ : by < argby_by_;
        }
        if (take) {
            argby_by_ = by;
            argby_payload_ = payload;
            argby_has_ = true;
        }
    }

    std::int64_t argby_payload() const { return argby_payload_; }

    // Keep the entire payload row (payload_n int64 slots) at the extreme `by`.
    // A `by` tie breaks lexicographically on the slots, so the kept row is
    // independent of contribution and merge order.
    void add_argrow(double by, const std::int64_t* payload,
                    std::uint32_t payload_n) {
        if (!monoid_is_argrow(kind_) || (payload_n && !payload)) return;
        bool take;
        if (!argrow_has_) {
            take = true;
        } else if (by == argrow_by_) {
            take = std::lexicographical_compare(
                payload, payload + payload_n, argrow_payload_.data(),
                argrow_payload_.data() + argrow_payload_.size());
        } else {
            take =
                monoid_argrow_is_max(kind_) ? by > argrow_by_ : by < argrow_by_;
        }
        if (take) {
            argrow_by_ = by;
            argrow_payload_.assign(payload, payload + payload_n);
            argrow_has_ = true;
        }
    }

    bool argrow_has() const { return argrow_has_; }
    double argrow_by() const { return argrow_by_; }
    const std::vector<std::int64_t>& argrow_payload() const {
        return argrow_payload_;
    }

    std::uint64_t sketch_count() const { return sketch_.count(); }
    double sketch_quantile(double q) const { return sketch_.quantile(q); }

    void add_topk(std::uint32_t k, double by, std::int64_t payload) {
        if (!monoid_is_topk(kind_)) return;
        if (k > topk_k_) topk_k_ = k;
        topk_.emplace_back(by, payload);
        topk_bound();
    }

    // SpaceSaving with k counters: when full, replace the min-count counter,
    // taking its count + 1 with error = the evicted count (bounds the
    // over-estimate).
    void add_approx_topk(std::uint32_t k, std::int64_t value) {
        if (!monoid_is_approx_topk(kind_)) return;
        if (k > approx_k_) approx_k_ = k;
        for (ApproxEntry& e : approx_)
            if (e.value == value) {
                ++e.count;
                return;
            }
        if (approx_.size() < approx_k_) {
            approx_.push_back({value, 1, 0});
            return;
        }
        if (approx_.empty()) return;
        ApproxEntry* victim = &approx_[0];
        for (ApproxEntry& e : approx_)
            if (e.count < victim->count ||
                (e.count == victim->count && e.value < victim->value))
                victim = &e;
        const std::uint64_t evicted = victim->count;
        victim->value = value;
        victim->count = evicted + 1;
        victim->error = evicted;
    }

    void add_sample(std::uint32_t k, std::int64_t item) {
        if (!monoid_is_sample(kind_)) return;
        if (k > sample_k_) sample_k_ = k;
        sample_insert(sample_hash(item), item);
    }

    // Sampled item ids sorted ascending, not in the hash-selection order, so
    // the column is stable.
    std::vector<std::int64_t> sample_items() const {
        std::vector<std::int64_t> out;
        out.reserve(sample_.size());
        for (const SampleEntry& e : sample_) out.push_back(e.item);
        std::sort(out.begin(), out.end());
        return out;
    }

    // (value, count) pairs ordered by count descending, value id breaking ties:
    // deterministic regardless of merge order.
    std::vector<std::pair<std::int64_t, std::uint64_t>> approx_topk_entries()
        const {
        std::vector<std::pair<std::int64_t, std::uint64_t>> out;
        out.reserve(approx_.size());
        for (const ApproxEntry& e : approx_) out.emplace_back(e.value, e.count);
        std::sort(out.begin(), out.end(),
                  [](const std::pair<std::int64_t, std::uint64_t>& a,
                     const std::pair<std::int64_t, std::uint64_t>& b) {
                      if (a.second != b.second) return a.second > b.second;
                      return a.first < b.first;
                  });
        return out;
    }

    // Kept payloads in `by` order (descending TOPK, ascending BOTTOMK), equal
    // `by` by smaller payload id, so set and order are merge-order independent.
    std::vector<std::int64_t> topk_elements() const {
        std::vector<std::pair<double, std::int64_t>> s = topk_;
        topk_sort(s);
        std::vector<std::int64_t> out;
        out.reserve(s.size());
        for (const auto& p : s) out.push_back(p.second);
        return out;
    }

    void merge(const MonoidAccumulator& o) {
        if (o.kind_ != kind_) return;
        MinMaxKind mm = minmax_kind(kind_);
        if (mm.family == MinMaxFamily::SIGNED) {
            std::int64_t ov = static_cast<std::int64_t>(o.u64_);
            std::int64_t cur = static_cast<std::int64_t>(u64_);
            if (mm.is_min ? ov < cur : ov > cur) u64_ = o.u64_;
            return;
        }
        if (mm.family == MinMaxFamily::UNSIGNED) {
            if (mm.is_min ? o.u64_ < u64_ : o.u64_ > u64_) u64_ = o.u64_;
            return;
        }
        if (mm.family == MinMaxFamily::FLOAT) {
            if (mm.is_min ? o.f64_ < f64_ : o.f64_ > f64_) f64_ = o.f64_;
            return;
        }
        switch (kind_) {
            case DFTU_MONOID_COUNTER:
                u64_ += o.u64_;
                break;
            case DFTU_MONOID_SUM_F64:
                f64_ += o.f64_;
                break;
            case DFTU_MONOID_SKETCH:
                sketch_.merge(o.sketch_);
                sum_ += o.sum_;
                break;
            case DFTU_MONOID_BOOL_AND:
                u64_ = (u64_ != 0 && o.u64_ != 0) ? 1 : 0;
                break;
            case DFTU_MONOID_BOOL_OR:
                u64_ = (u64_ != 0 || o.u64_ != 0) ? 1 : 0;
                break;
            case DFTU_MONOID_BITSET_OR:
                u64_ |= o.u64_;
                break;
            case DFTU_MONOID_DISTINCT:
                distinct_.merge_from(o.distinct_);
                break;
            case DFTU_MONOID_SET_STR:
            case DFTU_MONOID_SET_I64:
                set_.insert(o.set_.begin(), o.set_.end());
                break;
            case DFTU_MONOID_LIST_STR:
            case DFTU_MONOID_LIST_I64:
                list_.insert(list_.end(), o.list_.begin(), o.list_.end());
                break;
            case DFTU_MONOID_ARGMIN_I64:
            case DFTU_MONOID_ARGMAX_I64:
            case DFTU_MONOID_ARGMIN_STR:
            case DFTU_MONOID_ARGMAX_STR:
                if (o.argby_has_) add_argby(o.argby_by_, o.argby_payload_);
                break;
            case DFTU_MONOID_ARGMIN_ROW:
            case DFTU_MONOID_ARGMAX_ROW:
                if (o.argrow_has_)
                    add_argrow(
                        o.argrow_by_, o.argrow_payload_.data(),
                        static_cast<std::uint32_t>(o.argrow_payload_.size()));
                break;
            case DFTU_MONOID_TOPK_I64:
            case DFTU_MONOID_TOPK_STR:
            case DFTU_MONOID_BOTTOMK_I64:
            case DFTU_MONOID_BOTTOMK_STR:
                if (o.topk_k_ > topk_k_) topk_k_ = o.topk_k_;
                topk_.insert(topk_.end(), o.topk_.begin(), o.topk_.end());
                topk_bound();
                break;
            case DFTU_MONOID_APPROX_TOPK_I64:
            case DFTU_MONOID_APPROX_TOPK_STR:
                approx_merge(o);
                break;
            case DFTU_MONOID_SAMPLE_I64:
            case DFTU_MONOID_SAMPLE_STR:
                if (o.sample_k_ > sample_k_) sample_k_ = o.sample_k_;
                for (const SampleEntry& e : o.sample_)
                    sample_insert(e.hash, e.item);
                break;
            case DFTU_MONOID_MEAN:
            case DFTU_MONOID_VARIANCE:
            case DFTU_MONOID_STDDEV:
            case DFTU_MONOID_SKEWNESS:
            case DFTU_MONOID_KURTOSIS:
                fstat_.merge(o.fstat_);
                break;
            case DFTU_MONOID_CORR:
            case DFTU_MONOID_COVAR_POP:
            case DFTU_MONOID_COVAR_SAMP:
            case DFTU_MONOID_REGR_SLOPE:
            case DFTU_MONOID_REGR_INTERCEPT:
            case DFTU_MONOID_REGR_R2:
                co_n_ += o.co_n_;
                co_sx_ += o.co_sx_;
                co_sy_ += o.co_sy_;
                co_sxx_ += o.co_sxx_;
                co_syy_ += o.co_syy_;
                co_sxy_ += o.co_sxy_;
                break;
            default:
                break;
        }
    }

    // SET_STR/SET_I64 have no scalar to_value; the materializer reads ids here.
    const std::unordered_set<std::int64_t>& elements() const { return set_; }

    // SET_I64 elements sorted ascending: deterministic regardless of merge
    // order.
    std::vector<std::int64_t> sorted_elements() const {
        std::vector<std::int64_t> out(set_.begin(), set_.end());
        std::sort(out.begin(), out.end());
        return out;
    }

    // LIST element ids sorted by (order_key, element id): stable regardless of
    // merge order.
    std::vector<std::int64_t> ordered_elements() const {
        std::vector<std::pair<std::int64_t, std::int64_t>> s = list_;
        std::sort(s.begin(), s.end());
        std::vector<std::int64_t> out;
        out.reserve(s.size());
        for (const auto& p : s) out.push_back(p.second);
        return out;
    }

    dftu_monoid_value to_value() const {
        dftu_monoid_value out{};
        out.kind = kind_;
        MinMaxKind mm = minmax_kind(kind_);
        if (mm.family == MinMaxFamily::FLOAT) {
            out.as.f64 = f64_;
            return out;
        }
        if (mm.family == MinMaxFamily::SIGNED ||
            mm.family == MinMaxFamily::UNSIGNED) {
            out.as.u64 = u64_;
            return out;
        }
        switch (kind_) {
            case DFTU_MONOID_COUNTER:
            case DFTU_MONOID_BOOL_AND:
            case DFTU_MONOID_BOOL_OR:
            case DFTU_MONOID_BITSET_OR:
                out.as.u64 = u64_;
                break;
            case DFTU_MONOID_SUM_F64:
                out.as.f64 = f64_;
                break;
            case DFTU_MONOID_SKETCH:
                out.as.quant = quantiles();
                break;
            case DFTU_MONOID_DISTINCT:
                out.as.u64 = distinct_.estimate();
                break;
            case DFTU_MONOID_ARGMIN_I64:
            case DFTU_MONOID_ARGMAX_I64:
            case DFTU_MONOID_ARGMIN_STR:
            case DFTU_MONOID_ARGMAX_STR:
                out.as.u64 = static_cast<std::uint64_t>(argby_payload_);
                break;
            case DFTU_MONOID_MEAN:
                out.as.f64 = mean();
                break;
            case DFTU_MONOID_VARIANCE:
                out.as.f64 = variance();
                break;
            case DFTU_MONOID_STDDEV:
                out.as.f64 = stddev();
                break;
            case DFTU_MONOID_SKEWNESS:
                out.as.f64 = skewness();
                break;
            case DFTU_MONOID_KURTOSIS:
                out.as.f64 = kurtosis();
                break;
            case DFTU_MONOID_COVAR_POP:
                out.as.f64 = covar(false);
                break;
            case DFTU_MONOID_COVAR_SAMP:
                out.as.f64 = covar(true);
                break;
            case DFTU_MONOID_CORR:
                out.as.f64 = corr();
                break;
            case DFTU_MONOID_REGR_SLOPE:
                out.as.f64 = regr_slope();
                break;
            case DFTU_MONOID_REGR_INTERCEPT:
                out.as.f64 = regr_intercept();
                break;
            case DFTU_MONOID_REGR_R2:
                out.as.f64 = regr_r2();
                break;
            case DFTU_MONOID_SET_STR:
            case DFTU_MONOID_LIST_STR:
            case DFTU_MONOID_SET_I64:
            case DFTU_MONOID_LIST_I64:
            case DFTU_MONOID_SAMPLE_I64:
            case DFTU_MONOID_SAMPLE_STR:
                break;
            default:
                break;
        }
        return out;
    }

    // Append this monoid's state to out, paired with deserialize. Native-endian
    // and process-internal; not a persisted or versioned format.
    void serialize(std::string& out) const {
        MinMaxFamily fam = minmax_kind(kind_).family;
        if (fam == MinMaxFamily::FLOAT) {
            detail::put_f64(out, f64_);
            return;
        }
        if (fam == MinMaxFamily::SIGNED || fam == MinMaxFamily::UNSIGNED) {
            detail::put_u64(out, u64_);
            return;
        }
        switch (kind_) {
            case DFTU_MONOID_COUNTER:
            case DFTU_MONOID_BOOL_AND:
            case DFTU_MONOID_BOOL_OR:
            case DFTU_MONOID_BITSET_OR:
                detail::put_u64(out, u64_);
                break;
            case DFTU_MONOID_SUM_F64:
                detail::put_f64(out, f64_);
                break;
            case DFTU_MONOID_SKETCH: {
                detail::put_f64(out, sum_);
                std::vector<std::uint8_t> buf;
                sketch_.serialize_into(buf);
                detail::put_u64(out, static_cast<std::uint64_t>(buf.size()));
                out.append(reinterpret_cast<const char*>(buf.data()),
                           buf.size());
                break;
            }
            case DFTU_MONOID_DISTINCT: {
                const auto& dense = distinct_.dense_registers();
                const auto& sparse = distinct_.sparse_hashes();
                if (!dense.empty()) {
                    out.push_back(static_cast<char>(1));
                    detail::put_u64(out,
                                    static_cast<std::uint64_t>(dense.size()));
                    out.append(reinterpret_cast<const char*>(dense.data()),
                               dense.size());
                } else {
                    out.push_back(static_cast<char>(0));
                    detail::put_u64(out,
                                    static_cast<std::uint64_t>(sparse.size()));
                    for (std::uint64_t h : sparse) detail::put_u64(out, h);
                }
                break;
            }
            case DFTU_MONOID_SET_STR:
            case DFTU_MONOID_SET_I64: {
                std::vector<std::int64_t> s = sorted_elements();
                detail::put_u64(out, static_cast<std::uint64_t>(s.size()));
                for (std::int64_t v : s)
                    detail::put_u64(out, static_cast<std::uint64_t>(v));
                break;
            }
            case DFTU_MONOID_LIST_STR:
            case DFTU_MONOID_LIST_I64: {
                detail::put_u64(out, static_cast<std::uint64_t>(list_.size()));
                for (const auto& pr : list_) {
                    detail::put_u64(out, static_cast<std::uint64_t>(pr.first));
                    detail::put_u64(out, static_cast<std::uint64_t>(pr.second));
                }
                break;
            }
            case DFTU_MONOID_ARGMIN_I64:
            case DFTU_MONOID_ARGMAX_I64:
            case DFTU_MONOID_ARGMIN_STR:
            case DFTU_MONOID_ARGMAX_STR:
                detail::put_f64(out, argby_by_);
                detail::put_u64(out,
                                static_cast<std::uint64_t>(argby_payload_));
                out.push_back(static_cast<char>(argby_has_ ? 1 : 0));
                break;
            case DFTU_MONOID_ARGMIN_ROW:
            case DFTU_MONOID_ARGMAX_ROW:
                out.push_back(static_cast<char>(argrow_has_ ? 1 : 0));
                detail::put_f64(out, argrow_by_);
                detail::put_u64(
                    out, static_cast<std::uint64_t>(argrow_payload_.size()));
                for (std::int64_t s : argrow_payload_)
                    detail::put_u64(out, static_cast<std::uint64_t>(s));
                break;
            case DFTU_MONOID_TOPK_I64:
            case DFTU_MONOID_TOPK_STR:
            case DFTU_MONOID_BOTTOMK_I64:
            case DFTU_MONOID_BOTTOMK_STR:
                detail::put_u64(out, static_cast<std::uint64_t>(topk_k_));
                detail::put_u64(out, static_cast<std::uint64_t>(topk_.size()));
                for (const auto& pr : topk_) {
                    detail::put_f64(out, pr.first);
                    detail::put_u64(out, static_cast<std::uint64_t>(pr.second));
                }
                break;
            case DFTU_MONOID_APPROX_TOPK_I64:
            case DFTU_MONOID_APPROX_TOPK_STR:
                detail::put_u64(out, static_cast<std::uint64_t>(approx_k_));
                detail::put_u64(out,
                                static_cast<std::uint64_t>(approx_.size()));
                for (const ApproxEntry& e : approx_) {
                    detail::put_u64(out, static_cast<std::uint64_t>(e.value));
                    detail::put_u64(out, e.count);
                    detail::put_u64(out, e.error);
                }
                break;
            case DFTU_MONOID_SAMPLE_I64:
            case DFTU_MONOID_SAMPLE_STR:
                detail::put_u64(out, static_cast<std::uint64_t>(sample_k_));
                detail::put_u64(out,
                                static_cast<std::uint64_t>(sample_.size()));
                for (const SampleEntry& e : sample_) {
                    detail::put_u64(out, e.hash);
                    detail::put_u64(out, static_cast<std::uint64_t>(e.item));
                }
                break;
            case DFTU_MONOID_MEAN:
            case DFTU_MONOID_VARIANCE:
            case DFTU_MONOID_STDDEV:
            case DFTU_MONOID_SKEWNESS:
            case DFTU_MONOID_KURTOSIS:
                detail::put_u64(out, fstat_.n);
                detail::put_f64(out, fstat_.sum);
                detail::put_f64(out, fstat_.sumsq);
                detail::put_f64(out, fstat_.m3);
                detail::put_f64(out, fstat_.m4);
                detail::put_f64(out, fstat_.min);
                detail::put_f64(out, fstat_.max);
                break;
            case DFTU_MONOID_CORR:
            case DFTU_MONOID_COVAR_POP:
            case DFTU_MONOID_COVAR_SAMP:
            case DFTU_MONOID_REGR_SLOPE:
            case DFTU_MONOID_REGR_INTERCEPT:
            case DFTU_MONOID_REGR_R2:
                detail::put_u64(out, co_n_);
                detail::put_f64(out, co_sx_);
                detail::put_f64(out, co_sy_);
                detail::put_f64(out, co_sxx_);
                detail::put_f64(out, co_syy_);
                detail::put_f64(out, co_sxy_);
                break;
            default:
                break;
        }
    }

    // Read state written by serialize into this already-kind-constructed
    // monoid. Returns bytes consumed, or 0 on a short or invalid buffer.
    std::size_t deserialize(const std::byte* p, std::size_t n) {
        const std::byte* cur = p;
        std::size_t rem = n;
        MinMaxFamily fam = minmax_kind(kind_).family;
        if (fam == MinMaxFamily::FLOAT) {
            double v;
            if (!detail::take_f64(cur, rem, v)) return 0;
            f64_ = v;
            return n - rem;
        }
        if (fam == MinMaxFamily::SIGNED || fam == MinMaxFamily::UNSIGNED) {
            std::uint64_t v;
            if (!detail::take_u64(cur, rem, v)) return 0;
            u64_ = v;
            return n - rem;
        }
        switch (kind_) {
            case DFTU_MONOID_COUNTER:
            case DFTU_MONOID_BOOL_AND:
            case DFTU_MONOID_BOOL_OR:
            case DFTU_MONOID_BITSET_OR: {
                std::uint64_t v;
                if (!detail::take_u64(cur, rem, v)) return 0;
                u64_ = v;
                break;
            }
            case DFTU_MONOID_SUM_F64: {
                double v;
                if (!detail::take_f64(cur, rem, v)) return 0;
                f64_ = v;
                break;
            }
            case DFTU_MONOID_SKETCH: {
                double sm;
                std::uint64_t len;
                if (!detail::take_f64(cur, rem, sm)) return 0;
                if (!detail::take_u64(cur, rem, len)) return 0;
                if (rem < len) return 0;
                sketch_ = utilities::common::statistics::DDSketch::deserialize(
                    reinterpret_cast<const std::uint8_t*>(cur), len);
                cur += len;
                rem -= len;
                sum_ = sm;
                break;
            }
            case DFTU_MONOID_DISTINCT: {
                if (rem < 1) return 0;
                std::uint8_t tag = static_cast<std::uint8_t>(*cur);
                ++cur;
                --rem;
                std::uint64_t cnt;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (tag == 1) {
                    if (rem < cnt) return 0;
                    std::vector<std::uint8_t> regs(cnt);
                    std::memcpy(regs.data(), cur, cnt);
                    cur += cnt;
                    rem -= cnt;
                    distinct_.set_dense_registers(std::move(regs));
                } else {
                    if (cnt > rem / sizeof(std::uint64_t)) return 0;
                    std::vector<std::uint64_t> hashes;
                    hashes.reserve(cnt);
                    for (std::uint64_t i = 0; i < cnt; ++i) {
                        std::uint64_t h;
                        if (!detail::take_u64(cur, rem, h)) return 0;
                        hashes.push_back(h);
                    }
                    distinct_.set_sparse_hashes(std::move(hashes));
                }
                break;
            }
            case DFTU_MONOID_SET_STR:
            case DFTU_MONOID_SET_I64: {
                std::uint64_t cnt;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (cnt > rem / sizeof(std::uint64_t)) return 0;
                std::unordered_set<std::int64_t> s;
                s.reserve(cnt);
                for (std::uint64_t i = 0; i < cnt; ++i) {
                    std::uint64_t v;
                    if (!detail::take_u64(cur, rem, v)) return 0;
                    s.insert(static_cast<std::int64_t>(v));
                }
                set_ = std::move(s);
                break;
            }
            case DFTU_MONOID_LIST_STR:
            case DFTU_MONOID_LIST_I64: {
                std::uint64_t cnt;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (cnt > rem / (2 * sizeof(std::uint64_t))) return 0;
                std::vector<std::pair<std::int64_t, std::int64_t>> l;
                l.reserve(cnt);
                for (std::uint64_t i = 0; i < cnt; ++i) {
                    std::uint64_t order, elem;
                    if (!detail::take_u64(cur, rem, order)) return 0;
                    if (!detail::take_u64(cur, rem, elem)) return 0;
                    l.emplace_back(static_cast<std::int64_t>(order),
                                   static_cast<std::int64_t>(elem));
                }
                list_ = std::move(l);
                break;
            }
            case DFTU_MONOID_ARGMIN_I64:
            case DFTU_MONOID_ARGMAX_I64:
            case DFTU_MONOID_ARGMIN_STR:
            case DFTU_MONOID_ARGMAX_STR: {
                double by;
                std::uint64_t payload;
                if (!detail::take_f64(cur, rem, by)) return 0;
                if (!detail::take_u64(cur, rem, payload)) return 0;
                if (rem < 1) return 0;
                std::uint8_t has = static_cast<std::uint8_t>(*cur);
                ++cur;
                --rem;
                argby_by_ = by;
                argby_payload_ = static_cast<std::int64_t>(payload);
                argby_has_ = has != 0;
                break;
            }
            case DFTU_MONOID_ARGMIN_ROW:
            case DFTU_MONOID_ARGMAX_ROW: {
                if (rem < 1) return 0;
                std::uint8_t has = static_cast<std::uint8_t>(*cur);
                ++cur;
                --rem;
                double by;
                std::uint64_t cnt;
                if (!detail::take_f64(cur, rem, by)) return 0;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (cnt > rem / sizeof(std::uint64_t)) return 0;
                std::vector<std::int64_t> pl;
                pl.reserve(cnt);
                for (std::uint64_t i = 0; i < cnt; ++i) {
                    std::uint64_t v;
                    if (!detail::take_u64(cur, rem, v)) return 0;
                    pl.push_back(static_cast<std::int64_t>(v));
                }
                argrow_has_ = has != 0;
                argrow_by_ = by;
                argrow_payload_ = std::move(pl);
                break;
            }
            case DFTU_MONOID_TOPK_I64:
            case DFTU_MONOID_TOPK_STR:
            case DFTU_MONOID_BOTTOMK_I64:
            case DFTU_MONOID_BOTTOMK_STR: {
                std::uint64_t k, cnt;
                if (!detail::take_u64(cur, rem, k)) return 0;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (cnt > rem / (sizeof(double) + sizeof(std::uint64_t)))
                    return 0;
                std::vector<std::pair<double, std::int64_t>> v;
                v.reserve(cnt);
                for (std::uint64_t i = 0; i < cnt; ++i) {
                    double by;
                    std::uint64_t payload;
                    if (!detail::take_f64(cur, rem, by)) return 0;
                    if (!detail::take_u64(cur, rem, payload)) return 0;
                    v.emplace_back(by, static_cast<std::int64_t>(payload));
                }
                topk_k_ = static_cast<std::uint32_t>(k);
                topk_ = std::move(v);
                break;
            }
            case DFTU_MONOID_APPROX_TOPK_I64:
            case DFTU_MONOID_APPROX_TOPK_STR: {
                std::uint64_t k, cnt;
                if (!detail::take_u64(cur, rem, k)) return 0;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (cnt > rem / (3 * sizeof(std::uint64_t))) return 0;
                std::vector<ApproxEntry> v;
                v.reserve(cnt);
                for (std::uint64_t i = 0; i < cnt; ++i) {
                    std::uint64_t value, count, error;
                    if (!detail::take_u64(cur, rem, value)) return 0;
                    if (!detail::take_u64(cur, rem, count)) return 0;
                    if (!detail::take_u64(cur, rem, error)) return 0;
                    v.push_back(
                        {static_cast<std::int64_t>(value), count, error});
                }
                approx_k_ = static_cast<std::uint32_t>(k);
                approx_ = std::move(v);
                break;
            }
            case DFTU_MONOID_SAMPLE_I64:
            case DFTU_MONOID_SAMPLE_STR: {
                std::uint64_t k, cnt;
                if (!detail::take_u64(cur, rem, k)) return 0;
                if (!detail::take_u64(cur, rem, cnt)) return 0;
                if (cnt > rem / (2 * sizeof(std::uint64_t))) return 0;
                std::vector<SampleEntry> v;
                v.reserve(cnt);
                for (std::uint64_t i = 0; i < cnt; ++i) {
                    std::uint64_t hash, item;
                    if (!detail::take_u64(cur, rem, hash)) return 0;
                    if (!detail::take_u64(cur, rem, item)) return 0;
                    v.push_back({hash, static_cast<std::int64_t>(item)});
                }
                sample_k_ = static_cast<std::uint32_t>(k);
                sample_ = std::move(v);
                break;
            }
            case DFTU_MONOID_MEAN:
            case DFTU_MONOID_VARIANCE:
            case DFTU_MONOID_STDDEV:
            case DFTU_MONOID_SKEWNESS:
            case DFTU_MONOID_KURTOSIS: {
                std::uint64_t nn;
                double sum, sumsq, m3, m4, mn, mx;
                if (!detail::take_u64(cur, rem, nn)) return 0;
                if (!detail::take_f64(cur, rem, sum)) return 0;
                if (!detail::take_f64(cur, rem, sumsq)) return 0;
                if (!detail::take_f64(cur, rem, m3)) return 0;
                if (!detail::take_f64(cur, rem, m4)) return 0;
                if (!detail::take_f64(cur, rem, mn)) return 0;
                if (!detail::take_f64(cur, rem, mx)) return 0;
                fstat_.n = nn;
                fstat_.sum = sum;
                fstat_.sumsq = sumsq;
                fstat_.m3 = m3;
                fstat_.m4 = m4;
                fstat_.min = mn;
                fstat_.max = mx;
                break;
            }
            case DFTU_MONOID_CORR:
            case DFTU_MONOID_COVAR_POP:
            case DFTU_MONOID_COVAR_SAMP:
            case DFTU_MONOID_REGR_SLOPE:
            case DFTU_MONOID_REGR_INTERCEPT:
            case DFTU_MONOID_REGR_R2: {
                std::uint64_t nn;
                double sx, sy, sxx, syy, sxy;
                if (!detail::take_u64(cur, rem, nn)) return 0;
                if (!detail::take_f64(cur, rem, sx)) return 0;
                if (!detail::take_f64(cur, rem, sy)) return 0;
                if (!detail::take_f64(cur, rem, sxx)) return 0;
                if (!detail::take_f64(cur, rem, syy)) return 0;
                if (!detail::take_f64(cur, rem, sxy)) return 0;
                co_n_ = nn;
                co_sx_ = sx;
                co_sy_ = sy;
                co_sxx_ = sxx;
                co_syy_ = syy;
                co_sxy_ = sxy;
                break;
            }
            default:
                break;
        }
        return n - rem;
    }

    // Approximate in-memory byte size, a coarse budget for the spill counter;
    // not exact accounting.
    std::size_t state_bytes() const {
        if (minmax_kind(kind_).family != MinMaxFamily::NONE) return 8;
        switch (kind_) {
            case DFTU_MONOID_SET_STR:
            case DFTU_MONOID_SET_I64:
                return 16 + set_.size() * sizeof(std::int64_t);
            case DFTU_MONOID_LIST_STR:
            case DFTU_MONOID_LIST_I64:
                return 16 + list_.size() * 2 * sizeof(std::int64_t);
            case DFTU_MONOID_SKETCH:
            case DFTU_MONOID_DISTINCT:
                return 256;
            case DFTU_MONOID_ARGMIN_I64:
            case DFTU_MONOID_ARGMAX_I64:
            case DFTU_MONOID_ARGMIN_STR:
            case DFTU_MONOID_ARGMAX_STR:
                return sizeof(argby_by_) + sizeof(argby_payload_) + 1;
            case DFTU_MONOID_ARGMIN_ROW:
            case DFTU_MONOID_ARGMAX_ROW:
                return 16 + argrow_payload_.size() * sizeof(std::int64_t);
            case DFTU_MONOID_TOPK_I64:
            case DFTU_MONOID_TOPK_STR:
            case DFTU_MONOID_BOTTOMK_I64:
            case DFTU_MONOID_BOTTOMK_STR:
                return 16 +
                       topk_.size() * (sizeof(double) + sizeof(std::int64_t));
            case DFTU_MONOID_APPROX_TOPK_I64:
            case DFTU_MONOID_APPROX_TOPK_STR:
                return 16 + approx_.size() * 3 * sizeof(std::uint64_t);
            case DFTU_MONOID_SAMPLE_I64:
            case DFTU_MONOID_SAMPLE_STR:
                return 16 + sample_.size() * 2 * sizeof(std::uint64_t);
            case DFTU_MONOID_MEAN:
            case DFTU_MONOID_VARIANCE:
            case DFTU_MONOID_STDDEV:
            case DFTU_MONOID_SKEWNESS:
            case DFTU_MONOID_KURTOSIS:
                return sizeof(dftracer::utils::dataframe::FieldStat);
            case DFTU_MONOID_CORR:
            case DFTU_MONOID_COVAR_POP:
            case DFTU_MONOID_COVAR_SAMP:
            case DFTU_MONOID_REGR_SLOPE:
            case DFTU_MONOID_REGR_INTERCEPT:
            case DFTU_MONOID_REGR_R2:
                return sizeof(co_n_) + 5 * sizeof(double);
            default:
                return 8;
        }
    }

   private:
    // Sample (n-1) moments from the FieldStat power-sums, matching the
    // aggregator: variance/stddev is 0 below 2 samples, central sum of squares
    // clamped nonnegative against rounding.
    double mean() const {
        return fstat_.n ? fstat_.sum / static_cast<double>(fstat_.n) : 0.0;
    }
    double variance() const {
        if (fstat_.n < 2) return 0.0;
        const double dn = static_cast<double>(fstat_.n);
        const double mu = fstat_.sum / dn;
        double central = fstat_.sumsq - dn * mu * mu;
        if (central < 0.0) central = 0.0;
        return central / (dn - 1.0);
    }
    double stddev() const {
        const double v = variance();
        return v > 0.0 ? std::sqrt(v) : 0.0;
    }

    // Population skewness/excess-kurtosis, matching MetricStats::get_skewness/
    // get_kurtosis exactly: central-moment power sums (M2/M3/M4 unnormalized),
    // 0 below 3/4 samples, 0 when M2 is nonpositive (rounding can push it < 0).
    double skewness() const {
        if (fstat_.n < 3) return 0.0;
        const double n = static_cast<double>(fstat_.n);
        const double mu = fstat_.sum / n;
        const double M2 = fstat_.sumsq - n * mu * mu;
        if (M2 <= 0.0) return 0.0;
        const double M3 =
            fstat_.m3 - 3.0 * mu * fstat_.sumsq + 2.0 * n * mu * mu * mu;
        return std::sqrt(n) * M3 / std::pow(M2, 1.5);
    }
    double kurtosis() const {
        if (fstat_.n < 4) return 0.0;
        const double n = static_cast<double>(fstat_.n);
        const double mu = fstat_.sum / n;
        const double M2 = fstat_.sumsq - n * mu * mu;
        if (M2 <= 0.0) return 0.0;
        const double M4 = fstat_.m4 - 4.0 * mu * fstat_.m3 +
                          6.0 * mu * mu * fstat_.sumsq -
                          3.0 * n * mu * mu * mu * mu;
        return n * M4 / (M2 * M2) - 3.0;
    }

    // Two-variable co-moment readouts from the raw power sums, x independent, y
    // dependent. All 0 for n<2; ratios 0 when their denominator is nonpositive.
    double covar(bool sample) const {
        if (co_n_ < 2) return 0.0;
        const double n = static_cast<double>(co_n_);
        const double sxy = co_sxy_ - co_sx_ * co_sy_ / n;
        return sxy / (sample ? n - 1.0 : n);
    }
    double corr() const {
        if (co_n_ < 2) return 0.0;
        const double n = static_cast<double>(co_n_);
        const double sxy = co_sxy_ - co_sx_ * co_sy_ / n;
        const double sxx = co_sxx_ - co_sx_ * co_sx_ / n;
        const double syy = co_syy_ - co_sy_ * co_sy_ / n;
        const double denom = sxx * syy;
        if (denom <= 0.0) return 0.0;
        return sxy / std::sqrt(denom);
    }
    double regr_slope() const {
        if (co_n_ < 2) return 0.0;
        const double n = static_cast<double>(co_n_);
        const double sxy = co_sxy_ - co_sx_ * co_sy_ / n;
        const double sxx = co_sxx_ - co_sx_ * co_sx_ / n;
        if (sxx <= 0.0) return 0.0;
        return sxy / sxx;
    }
    double regr_intercept() const {
        if (co_n_ < 2) return 0.0;
        const double n = static_cast<double>(co_n_);
        return co_sy_ / n - regr_slope() * co_sx_ / n;
    }
    double regr_r2() const {
        const double r = corr();
        return r * r;
    }

    void topk_sort(std::vector<std::pair<double, std::int64_t>>& v) const {
        const bool bottom = monoid_topk_is_bottom(kind_);
        std::sort(v.begin(), v.end(),
                  [bottom](const std::pair<double, std::int64_t>& a,
                           const std::pair<double, std::int64_t>& b) {
                      if (a.first != b.first)
                          return bottom ? a.first < b.first : a.first > b.first;
                      return a.second < b.second;
                  });
    }

    // Sort by readout order and truncate to k, so which element is dropped is
    // deterministic and order-independent.
    void topk_bound() {
        if (topk_.size() <= topk_k_) return;
        topk_sort(topk_);
        topk_.resize(topk_k_);
    }

    // Union the counter sets, summing count and error where both monitor a
    // value, then keep the top k. Symmetric, so the merge stays commutative.
    void approx_merge(const MonoidAccumulator& o) {
        if (o.approx_k_ > approx_k_) approx_k_ = o.approx_k_;
        for (const ApproxEntry& oe : o.approx_) {
            bool found = false;
            for (ApproxEntry& e : approx_)
                if (e.value == oe.value) {
                    e.count += oe.count;
                    e.error += oe.error;
                    found = true;
                    break;
                }
            if (!found) approx_.push_back(oe);
        }
        if (approx_.size() <= approx_k_) return;
        std::sort(approx_.begin(), approx_.end(),
                  [](const ApproxEntry& a, const ApproxEntry& b) {
                      if (a.count != b.count) return a.count > b.count;
                      return a.value < b.value;
                  });
        approx_.resize(approx_k_);
    }

    // Same FNV-1a as map keys, so the sample and map partitioning share one
    // stable hash.
    static std::uint64_t sample_hash(std::int64_t item) {
        return dftracer::utils::hash::fnv1a_hash(&item, sizeof(item));
    }

    // Keep the k smallest distinct items by (hash, item); a repeat is ignored.
    // The tie-break on equal hash keeps the smaller item id, so the kept set is
    // independent of add and merge order.
    void sample_insert(std::uint64_t h, std::int64_t item) {
        for (const SampleEntry& e : sample_)
            if (e.item == item) return;
        if (sample_.size() < sample_k_) {
            sample_.push_back({h, item});
            return;
        }
        if (sample_.empty()) return;
        SampleEntry* worst = &sample_[0];
        for (SampleEntry& e : sample_)
            if (e.hash > worst->hash ||
                (e.hash == worst->hash && e.item > worst->item))
                worst = &e;
        if (h < worst->hash || (h == worst->hash && item < worst->item)) {
            worst->hash = h;
            worst->item = item;
        }
    }

    // DDSketch lacks a mean, so the weighted sum is tracked alongside.
    dftu_quantiles quantiles() const {
        dftu_quantiles q{};
        q.count = sketch_.count();
        q.min = sketch_.min();
        q.max = sketch_.max();
        q.mean = q.count ? sum_ / static_cast<double>(q.count) : 0.0;
        q.p50 = sketch_.quantile(0.5);
        q.p90 = sketch_.quantile(0.9);
        q.p95 = sketch_.quantile(0.95);
        q.p99 = sketch_.quantile(0.99);
        return q;
    }

    dftu_monoid_kind kind_;
    std::uint64_t u64_ = 0;
    double f64_ = 0.0;
    double sum_ = 0.0;
    utilities::common::statistics::DDSketch sketch_;
    utilities::common::statistics::DistinctSketch distinct_;
    std::unordered_set<std::int64_t> set_;
    std::vector<std::pair<std::int64_t, std::int64_t>> list_;
    dftracer::utils::dataframe::FieldStat fstat_;

    // Two-variable co-moment power sums (n and the five second-order sums).
    std::uint64_t co_n_ = 0;
    double co_sx_ = 0.0;
    double co_sy_ = 0.0;
    double co_sxx_ = 0.0;
    double co_syy_ = 0.0;
    double co_sxy_ = 0.0;

    double argby_by_ = 0.0;
    std::int64_t argby_payload_ = 0;
    bool argby_has_ = false;

    // Kept payload row (int64 slots) at the extreme `by`; the schema lives on
    // the MapAccum, the slots are stored raw here.
    double argrow_by_ = 0.0;
    std::vector<std::int64_t> argrow_payload_;
    bool argrow_has_ = false;
    std::vector<std::pair<double, std::int64_t>> topk_;
    std::uint32_t topk_k_ = 0;

    // One SpaceSaving counter; the true count is in [count - error, count].
    struct ApproxEntry {
        std::int64_t value;
        std::uint64_t count;
        std::uint64_t error;
    };
    std::vector<ApproxEntry> approx_;
    std::uint32_t approx_k_ = 0;

    // Kept sample element; hash cached so merge orders without recomputing.
    struct SampleEntry {
        std::uint64_t hash;
        std::int64_t item;
    };
    std::vector<SampleEntry> sample_;
    std::uint32_t sample_k_ = 0;
};

// cap-id hash -> merged monoid value. merge and finalize are serialized, so no
// locking is needed.
struct SharedResultRegistry {
    std::unordered_map<std::uint64_t, dftu_monoid_value> values;
};

}  // namespace dftracer::utils::plugins

#endif  // DFTRACER_UTILS_PLUGINS_MONOID_H
