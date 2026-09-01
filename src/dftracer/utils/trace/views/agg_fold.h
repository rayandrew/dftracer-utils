#ifndef DFTRACER_UTILS_TRACE_VIEWS_AGG_FOLD_H
#define DFTRACER_UTILS_TRACE_VIEWS_AGG_FOLD_H

#include <dftracer/utils/core/common/to_chars.h>
#include <dftracer/utils/dataframe/sketch.h>  // sketch_bucket_keys, DDSketch
#include <dftracer/utils/trace/aggregators/reserved_args.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/event_source.h>
#include <dftracer/utils/trace/views/view_aggregate.h>

#include <cctype>
#include <cmath>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// The aggregation fold, written once over an EventSource so it runs identically
// over a live simdjson element (DomSource) and the interned POD (PodSource).
namespace dftracer::utils::trace::views::detail {

inline void lower_ascii(std::string& s, std::size_t from) {
    for (std::size_t i = from; i < s.size(); ++i)
        s[i] =
            static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
}

// "size" is the io-cat-derived byte size, ret becoming size for POSIX/STDIO
// read|write, so aggregates match dfanalyzer.
template <class Src>
std::optional<double> derived_size_t(const Src& src) {
    namespace dfi = trace::internal;
    auto as_i64 = [&](std::string_view k) -> std::optional<std::int64_t> {
        if (auto n = src.number(k)) return static_cast<std::int64_t>(*n);
        return std::nullopt;
    };
    auto s = dfi::derive_io_size(src.value("cat"), src.value("name"),
                                 as_i64("size_sum"), as_i64("ret"),
                                 as_i64("image_size"));
    return s ? std::optional<double>(static_cast<double>(*s)) : std::nullopt;
}

template <class Src>
std::optional<double> agg_field_t(const Src& src, std::string_view field) {
    if (field == "size") return derived_size_t(src);
    if (auto n = src.number(field)) return n;
    if (field == "te") {
        auto ts = src.number("ts");
        auto dur = src.number("dur");
        if (ts && dur) return *ts + *dur;
    }
    return std::nullopt;
}

// Like agg_field_t but keeps the value's numeric domain (so integer fields stay
// exact). Derived fields (size, te) are computed in double.
template <class Src>
std::optional<dftracer::utils::dataframe::FieldNum> agg_field_typed_t(
    const Src& src, std::string_view field) {
    // size and te are integer-valued (byte counts, timestamps), so they use the
    // U64 domain to match the tier, which stores them as uint64.
    if (field == "size") {
        if (auto s = derived_size_t(src))
            return dftracer::utils::dataframe::FieldNum::of(
                static_cast<std::uint64_t>(*s));
        return std::nullopt;
    }
    if (auto n = src.number_typed(field)) return n;
    if (field == "te") {
        auto ts = src.number_typed("ts");
        auto dur = src.number_typed("dur");
        if (ts && dur)
            return dftracer::utils::dataframe::FieldNum::of(ts->as_u64() +
                                                            dur->as_u64());
    }
    return std::nullopt;
}

// The per-dimension group value, appended straight into the key buffer. Group
// dims fhash/hhash/arg are args-only; name/cat/pid/tid are top-then-args; cat
// is lowercased so a raw scan and the lowercased tier agree.
template <class Src>
void append_group_dim(std::string& out, const Src& src, const GroupKey& gk) {
    switch (gk.kind) {
        case GroupKey::Kind::Name:
            src.append_value(out, "name");
            break;
        case GroupKey::Kind::Cat: {
            const std::size_t start = out.size();
            src.append_value(out, "cat");
            lower_ascii(out, start);
            break;
        }
        case GroupKey::Kind::Pid:
            src.append_value(out, "pid");
            break;
        case GroupKey::Kind::Tid:
            src.append_value(out, "tid");
            break;
        case GroupKey::Kind::Fhash:
        case GroupKey::Kind::FilePath:
        case GroupKey::Kind::FileName:
            src.append_arg(out, "fhash");
            break;
        case GroupKey::Kind::Hhash:
        case GroupKey::Kind::HostName:
            src.append_arg(out, "hhash");
            break;
        case GroupKey::Kind::IoCat: {
            char buf[8];
            char* p = to_chars_i64(
                buf, buf + sizeof(buf),
                static_cast<int>(
                    trace::internal::io_category(src.value("name"))));
            out.append(buf, static_cast<std::size_t>(p - buf));
            break;
        }
        case GroupKey::Kind::AccPat:
            out.push_back('0');
            break;
        case GroupKey::Kind::Rank:
            // Group on pid; the post-aggregation resolver relabels it to rank.
            src.append_value(out, "pid");
            break;
        case GroupKey::Kind::Arg:
            src.append_arg(out, gk.arg);
            break;
        case GroupKey::Kind::Field:
            src.append_value(out, gk.arg);
            break;
    }
}

template <class Src>
std::string group_dim_str(const Src& src, const GroupKey& gk) {
    std::string s;
    append_group_dim(s, src, gk);
    return s;
}

template <class Src>
void fold_numeric_args_t(AggAccum& a, const Src& src, bool want_sketch) {
    namespace agg = trace::aggregators;
    auto feed = [&](std::string_view key, double num) {
        a.dyn[std::string(key)].add(num);
        if (want_sketch) a.dyn_sketches[std::string(key)].add(num);
    };
    if (auto sz = derived_size_t(src)) feed("size", *sz);
    src.for_each_numeric_arg([&](std::string_view key, double num) {
        if (agg::is_reserved_arg(key) || agg::is_preagg_suffix(key)) return;
        feed(key, num);
    });
}

// Deferred per-batch DDSketch updates. Stores the group's value-INDEX (not a
// pointer): unordered_dense inserts are append-only and the fold never erases,
// so the index survives map growth where a pointer would dangle.
struct SketchBatch {
    std::vector<double> vals;
    std::vector<std::uint32_t> gidx;
    std::vector<std::int32_t> slot;
    std::vector<std::int32_t> keys;  // scratch out
    bool empty() const { return vals.empty(); }
    void record(double v, std::uint32_t g, std::int32_t s) {
        vals.push_back(v);
        gidx.push_back(g);
        slot.push_back(s);
    }
    void clear() {
        vals.clear();
        gidx.clear();
        slot.clear();
    }
};

inline void flush_sketch_batch(GroupMap& map, SketchBatch& sb) {
    if (sb.empty()) return;
    const std::size_t n = sb.vals.size();
    sb.keys.resize(n);
    dftracer::utils::dataframe::sketch_bucket_keys(
        sb.vals.data(), static_cast<std::int64_t>(n),
        dftracer::utils::dataframe::DDSketch{}.log_gamma(), sb.keys.data());
    for (std::size_t i = 0; i < n; ++i)
        (map.begin() + static_cast<std::ptrdiff_t>(sb.gidx[i]))
            ->second.sketches[static_cast<std::size_t>(sb.slot[i])]
            .add_key(sb.keys[i]);
    sb.clear();
}

// One event folded into `map`. `keybuf` is caller-owned scratch reused across
// events so the hot path allocates no per-event key. A non-null `sketch_batch`
// defers DDSketch updates into it (batched, SIMD-keyed at flush); null runs
// them inline (scalar log).
template <class Src>
void fold_event_over(GroupMap& map, const Src& src, const ViewPlan& plan,
                     std::string& keybuf, SketchBatch* sketch_batch = nullptr) {
    const AggSchema& sch = *plan.schema;
    keybuf.clear();
    std::int64_t bucket = 0;
    const bool has_bucket = plan.time_bucket_us > 0;

    if (has_bucket) {
        auto ts = agg_field_t(src, "ts");
        if (!ts) return;
        const auto interval = static_cast<double>(plan.time_bucket_us);
        const auto w = static_cast<std::int64_t>(plan.time_bucket_us);
        const auto origin = static_cast<std::int64_t>(plan.bucket_origin_us);
        // Floor (ts_scaled - origin) toward -inf so buckets tile evenly on
        // either side of the origin, then shift back by origin.
        const double rel = *ts * plan.time_scale - static_cast<double>(origin);
        bucket =
            static_cast<std::int64_t>(std::floor(rel / interval)) * w + origin;
        char b[24];
        char* p = to_chars_i64(b, b + sizeof(b), bucket);
        keybuf.append(b, static_cast<std::size_t>(p - b));
        keybuf += GROUP_SEP;
    }
    for (const auto& gk : plan.group_by) {
        append_group_dim(keybuf, src, gk);
        keybuf += GROUP_SEP;
    }

    auto it = map.find(keybuf);
    if (it == map.end()) {
        AggAccum a;
        a.keys.reserve(plan.group_by.size() + (has_bucket ? 1 : 0));
        if (has_bucket) a.keys.push_back(std::to_string(bucket));
        for (const auto& gk : plan.group_by)
            a.keys.push_back(group_dim_str(src, gk));
        a.fields.resize(sch.fields.size());
        a.argmax.resize(sch.argmax_count);
        a.sketches.resize(sch.sketch_count);
        a.sets.resize(sch.set_count);
        it = map.emplace(keybuf, std::move(a)).first;
    }
    AggAccum& a = it->second;
    ++a.count;
    const std::uint32_t gidx =
        sketch_batch ? static_cast<std::uint32_t>(it - map.begin()) : 0;

    // Occupancy (time-window reduction): record the event's endpoints as a
    // +1/-1 delta pair; finalize sweeps for the exact interval union.
    if (sch.want_occupancy) {
        auto ts = src.number("ts");
        auto dur = src.number("dur");
        if (ts && dur && *dur > 0) {
            std::uint64_t s = static_cast<std::uint64_t>(*ts);
            std::uint64_t e = s + static_cast<std::uint64_t>(*dur);
            const std::uint64_t cell = sch.occ_cell_us;
            if (cell) {  // optional tolerance: snap start down, end up
                s = s / cell * cell;
                e = (e + cell - 1) / cell * cell;
            }
            a.occ_cell_us = cell;
            a.occ_total += static_cast<std::uint64_t>(*dur);
            if (s < a.occ_ts) a.occ_ts = s;
            if (e > a.occ_te) a.occ_te = e;
            a.occ_deltas[s] += 1;
            a.occ_deltas[e] -= 1;
        }
    }

    for (std::size_t fi = 0; fi < sch.fields.size(); ++fi) {
        auto v = agg_field_typed_t(src, sch.fields[fi]);
        if (!v) continue;
        // A fractional time_scale turns an integer ts/dur into a real value, so
        // that field accumulates in double; otherwise the native domain is
        // kept.
        const int sk = sch.field_sketch[fi];
        if (sch.field_scaled[fi] && plan.time_scale != 1.0) {
            const double x = v->as_double() * plan.time_scale;
            a.fields[fi].add(x);
            if (sk >= 0) {
                if (sketch_batch)
                    sketch_batch->record(x, gidx, sk);
                else
                    a.sketches[static_cast<std::size_t>(sk)].add(x);
            }
        } else {
            a.fields[fi].add(*v);
            if (sk >= 0) {
                const double x = v->as_double();
                if (sketch_batch)
                    sketch_batch->record(x, gidx, sk);
                else
                    a.sketches[static_cast<std::size_t>(sk)].add(x);
            }
        }
    }
    for (std::size_t i = 0; i < plan.agg.size(); ++i) {
        const int slot = sch.spec_argmax[i];
        if (slot < 0) continue;
        auto by = agg_field_t(src, plan.agg[i].by);
        if (!by) continue;
        auto& am = a.argmax[slot];
        if (!am.has || *by > am.by) {
            am.by = *by;
            am.repr = src.value(plan.agg[i].field);
            am.has = true;
        }
    }
    for (std::size_t i = 0; i < plan.agg.size(); ++i) {
        const int slot = sch.spec_set[i];
        if (slot < 0) continue;
        std::string v = src.value(plan.agg[i].field);
        if (!v.empty()) a.sets[slot].insert(std::move(v));
    }
    if (plan.auto_numeric_metrics) fold_numeric_args_t(a, src, sch.dyn_sketch);
}

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_AGG_FOLD_H
