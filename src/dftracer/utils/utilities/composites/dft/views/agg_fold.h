#ifndef DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_AGG_FOLD_H
#define DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_AGG_FOLD_H

#include <dftracer/utils/core/common/to_chars.h>
#include <dftracer/utils/utilities/composites/dft/aggregators/reserved_args.h>
#include <dftracer/utils/utilities/composites/dft/internal/utils.h>
#include <dftracer/utils/utilities/composites/dft/views/event_source.h>
#include <dftracer/utils/utilities/composites/dft/views/view_aggregate.h>

#include <cctype>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

// The aggregation fold, written once over an EventSource so it runs identically
// over a live simdjson element (DomSource) and the interned POD (PodSource).
namespace dftracer::utils::utilities::composites::dft::views::detail {

inline void lower_ascii(std::string& s, std::size_t from) {
    for (std::size_t i = from; i < s.size(); ++i)
        s[i] =
            static_cast<char>(std::tolower(static_cast<unsigned char>(s[i])));
}

// "size" is the io-cat-derived byte size, ret becoming size for POSIX/STDIO
// read|write, so aggregates match dfanalyzer.
template <class Src>
std::optional<double> derived_size_t(const Src& src) {
    namespace dfi = composites::dft::internal;
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
                    composites::dft::internal::io_category(src.value("name"))));
            out.append(buf, static_cast<std::size_t>(p - buf));
            break;
        }
        case GroupKey::Kind::AccPat:
            out.push_back('0');
            break;
        case GroupKey::Kind::Arg:
            src.append_arg(out, gk.arg);
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
void fold_numeric_args_t(AggAccum& a, const Src& src) {
    namespace agg = composites::dft::aggregators;
    if (auto sz = derived_size_t(src)) a.dyn["size"].add(*sz);
    src.for_each_numeric_arg([&](std::string_view key, double num) {
        if (agg::is_reserved_arg(key) || agg::is_preagg_suffix(key)) return;
        a.dyn[std::string(key)].add(num);
    });
}

// One event folded into `map`. `keybuf` is caller-owned scratch reused across
// events so the hot path allocates no per-event key.
template <class Src>
void fold_event_over(GroupMap& map, const Src& src, const ViewPlan& plan,
                     std::string& keybuf) {
    const AggSchema& sch = *plan.schema;
    keybuf.clear();
    std::int64_t bucket = 0;
    const bool has_bucket = plan.time_bucket_us > 0;

    if (has_bucket) {
        auto ts = agg_field_t(src, "ts");
        if (!ts) return;
        auto interval = static_cast<double>(plan.time_bucket_us);
        bucket = static_cast<std::int64_t>(*ts * plan.time_scale / interval) *
                 static_cast<std::int64_t>(plan.time_bucket_us);
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

    for (std::size_t fi = 0; fi < sch.fields.size(); ++fi) {
        auto v = agg_field_t(src, sch.fields[fi]);
        if (!v) continue;
        double x = *v;
        if (sch.field_scaled[fi] && plan.time_scale != 1.0)
            x *= plan.time_scale;
        a.fields[fi].add(x);
        if (sch.field_sketch[fi] >= 0) a.sketches[sch.field_sketch[fi]].add(x);
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
    if (plan.auto_numeric_metrics) fold_numeric_args_t(a, src);
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail

#endif  // DFTRACER_UTILS_UTILITIES_COMPOSITES_DFT_VIEWS_AGG_FOLD_H
