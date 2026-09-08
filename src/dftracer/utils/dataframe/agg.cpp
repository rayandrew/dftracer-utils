#include <dftracer/utils/dataframe/agg/detail.h>

#include <bit>
#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

namespace dftracer::utils::dataframe {

std::int64_t agg_num_groups(const AggState& st) { return st.ngroups(); }

const std::vector<AggSpec>& agg_specs(const AggState& st) { return st.specs; }

std::vector<std::string> agg_group_key(const AggState& st, std::int64_t g) {
    std::vector<std::string> out;
    out.reserve(st.nkeys);
    for (std::size_t k = 0; k < st.nkeys; ++k) {
        if (st.key_is_str[k]) {
            out.push_back(st.skey_cols[k][static_cast<std::size_t>(g)]);
            continue;
        }
        const std::int64_t bits = st.ikey_cols[k][static_cast<std::size_t>(g)];
        const FieldStatDomain kd =
            k < st.key_domain.size() ? st.key_domain[k] : FieldStatDomain::I64;
        if (kd == FieldStatDomain::U64)
            out.push_back(std::to_string(std::bit_cast<std::uint64_t>(bits)));
        else if (kd == FieldStatDomain::F64)
            out.push_back(std::to_string(std::bit_cast<double>(bits)));
        else
            out.push_back(std::to_string(bits));
    }
    return out;
}

std::size_t agg_approx_bytes(const AggState& st) {
    std::size_t total = 0;
    for (std::size_t k = 0; k < st.nkeys; ++k) {
        if (st.key_is_str[k])
            for (const std::string& v : st.skey_cols[k])
                total += v.size() + sizeof(std::string);
        else
            total += st.ikey_cols[k].size() * sizeof(std::int64_t);
    }
    total += st.counts.size() * sizeof(std::uint64_t);
    total += st.fstats.size() * sizeof(FieldStat);
    if (st.has_fl) {
        total +=
            (st.fl_first.size() + st.fl_last.size()) * sizeof(std::uint64_t);
        total += (st.fl_first_idx.size() + st.fl_last_idx.size()) *
                 sizeof(std::int64_t);
        for (const std::string& v : st.fl_first_s) total += v.size();
        for (const std::string& v : st.fl_last_s) total += v.size();
    }
    if (st.has_sketch)
        for (const DDSketch& sk : st.sketches)
            total += sk.bins().size() * 24 + 64;
    if (st.has_arg) {
        total += st.arg_by.size() * sizeof(double) + st.arg_has.size();
        for (const std::string& v : st.arg_repr) total += v.size();
    }
    if (st.has_bitor) total += st.bitor_acc.size() * sizeof(std::uint64_t);
    if (st.has_kmv)
        for (const KmvMap& m : st.kmv)
            for (const auto& [h, v] : m)
                total += sizeof(std::uint64_t) + v.size() + 48;
    if (st.has_lst)
        for (const ListItems& items : st.lst)
            for (const auto& [by, repr] : items)
                total += sizeof(double) + repr.size() + sizeof(std::string);
    if (st.has_ss)
        for (const SpaceSavingMap& m : st.ss_counters)
            for (const auto& [v, c] : m)
                total += v.size() + sizeof(std::uint64_t) + 48;
    if (st.has_co)
        total += (st.co_n.size() + st.co_sx.size() + st.co_sy.size() +
                  st.co_sxx.size() + st.co_syy.size() + st.co_sxy.size()) *
                 sizeof(double);
    if (st.has_set)
        for (const std::set<std::string>& gset : st.sets)
            for (const std::string& v : gset) total += v.size() + 32;
    if (st.has_occ) {
        total += (st.occ_total.size() + st.occ_ts.size() + st.occ_te.size()) *
                 sizeof(std::uint64_t);
        for (const AggState::OccDeltas& d : st.occ_deltas)
            total += d.size() * (sizeof(std::uint64_t) + sizeof(std::int64_t));
    }
    if (st.has_dyn) {
        for (const std::map<std::string, FieldStat>& gmap : st.dyn_fs)
            for (const auto& [name, fs] : gmap)
                total += name.size() + sizeof(FieldStat) + 48;
        if (st.dyn_has_sketch)
            for (const std::map<std::string, DDSketch>& gsk : st.dyn_sketch)
                for (const auto& [name, sk] : gsk)
                    total += name.size() + sk.bins().size() * 24 + 64;
    }
    return total;
}

int agg_key_cmp(const AggState& a, std::int64_t ga, const AggState& b,
                std::int64_t gb) {
    for (std::size_t k = 0; k < a.nkeys; ++k) {
        if (a.key_is_str[k]) {
            const std::string& x = a.skey_cols[k][static_cast<std::size_t>(ga)];
            const std::string& y = b.skey_cols[k][static_cast<std::size_t>(gb)];
            if (x != y) return x < y ? -1 : 1;
        } else {
            const std::int64_t x = a.ikey_cols[k][static_cast<std::size_t>(ga)];
            const std::int64_t y = b.ikey_cols[k][static_cast<std::size_t>(gb)];
            if (x != y) return x < y ? -1 : 1;
        }
    }
    return 0;
}

}  // namespace dftracer::utils::dataframe
