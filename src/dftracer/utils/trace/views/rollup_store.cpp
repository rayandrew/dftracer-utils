#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/common/hash/fnv1a.h>
#include <dftracer/utils/core/rocksdb/column_families.h>
#include <dftracer/utils/core/rocksdb/db_manager.h>
#include <dftracer/utils/trace/views/rollup_store.h>
#include <dftracer/utils/trace/views/view_plan.h>
#include <dftracer/utils/trace/views/view_spill.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <system_error>
#include <utility>
#include <vector>

namespace dftracer::utils::trace::views::detail {

namespace rdb = dftracer::utils::rocksdb;
namespace codec = utilities::common::serialization;

namespace {

void put_be64(std::string& out, std::uint64_t v) {
    for (int shift = 56; shift >= 0; shift -= 8)
        out.push_back(static_cast<char>((v >> shift) & 0xFF));
}

std::uint64_t get_be64(std::string_view b) {
    std::uint64_t v = 0;
    for (unsigned char c : b) v = (v << 8) | c;
    return v;
}

// Every plan field except group_by - the basis shared by plan_signature and
// rest_signature.
void add_rest_fields(std::string& sig, const ViewPlan& plan) {
    auto add = [&](std::string_view s) {
        sig.append(s);
        sig.push_back('\0');
    };
    std::vector<const ViewFile*> files;
    files.reserve(plan.files.size());
    for (const auto& f : plan.files) files.push_back(&f);
    std::sort(files.begin(), files.end(),
              [](auto* a, auto* b) { return a->file_path < b->file_path; });
    for (auto* f : files) {
        add(f->file_path);
        std::error_code ec;
        add(std::to_string(fs::file_size(f->file_path, ec)));
        auto wt = fs::last_write_time(f->file_path, ec);
        add(std::to_string(
            static_cast<long long>(ec ? 0 : wt.time_since_epoch().count())));
    }
    add(plan.query ? plan.query->source() : "");
    add(std::to_string(static_cast<int>(plan.phase)));
    // time_bucket_us is intentionally NOT part of the rest basis: a coarser
    // query can be served by re-bucketing a finer rollup, so buckets must not
    // gate subsumption. It lands in plan_signature (row-key identity) only.
    add(std::to_string(plan.time_scale));
    if (plan.time_range) {
        add(std::to_string(plan.time_range->first));
        add(std::to_string(plan.time_range->second));
    }
    for (const auto& a : plan.agg) {
        add(std::to_string(static_cast<int>(a.op)));
        add(a.field);
        add(a.out_name);
        add(a.by);
    }
    add(plan.auto_numeric_metrics ? "1" : "0");
    for (const auto& s : plan.select) add(s);
}

// Roll `src` (grouped by `src_gb` at `src_bucket` time grain, key layout
// [time_bucket?] + dims) up to `dst`'s grouping, a subset, merging rows that
// collapse to the same key. When `dst`'s bucket is coarser than `src_bucket`,
// the retained bucket-start key is re-floored to `dst`'s grain so finer buckets
// fold together. Empty if `dst`'s grouping is not a subset of `src_gb`.
GroupMap reaggregate_to(const GroupMap& src,
                        const std::vector<GroupKey>& src_gb,
                        std::uint64_t src_bucket, const ViewPlan& dst) {
    const std::size_t off = dst.time_bucket_us > 0 ? 1 : 0;
    // src_bucket evenly divides dst.time_bucket_us (guaranteed by the matcher),
    // and stored bucket-starts are multiples of src_bucket, so an integer floor
    // to the coarser grain is exact - no re-scaling by time_scale needed.
    const bool recut =
        off && src_bucket > 0 && dst.time_bucket_us != src_bucket;
    std::vector<std::size_t> pos;
    pos.reserve(dst.group_by.size());
    for (const auto& dg : dst.group_by) {
        std::size_t at = src_gb.size();
        for (std::size_t i = 0; i < src_gb.size(); ++i)
            if (src_gb[i].kind == dg.kind && src_gb[i].arg == dg.arg) {
                at = i;
                break;
            }
        if (at == src_gb.size()) return {};
        pos.push_back(at);
    }
    GroupMap out;
    std::string newkey;
    for (const auto& [k, accum] : src) {
        AggAccum a = accum;
        std::vector<std::string> keys;
        keys.reserve(off + pos.size());
        if (off && !accum.keys.empty()) {
            if (recut) {
                const std::int64_t b = std::stoll(accum.keys[0]);
                const std::int64_t q =
                    static_cast<std::int64_t>(dst.time_bucket_us);
                keys.push_back(std::to_string((b / q) * q));
            } else {
                keys.push_back(accum.keys[0]);
            }
        }
        for (std::size_t p : pos)
            keys.push_back(off + p < accum.keys.size() ? accum.keys[off + p]
                                                       : std::string());
        a.keys = keys;
        newkey.clear();
        for (const auto& part : keys) {
            newkey += part;
            newkey += GROUP_SEP;
        }
        merge_accum(out[newkey], a, dst);
    }
    return out;
}

}  // namespace

std::string rollup_row_key(std::uint64_t sig, std::string_view group_key) {
    std::string key;
    key.reserve(9 + group_key.size());
    key.push_back('\x01');
    put_be64(key, sig);
    key.append(group_key);
    return key;
}

std::string rollup_desc_key(std::uint64_t sig) {
    std::string key;
    key.reserve(9);
    key.push_back('\x00');
    put_be64(key, sig);
    return key;
}

void merge_accum_free(AggAccum& da, const AggAccum& sa) {
    if (da.count == 0 && da.fields.empty() && da.argmax.empty() &&
        da.sketches.empty() && da.dyn.empty() && da.sets.empty()) {
        da = sa;
        return;
    }
    da.count += sa.count;
    for (std::size_t i = 0; i < da.fields.size() && i < sa.fields.size(); ++i)
        da.fields[i].merge(sa.fields[i]);
    for (std::size_t i = 0; i < da.argmax.size() && i < sa.argmax.size(); ++i) {
        const auto& s = sa.argmax[i];
        if (!s.has) continue;
        auto& d = da.argmax[i];
        if (!d.has || s.by > d.by) d = s;
    }
    for (std::size_t i = 0; i < da.sketches.size() && i < sa.sketches.size();
         ++i)
        da.sketches[i].merge(sa.sketches[i]);
    if (da.sets.size() < sa.sets.size()) da.sets.resize(sa.sets.size());
    for (std::size_t i = 0; i < sa.sets.size(); ++i)
        da.sets[i].insert(sa.sets[i].begin(), sa.sets[i].end());
    for (const auto& [name, sm] : sa.dyn) da.dyn[name].merge(sm);
}

std::shared_ptr<rdb::RocksDatabase> open_rollup_db(
    const std::string& index_path, rdb::RocksDatabase::OpenMode mode) {
    return rdb::RocksDBManager::instance().get_or_open(index_path, mode);
}

void persist_rollup(rdb::RocksDatabase& db, std::uint64_t sig,
                    std::uint64_t rest_sig, std::uint64_t time_bucket_us,
                    const std::vector<GroupKey>& group_by,
                    const GroupMap& map) {
    std::string val;
    for (const auto& [group_key, accum] : map) {
        val.clear();
        serialize_accum(val, group_key, accum);
        if (const auto st =
                db.put(rollup_row_key(sig, group_key), val, rdb::cf::ROLLUP);
            !st.ok())
            throw DFTUtilsException(ErrorCode::IO,
                                    "rollup persist: " + st.ToString());
    }
    // Shape descriptor the planner matches against: rest hash + this rollup's
    // time bucket (for coarsening) + this view's grouping (kind + arg per key,
    // in order).
    std::string desc;
    codec::put_be64(desc, rest_sig);
    codec::put_be64(desc, time_bucket_us);
    codec::put_be32(desc, static_cast<std::uint32_t>(group_by.size()));
    for (const auto& gk : group_by) {
        codec::put_be32(desc, static_cast<std::uint32_t>(gk.kind));
        codec::put_str(desc, gk.arg);
    }
    if (const auto st = db.put(rollup_desc_key(sig), desc, rdb::cf::ROLLUP);
        !st.ok())
        throw DFTUtilsException(ErrorCode::IO,
                                "rollup descriptor: " + st.ToString());
}

bool rollup_exists(const rdb::RocksDatabase& db, std::uint64_t sig) {
    std::string val;
    return db.get(rollup_desc_key(sig), &val, rdb::cf::ROLLUP).ok();
}

GroupMap read_rollup(const rdb::RocksDatabase& db, std::uint64_t sig) {
    GroupMap out;
    const std::string prefix = rollup_row_key(sig, "");  // 0x01 | sig
    auto it = db.new_iterator(rdb::cf::ROLLUP);
    for (it->Seek(prefix); it->Valid(); it->Next()) {
        const std::string_view k(it->key().data(), it->key().size());
        if (k.size() < prefix.size() || k.substr(0, prefix.size()) != prefix)
            break;
        const std::string_view v(it->value().data(), it->value().size());
        codec::BinaryReader br(v);
        std::string row_key;
        AggAccum a;
        deserialize_accum(br, row_key, a);
        out.emplace(std::string(k.substr(prefix.size())), std::move(a));
    }
    return out;
}

std::uint64_t rest_signature(const ViewPlan& plan) {
    std::string sig;
    add_rest_fields(sig, plan);
    return dftracer::utils::hash::fnv1a_hash(sig);
}

std::uint64_t plan_signature(const ViewPlan& plan) {
    std::string sig;
    add_rest_fields(sig, plan);
    sig.append(std::to_string(plan.time_bucket_us));
    sig.push_back('\0');
    for (const auto& g : plan.group_by) {
        sig.append(std::to_string(static_cast<int>(g.kind)));
        sig.push_back('\0');
        sig.append(g.arg);
        sig.push_back('\0');
    }
    return dftracer::utils::hash::fnv1a_hash(sig);
}

std::optional<GroupMap> find_subsuming_rollup(const rdb::RocksDatabase& db,
                                              const ViewPlan& plan) {
    const std::uint64_t q_rest = rest_signature(plan);
    const char desc_tag = '\x00';

    // Pick the tightest subsuming rollup: fewest group dims means fewest rows
    // to re-aggregate, and an exact match (same dims as the query) is optimal.
    std::uint64_t best_sig = 0;
    std::uint64_t best_bucket = 0;
    std::vector<GroupKey> best_gb;
    bool have_best = false;

    auto it = db.new_iterator(rdb::cf::ROLLUP);
    for (it->Seek(std::string_view(&desc_tag, 1)); it->Valid(); it->Next()) {
        const std::string_view k(it->key().data(), it->key().size());
        if (k.empty() || k[0] != '\x00') break;
        if (k.size() != 9) continue;

        const std::string_view v(it->value().data(), it->value().size());
        if (v.size() < 20) continue;
        codec::BinaryReader br(v);
        if (br.be64() != q_rest) continue;
        const std::uint64_t r_bucket = br.be64();

        // Bucket compatibility: a query with a time bucket can only be served
        // by a rollup whose bucket evenly divides it (coarsen finer ->
        // coarser). A query without a bucket collapses any rollup's buckets, so
        // it always matches. The reverse (finer query, coarser rollup) is
        // unserviceable.
        if (plan.time_bucket_us > 0 &&
            (r_bucket == 0 || plan.time_bucket_us % r_bucket != 0))
            continue;

        std::vector<GroupKey> r_gb;
        const std::uint32_t n = br.be32();
        r_gb.reserve(n);
        for (std::uint32_t i = 0; i < n; ++i) {
            GroupKey gk;
            gk.kind = static_cast<GroupKey::Kind>(br.be32());
            gk.arg = std::string(br.str());
            r_gb.push_back(std::move(gk));
        }
        if (have_best && r_gb.size() >= best_gb.size()) continue;

        bool subset = true;
        for (const auto& qg : plan.group_by) {
            bool found = false;
            for (const auto& rg : r_gb)
                if (rg.kind == qg.kind && rg.arg == qg.arg) {
                    found = true;
                    break;
                }
            if (!found) {
                subset = false;
                break;
            }
        }
        if (!subset) continue;

        best_sig = get_be64(k.substr(1, 8));
        best_bucket = r_bucket;
        best_gb = std::move(r_gb);
        have_best = true;
        if (best_gb.size() == plan.group_by.size() &&
            best_bucket == plan.time_bucket_us)
            break;  // exact: optimal
    }

    if (!have_best) return std::nullopt;
    GroupMap rows = read_rollup(db, best_sig);
    if (rows.empty()) return std::nullopt;
    return reaggregate_to(rows, best_gb, best_bucket, plan);
}

std::string rollup_index_path(const ViewPlan& plan) {
    // A Rank group key resolves pid -> rank from PR metadata harvested during
    // the scan; a rollup read-back never scans, so a rank plan cannot use a
    // rollup. An empty path keeps it on the scan.
    for (const auto& gk : plan.group_by)
        if (gk.kind == GroupKey::Kind::Rank) return {};
    // A caller-set shared root anchors a cross-file rollup (dask sets it for a
    // multi-file query, whose per-file index paths differ); otherwise the one
    // index all files share, if any (single-file and single-index cases).
    if (!plan.rollup_root.empty()) return plan.rollup_root;
    if (plan.files.empty()) return {};
    const std::string& p = plan.files.front().index_path;
    if (p.empty()) return {};
    for (const auto& f : plan.files)
        if (f.index_path != p) return {};
    return p;
}

}  // namespace dftracer::utils::trace::views::detail
