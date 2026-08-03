#include <dftracer/utils/utilities/composites/dft/views/view_spill.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace dftracer::utils::utilities::composites::dft::views::detail {

namespace {
void put_stat(std::string& out, const FieldStat& f) {
    codec::put_be64(out, f.n);
    codec::put_double(out, f.sum);
    codec::put_double(out, f.sumsq);
    codec::put_double(out, f.m3);
    codec::put_double(out, f.m4);
    codec::put_double(out, f.min);
    codec::put_double(out, f.max);
}
FieldStat get_stat(codec::BinaryReader& br) {
    FieldStat f;
    f.n = br.be64();
    f.sum = br.f64();
    f.sumsq = br.f64();
    f.m3 = br.f64();
    f.m4 = br.f64();
    f.min = br.f64();
    f.max = br.f64();
    return f;
}
}  // namespace

void serialize_accum(std::string& out, const std::string& key,
                     const AggAccum& a) {
    codec::put_str(out, key);
    codec::put_be64(out, a.count);
    codec::put_be64(out, a.keys.size());
    for (const auto& k : a.keys) codec::put_str(out, k);
    codec::put_be64(out, a.fields.size());
    for (const auto& f : a.fields) put_stat(out, f);
    codec::put_be64(out, a.argmax.size());
    for (const auto& am : a.argmax) {
        codec::put_u8(out, am.has ? 1 : 0);
        codec::put_double(out, am.by);
        codec::put_str(out, am.repr);
    }
    codec::put_be64(out, a.sketches.size());
    for (const auto& sk : a.sketches) {
        auto bytes = sk.serialize();
        codec::put_str(out, std::string(bytes.begin(), bytes.end()));
    }
    codec::put_be64(out, a.dyn.size());
    for (const auto& [name, m] : a.dyn) {
        codec::put_str(out, name);
        put_stat(out, m);
    }
    codec::put_be64(out, a.sets.size());
    for (const auto& s : a.sets) {
        codec::put_be64(out, s.size());
        for (const auto& v : s) codec::put_str(out, v);
    }
}

void deserialize_accum(codec::BinaryReader& br, std::string& key, AggAccum& a) {
    key = std::string(br.str());
    a.count = br.be64();
    a.keys.resize(br.be64());
    for (auto& k : a.keys) k = std::string(br.str());
    a.fields.resize(br.be64());
    for (auto& f : a.fields) f = get_stat(br);
    a.argmax.resize(br.be64());
    for (auto& am : a.argmax) {
        am.has = br.u8() != 0;
        am.by = br.f64();
        am.repr = std::string(br.str());
    }
    const std::uint64_t nsk = br.be64();
    a.sketches.clear();
    a.sketches.reserve(nsk);
    for (std::uint64_t i = 0; i < nsk; ++i) {
        auto sv = br.str();
        a.sketches.push_back(common::statistics::DDSketch::deserialize(
            reinterpret_cast<const std::uint8_t*>(sv.data()), sv.size()));
    }
    const std::uint64_t ndyn = br.be64();
    for (std::uint64_t i = 0; i < ndyn; ++i) {
        std::string name(br.str());
        a.dyn.emplace(std::move(name), get_stat(br));
    }
    a.sets.resize(br.be64());
    for (auto& s : a.sets) {
        const std::uint64_t n = br.be64();
        for (std::uint64_t j = 0; j < n; ++j) s.insert(std::string(br.str()));
    }
}

namespace {

using Entry = std::pair<const std::string, AggAccum>;

void write_run(const std::string& path, std::vector<const Entry*>& ents) {
    std::sort(ents.begin(), ents.end(),
              [](auto* a, auto* b) { return a->first < b->first; });
    std::ofstream os(path, std::ios::binary);
    std::string rec, hdr;
    for (auto* e : ents) {
        rec.clear();
        serialize_accum(rec, e->first, e->second);
        hdr.clear();
        codec::put_be64(hdr, rec.size());
        os.write(hdr.data(), static_cast<std::streamsize>(hdr.size()));
        os.write(rec.data(), static_cast<std::streamsize>(rec.size()));
    }
}

}  // namespace

void spill_run(GroupMap& map, const std::string& path) {
    std::vector<const Entry*> ents;
    ents.reserve(map.size());
    for (const auto& kv : map) ents.push_back(&kv);
    write_run(path, ents);
    map.clear();
}

RunReader::RunReader(const std::string& path) : is(path, std::ios::binary) {
    advance();
}

void RunReader::advance() {
    accum = AggAccum{};
    char hdr[8];
    if (!is.read(hdr, 8)) {
        valid = false;
        return;
    }
    std::uint64_t len = 0;
    for (int i = 0; i < 8; ++i)
        len = (len << 8) | static_cast<std::uint8_t>(hdr[i]);
    buf.resize(len);
    if (len && !is.read(buf.data(), static_cast<std::streamsize>(len))) {
        valid = false;
        return;
    }
    codec::BinaryReader br(buf);
    deserialize_accum(br, key, accum);
    valid = true;
}

}  // namespace dftracer::utils::utilities::composites::dft::views::detail
