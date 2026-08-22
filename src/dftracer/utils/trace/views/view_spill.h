#ifndef DFTRACER_UTILS_TRACE_VIEWS_VIEW_SPILL_H
#define DFTRACER_UTILS_TRACE_VIEWS_VIEW_SPILL_H

#include <dftracer/utils/trace/views/view_aggregate.h>
#include <dftracer/utils/utilities/common/serialization/binary_codec.h>

#include <cstddef>
#include <fstream>
#include <string>

// Out-of-core aggregation support: (de)serialize one group accum, spill a group
// map to a sorted "run" on disk, and stream a run back one record at a time for
// a bounded k-way merge. Also used to ship rank-local partials between ranks.
namespace dftracer::utils::trace::views::detail {

namespace codec = utilities::common::serialization;

// Rough in-memory size of a group map, used only to decide when to spill. Exact
// accounting is not needed: the budget just bounds peak memory.
inline std::size_t approx_bytes(const GroupMap& m) { return m.size() * 256; }

// Encode/decode one (key, accum) with the shared binary codec.
void serialize_accum(std::string& out, const std::string& key,
                     const AggAccum& a);
void deserialize_accum(codec::BinaryReader& br, std::string& key, AggAccum& a);

// Spill a group map to `path` as length-prefixed records sorted by key (a
// "run"), then clear it. The length prefix lets a reader stream one record at a
// time, so the k-way merge stays bounded.
void spill_run(GroupMap& map, const std::string& path);

// A cursor streaming one sorted run, one record at a time.
struct RunReader {
    std::ifstream is;
    std::string buf;
    std::string key;
    AggAccum accum;
    bool valid = false;

    explicit RunReader(const std::string& path);
    void advance();
};

}  // namespace dftracer::utils::trace::views::detail

#endif  // DFTRACER_UTILS_TRACE_VIEWS_VIEW_SPILL_H
