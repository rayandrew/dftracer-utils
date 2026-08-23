#ifndef DFTRACER_UTILS_QUERY_PATTERN_H
#define DFTRACER_UTILS_QUERY_PATTERN_H

#include <regex>

namespace dftracer::utils::query {

/// Compiled matcher backing a MatchNode.
struct CompiledPattern {
    std::regex re;
};

}  // namespace dftracer::utils::query

#endif  // DFTRACER_UTILS_QUERY_PATTERN_H
