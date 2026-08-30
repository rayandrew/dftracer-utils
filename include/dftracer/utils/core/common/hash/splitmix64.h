#ifndef DFTRACER_UTILS_CORE_COMMON_HASH_SPLITMIX64_H
#define DFTRACER_UTILS_CORE_COMMON_HASH_SPLITMIX64_H

#include <cstdint>

namespace dftracer::utils::hash {

/// SplitMix64 finalizer: a fast, well-avalanched hash of a 64-bit value.
///
/// The finalizer from Sebastiano Vigna's SplitMix64 generator (public domain /
/// CC0): https://prng.di.unimi.it/splitmix64.c. The plugin ABI keeps its own
/// self-contained copy as dftu_mix64 (plugins/prims.h); the constants match, so
/// the two agree bit-for-bit.
inline std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

}  // namespace dftracer::utils::hash

#endif  // DFTRACER_UTILS_CORE_COMMON_HASH_SPLITMIX64_H
