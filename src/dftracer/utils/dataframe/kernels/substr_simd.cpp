#include <dftracer/utils/dataframe/internal/substr_simd.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#undef HWY_TARGET_INCLUDE
#define HWY_TARGET_INCLUDE "dftracer/utils/dataframe/kernels/substr_simd.cpp"
#include <hwy/foreach_target.h>  // must precede highway.h
#include <hwy/highway.h>

HWY_BEFORE_NAMESPACE();
namespace dftracer::utils::dataframe {
namespace HWY_NAMESPACE {
namespace hn = hwy::HWY_NAMESPACE;

// Needles at or below this length take the SIMD first-byte scan; longer ones
// fall back to a scalar search (the per-candidate memcmp dominates, so the
// vector filter stops paying off). The trailing tail shorter than a vector is
// likewise scalar.
constexpr std::int64_t SUBSTR_SIMD_MAX_NEEDLE = 64;

// First index in [0, hay_len - needle_len] whose byte equals needle[0] and
// whose following needle_len bytes match, or -1. Callers handle the empty
// needle and needle-longer-than-haystack cases before dispatch.
std::int64_t SubstrFind(const char* hay, std::int64_t hay_len,
                        const char* needle, std::int64_t needle_len) {
    const std::uint8_t* hp = reinterpret_cast<const std::uint8_t*>(hay);
    const std::int64_t last = hay_len - needle_len;  // inclusive last start
    const std::int64_t m = needle_len;

    if (m > SUBSTR_SIMD_MAX_NEEDLE) {
        for (std::int64_t i = 0; i <= last; ++i)
            if (hp[i] == static_cast<std::uint8_t>(needle[0]) &&
                std::memcmp(hp + i, needle, static_cast<std::size_t>(m)) == 0)
                return i;
        return -1;
    }

    const hn::ScalableTag<std::uint8_t> d;
    const std::size_t lanes = hn::Lanes(d);
    const auto first = hn::Set(d, static_cast<std::uint8_t>(needle[0]));
    std::int64_t i = 0;
    // The candidate offsets in a block are read from a 64-bit mask accumulator;
    // skip the vector loop when a target's u8 vector exceeds 64 lanes (some
    // scalable ISAs), leaving the scalar tail to cover every position.
    const bool vec_ok = lanes <= 64;
    // Load `lanes` bytes at a time; require i + lanes <= last + 1 so the load
    // stays in bounds (last + 1 <= hay_len for m >= 1) and every candidate has
    // a full needle after it.
    for (; vec_ok && i + static_cast<std::int64_t>(lanes) <= last + 1;
         i += static_cast<std::int64_t>(lanes)) {
        const auto v = hn::LoadU(d, hp + i);
        const auto eq = hn::Eq(v, first);
        alignas(8) std::uint8_t mbytes[8] = {0};
        hn::StoreMaskBits(d, eq, mbytes);
        std::uint64_t bits;
        std::memcpy(&bits, mbytes, sizeof(bits));
        while (bits != 0) {
            const std::int64_t cand =
                i + static_cast<std::int64_t>(
                        hwy::Num0BitsBelowLS1Bit_Nonzero64(bits));
            if (std::memcmp(hp + cand, needle, static_cast<std::size_t>(m)) ==
                0)
                return cand;
            bits &= bits - 1;
        }
    }
    for (; i <= last; ++i)
        if (hp[i] == static_cast<std::uint8_t>(needle[0]) &&
            std::memcmp(hp + i, needle, static_cast<std::size_t>(m)) == 0)
            return i;
    return -1;
}

}  // namespace HWY_NAMESPACE
}  // namespace dftracer::utils::dataframe
HWY_AFTER_NAMESPACE();

#if HWY_ONCE
namespace dftracer::utils::dataframe {

HWY_EXPORT(SubstrFind);

std::int64_t substr_find(const char* hay, std::int64_t hay_len,
                         const char* needle, std::int64_t needle_len) {
    if (needle_len <= 0)
        return 0;  // empty needle matches at 0 (find semantics)
    if (needle_len > hay_len) return -1;
    return HWY_DYNAMIC_DISPATCH(SubstrFind)(hay, hay_len, needle, needle_len);
}

}  // namespace dftracer::utils::dataframe
#endif  // HWY_ONCE
