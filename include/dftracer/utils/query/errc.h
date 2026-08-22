#ifndef DFTRACER_UTILS_QUERY_ERRC_H
#define DFTRACER_UTILS_QUERY_ERRC_H

#include <dftracer/utils/core/common/error.h>

#include <cstdint>

namespace dftracer::utils::query {

/// The query library's error domain (domain #1 for the extensible error id).
inline constexpr dftracer::utils::ErrorDomain ERROR_DOMAIN =
    dftracer::utils::make_error_domain("dftracer.query");

/// Query-specific error codes; portable conditions come from error_condition.
enum class QueryErrc : std::int32_t {
    Parse,        ///< syntax / tokenize failure
    Pattern,      ///< invalid match pattern (like / regex)
    Unsupported,  ///< predicate has no columnar (vec-mask) lowering
};

/// ADL opt-in: binds each code to the query domain and a portable condition, so
/// dftracer::utils::make_error(QueryErrc::...) deduces both.
constexpr dftracer::utils::ErrorDomain error_domain(QueryErrc) noexcept {
    return ERROR_DOMAIN;
}
constexpr dftracer::utils::Condition error_condition(QueryErrc e) noexcept {
    switch (e) {
        case QueryErrc::Parse:
            return dftracer::utils::Condition::Parse;
        case QueryErrc::Pattern:
            return dftracer::utils::Condition::InvalidArgument;
        case QueryErrc::Unsupported:
            return dftracer::utils::Condition::Unsupported;
    }
    return dftracer::utils::Condition::Unknown;
}

}  // namespace dftracer::utils::query

#endif  // DFTRACER_UTILS_QUERY_ERRC_H
