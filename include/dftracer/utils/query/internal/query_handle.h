#ifndef DFTRACER_UTILS_QUERY_INTERNAL_QUERY_HANDLE_H
#define DFTRACER_UTILS_QUERY_INTERNAL_QUERY_HANDLE_H

#include <dftracer/utils/query/abi.h>
#include <dftracer/utils/query/query.h>

namespace dftracer::utils::query {

// Access the Query wrapped by a dftu_query handle. Internal bridge so a C ABI
// in another translation unit (the dataframe columnar backend) can execute a
// parsed query without re-declaring the opaque handle struct.
const Query& query_handle_unwrap(const dftu_query* h);

}  // namespace dftracer::utils::query

#endif  // DFTRACER_UTILS_QUERY_INTERNAL_QUERY_HANDLE_H
