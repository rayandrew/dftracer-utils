#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/mask.h>
#include <dftracer/utils/query/abi.h>
#include <dftracer/utils/query/internal/query_handle.h>

#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>

namespace dataframe = dftracer::utils::dataframe;

namespace {}  // namespace

namespace dftracer::utils::dataframe {

Series DataFrame::mask(const query::Query& q) const {
    std::vector<const dftu_series*> cols;
    std::vector<const char*> col_names;
    cols.reserve(columns.size());
    col_names.reserve(names.size());
    for (const Series& c : columns) cols.push_back(c.handle());
    for (const std::string& n : names) col_names.push_back(n.c_str());

    dftu_query* handle = dftu_query_parse(q.source().c_str());
    if (!handle)
        throw std::runtime_error("DataFrame::mask: failed to compile query");
    dftu_series* out =
        dftu_dataframe_mask(handle, cols.data(), col_names.data(),
                            static_cast<std::int32_t>(cols.size()));
    dftu_query_free(handle);
    if (!out)
        throw std::runtime_error(
            "DataFrame::mask: predicate has no columnar lowering for this "
            "frame");
    return Series{out};
}

}  // namespace dftracer::utils::dataframe

extern "C" {

dftu_series* dftu_dataframe_mask(const dftu_query* q,
                                 const dftu_series* const* columns,
                                 const char* const* names, int32_t n) {
    if (!q || n < 0) return nullptr;
    dataframe::DataFrame b;
    b.names.reserve(static_cast<std::size_t>(n));
    b.columns.reserve(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        b.names.emplace_back(names[i] ? names[i] : "");
        b.columns.emplace_back(const_cast<dftu_series*>(columns[i]));
    }
    dftu_series* out = nullptr;
    try {
        const auto& query = dftracer::utils::query::query_handle_unwrap(q);
        out = dftracer::utils::dataframe::evaluate_mask(query.root(), b)
                  .release();
    } catch (const std::exception&) {
        out = nullptr;
    }
    for (dataframe::Series& c : b.columns)
        c.release();  // borrowed, do not free
    return out;
}
}
