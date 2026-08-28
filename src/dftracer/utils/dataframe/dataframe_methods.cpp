#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/dataframe/dataframe.h>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

// The dfops alias qualifies the batch_ops free-function calls below so they are
// not shadowed by the same-named DataFrame members.
namespace dfops = dftracer::utils::dataframe;

namespace dftracer::utils::dataframe {

std::int64_t DataFrame::column_index(std::string_view name) const {
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (names[i] == name) return static_cast<std::int64_t>(i);
    }
    return -1;
}

Series DataFrame::column(std::string_view name) const {
    const std::int64_t i = column_index(name);
    return i < 0 ? Series{} : columns[static_cast<std::size_t>(i)].share();
}

DataFrame DataFrame::take(const std::vector<std::int64_t>& indices) const {
    return dfops::take(*this, indices);
}
DataFrame DataFrame::filter(const Series& mask) const {
    return dfops::filter(*this, mask);
}
DataFrame DataFrame::slice(std::int64_t offset, std::int64_t len) const {
    return dfops::slice(*this, offset, len);
}
DataFrame DataFrame::head(std::int64_t n) const {
    return dfops::head(*this, n);
}
DataFrame DataFrame::tail(std::int64_t n) const {
    return dfops::tail(*this, n);
}
DataFrame DataFrame::reverse() const { return dfops::reverse(*this); }
DataFrame DataFrame::sort_by(const std::string& name, bool descending) const {
    return dfops::sort_by(*this, name, descending);
}
DataFrame DataFrame::sort_by_multi(const std::vector<std::string>& by,
                                   bool descending) const {
    return dfops::sort_by_multi(*this, by, descending);
}
DataFrame DataFrame::drop_nulls() const { return dfops::drop_nulls(*this); }
DataFrame DataFrame::fill_null(dftu_scalar value) const {
    return dfops::fill_null(*this, value);
}
DataFrame DataFrame::unique() const { return dfops::unique(*this); }
DataFrame DataFrame::drop_duplicates() const { return dfops::unique(*this); }
DataFrame DataFrame::sample(std::int64_t n, std::uint64_t seed) const {
    return dfops::sample(*this, n, seed);
}
DataFrame DataFrame::with_row_index(const std::string& name) const {
    return dfops::with_row_index(*this, name);
}
DataFrame DataFrame::describe() const { return dfops::describe(*this); }
DataFrame DataFrame::null_count() const { return dfops::null_count(*this); }
Series DataFrame::is_duplicated() const { return dfops::is_duplicated(*this); }
Series DataFrame::is_unique() const { return dfops::is_unique(*this); }

DataFrame Series::value_counts() const { return dfops::value_counts(*this); }
DataFrame DataFrame::topk(const std::string& name, std::int64_t k,
                          bool largest) const {
    return dfops::topk(*this, name, k, largest);
}

DataFrame DataFrame::select(const std::vector<std::string>& cols) const {
    return dfops::select(*this, cols);
}
DataFrame DataFrame::rename(const std::vector<std::string>& new_names) const {
    return dfops::rename(*this, new_names);
}
DataFrame DataFrame::with_column(const std::string& name,
                                 const Series& col) const {
    return dfops::with_column(*this, name, col);
}

DataFrame DataFrame::group_by(const std::string& key,
                              const std::vector<GroupAgg>& aggs) const {
    return dfops::group_by(*this, key, aggs);
}

DataFrame DataFrame::unpivot(const std::vector<std::string>& id_vars,
                             const std::vector<std::string>& value_vars) const {
    return dfops::unpivot(*this, id_vars, value_vars);
}
DataFrame DataFrame::melt(const std::vector<std::string>& id_vars,
                          const std::vector<std::string>& value_vars) const {
    return dfops::unpivot(*this, id_vars, value_vars);
}
DataFrame DataFrame::explode(const std::string& column) const {
    return dfops::explode(*this, column);
}
DataFrame DataFrame::to_dummies(const std::string& column) const {
    return dfops::to_dummies(*this, column);
}
DataFrame DataFrame::pivot(const std::string& index, const std::string& on,
                           const std::string& values,
                           const std::string& agg) const {
    return dfops::pivot(*this, index, on, values, agg);
}
DataFrame DataFrame::pivot(const std::string& index, const std::string& on,
                           const std::string& values, Agg agg) const {
    return dfops::pivot(*this, index, on, values, to_string(agg));
}
DataFrame DataFrame::group_by_dynamic(const std::string& time_col,
                                      std::int64_t every, std::int64_t period,
                                      const std::vector<GroupAgg>& aggs,
                                      std::int64_t origin,
                                      bool origin_min) const {
    return dfops::group_by_dynamic(*this, time_col, every, period, aggs, origin,
                                   origin_min);
}

}  // namespace dftracer::utils::dataframe
