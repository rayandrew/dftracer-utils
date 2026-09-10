#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/internal/dataframe_handle.h>
#include <dftracer/utils/dataframe/internal/expr_handle.h>
#include <dftracer/utils/dataframe/internal/lazyframe_handle.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/query/internal/query_handle.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

// The opaque dftu_dataframe handle owns a C++ DataFrame.
struct dftu_dataframe {
    dftracer::utils::dataframe::DataFrame df;
};

// The opaque dftu_lazyframe handle owns a C++ LazyFrame.
struct dftu_lazyframe {
    dftracer::utils::dataframe::LazyFrame lf;
};

namespace dftracer::utils::dataframe {
LazyFrame& lazyframe_handle_unwrap(dftu_lazyframe* h) { return h->lf; }
dftu_lazyframe* lazyframe_handle_wrap(LazyFrame&& lf) {
    return new dftu_lazyframe{std::move(lf)};
}
dftu_dataframe* dataframe_handle_wrap(DataFrame&& df) {
    return new dftu_dataframe{std::move(df)};
}
DataFrame dataframe_handle_take(dftu_dataframe* h) {
    DataFrame df = std::move(h->df);
    delete h;
    return df;
}
}  // namespace dftracer::utils::dataframe

namespace {
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::dataframe::GroupAgg;
using dftracer::utils::dataframe::LazyFrame;
using dftracer::utils::dataframe::Series;

dftu_dataframe* wrap(DataFrame&& df) {
    return new dftu_dataframe{std::move(df)};
}

dftu_lazyframe* wrap_lazy(LazyFrame&& lf) {
    return new dftu_lazyframe{std::move(lf)};
}

char* dup_string(const std::string& s) {
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

// Run `fn(borrowed)` with a non-owning Series over `h`, without freeing `h`.
template <class Fn>
auto with_borrowed(const dftu_series* h, Fn&& fn) {
    Series s{const_cast<dftu_series*>(h)};
    auto out = fn(s);
    s.release();
    return out;
}
}  // namespace

extern "C" {

dftu_dataframe* dftu_dataframe_new(const char* const* names,
                                   dftu_series* const* columns, int32_t n) {
    if (n < 0 || (n > 0 && (!names || !columns))) return nullptr;
    DataFrame df;
    df.names.reserve(static_cast<std::size_t>(n));
    df.columns.reserve(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        df.names.emplace_back(names[i]);
        df.columns.emplace_back(Series{columns[i]});  // takes ownership
    }
    return wrap(std::move(df));
}

void dftu_dataframe_free(dftu_dataframe* df) { delete df; }

int64_t dftu_dataframe_num_rows(const dftu_dataframe* df) {
    return df ? df->df.num_rows() : 0;
}
int32_t dftu_dataframe_num_columns(const dftu_dataframe* df) {
    return df ? static_cast<int32_t>(df->df.num_columns()) : 0;
}
const char* dftu_dataframe_column_name(const dftu_dataframe* df, int32_t i) {
    if (!df || i < 0 || static_cast<std::size_t>(i) >= df->df.names.size())
        return nullptr;
    return df->df.names[static_cast<std::size_t>(i)].c_str();
}
dftu_series* dftu_dataframe_column(const dftu_dataframe* df, const char* name) {
    if (!df || !name) return nullptr;
    Series col = df->df.column(name);
    return col.valid() ? col.release() : nullptr;
}

dftu_dataframe* dftu_dataframe_take(const dftu_dataframe* df,
                                    const int64_t* idx, int64_t n) {
    if (!df) return nullptr;
    std::vector<std::int64_t> v(idx, idx + n);
    return wrap(df->df.take(v));
}
dftu_dataframe* dftu_dataframe_filter(const dftu_dataframe* df,
                                      const dftu_series* mask) {
    if (!df) return nullptr;
    return wrap(
        with_borrowed(mask, [&](const Series& m) { return df->df.filter(m); }));
}
dftu_dataframe* dftu_dataframe_slice(const dftu_dataframe* df, int64_t offset,
                                     int64_t len) {
    return df ? wrap(df->df.slice(offset, len)) : nullptr;
}
dftu_dataframe* dftu_dataframe_head(const dftu_dataframe* df, int64_t n) {
    return df ? wrap(df->df.head(n)) : nullptr;
}
dftu_dataframe* dftu_dataframe_sort_by(const dftu_dataframe* df,
                                       const char* name, int32_t descending) {
    return df ? wrap(df->df.sort_by(name, descending != 0)) : nullptr;
}
dftu_dataframe* dftu_dataframe_topk(const dftu_dataframe* df, const char* name,
                                    int64_t k, int32_t largest) {
    return df ? wrap(df->df.topk(name, k, largest != 0)) : nullptr;
}

dftu_dataframe* dftu_dataframe_select(const dftu_dataframe* df,
                                      const char* const* names, int32_t n) {
    if (!df) return nullptr;
    std::vector<std::string> cols;
    cols.reserve(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) cols.emplace_back(names[i]);
    return wrap(df->df.select(cols));
}
dftu_dataframe* dftu_dataframe_rename(const dftu_dataframe* df,
                                      const char* const* new_names, int32_t n) {
    if (!df) return nullptr;
    std::vector<std::string> names;
    names.reserve(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) names.emplace_back(new_names[i]);
    return wrap(df->df.rename(names));
}
dftu_dataframe* dftu_dataframe_with_column(const dftu_dataframe* df,
                                           const char* name,
                                           const dftu_series* col) {
    if (!df) return nullptr;
    return wrap(with_borrowed(
        col, [&](const Series& c) { return df->df.with_column(name, c); }));
}

dftu_dataframe* dftu_dataframe_drop_nulls(const dftu_dataframe* df) {
    return df ? wrap(df->df.drop_nulls()) : nullptr;
}
dftu_dataframe* dftu_dataframe_fill_null(const dftu_dataframe* df,
                                         dftu_scalar value) {
    return df ? wrap(df->df.fill_null(value)) : nullptr;
}
dftu_dataframe* dftu_dataframe_unique(const dftu_dataframe* df) {
    return df ? wrap(df->df.unique()) : nullptr;
}
dftu_dataframe* dftu_dataframe_drop_duplicates(const dftu_dataframe* df) {
    return df ? wrap(df->df.drop_duplicates()) : nullptr;
}
dftu_dataframe* dftu_dataframe_sort_by_multi(const dftu_dataframe* df,
                                             const char* const* names,
                                             int32_t n, int32_t descending) {
    if (!df) return nullptr;
    std::vector<std::string> cols;
    cols.reserve(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) cols.emplace_back(names[i]);
    return wrap(df->df.sort_by_multi(cols, descending != 0));
}
dftu_dataframe* dftu_dataframe_sort_by_multi_per_col(const dftu_dataframe* df,
                                                     const char* const* names,
                                                     int32_t n,
                                                     const int32_t* descending,
                                                     int32_t descending_n) {
    if (!df) return nullptr;
    std::vector<std::string> cols;
    cols.reserve(static_cast<std::size_t>(n));
    for (int32_t i = 0; i < n; ++i) cols.emplace_back(names[i]);
    std::vector<bool> desc;
    desc.reserve(static_cast<std::size_t>(descending_n));
    for (int32_t i = 0; i < descending_n; ++i)
        desc.push_back(descending[i] != 0);
    return wrap(df->df.sort_by_multi(cols, desc));
}
dftu_dataframe* dftu_dataframe_tail(const dftu_dataframe* df, int64_t n) {
    return df ? wrap(df->df.tail(n)) : nullptr;
}
dftu_dataframe* dftu_dataframe_reverse(const dftu_dataframe* df) {
    return df ? wrap(df->df.reverse()) : nullptr;
}
dftu_dataframe* dftu_dataframe_sample(const dftu_dataframe* df, int64_t n,
                                      uint64_t seed) {
    return df ? wrap(df->df.sample(n, seed)) : nullptr;
}
dftu_dataframe* dftu_dataframe_with_row_index(const dftu_dataframe* df,
                                              const char* name) {
    return df ? wrap(df->df.with_row_index(name)) : nullptr;
}
dftu_dataframe* dftu_dataframe_describe(const dftu_dataframe* df) {
    return df ? wrap(df->df.describe()) : nullptr;
}
dftu_dataframe* dftu_dataframe_null_count(const dftu_dataframe* df) {
    return df ? wrap(df->df.null_count()) : nullptr;
}
dftu_series* dftu_dataframe_is_duplicated(const dftu_dataframe* df) {
    return df ? df->df.is_duplicated().release() : nullptr;
}
dftu_series* dftu_dataframe_is_unique(const dftu_dataframe* df) {
    return df ? df->df.is_unique().release() : nullptr;
}

dftu_series* dftu_dataframe_mask_frame(const dftu_dataframe* df,
                                       const dftu_query* q) {
    if (!df || !q) return nullptr;
    try {
        return df->df.mask(dftracer::utils::query::query_handle_unwrap(q))
            .release();
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_dataframe* dftu_series_value_counts(const dftu_series* v) {
    if (!v) return nullptr;
    return wrap(with_borrowed(v, [&](const Series& c) {
        return dftracer::utils::dataframe::value_counts(c);
    }));
}

dftu_dataframe* dftu_dataframe_unpivot(const dftu_dataframe* df,
                                       const char* const* id_vars, int32_t n_id,
                                       const char* const* value_vars,
                                       int32_t n_val) {
    if (!df || n_id < 0 || n_val < 0) return nullptr;
    std::vector<std::string> ids, vals;
    ids.reserve(static_cast<std::size_t>(n_id));
    vals.reserve(static_cast<std::size_t>(n_val));
    for (int32_t i = 0; i < n_id; ++i) ids.emplace_back(id_vars[i]);
    for (int32_t i = 0; i < n_val; ++i) vals.emplace_back(value_vars[i]);
    try {
        return wrap(df->df.unpivot(ids, vals));
    } catch (const std::exception&) {
        return nullptr;
    }
}
dftu_dataframe* dftu_dataframe_explode(const dftu_dataframe* df,
                                       const char* column) {
    if (!df || !column) return nullptr;
    try {
        return wrap(df->df.explode(column));
    } catch (const std::exception&) {
        return nullptr;
    }
}
dftu_dataframe* dftu_dataframe_to_dummies(const dftu_dataframe* df,
                                          const char* column) {
    if (!df || !column) return nullptr;
    try {
        return wrap(df->df.to_dummies(column));
    } catch (const std::exception&) {
        return nullptr;
    }
}
dftu_dataframe* dftu_dataframe_pivot(const dftu_dataframe* df,
                                     const char* index, const char* columns,
                                     const char* values, const char* agg) {
    if (!df || !index || !columns || !values) return nullptr;
    try {
        return wrap(df->df.pivot(index, columns, values, agg ? agg : "first"));
    } catch (const std::exception&) {
        return nullptr;
    }
}
dftu_dataframe* dftu_dataframe_group_by_dynamic(const dftu_dataframe* df,
                                                const char* time_col,
                                                int64_t every, int64_t period,
                                                const dftu_group_agg* aggs,
                                                int32_t n_aggs) {
    if (!df || !time_col || n_aggs < 0 || (n_aggs > 0 && !aggs)) return nullptr;
    try {
        std::vector<dftracer::utils::dataframe::GroupAgg> v;
        v.reserve(static_cast<std::size_t>(n_aggs));
        for (int32_t i = 0; i < n_aggs; ++i) {
            dftracer::utils::dataframe::GroupAgg a;
            a.op = dftracer::utils::dataframe::agg_from_string(
                aggs[i].op ? aggs[i].op : "");
            a.column = aggs[i].column ? aggs[i].column : "";
            a.out = aggs[i].out ? aggs[i].out : "";
            v.push_back(std::move(a));
        }
        return wrap(df->df.group_by_dynamic(time_col, every, period, v));
    } catch (const std::exception&) {
        return nullptr;
    }
}
dftu_dataframe* dftu_dataframe_group_by(const dftu_dataframe* df,
                                        const char* const* keys, int32_t n_keys,
                                        const dftu_group_agg* aggs,
                                        int32_t n_aggs) {
    if (!df || n_keys < 0 || n_aggs < 0 || (n_keys > 0 && !keys) ||
        (n_aggs > 0 && !aggs))
        return nullptr;
    try {
        std::vector<std::string> ks;
        ks.reserve(static_cast<std::size_t>(n_keys));
        for (int32_t i = 0; i < n_keys; ++i) ks.emplace_back(keys[i]);
        std::vector<GroupAgg> ag;
        ag.reserve(static_cast<std::size_t>(n_aggs));
        for (int32_t i = 0; i < n_aggs; ++i) {
            GroupAgg a;
            a.op = dftracer::utils::dataframe::agg_from_string(
                aggs[i].op ? aggs[i].op : "");
            a.column = aggs[i].column ? aggs[i].column : "";
            a.out = aggs[i].out ? aggs[i].out : "";
            ag.push_back(std::move(a));
        }
        return wrap(df->df.group_by(std::move(ks), std::move(ag)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_dataframe_lazy(const dftu_dataframe* df) {
    if (!df) return nullptr;
    try {
        return wrap_lazy(df->df.lazy());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_dataframe* dftu_lazyframe_collect(dftu_lazyframe* lf,
                                       int64_t morsel_rows) {
    if (!lf) return nullptr;
    try {
        DataFrame out = dftracer::utils::default_runtime()
                            .submit(lf->lf.collect(morsel_rows))
                            .get();
        return wrap(std::move(out));
    } catch (const std::exception&) {
        return nullptr;
    }
}

void dftu_lazyframe_free(dftu_lazyframe* lf) { delete lf; }

char* dftu_lazyframe_schema(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        std::vector<std::string> names = lf->lf.schema();
        std::string joined;
        for (std::size_t i = 0; i < names.size(); ++i) {
            if (i) joined.push_back('\n');
            joined += names[i];
        }
        return dup_string(joined);
    } catch (const std::exception&) {
        return nullptr;
    }
}

char* dftu_lazyframe_explain(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return dup_string(lf->lf.explain());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_filter(const dftu_lazyframe* lf,
                                      const dftu_expr* pred) {
    if (!lf || !pred) return nullptr;
    try {
        return wrap_lazy(lf->lf.filter(
            dftracer::utils::dataframe::expr_handle_unwrap(pred)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_select(const dftu_lazyframe* lf,
                                      const char* const* names, int32_t n) {
    if (!lf || n < 0 || (n > 0 && !names)) return nullptr;
    try {
        std::vector<std::string> cols;
        cols.reserve(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) cols.emplace_back(names[i]);
        return wrap_lazy(lf->lf.select(std::move(cols)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_with_column(const dftu_lazyframe* lf,
                                           const char* name,
                                           const dftu_expr* expr) {
    if (!lf || !name || !expr) return nullptr;
    try {
        return wrap_lazy(lf->lf.with_column(
            name, dftracer::utils::dataframe::expr_handle_unwrap(expr)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_group_by(const dftu_lazyframe* lf,
                                        const char* const* keys, int32_t n_keys,
                                        const dftu_group_agg* aggs,
                                        int32_t n_aggs) {
    if (!lf || n_keys < 0 || n_aggs < 0 || (n_keys > 0 && !keys) ||
        (n_aggs > 0 && !aggs))
        return nullptr;
    try {
        std::vector<std::string> ks;
        ks.reserve(static_cast<std::size_t>(n_keys));
        for (int32_t i = 0; i < n_keys; ++i) ks.emplace_back(keys[i]);
        std::vector<GroupAgg> ag;
        ag.reserve(static_cast<std::size_t>(n_aggs));
        for (int32_t i = 0; i < n_aggs; ++i) {
            GroupAgg a;
            a.op = dftracer::utils::dataframe::agg_from_string(
                aggs[i].op ? aggs[i].op : "");
            a.column = aggs[i].column ? aggs[i].column : "";
            a.out = aggs[i].out ? aggs[i].out : "";
            ag.push_back(std::move(a));
        }
        return wrap_lazy(lf->lf.group_by(std::move(ks), std::move(ag)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_sort_by(const dftu_lazyframe* lf,
                                       const char* name, int32_t descending) {
    if (!lf || !name) return nullptr;
    try {
        return wrap_lazy(lf->lf.sort_by(name, descending != 0));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_head(const dftu_lazyframe* lf, int64_t n) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.head(n));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_tail(const dftu_lazyframe* lf, int64_t n) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.tail(n));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_drop_nulls(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.drop_nulls());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_unique(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.unique());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_drop_duplicates(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.drop_duplicates());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_rename(const dftu_lazyframe* lf,
                                      const char* const* names, int32_t n) {
    if (!lf || n < 0 || (n > 0 && !names)) return nullptr;
    try {
        std::vector<std::string> v;
        v.reserve(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) v.emplace_back(names[i]);
        return wrap_lazy(lf->lf.rename(std::move(v)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_slice(const dftu_lazyframe* lf, int64_t offset,
                                     int64_t len) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.slice(offset, len));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_fill_null(const dftu_lazyframe* lf,
                                         dftu_scalar value) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.fill_null(value));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_with_row_index(const dftu_lazyframe* lf,
                                              const char* name) {
    if (!lf || !name) return nullptr;
    try {
        return wrap_lazy(lf->lf.with_row_index(name));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_null_count(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.null_count());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_explode(const dftu_lazyframe* lf,
                                       const char* column) {
    if (!lf || !column) return nullptr;
    try {
        return wrap_lazy(lf->lf.explode(column));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_unpivot(const dftu_lazyframe* lf,
                                       const char* const* id_vars, int32_t n_id,
                                       const char* const* value_vars,
                                       int32_t n_val) {
    if (!lf || n_id < 0 || n_val < 0) return nullptr;
    try {
        std::vector<std::string> ids, vals;
        ids.reserve(static_cast<std::size_t>(n_id));
        vals.reserve(static_cast<std::size_t>(n_val));
        for (int32_t i = 0; i < n_id; ++i) ids.emplace_back(id_vars[i]);
        for (int32_t i = 0; i < n_val; ++i) vals.emplace_back(value_vars[i]);
        return wrap_lazy(lf->lf.unpivot(std::move(ids), std::move(vals)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_melt(const dftu_lazyframe* lf,
                                    const char* const* id_vars, int32_t n_id,
                                    const char* const* value_vars,
                                    int32_t n_val) {
    if (!lf || n_id < 0 || n_val < 0) return nullptr;
    try {
        std::vector<std::string> ids, vals;
        ids.reserve(static_cast<std::size_t>(n_id));
        vals.reserve(static_cast<std::size_t>(n_val));
        for (int32_t i = 0; i < n_id; ++i) ids.emplace_back(id_vars[i]);
        for (int32_t i = 0; i < n_val; ++i) vals.emplace_back(value_vars[i]);
        return wrap_lazy(lf->lf.melt(std::move(ids), std::move(vals)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_topk(const dftu_lazyframe* lf, const char* name,
                                    int64_t k, int32_t largest) {
    if (!lf || !name) return nullptr;
    try {
        return wrap_lazy(lf->lf.topk(name, k, largest != 0));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_sample(const dftu_lazyframe* lf, int64_t n,
                                      uint64_t seed) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.sample(n, seed));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_is_duplicated(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.is_duplicated());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_is_unique(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.is_unique());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_group_by_dynamic(const dftu_lazyframe* lf,
                                                const char* time_col,
                                                int64_t every, int64_t period,
                                                const dftu_group_agg* aggs,
                                                int32_t n_aggs, int64_t origin,
                                                int32_t origin_min) {
    if (!lf || !time_col || n_aggs < 0 || (n_aggs > 0 && !aggs)) return nullptr;
    try {
        std::vector<GroupAgg> v;
        v.reserve(static_cast<std::size_t>(n_aggs));
        for (int32_t i = 0; i < n_aggs; ++i) {
            GroupAgg a;
            a.op = dftracer::utils::dataframe::agg_from_string(
                aggs[i].op ? aggs[i].op : "");
            a.column = aggs[i].column ? aggs[i].column : "";
            a.out = aggs[i].out ? aggs[i].out : "";
            v.push_back(std::move(a));
        }
        return wrap_lazy(lf->lf.group_by_dynamic(
            time_col, every, period, std::move(v), origin, origin_min != 0));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_pivot(const dftu_lazyframe* lf,
                                     const char* index, const char* on,
                                     const char* values, const char* agg) {
    if (!lf || !index || !on || !values) return nullptr;
    try {
        return wrap_lazy(lf->lf.pivot(index, on, values, agg ? agg : "first"));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_to_dummies(const dftu_lazyframe* lf,
                                          const char* column) {
    if (!lf || !column) return nullptr;
    try {
        return wrap_lazy(lf->lf.to_dummies(column));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_describe(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.describe());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_memory_budget(const dftu_lazyframe* lf,
                                             uint64_t bytes) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.memory_budget(bytes));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_auto_spill(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.auto_spill());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_filter_mask(const dftu_lazyframe* lf,
                                           const dftu_series* mask) {
    if (!lf || !mask) return nullptr;
    try {
        return with_borrowed(mask, [&](const Series& m) {
            return wrap_lazy(lf->lf.filter_mask(m.share()));
        });
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_reverse(const dftu_lazyframe* lf) {
    if (!lf) return nullptr;
    try {
        return wrap_lazy(lf->lf.reverse());
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_take(const dftu_lazyframe* lf,
                                    const int64_t* idx, int32_t n) {
    if (!lf || n < 0 || (n > 0 && !idx)) return nullptr;
    try {
        std::vector<std::int64_t> v(idx, idx + n);
        return wrap_lazy(lf->lf.take(std::move(v)));
    } catch (const std::exception&) {
        return nullptr;
    }
}

dftu_lazyframe* dftu_lazyframe_sort_by_multi(const dftu_lazyframe* lf,
                                             const char* const* by, int32_t n,
                                             int32_t descending) {
    if (!lf || n < 0 || (n > 0 && !by)) return nullptr;
    try {
        std::vector<std::string> cols;
        cols.reserve(static_cast<std::size_t>(n));
        for (int32_t i = 0; i < n; ++i) cols.emplace_back(by[i]);
        return wrap_lazy(
            lf->lf.sort_by_multi(std::move(cols), descending != 0));
    } catch (const std::exception&) {
        return nullptr;
    }
}

}  // extern "C"
