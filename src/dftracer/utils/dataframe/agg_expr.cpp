#include <dftracer/utils/dataframe/agg_expr.h>
#include <dftracer/utils/dataframe/internal/expr_handle.h>

#include <cstdint>
#include <exception>
#include <string>
#include <unordered_map>
#include <utility>

namespace dataframe = dftracer::utils::dataframe;

namespace dftracer::utils::dataframe {

AggExprSpec agg_count(std::string out) {
    return {AggOp::Count, Expr{}, std::move(out)};
}
AggExprSpec agg_sum(Expr value, std::string out) {
    return {AggOp::Sum, std::move(value), std::move(out)};
}
AggExprSpec agg_min(Expr value, std::string out) {
    return {AggOp::Min, std::move(value), std::move(out)};
}
AggExprSpec agg_max(Expr value, std::string out) {
    return {AggOp::Max, std::move(value), std::move(out)};
}
AggExprSpec agg_mean(Expr value, std::string out) {
    return {AggOp::Mean, std::move(value), std::move(out)};
}
AggExprSpec agg_var(Expr value, std::string out) {
    return {AggOp::Var, std::move(value), std::move(out)};
}
AggExprSpec agg_std(Expr value, std::string out) {
    return {AggOp::Std, std::move(value), std::move(out)};
}
AggExprSpec agg_skew(Expr value, std::string out) {
    return {AggOp::Skew, std::move(value), std::move(out)};
}
AggExprSpec agg_kurt(Expr value, std::string out) {
    return {AggOp::Kurt, std::move(value), std::move(out)};
}
AggExprSpec agg_pct(Expr value, double q, std::string out) {
    return {AggOp::Pct, std::move(value), std::move(out), q};
}
AggExprSpec agg_hist(Expr value, std::string out) {
    return {AggOp::Hist, std::move(value), std::move(out)};
}

DataFrame group_agg_expr(const Expr& key, const std::vector<AggExprSpec>& specs,
                         const std::vector<const Series*>& inputs,
                         const std::string& key_name) {
    // The value expressions (all numeric) compile into one program so a shared
    // subexpression is computed once (CSE); deduping identical value nodes maps
    // them to one evaluated column, folded into a single shared FieldStat.
    std::vector<Expr> value_roots;
    std::unordered_map<const ExprNode*, std::int32_t> value_col;
    std::vector<AggSpec> col_specs;
    col_specs.reserve(specs.size());
    for (const AggExprSpec& sp : specs) {
        AggSpec cs;
        cs.op = sp.op;
        cs.out = sp.out;
        cs.param = sp.param;
        if (sp.op == AggOp::Count || !sp.value.valid()) {
            cs.value_col = -1;
        } else {
            const ExprNode* np = sp.value.node().get();
            auto it = value_col.find(np);
            if (it != value_col.end()) {
                cs.value_col = it->second;
            } else {
                cs.value_col = static_cast<std::int32_t>(value_roots.size());
                value_roots.push_back(sp.value);
                value_col.emplace(np, cs.value_col);
            }
        }
        col_specs.push_back(std::move(cs));
    }

    std::vector<Series> value_cols;
    if (!value_roots.empty()) value_cols = eval_many(value_roots, inputs);
    std::vector<const Series*> values;
    values.reserve(value_cols.size());
    for (Series& c : value_cols) values.push_back(&c);

    // The key may be any type (e.g. a string category): take a bare column
    // reference directly, only routing a computed key through the evaluator.
    const std::int32_t ki = expr_col_index(key);
    Series key_col = ki >= 0 ? inputs[static_cast<std::size_t>(ki)]->share()
                             : eval(key, inputs);
    return group_agg(key_col, values, std::move(col_specs), key_name);
}

}  // namespace dftracer::utils::dataframe

namespace {
dftu_agg_spec make_spec(int32_t op, const dftu_expr* value, const char* out,
                        double param = 0.0) {
    return {op, value, out, param};
}
}  // namespace

extern "C" {

dftu_agg_spec dftu_agg_count(const char* out) {
    return make_spec(DFTU_AGG_COUNT, nullptr, out);
}
dftu_agg_spec dftu_agg_sum(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_SUM, value, out);
}
dftu_agg_spec dftu_agg_min(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_MIN, value, out);
}
dftu_agg_spec dftu_agg_max(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_MAX, value, out);
}
dftu_agg_spec dftu_agg_mean(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_MEAN, value, out);
}
dftu_agg_spec dftu_agg_var(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_VAR, value, out);
}
dftu_agg_spec dftu_agg_std(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_STD, value, out);
}
dftu_agg_spec dftu_agg_skew(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_SKEW, value, out);
}
dftu_agg_spec dftu_agg_kurt(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_KURT, value, out);
}
dftu_agg_spec dftu_agg_first(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_FIRST, value, out);
}
dftu_agg_spec dftu_agg_last(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_LAST, value, out);
}
dftu_agg_spec dftu_agg_pct(const dftu_expr* value, double q, const char* out) {
    return make_spec(DFTU_AGG_PCT, value, out, q);
}
dftu_agg_spec dftu_agg_hist(const dftu_expr* value, const char* out) {
    return make_spec(DFTU_AGG_HIST, value, out);
}

int32_t dftu_dataframe_group_agg_expr(const dftu_expr* key,
                                      const dftu_agg_spec* specs,
                                      int32_t n_specs,
                                      const dftu_series* const* inputs,
                                      int32_t n_inputs, dftu_series** out_key,
                                      dftu_series** out_values) {
    if (!key || !specs || n_specs <= 0) return -1;
    std::vector<dataframe::AggExprSpec> cxx;
    cxx.reserve(static_cast<std::size_t>(n_specs));
    for (int32_t i = 0; i < n_specs; ++i) {
        dataframe::AggExprSpec s;
        s.op = static_cast<dataframe::AggOp>(specs[i].op);
        if (specs[i].value)
            s.value = dataframe::expr_handle_unwrap(specs[i].value);
        s.out = specs[i].out ? specs[i].out : "";
        s.param = specs[i].param;
        cxx.push_back(std::move(s));
    }
    std::vector<dataframe::Series> owned;
    std::vector<const dataframe::Series*> cols;
    owned.reserve(static_cast<std::size_t>(n_inputs));
    cols.reserve(static_cast<std::size_t>(n_inputs));
    for (int32_t i = 0; i < n_inputs; ++i) {
        owned.emplace_back(const_cast<dftu_series*>(inputs[i]));
        cols.push_back(&owned.back());
    }
    int32_t written = -1;
    try {
        dataframe::DataFrame out = dataframe::group_agg_expr(
            dataframe::expr_handle_unwrap(key), cxx, cols, "key");
        *out_key = out.columns[0].release();
        for (std::size_t i = 1; i < out.columns.size(); ++i)
            out_values[i - 1] = out.columns[i].release();
        written = n_specs;
    } catch (const std::exception&) {
        written = -1;
    }
    for (dataframe::Series& c : owned)
        c.release();  // borrowed inputs, do not free
    return written;
}
}
