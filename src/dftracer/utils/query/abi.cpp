#include <dftracer/utils/query/abi.h>
#include <dftracer/utils/query/builder.h>
#include <dftracer/utils/query/internal/query_handle.h>
#include <dftracer/utils/query/query.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

struct dftu_query {
    dftracer::utils::query::Query q;
};

namespace dftracer::utils::query {
const Query& query_handle_unwrap(const dftu_query* h) { return h->q; }
}  // namespace dftracer::utils::query

namespace {

using dftracer::utils::query::CompareOp;
using dftracer::utils::query::Expr;
using dftracer::utils::query::MatchOp;
using dftracer::utils::query::Query;

dftu_query* wrap_expr(const Expr& e) {
    auto q = e.build();
    if (!q) return nullptr;
    return new dftu_query{std::move(*q)};
}

dftu_query* wrap_string(const std::string& s) {
    auto q = Query::from_string(s);
    if (!q) return nullptr;
    return new dftu_query{std::move(*q)};
}

dftu_query* build_in_i64(const char* field, const int64_t* values, int32_t n,
                         bool negate) {
    if (!field || (n > 0 && !values)) return nullptr;
    std::vector<std::int64_t> vals;
    for (int32_t i = 0; i < n; ++i) vals.push_back(values[i]);
    auto e = negate ? dftracer::utils::query::field_not_in(field, vals)
                    : dftracer::utils::query::field_in(field, vals);
    return wrap_expr(e);
}

dftu_query* build_in_str(const char* field, const char* const* values,
                         int32_t n, bool negate) {
    if (!field || (n > 0 && !values)) return nullptr;
    std::vector<std::string> vals;
    for (int32_t i = 0; i < n; ++i) {
        if (!values[i]) return nullptr;
        vals.emplace_back(values[i]);
    }
    auto e = negate ? dftracer::utils::query::field_not_in(field, vals)
                    : dftracer::utils::query::field_in(field, vals);
    return wrap_expr(e);
}

}  // namespace

extern "C" {

dftu_query* dftu_query_parse(const char* text) {
    if (!text) return nullptr;
    auto r = dftracer::utils::query::Query::from_string(text);
    if (!r) return nullptr;
    return new dftu_query{std::move(*r)};
}

void dftu_query_free(dftu_query* q) { delete q; }

char* dftu_query_to_string(const dftu_query* q) {
    if (!q) return nullptr;
    std::string s = q->q.to_string();
    char* out = static_cast<char*>(std::malloc(s.size() + 1));
    if (!out) return nullptr;
    std::memcpy(out, s.c_str(), s.size() + 1);
    return out;
}

void dftu_query_string_free(char* s) { std::free(s); }

dftu_query* dftu_query_cmp_i64(const char* field, dftu_query_cmp_op op,
                               int64_t value) {
    if (!field) return nullptr;
    return wrap_expr(dftracer::utils::query::field_cmp(
        field, static_cast<CompareOp>(op),
        dftracer::utils::query::detail::literal(
            static_cast<std::int64_t>(value))));
}

dftu_query* dftu_query_cmp_f64(const char* field, dftu_query_cmp_op op,
                               double value) {
    if (!field) return nullptr;
    return wrap_expr(dftracer::utils::query::field_cmp(
        field, static_cast<CompareOp>(op),
        dftracer::utils::query::detail::literal(value)));
}

dftu_query* dftu_query_cmp_str(const char* field, dftu_query_cmp_op op,
                               const char* value) {
    if (!field || !value) return nullptr;
    return wrap_expr(dftracer::utils::query::field_cmp(
        field, static_cast<CompareOp>(op),
        dftracer::utils::query::detail::literal(std::string_view(value))));
}

dftu_query* dftu_query_in_i64(const char* field, const int64_t* values,
                              int32_t n) {
    return build_in_i64(field, values, n, /*negate=*/false);
}

dftu_query* dftu_query_in_str(const char* field, const char* const* values,
                              int32_t n) {
    return build_in_str(field, values, n, /*negate=*/false);
}

dftu_query* dftu_query_not_in_i64(const char* field, const int64_t* values,
                                  int32_t n) {
    return build_in_i64(field, values, n, /*negate=*/true);
}

dftu_query* dftu_query_not_in_str(const char* field, const char* const* values,
                                  int32_t n) {
    return build_in_str(field, values, n, /*negate=*/true);
}

dftu_query* dftu_query_match(const char* field, dftu_query_match_op match_op,
                             const char* pattern) {
    if (!field || !pattern) return nullptr;
    return wrap_expr(dftracer::utils::query::field_match(
        field, static_cast<MatchOp>(match_op), pattern));
}

dftu_query* dftu_query_and(dftu_query* a, dftu_query* b) {
    if (!a || !b) {
        dftu_query_free(a);
        dftu_query_free(b);
        return nullptr;
    }
    std::string s = "(" + a->q.to_string() + " and " + b->q.to_string() + ")";
    dftu_query_free(a);
    dftu_query_free(b);
    return wrap_string(s);
}

dftu_query* dftu_query_or(dftu_query* a, dftu_query* b) {
    if (!a || !b) {
        dftu_query_free(a);
        dftu_query_free(b);
        return nullptr;
    }
    std::string s = "(" + a->q.to_string() + " or " + b->q.to_string() + ")";
    dftu_query_free(a);
    dftu_query_free(b);
    return wrap_string(s);
}

dftu_query* dftu_query_not(dftu_query* a) {
    if (!a) return nullptr;
    std::string s = "not (" + a->q.to_string() + ")";
    dftu_query_free(a);
    return wrap_string(s);
}
}
