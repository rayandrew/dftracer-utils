#ifndef DFTRACER_UTILS_DATAFRAME_EXPR_H
#define DFTRACER_UTILS_DATAFRAME_EXPR_H

#include <dftracer/utils/core/common/export.h>
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/series.h>

#include <cstdint>
#include <memory>
#include <vector>

// A native columnar expression engine. Expressions are built as a DAG (col
// refs, literals, arithmetic, primitives, comparisons, logical ops, casts),
// then compiled - type inference + common-subexpression elimination + lowering
// to a flat slot program - and evaluated in one chunked pass (parallel via the
// injected backend, see parallel.h). The whole compiler lives here in C++ so
// every consumer (the Python DSL, plugins, the distributed engine) gets the
// same fusion and CSE; language frontends only build the DAG.
namespace dftracer::utils::dataframe {

struct ExprNode;

/// A handle to an expression DAG node (shared, so subexpressions can be reused
/// and are naturally de-duplicated; the compiler also hash-conses structurally
/// identical nodes). Build with the free functions below.
class Expr {
   public:
    Expr() = default;
    explicit Expr(std::shared_ptr<const ExprNode> node)
        : node_(std::move(node)) {}
    const std::shared_ptr<const ExprNode>& node() const noexcept {
        return node_;
    }
    bool valid() const noexcept { return node_ != nullptr; }

   private:
    std::shared_ptr<const ExprNode> node_;
};

/// Binary arithmetic op codes (match dftu_expr_binary).
enum class BinaryOp { Add = 0, Sub = 1, Mul = 2, Div = 3 };

Expr expr_col(std::int32_t index);

/// If `e` is a bare column reference, its column index; otherwise -1. Lets a
/// caller take the input column directly (e.g. a string group key) instead of
/// routing it through the numeric evaluator.
std::int32_t expr_col_index(const Expr& e);

/// True if `e` reads input column `index` anywhere in its tree. Lets the query
/// planner tell whether a predicate depends on a given column.
bool expr_references(const Expr& e, std::int32_t index);

/// If `e` is `col <cmp> scalar`, fill *col/*op/*rhs and return true. Lets the
/// planner run a trivial predicate as a direct Series kernel, skipping the
/// expression compiler.
bool expr_as_col_cmp(const Expr& e, std::int32_t* col, CmpOp* op, Scalar* rhs);

/// If `e` is `col <binary> col`, fill *op/*a/*b and return true.
bool expr_as_col_binary(const Expr& e, BinaryOp* op, std::int32_t* a,
                        std::int32_t* b);

/// Rewrite every column reference `col(i)` to `col(old_to_new[i])`, leaving an
/// out-of-range index unchanged. Lets the planner renumber a predicate after
/// columns are dropped or reordered (projection pushdown). Returns a new Expr;
/// unchanged subtrees are shared.
Expr expr_remap_cols(const Expr& e,
                     const std::vector<std::int32_t>& old_to_new);
Expr expr_lit(std::int64_t value);
Expr expr_lit(double value);
Expr expr_binary(BinaryOp op, const Expr& a, const Expr& b);
Expr expr_prim(PrimOp prim, const Expr& a);
/// Unary numeric op (UnaryOp: abs/round/floor/ceil/log/sqrt/exp/sign/negate/
/// trunc/is_nan/is_finite/is_infinite).
Expr expr_unary(UnaryOp op, const Expr& a);
/// Clamp each value to [lo, hi].
Expr expr_clip(const Expr& a, Scalar lo, Scalar hi);
/// Replace nulls (validity bitmap) with a fill value.
Expr expr_fillna(const Expr& a, Scalar fill);
/// ASCII-lowercase a String column (Series::to_lowercase). Throws at compile
/// time if the operand is not a String column.
Expr expr_lower(const Expr& a);
Expr expr_cmp(CmpOp cmp, const Expr& a, Scalar rhs);
Expr expr_logical(LogicalOp op, const Expr& a, const Expr& b);
Expr expr_not(const Expr& a);
Expr expr_cast(TypeId type, const Expr& a);

/// Short aliases: reference input column `index`, or a literal.
inline Expr col(std::int32_t index) { return expr_col(index); }
inline Expr lit(std::int64_t value) { return expr_lit(value); }
inline Expr lit(double value) { return expr_lit(value); }

// Comparison operators take a scalar right-hand side (no Expr-vs-Expr compare).
inline Expr operator+(const Expr& a, const Expr& b) {
    return expr_binary(BinaryOp::Add, a, b);
}
inline Expr operator-(const Expr& a, const Expr& b) {
    return expr_binary(BinaryOp::Sub, a, b);
}
inline Expr operator*(const Expr& a, const Expr& b) {
    return expr_binary(BinaryOp::Mul, a, b);
}
inline Expr operator/(const Expr& a, const Expr& b) {
    return expr_binary(BinaryOp::Div, a, b);
}
inline Expr operator&(const Expr& a, const Expr& b) {
    return expr_logical(LogicalOp::And, a, b);
}
inline Expr operator|(const Expr& a, const Expr& b) {
    return expr_logical(LogicalOp::Or, a, b);
}
inline Expr operator~(const Expr& a) { return expr_not(a); }

namespace detail {
inline dftu_scalar expr_scalar_i(std::int64_t v) {
    dftu_scalar s;
    s.kind = DFTU_SCALAR_TAG_I64;
    s.value.i = v;
    return s;
}
inline dftu_scalar expr_scalar_d(double v) {
    dftu_scalar s;
    s.kind = DFTU_SCALAR_TAG_F64;
    s.value.d = v;
    return s;
}
}  // namespace detail

inline Expr operator>(const Expr& a, std::int64_t v) {
    return expr_cmp(CmpOp::Gt, a, detail::expr_scalar_i(v));
}
inline Expr operator>=(const Expr& a, std::int64_t v) {
    return expr_cmp(CmpOp::Ge, a, detail::expr_scalar_i(v));
}
inline Expr operator<(const Expr& a, std::int64_t v) {
    return expr_cmp(CmpOp::Lt, a, detail::expr_scalar_i(v));
}
inline Expr operator<=(const Expr& a, std::int64_t v) {
    return expr_cmp(CmpOp::Le, a, detail::expr_scalar_i(v));
}
inline Expr operator>(const Expr& a, double v) {
    return expr_cmp(CmpOp::Gt, a, detail::expr_scalar_d(v));
}
inline Expr operator>=(const Expr& a, double v) {
    return expr_cmp(CmpOp::Ge, a, detail::expr_scalar_d(v));
}
inline Expr operator<(const Expr& a, double v) {
    return expr_cmp(CmpOp::Lt, a, detail::expr_scalar_d(v));
}
inline Expr operator<=(const Expr& a, double v) {
    return expr_cmp(CmpOp::Le, a, detail::expr_scalar_d(v));
}

/// Compile `root` (type inference + CSE + lowering) and evaluate it over
/// `inputs` in one chunked pass. Throws std::invalid_argument on a malformed
/// expression or an out-of-range column reference.
Series eval(const Expr& root, const std::vector<const Series*>& inputs);

/// Compile `roots` into ONE slot program - CSE spans all of them, so a
/// subexpression shared across outputs (e.g. `a+b` in both `sum(a+b)` and
/// `var(a+b)`) is computed once - and evaluate them in a single chunked pass.
/// Only the input columns the program references are materialized (the pruner).
/// Returns one column per root, aligned to `roots`. Same throwing contract as
/// eval.
std::vector<Series> eval_many(const std::vector<Expr>& roots,
                              const std::vector<const Series*>& inputs);

}  // namespace dftracer::utils::dataframe

// C ABI: build and evaluate an expression from any C/C++ consumer (the compiler
// - type inference + CSE + lowering - runs in dftu_expr_eval, so every consumer
// gets the same fusion). Builders return an owned handle; children are shared,
// so reusing a handle de-duplicates naturally.
extern "C" {

typedef struct dftu_expr dftu_expr;

DFTU_EXPORT dftu_expr* dftu_expr_col(int32_t index);
DFTU_EXPORT dftu_expr* dftu_expr_lit_i64(int64_t value);
DFTU_EXPORT dftu_expr* dftu_expr_lit_f64(double value);
DFTU_EXPORT dftu_expr* dftu_expr_binary(int32_t op, const dftu_expr* a,
                                        const dftu_expr* b);
DFTU_EXPORT dftu_expr* dftu_expr_prim(int32_t prim, const dftu_expr* a);
DFTU_EXPORT dftu_expr* dftu_expr_unary(int32_t op, const dftu_expr* a);
DFTU_EXPORT dftu_expr* dftu_expr_clip(const dftu_expr* a, dftu_scalar lo,
                                      dftu_scalar hi);
DFTU_EXPORT dftu_expr* dftu_expr_cmp(int32_t cmp, const dftu_expr* a,
                                     dftu_scalar rhs);
DFTU_EXPORT dftu_expr* dftu_expr_logical(int32_t op, const dftu_expr* a,
                                         const dftu_expr* b);
DFTU_EXPORT dftu_expr* dftu_expr_not(const dftu_expr* a);
DFTU_EXPORT dftu_expr* dftu_expr_cast(int32_t type, const dftu_expr* a);
DFTU_EXPORT dftu_expr* dftu_expr_lower(const dftu_expr* a);
DFTU_EXPORT void dftu_expr_free(dftu_expr* e);

/** Compile and evaluate `root` over `n_inputs` columns. Returns an owned
 * column, or NULL on a malformed expression. */
DFTU_EXPORT dftu_series* dftu_expr_eval(const dftu_expr* root,
                                        const dftu_series* const* inputs,
                                        int32_t n_inputs);

/** Compile `n_roots` expressions into one program (CSE spans them; only
 * referenced inputs are materialized) and evaluate them in a single pass.
 * Writes one owned column per root into `out[0..n_roots)` and returns n_roots,
 * or -1 on a malformed expression (writing nothing). */
DFTU_EXPORT int32_t dftu_expr_eval_many(const dftu_expr* const* roots,
                                        int32_t n_roots,
                                        const dftu_series* const* inputs,
                                        int32_t n_inputs, dftu_series** out);
}

#endif  // DFTRACER_UTILS_DATAFRAME_EXPR_H
