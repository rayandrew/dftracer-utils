#include <dftracer/utils/dataframe/batch_ops.h>  // concat_columns
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/internal/expr_handle.h>
#include <dftracer/utils/dataframe/parallel.h>

#include <bit>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <tuple>
#include <vector>

namespace dataframe = dftracer::utils::dataframe;

namespace dftracer::utils::dataframe {

enum class ExprKind {
    LitI64,
    LitF64,
    Col,
    Binary,
    Prim,
    Unary,
    Clip,
    Fillna,
    Cmp,
    Logical,
    Not,
    Cast,
    Lower
};

struct ExprNode {
    ExprKind kind;
    std::int32_t i = 0;     // col index / prim / unary / cmp op / logical op /
                            // cast type / binary op
    dftu_scalar scalar{};   // literal value / cmp rhs / clip lo
    dftu_scalar scalar2{};  // clip hi
    std::shared_ptr<const ExprNode> a;  // first child
    std::shared_ptr<const ExprNode> b;  // second child
};

namespace {

Expr make(ExprKind k, std::int32_t i, dftu_scalar s,
          std::shared_ptr<const ExprNode> a,
          std::shared_ptr<const ExprNode> b) {
    auto n = std::make_shared<ExprNode>();
    n->kind = k;
    n->i = i;
    n->scalar = s;
    n->a = std::move(a);
    n->b = std::move(b);
    return Expr{std::move(n)};
}

}  // namespace

Expr expr_col(std::int32_t index) {
    return make(ExprKind::Col, index, {}, nullptr, nullptr);
}
std::int32_t expr_col_index(const Expr& e) {
    const auto& n = e.node();
    return n && n->kind == ExprKind::Col ? n->i : -1;
}

bool expr_as_col_cmp(const Expr& e, std::int32_t* col, CmpOp* op, Scalar* rhs) {
    const auto& n = e.node();
    if (!n || n->kind != ExprKind::Cmp || !n->a || n->a->kind != ExprKind::Col)
        return false;
    *col = n->a->i;
    *op = static_cast<CmpOp>(n->i);
    *rhs = n->scalar;
    return true;
}

bool expr_as_col_binary(const Expr& e, BinaryOp* op, std::int32_t* a,
                        std::int32_t* b) {
    const auto& n = e.node();
    if (!n || n->kind != ExprKind::Binary || !n->a || !n->b ||
        n->a->kind != ExprKind::Col || n->b->kind != ExprKind::Col)
        return false;
    *op = static_cast<BinaryOp>(n->i);
    *a = n->a->i;
    *b = n->b->i;
    return true;
}

namespace {
bool node_references(const std::shared_ptr<const ExprNode>& n,
                     std::int32_t index) {
    if (!n) return false;
    if (n->kind == ExprKind::Col) return n->i == index;
    return node_references(n->a, index) || node_references(n->b, index);
}
}  // namespace

bool expr_references(const Expr& e, std::int32_t index) {
    return node_references(e.node(), index);
}

namespace {
std::shared_ptr<const ExprNode> node_remap(
    const std::shared_ptr<const ExprNode>& n,
    const std::vector<std::int32_t>& old_to_new) {
    if (!n) return nullptr;
    if (n->kind == ExprKind::Col) {
        const bool in_range =
            n->i >= 0 && static_cast<std::size_t>(n->i) < old_to_new.size();
        auto c = std::make_shared<ExprNode>(*n);
        c->i = in_range ? old_to_new[static_cast<std::size_t>(n->i)] : n->i;
        return c;
    }
    auto a = node_remap(n->a, old_to_new);
    auto b = node_remap(n->b, old_to_new);
    if (a == n->a && b == n->b) return n;  // unchanged subtree: share it
    auto c = std::make_shared<ExprNode>(*n);
    c->a = std::move(a);
    c->b = std::move(b);
    return c;
}
}  // namespace

Expr expr_remap_cols(const Expr& e,
                     const std::vector<std::int32_t>& old_to_new) {
    return Expr{node_remap(e.node(), old_to_new)};
}
Expr expr_lit(std::int64_t value) {
    dftu_scalar s{};
    s.kind = DFTU_SCALAR_TAG_I64;
    s.value.i = value;
    return make(ExprKind::LitI64, 0, s, nullptr, nullptr);
}
Expr expr_lit(double value) {
    dftu_scalar s{};
    s.kind = DFTU_SCALAR_TAG_F64;
    s.value.d = value;
    return make(ExprKind::LitF64, 0, s, nullptr, nullptr);
}
Expr expr_binary(BinaryOp op, const Expr& a, const Expr& b) {
    return make(ExprKind::Binary, static_cast<std::int32_t>(op), {}, a.node(),
                b.node());
}
Expr expr_prim(std::int32_t prim, const Expr& a) {
    return make(ExprKind::Prim, prim, {}, a.node(), nullptr);
}
Expr expr_unary(std::int32_t op, const Expr& a) {
    return make(ExprKind::Unary, op, {}, a.node(), nullptr);
}
Expr expr_clip(const Expr& a, dftu_scalar lo, dftu_scalar hi) {
    auto n = std::make_shared<ExprNode>();
    n->kind = ExprKind::Clip;
    n->scalar = lo;
    n->scalar2 = hi;
    n->a = a.node();
    return Expr{std::move(n)};
}
Expr expr_fillna(const Expr& a, dftu_scalar fill) {
    return make(ExprKind::Fillna, 0, fill, a.node(), nullptr);
}
Expr expr_cmp(std::int32_t cmp, const Expr& a, dftu_scalar rhs) {
    return make(ExprKind::Cmp, cmp, rhs, a.node(), nullptr);
}
Expr expr_logical(std::int32_t op, const Expr& a, const Expr& b) {
    return make(ExprKind::Logical, op, {}, a.node(), b.node());
}
Expr expr_not(const Expr& a) {
    return make(ExprKind::Not, 0, {}, a.node(), nullptr);
}
Expr expr_cast(TypeId type, const Expr& a) {
    return make(ExprKind::Cast, static_cast<std::int32_t>(type), {}, a.node(),
                nullptr);
}
Expr expr_lower(const Expr& a) {
    return make(ExprKind::Lower, 0, {}, a.node(), nullptr);
}

// ---- compiler: type inference + CSE + lowering to a slot program ----------

namespace {

// Slot-IR opcodes.
enum {
    OP_LOAD,
    OP_ADD,
    OP_SUB,
    OP_MUL,
    OP_DIV,
    OP_ADDS,
    OP_SUBS,
    OP_MULS,
    OP_DIVS,
    OP_PRIM,
    OP_UNARY,
    OP_CLIP,
    OP_FILLNA,
    OP_CMP,
    OP_LOGICAL,
    OP_NOT,
    OP_CAST,
    OP_LOWER
};

struct SlotOp {
    int opcode;
    int a = -1;
    int b = -1;
    std::int32_t param = 0;
    dftu_scalar scalar{};
    dftu_scalar scalar2{};  // clip hi
};

const int COL_OP[4] = {OP_ADD, OP_SUB, OP_MUL, OP_DIV};
const int SCALAR_OP[4] = {OP_ADDS, OP_SUBS, OP_MULS, OP_DIVS};

// A compiled subexpression: either a compile-time scalar or a column in slot
// `slot` of type `type`.
struct Val {
    bool is_scalar;
    int slot;
    TypeId type;
    dftu_scalar scalar;
};

bool is_float(const Val& v) {
    if (v.is_scalar) return v.scalar.kind == DFTU_SCALAR_TAG_F64;
    return v.type == TypeId::Float32 || v.type == TypeId::Float64;
}

double scalar_to_double(dftu_scalar s) {
    if (s.kind == DFTU_SCALAR_TAG_I64) return static_cast<double>(s.value.i);
    if (s.kind == DFTU_SCALAR_TAG_U64) return static_cast<double>(s.value.u);
    return s.value.d;
}

dftu_scalar to_f64_scalar(dftu_scalar s) {
    dftu_scalar r{};
    r.kind = DFTU_SCALAR_TAG_F64;
    r.value.d = scalar_to_double(s);
    return r;
}

class Compiler {
   public:
    explicit Compiler(const std::vector<const Series*>& inputs)
        : inputs_(inputs) {}

    Val compile(const ExprNode* n) {
        switch (n->kind) {
            case ExprKind::LitI64:
            case ExprKind::LitF64:
                return {true, -1, TypeId::Int64, n->scalar};
            case ExprKind::Col: {
                if (n->i < 0 ||
                    static_cast<std::size_t>(n->i) >= inputs_.size())
                    throw std::invalid_argument(
                        "expr: column index out of range");
                int slot = emit(OP_LOAD, -1, -1, n->i, {});
                return {false,
                        slot,
                        inputs_[static_cast<std::size_t>(n->i)]->type(),
                        {}};
            }
            case ExprKind::Binary:
                return compile_binary(n);
            case ExprKind::Prim: {
                Val a = as_col(compile(n->a.get()), "prim");
                if (a.type != TypeId::Int64 && a.type != TypeId::Uint64)
                    a = cast(a, TypeId::Int64);
                return {false,
                        emit(OP_PRIM, a.slot, -1, n->i, {}),
                        TypeId::Int64,
                        {}};
            }
            case ExprKind::Unary: {
                Val a = as_col(compile(n->a.get()), "unary");
                // is_nan/is_finite/is_infinite yield a Bool mask; log/sqrt/exp
                // widen to Float64; the rest keep the input type (integer
                // floor/ceil/round/trunc are identities).
                const auto op = static_cast<UnaryOp>(n->i);
                TypeId t;
                switch (op) {
                    case UnaryOp::IsNan:
                    case UnaryOp::IsFinite:
                    case UnaryOp::IsInfinite:
                        t = TypeId::Bool;
                        break;
                    case UnaryOp::Log:
                    case UnaryOp::Sqrt:
                    case UnaryOp::Exp:
                        t = TypeId::Float64;
                        break;
                    default:
                        t = a.type;
                        break;
                }
                return {false, emit(OP_UNARY, a.slot, -1, n->i, {}), t, {}};
            }
            case ExprKind::Clip: {
                Val a = as_col(compile(n->a.get()), "clip");
                return {false,
                        emit(OP_CLIP, a.slot, -1, 0, n->scalar, n->scalar2),
                        a.type,
                        {}};
            }
            case ExprKind::Fillna: {
                Val a = as_col(compile(n->a.get()), "fillna");
                return {false,
                        emit(OP_FILLNA, a.slot, -1, 0, n->scalar),
                        a.type,
                        {}};
            }
            case ExprKind::Cmp: {
                Val a = as_col(compile(n->a.get()), "compare");
                return {false,
                        emit(OP_CMP, a.slot, -1, n->i, n->scalar),
                        TypeId::Bool,
                        {}};
            }
            case ExprKind::Logical: {
                Val a = as_col(compile(n->a.get()), "logical");
                Val b = as_col(compile(n->b.get()), "logical");
                return {false,
                        emit(OP_LOGICAL, a.slot, b.slot, n->i, {}),
                        TypeId::Bool,
                        {}};
            }
            case ExprKind::Not: {
                Val a = as_col(compile(n->a.get()), "not");
                return {
                    false, emit(OP_NOT, a.slot, -1, 0, {}), TypeId::Bool, {}};
            }
            case ExprKind::Cast: {
                Val a = as_col(compile(n->a.get()), "cast");
                return cast(a, static_cast<TypeId>(n->i));
            }
            case ExprKind::Lower: {
                Val a = as_col(compile(n->a.get()), "lower");
                if (a.type != TypeId::String)
                    throw std::invalid_argument(
                        "expr: lower needs a String column");
                return {false,
                        emit(OP_LOWER, a.slot, -1, 0, {}),
                        TypeId::String,
                        {}};
            }
        }
        throw std::invalid_argument("expr: unknown node");
    }

    std::vector<SlotOp> program;

   private:
    Val as_col(Val v, const char* who) {
        if (v.is_scalar)
            throw std::invalid_argument(std::string("expr: ") + who +
                                        " needs a column operand");
        return v;
    }

    Val cast(Val v, TypeId t) {
        return {false,
                emit(OP_CAST, v.slot, -1, static_cast<std::int32_t>(t), {}),
                t,
                {}};
    }

    Val compile_binary(const ExprNode* n) {
        const int op = n->i;  // BinaryOp
        Val a = compile(n->a.get());
        Val b = compile(n->b.get());
        if (a.is_scalar && b.is_scalar) return fold(op, a.scalar, b.scalar);

        const bool rf =
            op == static_cast<int>(BinaryOp::Div) || is_float(a) || is_float(b);
        TypeId out_type;
        if (rf) {
            promote_float(a);
            promote_float(b);
            out_type = TypeId::Float64;
        } else if (!a.is_scalar && !b.is_scalar && a.type != b.type) {
            a = cast(a, TypeId::Int64);
            b = cast(b, TypeId::Int64);
            out_type = TypeId::Int64;
        } else {
            out_type = a.is_scalar ? b.type : a.type;
        }

        if (!a.is_scalar && !b.is_scalar)
            return {
                false, emit(COL_OP[op], a.slot, b.slot, 0, {}), out_type, {}};
        if (!a.is_scalar)  // col op scalar
            return {false,
                    emit(SCALAR_OP[op], a.slot, -1, 0, b.scalar),
                    out_type,
                    {}};
        if (op == static_cast<int>(BinaryOp::Add) ||
            op == static_cast<int>(BinaryOp::Mul))  // scalar op col commutes
            return {false,
                    emit(SCALAR_OP[op], b.slot, -1, 0, a.scalar),
                    out_type,
                    {}};
        throw std::invalid_argument("expr: scalar - / column has no kernel");
    }

    void promote_float(Val& v) {
        if (v.is_scalar) {
            v.scalar = to_f64_scalar(v.scalar);
        } else if (v.type != TypeId::Float32 && v.type != TypeId::Float64) {
            v = cast(v, TypeId::Float64);
        }
    }

    Val fold(int op, dftu_scalar a, dftu_scalar b) {
        const bool f = op == static_cast<int>(BinaryOp::Div) ||
                       a.kind == DFTU_SCALAR_TAG_F64 ||
                       b.kind == DFTU_SCALAR_TAG_F64;
        dftu_scalar r{};
        if (f) {
            double x = scalar_to_double(a), y = scalar_to_double(b);
            r.kind = DFTU_SCALAR_TAG_F64;
            r.value.d = op == 0   ? x + y
                        : op == 1 ? x - y
                        : op == 2 ? x * y
                                  : x / y;
        } else {
            std::int64_t x = a.value.i, y = b.value.i;
            r.kind = DFTU_SCALAR_TAG_I64;
            r.value.i = op == 0   ? x + y
                        : op == 1 ? x - y
                        : op == 2 ? x * y
                                  : (y != 0 ? x / y : 0);
        }
        return {true, -1, TypeId::Int64, r};
    }

    // Emit an op, hash-consing structurally identical ops to the same slot
    // (common-subexpression elimination).
    int emit(int opcode, int a, int b, std::int32_t param, dftu_scalar s,
             dftu_scalar s2 = {}) {
        auto bits = [](dftu_scalar x) {
            return x.kind == DFTU_SCALAR_TAG_F64
                       ? std::bit_cast<std::int64_t>(x.value.d)
                       : x.value.i;
        };
        auto key = std::make_tuple(opcode, a, b, static_cast<int>(param),
                                   static_cast<int>(s.kind), bits(s), bits(s2));
        auto it = memo_.find(key);
        if (it != memo_.end()) return it->second;
        int slot = static_cast<int>(program.size());
        program.push_back({opcode, a, b, param, s, s2});
        memo_.emplace(key, slot);
        return slot;
    }

    const std::vector<const Series*>& inputs_;
    std::map<std::tuple<int, int, int, int, int, std::int64_t, std::int64_t>,
             int>
        memo_;
};

// Series::slice only supports fixed-width types (dftu_series_slice returns
// null for String/Binary); fall back to a row-index take for those.
Series load_slice(const Series& in, std::int64_t offset, std::int64_t len) {
    if (in.type() != TypeId::String && in.type() != TypeId::Binary)
        return in.slice(offset, len);
    std::vector<std::int64_t> idx(static_cast<std::size_t>(len));
    for (std::int64_t i = 0; i < len; ++i)
        idx[static_cast<std::size_t>(i)] = offset + i;
    return in.take(idx);
}

// Evaluate the slot program over rows [offset, offset+len) and extract one
// column per requested final slot (shared, so distinct outputs that resolved to
// the same slot alias the one buffer).
std::vector<Series> eval_chunk(const std::vector<SlotOp>& prog,
                               const std::vector<Series>& inputs,
                               const std::vector<int>& finals,
                               std::int64_t offset, std::int64_t len) {
    std::vector<Series> s(prog.size());
    for (std::size_t k = 0; k < prog.size(); ++k) {
        const SlotOp& op = prog[k];
        auto A = [&]() { return s[static_cast<std::size_t>(op.a)].handle(); };
        auto B = [&]() { return s[static_cast<std::size_t>(op.b)].handle(); };
        switch (op.opcode) {
            case OP_LOAD:
                s[k] = load_slice(inputs[static_cast<std::size_t>(op.param)],
                                  offset, len);
                break;
            case OP_ADD:
                s[k] = Series{dftu_series_add(A(), B())};
                break;
            case OP_SUB:
                s[k] = Series{dftu_series_sub(A(), B())};
                break;
            case OP_MUL:
                s[k] = Series{dftu_series_mul(A(), B())};
                break;
            case OP_DIV:
                s[k] = Series{dftu_series_div(A(), B())};
                break;
            case OP_ADDS:
                s[k] = Series{dftu_series_add_scalar(A(), op.scalar)};
                break;
            case OP_SUBS:
                s[k] = Series{dftu_series_sub_scalar(A(), op.scalar)};
                break;
            case OP_MULS:
                s[k] = Series{dftu_series_mul_scalar(A(), op.scalar)};
                break;
            case OP_DIVS:
                s[k] = Series{dftu_series_div_scalar(A(), op.scalar)};
                break;
            case OP_PRIM:
                s[k] = Series{
                    dftu_series_prim(A(), static_cast<dftu_prim_op>(op.param))};
                break;
            case OP_UNARY: {
                dftu_series* r = nullptr;
                switch (static_cast<UnaryOp>(op.param)) {
                    case UnaryOp::Abs:
                        r = dftu_series_abs(A());
                        break;
                    case UnaryOp::Round:
                        r = dftu_series_round(A());
                        break;
                    case UnaryOp::Floor:
                        r = dftu_series_floor(A());
                        break;
                    case UnaryOp::Ceil:
                        r = dftu_series_ceil(A());
                        break;
                    case UnaryOp::Log:
                        r = dftu_series_log(A());
                        break;
                    case UnaryOp::Sqrt:
                        r = dftu_series_sqrt(A());
                        break;
                    case UnaryOp::Exp:
                        r = dftu_series_exp(A());
                        break;
                    case UnaryOp::Sign:
                        r = dftu_series_sign(A());
                        break;
                    case UnaryOp::Negate:
                        r = dftu_series_negate(A());
                        break;
                    case UnaryOp::Trunc:
                        r = dftu_series_trunc(A());
                        break;
                    case UnaryOp::IsNan:
                        r = dftu_series_is_nan(A());
                        break;
                    case UnaryOp::IsFinite:
                        r = dftu_series_is_finite(A());
                        break;
                    case UnaryOp::IsInfinite:
                        r = dftu_series_is_infinite(A());
                        break;
                }
                s[k] = Series{r};
                break;
            }
            case OP_CLIP:
                s[k] = Series{dftu_series_clip(A(), op.scalar, op.scalar2)};
                break;
            case OP_FILLNA:
                s[k] = Series{dftu_series_fillna(A(), op.scalar)};
                break;
            case OP_CMP:
                s[k] = Series{dftu_series_compare(
                    A(), static_cast<dftu_cmp_op>(op.param), op.scalar)};
                break;
            case OP_LOGICAL:
                s[k] = Series{dftu_series_logical(
                    A(), B(), static_cast<dftu_logical_op>(op.param))};
                break;
            case OP_NOT:
                s[k] = Series{dftu_series_logical_not(A())};
                break;
            case OP_CAST:
                s[k] = Series{
                    dftu_series_cast(A(), static_cast<dftu_dtype>(op.param))};
                break;
            case OP_LOWER:
                s[k] = Series{dftu_series_to_lowercase(A())};
                break;
            default:
                return {};
        }
    }
    std::vector<Series> outs;
    outs.reserve(finals.size());
    for (int f : finals) outs.push_back(s[static_cast<std::size_t>(f)].share());
    return outs;
}

Series ensure_flat(const Series& c) {
    if (c.encoding() == Encoding::Flat && c.null_count() == 0) return c.share();
    return Series{dftu_series_materialize(c.handle())};
}

constexpr std::int64_t GRAIN = 1 << 16;

}  // namespace

std::vector<Series> eval_many(const std::vector<Expr>& roots,
                              const std::vector<const Series*>& inputs) {
    if (roots.empty())
        throw std::invalid_argument("expr: needs at least one expression");
    if (inputs.empty())
        throw std::invalid_argument("expr: needs at least one input column");

    // One compiler for all roots: hash-consing (CSE) spans the whole program.
    Compiler c(inputs);
    std::vector<int> finals;
    finals.reserve(roots.size());
    for (const Expr& root : roots) {
        if (!root.valid()) throw std::invalid_argument("expr: null expression");
        Val out = c.compile(root.node().get());
        if (out.is_scalar)
            throw std::invalid_argument(
                "expr: a constant expression has no column");
        finals.push_back(out.slot);
    }

    // Pruner: materialize only the inputs the program actually loads.
    std::vector<bool> used(inputs.size(), false);
    for (const SlotOp& op : c.program)
        if (op.opcode == OP_LOAD)
            used[static_cast<std::size_t>(op.param)] = true;
    std::vector<Series> flat(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i)
        if (used[i]) flat[i] = ensure_flat(*inputs[i]);

    const std::int64_t n = inputs.front()->length();
    if (n <= GRAIN) return eval_chunk(c.program, flat, finals, 0, n);

    const std::int64_t chunks = (n + GRAIN - 1) / GRAIN;
    std::vector<std::vector<Series>> parts(static_cast<std::size_t>(chunks));
    parallel_for(n, GRAIN, [&](std::int64_t b, std::int64_t e) {
        parts[static_cast<std::size_t>(b / GRAIN)] =
            eval_chunk(c.program, flat, finals, b, e - b);
    });
    std::vector<Series> outs;
    outs.reserve(finals.size());
    for (std::size_t j = 0; j < finals.size(); ++j) {
        std::vector<const Series*> ptrs;
        ptrs.reserve(parts.size());
        for (const std::vector<Series>& part : parts) ptrs.push_back(&part[j]);
        outs.push_back(concat_columns(ptrs));
    }
    return outs;
}

Series eval(const Expr& root, const std::vector<const Series*>& inputs) {
    if (!root.valid()) throw std::invalid_argument("expr: null expression");
    std::vector<Series> outs = eval_many({root}, inputs);
    return std::move(outs.front());
}

}  // namespace dftracer::utils::dataframe

// ---- C ABI ----------------------------------------------------------------

struct dftu_expr {
    dftracer::utils::dataframe::Expr e;
};

namespace dftracer::utils::dataframe {
const Expr& expr_handle_unwrap(const dftu_expr* h) { return h->e; }
}  // namespace dftracer::utils::dataframe

namespace {
dftu_expr* wrap(dataframe::Expr e) { return new dftu_expr{std::move(e)}; }
const dataframe::Expr& unwrap(const dftu_expr* e) { return e->e; }
}  // namespace

extern "C" {

dftu_expr* dftu_expr_col(int32_t index) {
    return wrap(dataframe::expr_col(index));
}
dftu_expr* dftu_expr_lit_i64(int64_t value) {
    return wrap(dataframe::expr_lit(static_cast<std::int64_t>(value)));
}
dftu_expr* dftu_expr_lit_f64(double value) {
    return wrap(dataframe::expr_lit(value));
}
dftu_expr* dftu_expr_binary(int32_t op, const dftu_expr* a,
                            const dftu_expr* b) {
    return wrap(dataframe::expr_binary(static_cast<dataframe::BinaryOp>(op),
                                       unwrap(a), unwrap(b)));
}
dftu_expr* dftu_expr_prim(int32_t prim, const dftu_expr* a) {
    return wrap(dataframe::expr_prim(prim, unwrap(a)));
}
dftu_expr* dftu_expr_unary(int32_t op, const dftu_expr* a) {
    return wrap(dataframe::expr_unary(op, unwrap(a)));
}
dftu_expr* dftu_expr_clip(const dftu_expr* a, dftu_scalar lo, dftu_scalar hi) {
    return wrap(dataframe::expr_clip(unwrap(a), lo, hi));
}
dftu_expr* dftu_expr_cmp(int32_t cmp, const dftu_expr* a, dftu_scalar rhs) {
    return wrap(dataframe::expr_cmp(cmp, unwrap(a), rhs));
}
dftu_expr* dftu_expr_logical(int32_t op, const dftu_expr* a,
                             const dftu_expr* b) {
    return wrap(dataframe::expr_logical(op, unwrap(a), unwrap(b)));
}
dftu_expr* dftu_expr_not(const dftu_expr* a) {
    return wrap(dataframe::expr_not(unwrap(a)));
}
dftu_expr* dftu_expr_cast(int32_t type, const dftu_expr* a) {
    return wrap(
        dataframe::expr_cast(static_cast<dataframe::TypeId>(type), unwrap(a)));
}
dftu_expr* dftu_expr_lower(const dftu_expr* a) {
    return wrap(dataframe::expr_lower(unwrap(a)));
}
void dftu_expr_free(dftu_expr* e) { delete e; }

dftu_series* dftu_expr_eval(const dftu_expr* root,
                            const dftu_series* const* inputs,
                            int32_t n_inputs) {
    if (!root) return nullptr;
    std::vector<dataframe::Series> owned;
    std::vector<const dataframe::Series*> cols;
    owned.reserve(static_cast<std::size_t>(n_inputs));
    cols.reserve(static_cast<std::size_t>(n_inputs));
    for (int32_t i = 0; i < n_inputs; ++i) {
        owned.emplace_back(const_cast<dftu_series*>(inputs[i]));
        cols.push_back(&owned.back());
    }
    dftu_series* out = nullptr;
    try {
        out = dataframe::eval(unwrap(root), cols).release();
    } catch (const std::exception&) {
        out = nullptr;
    }
    for (dataframe::Series& c : owned)
        c.release();  // borrowed inputs, do not free
    return out;
}

int32_t dftu_expr_eval_many(const dftu_expr* const* roots, int32_t n_roots,
                            const dftu_series* const* inputs, int32_t n_inputs,
                            dftu_series** out) {
    if (!roots || n_roots <= 0) return -1;
    std::vector<dataframe::Expr> exprs;
    exprs.reserve(static_cast<std::size_t>(n_roots));
    for (int32_t i = 0; i < n_roots; ++i) {
        if (!roots[i]) return -1;
        exprs.push_back(unwrap(roots[i]));
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
        std::vector<dataframe::Series> res = dataframe::eval_many(exprs, cols);
        for (std::size_t i = 0; i < res.size(); ++i) out[i] = res[i].release();
        written = static_cast<int32_t>(res.size());
    } catch (const std::exception&) {
        written = -1;
    }
    for (dataframe::Series& c : owned)
        c.release();  // borrowed inputs, do not free
    return written;
}
}
