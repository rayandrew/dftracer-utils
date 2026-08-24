"""The unified expression DSL: one ``F`` for both column values and row filters.

``F.dur`` (or ``col("dur")`` / ``F("args.level")``) is an :class:`Expr` leaf.
Build from it in two directions with the same object:

- **Value expressions** - ``+ - * /``, scalar broadcast, and the numeric
  primitives (``.ilog2()``, ``.bit_width()``, ``.popcount()``, ``.clz()``,
  ``.ctz()``, ``.mix64()``). Evaluate in memory with ``.apply(source)``, which
  runs on our native SIMD columnar engine (a ``Series``, Highway-backed) and
  returns a native :class:`~dftracer.utils.Series`. Arrow appears only at the
  explicit ``.to_arrow()`` / ``.to_pandas()`` edge.
- **Filter predicates** - comparisons (``> >= < <= == !=``), string matching
  (``.like()``, ``.ilike()``, ``.regex()``, ``.contains()``), membership
  (``.is_in()`` / ``.not_in()``), ``resolved.*`` virtual fields, and ``& | ~``.
  A pure predicate serializes to the query DSL (``str(expr)`` / ``.to_query()``)
  that ``TraceViewer.filter()`` / ``.query()`` push down to the index.

Comparisons over a numeric column also evaluate in memory to a boolean-mask
Series via ``.apply()``. Predicate-only ops (string match, membership) and
string/bool comparisons have no in-memory form; they are filter-only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Dict, List, NamedTuple, Sequence, Union

from . import dftracer_utils_ext as _ext
from .dataframe import DataFrame
from .series import Series, _unwrap

if TYPE_CHECKING:
    import pyarrow as pa  # ty: ignore[unresolved-import]

# A source of columns for in-memory evaluation: a wrapped or native DataFrame, a
# {name: Series} mapping, or a pyarrow Table imported at the Arrow boundary.
_Source = Union["DataFrame", "_ext._DataFrame", Dict[str, "Series"], "pa.Table"]
# where(): a frame in, the same kind of frame out.
_Frame = Union["DataFrame", "_ext._DataFrame", "pa.Table"]

__all__ = [
    "col",
    "lit",
    "columnar",
    "eval_many",
    "Expr",
    "ColumnExpr",
    "Field",
    "F",
    "resolved",
    "Value",
    "where",
    "Agg",
    "GroupBy",
    "count",
]

_Scalar = Union[int, float]

#: A scalar the filter DSL can compare a field against.
Value = Union[str, int, float, bool]

# vec::TypeId (types.h enum order); only the ones the DSL requests.
_INT64 = 4
_UINT64 = 8
_FLOAT32 = 9
_FLOAT64 = 10
_FLOAT_TYPES = (_FLOAT32, _FLOAT64)

# vec::Prim codes (kernels/prims.h). Exposed as methods, e.g. F.dur.ilog2().
_PRIM_CODES = {
    "ilog2": 0,
    "bit_width": 1,
    "popcount": 2,
    "clz": 3,
    "ctz": 4,
    "mix64": 5,
}

# DFTU_VEC_CMP_* and DFTU_VEC_LOGICAL_* codes (vec/abi.h).
_CMP_CODES = {"gt": 0, "ge": 1, "lt": 2, "le": 3, "eq": 4, "ne": 5}
_CMP_SYMBOL = {"gt": ">", "ge": ">=", "lt": "<", "le": "<=", "eq": "==", "ne": "!="}
_LOGICAL_CODES = {"and": 0, "or": 1}

# Resolved virtual-field leaf names accepted after the resolved./r. prefix.
_RESOLVED_FIELDS = frozenset({"fpath", "cwd", "hostname", "host", "exec", "cmd"})

# vec_eval opcodes (mirror the Op enum in columnar_eval.cpp).
# Post-order AST node opcodes for vec_eval (mirror columnar_eval.cpp). This is a
# pure structural serialization - no type logic; the C++ compiler does type
# inference, CSE, and lowering, so C/C++ consumers get the same engine.
_AST_COL, _AST_LIT_I, _AST_LIT_F, _AST_BIN = 0, 1, 2, 3
_AST_PRIM, _AST_CMP, _AST_LOGICAL, _AST_NOT, _AST_CAST = 4, 5, 6, 7, 8
# BinaryOp codes (vec::BinaryOp).
_BIN_OP_CODE = {"+": 0, "-": 1, "*": 2, "/": 3}

# vec::AggOp codes (agg.h), mirrored by the C ABI DFTU_AGG_* and the C++ builders.
_AGG_CODE = {
    "count": 0,
    "sum": 1,
    "min": 2,
    "max": 3,
    "mean": 4,
    "var": 5,
    "std": 6,
    "skew": 7,
    "kurt": 8,
}

_NOT_PUSHABLE = (
    "not an index-pushable predicate; compute it with .apply() or filter the materialized frame"
)
_PREDICATE_ONLY = (
    "this predicate (like/ilike/regex/contains/is_in) is filter-only and has no "
    "in-memory .apply() form; push it down with TraceViewer.filter()/.query(), "
    "or use a Series string kernel (e.g. s.str_contains) on a materialized column"
)


class Expr:
    """A lazy expression. Build values with ``+ - * /`` and the numeric prims,
    or predicates with comparisons / string-match / membership / ``& | ~``.
    Evaluate a value or a numeric-comparison mask in memory with :meth:`apply`;
    serialize a pure predicate for index pushdown with :meth:`to_query`."""

    def __add__(self, other: object) -> "Expr":
        return _Bin("+", self, _wrap(other))

    def __sub__(self, other: object) -> "Expr":
        return _Bin("-", self, _wrap(other))

    def __mul__(self, other: object) -> "Expr":
        return _Bin("*", self, _wrap(other))

    def __truediv__(self, other: object) -> "Expr":
        return _Bin("/", self, _wrap(other))

    def __radd__(self, other: object) -> "Expr":
        return _Bin("+", _wrap(other), self)

    def __rsub__(self, other: object) -> "Expr":
        return _Bin("-", _wrap(other), self)

    def __rmul__(self, other: object) -> "Expr":
        return _Bin("*", _wrap(other), self)

    # Comparisons build predicate Exprs. Against a numeric value they also
    # evaluate in memory to a boolean mask; against a string/bool they are
    # filter-only.
    def __gt__(self, other: Value) -> "Expr":
        return _Cmp("gt", self, other)

    def __ge__(self, other: Value) -> "Expr":
        return _Cmp("ge", self, other)

    def __lt__(self, other: Value) -> "Expr":
        return _Cmp("lt", self, other)

    def __le__(self, other: Value) -> "Expr":
        return _Cmp("le", self, other)

    def __eq__(self, other: Value) -> "Expr":  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return _Cmp("eq", self, other)

    def __ne__(self, other: Value) -> "Expr":  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return _Cmp("ne", self, other)

    def eq(self, other: Value) -> "Expr":
        """Equality predicate (the method form of ``==``)."""
        return _Cmp("eq", self, other)

    def ne(self, other: Value) -> "Expr":
        """Inequality predicate (the method form of ``!=``)."""
        return _Cmp("ne", self, other)

    def count(self) -> "Agg":
        """Group row count as an :class:`Agg` (the field is ignored, matching the
        engine's count)."""
        return Agg("count", None)

    __hash__ = None  # type: ignore[assignment]  # comparisons build exprs, not bools

    # Combine predicates: &, |, ~ (Python and/or cannot be overloaded).
    def __and__(self, other: object) -> "Expr":
        return _Logical("and", self, _as_expr(other))

    def __or__(self, other: object) -> "Expr":
        return _Logical("or", self, _as_expr(other))

    def __invert__(self) -> "Expr":
        return _Logical("not", self, None)

    def apply(self, source: _Source) -> Series:
        """Evaluate this expression on ``source`` and return a
        :class:`~dftracer.utils.Series`.

        A value expression yields a value Series; a numeric comparison yields a
        boolean-mask Series. ``source`` is a native ``DataFrame`` (e.g. from
        ``view.collect()``), a ``{name: Series}`` mapping, or a
        ``pyarrow.Table``. Filter-only predicates (string match, membership,
        string/bool comparison) raise; push those down with
        ``TraceViewer.filter()`` instead. See also :meth:`DataFrame.apply` for
        the frame-first spelling ``df.apply(expr)``."""
        return Columnar(self).apply(source)

    def to_query(self) -> str:
        """Serialize this predicate to an index-pushable query-DSL string.

        Raises ``TypeError`` if the expression mixes value ops (arithmetic or
        numeric prims) into the predicate, i.e. it is not pushable; evaluate
        those in memory with :meth:`apply` instead."""
        return self._dsl()

    def _dsl(self) -> str:
        raise TypeError(_NOT_PUSHABLE)

    def __str__(self) -> str:
        return self._dsl()

    if TYPE_CHECKING:
        # Prims (dispatched via __getattr__) and aggs (installed by setattr
        # below), declared so consumers get precise types instead of Any.
        def ilog2(self) -> "Expr": ...
        def bit_width(self) -> "Expr": ...
        def popcount(self) -> "Expr": ...
        def clz(self) -> "Expr": ...
        def ctz(self) -> "Expr": ...
        def mix64(self) -> "Expr": ...
        def sum(self) -> "Agg": ...
        def min(self) -> "Agg": ...
        def max(self) -> "Agg": ...
        def mean(self) -> "Agg": ...
        def var(self) -> "Agg": ...
        def std(self) -> "Agg": ...
        def skew(self) -> "Agg": ...
        def kurt(self) -> "Agg": ...

    # Runtime prim-method seam; hidden from the checker so unknown attributes
    # are type errors, not Any (the real methods are declared above).
    if not TYPE_CHECKING:

        def __getattr__(self, name):
            # F.dur.ilog2(), F.hhash.mix64(), ... - a unary primitive as a method.
            if name in _PRIM_CODES:
                return lambda: _Prim(name, self)
            raise AttributeError(name)


ColumnExpr = Expr


def _wrap(x: object) -> Expr:
    if isinstance(x, Expr):
        return x
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError(f"expected a column expr or a number, got {x!r}")
    return _Lit(x)


class _Col(Expr):
    """A field / column reference. Its name drives both column lookup for
    :meth:`Expr.apply` and the field name in a pushed-down predicate."""

    def __init__(self, name: str) -> None:
        self.name = name

    def is_in(self, values: Sequence[Value]) -> Expr:
        """Membership predicate: the field is one of ``values``."""
        return _In(self.name, values, negate=False)

    def not_in(self, values: Sequence[Value]) -> Expr:
        """Exclusion predicate: the field is none of ``values``."""
        return _In(self.name, values, negate=True)

    def like(self, pattern: str) -> Expr:
        """SQL LIKE: ``%`` matches any run, ``_`` any single char."""
        return _Match(self.name, "like", pattern)

    def ilike(self, pattern: str) -> Expr:
        """Case-insensitive LIKE."""
        return _Match(self.name, "ilike", pattern)

    def regex(self, pattern: str) -> Expr:
        """ECMAScript regex match (anchored as written)."""
        return _Match(self.name, "~", pattern)

    def iregex(self, pattern: str) -> Expr:
        """Case-insensitive regex match."""
        return _Match(self.name, "~*", pattern)

    def contains(self, sub: str) -> Expr:
        """Unanchored substring search (``"sub" in field``)."""
        return _Contains(self.name, sub)


#: Back-compat alias: ``Field("dur")`` is the field leaf, same as ``F.dur``.
Field = _Col


class _Lit(Expr):
    def __init__(self, value: _Scalar) -> None:
        self.value = value


class _Bin(Expr):
    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self.op = op
        self.left = left
        self.right = right


class _Prim(Expr):
    def __init__(self, name: str, arg: Expr) -> None:
        self.name = name
        self.arg = arg


class _Cmp(Expr):
    def __init__(self, op: str, left: Expr, rhs: object) -> None:
        if isinstance(rhs, _Lit):
            rhs = rhs.value
        if isinstance(rhs, Expr):
            raise TypeError("comparison right side must be a scalar value, not an expression")
        self.op = op
        self.left = left
        self.rhs = rhs

    def _dsl(self) -> str:
        if not isinstance(self.left, _Col):
            raise TypeError(_NOT_PUSHABLE)
        return f"{self.left.name} {_CMP_SYMBOL[self.op]} {_format_value(self.rhs)}"


class _Logical(Expr):
    def __init__(self, op: str, left: Expr, right: "Expr | None") -> None:
        self.op = op
        self.left = left
        self.right = right

    def _dsl(self) -> str:
        if self.op == "not" or self.right is None:
            return f"not ({self.left._dsl()})"
        return f"({self.left._dsl()} {self.op} {self.right._dsl()})"


class _Match(Expr):
    """A string-match predicate: ``field <op> "pattern"`` where op is one of
    like / ilike / ~ (regex) / ~* (case-insensitive regex). Filter-only."""

    def __init__(self, field: str, op: str, pattern: str) -> None:
        self.field = field
        self.op = op
        self.pattern = pattern

    def _dsl(self) -> str:
        return f"{self.field} {self.op} {_format_value(self.pattern)}"


class _Contains(Expr):
    """A substring predicate, serialized literal-first as ``"sub" in field``.
    Filter-only."""

    def __init__(self, field: str, sub: str) -> None:
        self.field = field
        self.sub = sub

    def _dsl(self) -> str:
        return f"{_format_value(self.sub)} in {self.field}"


class _In(Expr):
    """A membership predicate (``field in [...]`` / ``field not in [...]``).
    Filter-only."""

    def __init__(self, field: str, values: Sequence[Value], negate: bool) -> None:
        self.field = field
        self.values = list(values)
        self.negate = negate

    def _dsl(self) -> str:
        items = ", ".join(_format_value(v) for v in self.values)
        kw = "not in" if self.negate else "in"
        return f"{self.field} {kw} [{items}]"


def _as_expr(x: object) -> Expr:
    if isinstance(x, Expr):
        return x
    raise TypeError("& / | combine expressions, not plain values")


def _format_value(v: object) -> str:
    if isinstance(v, str):
        escaped = v.replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


# Import Arrow data into the native columnar engine (the boundary for the DSL).
def _series_from_arrow(array: "pa.Array") -> "_ext._Series":
    return _ext._series_from_arrow(array)


def _dataframe_from_arrow(table: "pa.Table") -> "_ext._DataFrame":
    return _ext._dataframe_from_arrow(table)


def col(name: str) -> Expr:
    """A column / field reference by name."""
    return _Col(name)


def lit(value: _Scalar) -> Expr:
    """A scalar literal (kept distinct from a column name, Polars-style)."""
    return _Lit(value)


class _FAccessor:
    """The ``F`` entry point. ``F.dur`` / ``F("args.level")`` / ``F["args.level"]``
    all build a field-leaf :class:`Expr`; use the call/subscript forms for a
    nested or non-identifier field name."""

    def __call__(self, name: str) -> Expr:
        return _Col(name)

    def __getattr__(self, name: str) -> Expr:
        return _Col(name)

    def __getitem__(self, name: str) -> Expr:
        return _Col(name)


class _Wildcard(Expr):
    """``F.any``: the numeric-args wildcard for ``TraceViewer.agg``. ``.mean()``
    aggregates every numeric ``args.*`` field as a per-group mean (fields
    discovered at scan time); ``.count()`` is the group row count. The engine's
    wildcard path computes only the mean, so other reductions raise in
    ``TraceViewer.agg`` - name a field for those."""

    def count(self) -> "Agg":
        return Agg("count", self)


F = _FAccessor()
#: ``F.any`` - the numeric-args wildcard (see :class:`_Wildcard`).
F.any = _Wildcard()  # type: ignore[attr-defined]


def resolved(name: str) -> Expr:
    """A resolved virtual field (``resolved.<name>``), which the engine rewrites
    to a concrete hash lookup against the index. Known names: fpath, cwd,
    hostname (alias host), exec, cmd."""
    if name not in _RESOLVED_FIELDS:
        raise ValueError(
            f"unknown resolved field {name!r}; expected one of {sorted(_RESOLVED_FIELDS)}"
        )
    return _Col("resolved." + name)


def columnar(expr: Expr) -> "Columnar":
    """Wrap a value expression; call ``apply(source)`` to evaluate it on the
    DataFrame engine and get a native ``Series``."""
    if not isinstance(expr, Expr):
        raise TypeError("columnar() expects an expression (col()/F./lit())")
    return Columnar(expr)


def where(table: _Frame, pred: "Expr | str") -> _Frame:
    """Keep the rows where ``pred`` is true.

    ``pred`` is either an expression (numeric comparisons / ``& | ~``),
    evaluated on the DataFrame engine to a boolean mask, or a query DSL
    **string** (e.g. ``"dur > 100 and cat == 'io'"``), parsed by the query
    library and evaluated as a SIMD mask. A native ``DataFrame`` in stays fully
    columnar (returns a filtered ``DataFrame``); a ``pyarrow.Table`` in returns
    a filtered ``pyarrow.Table``."""
    native = _unwrap(table)  # a wrapped DataFrame -> its native handle
    if isinstance(pred, str):
        if isinstance(native, _ext._DataFrame):
            return DataFrame(native.query(pred))
        import pyarrow as pa

        return pa.table(_ext._dataframe_from_arrow(native).query(pred))
    mask = columnar(pred).apply(native)  # a wrapped Series
    if isinstance(native, _ext._DataFrame):
        return DataFrame(native.filter(_unwrap(mask)))
    import pyarrow as pa

    return native.filter(pa.array(_unwrap(mask)))


class Columnar:
    """A value expression ready to evaluate on the DataFrame engine."""

    def __init__(self, expr: Expr) -> None:
        self._expr = expr
        self._columns = _collect_columns(expr)

    @property
    def columns(self) -> List[str]:
        """Input column names this expression reads."""
        return list(self._columns)

    def apply(self, source: _Source) -> Series:
        """Evaluate on the DataFrame engine and return a native ``Series``.

        ``source`` is either a native ``DataFrame`` (e.g. ``view.collect()``) or
        a ``{name: Series}`` mapping - kept entirely columnar with no input
        Arrow; or a ``pyarrow.Table``, whose columns are imported into the engine
        at the boundary. The result is a ``Series``; call ``.to_arrow()`` /
        ``.to_pandas()`` only if you want Arrow.

        Python serializes the expression tree to a post-order AST and hands it
        to the native engine (``_ext.vec_eval``), which does all the compilation
        - type inference, CSE, lowering - and fused chunked + parallel
        evaluation. A pure-constant expression (no columns) broadcasts on the
        per-node path."""
        cols = self._to_vec_columns(source)
        if self._columns:
            ast: List[tuple] = []
            self._serialize(self._expr, ast)
            inputs = [cols[name] for name in self._columns]
            return Series(_ext.vec_eval(ast, inputs))
        # Constant expression (no columns): broadcast to the row count.
        out_float = _is_float(self._expr, cols)
        result = _eval(self._expr, cols, out_float)
        if isinstance(result, _ScalarVal):
            return Series(_broadcast_scalar(result.value, out_float, cols))
        return Series(result.value)

    # Serialize `expr` to a post-order AST (a list of (op, args...) tuples) that
    # the C++ engine rebuilds and compiles. Structural only - no type logic.
    def _serialize(self, expr: Expr, ast: List[tuple]) -> None:
        _emit_ast(expr, self._columns.index, ast)

    def _to_vec_columns(self, source: _Source) -> Dict[str, "_ext._Series"]:
        return _import_vec_columns(self._columns, source)


def _import_vec_columns(names: List[str], source: _Source) -> Dict[str, "_ext._Series"]:
    """Resolve `names` to native vec columns from `source` (a native batch or a
    pyarrow Table), importing at the Arrow boundary only when needed."""
    source = _unwrap(source)  # a wrapped DataFrame -> its native handle
    # A native batch (_DataFrame or a {name: Series} dict) is used directly.
    if isinstance(source, (dict, _ext._DataFrame)):
        return {name: _unwrap(source[name]) for name in names}
    # A pyarrow Table: import each needed column into vec at the boundary.
    import pyarrow as pa

    cols: Dict[str, "_ext._Series"] = {}
    for name in names:
        arr = source.column(name)
        if isinstance(arr, pa.ChunkedArray):
            arr = arr.combine_chunks()
        if arr.null_count:
            raise ValueError(f"columnar column '{name}' has nulls (unsupported)")
        cols[name] = _ext._series_from_arrow(arr)
    return cols


def eval_many(exprs: "List[Expr | Columnar]", source: _Source) -> "List[Series]":
    """Evaluate several value expressions in ONE compiled pass over `source`.

    CSE spans all of them, so a subexpression shared across outputs is computed
    once. Returns one :class:`Series` per input expression, aligned to `exprs`.
    `source` is a native vec batch or a pyarrow Table, as for :meth:`Columnar.apply`."""
    cs = [e if isinstance(e, Columnar) else columnar(e) for e in exprs]
    union: List[str] = []
    seen = set()
    for c in cs:
        for name in c.columns:
            if name not in seen:
                seen.add(name)
                union.append(name)
    if not union:  # all-constant expressions: no shared program to build
        return [c.apply(source) for c in cs]
    cols = _import_vec_columns(union, source)
    inputs = [cols[name] for name in union]
    index = {name: i for i, name in enumerate(union)}
    asts: List[List[tuple]] = []
    for c in cs:
        ast: List[tuple] = []
        _emit_ast(c._expr, index.__getitem__, ast)
        asts.append(ast)
    return [Series(s) for s in _ext.vec_eval_many(asts, inputs)]


def _broadcast_scalar(
    value: _Scalar, out_float: bool, cols: Dict[str, "_ext._Series"]
) -> "_ext._Series":
    # No constant-column kernel; derive one from any input column so a pure
    # constant expression stays in vec: (col * 0) + value, in the target type.
    if not cols:
        raise ValueError("a constant columnar expression needs at least one column")
    base = next(iter(cols.values())).cast(_FLOAT64 if out_float else _INT64)
    return base.mul_scalar(0).add_scalar(value)


# Serialize `expr` to a post-order AST, resolving column names to input indices
# via `resolve` (a name -> int callable). Structural only - the C++ compiler does
# type inference, CSE, and lowering, so every consumer gets the same engine.
def _emit_ast(expr: Expr, resolve: Callable[[str], int], ast: List[tuple]) -> None:
    if isinstance(expr, _Lit):
        if isinstance(expr.value, float):
            ast.append((_AST_LIT_F, float(expr.value)))
        else:
            ast.append((_AST_LIT_I, int(expr.value)))
    elif isinstance(expr, _Col):
        ast.append((_AST_COL, resolve(expr.name)))
    elif isinstance(expr, _Bin):
        _emit_ast(expr.left, resolve, ast)
        _emit_ast(expr.right, resolve, ast)
        ast.append((_AST_BIN, _BIN_OP_CODE[expr.op]))
    elif isinstance(expr, _Prim):
        _emit_ast(expr.arg, resolve, ast)
        ast.append((_AST_PRIM, _PRIM_CODES[expr.name]))
    elif isinstance(expr, _Cmp):
        if isinstance(expr.rhs, bool) or not isinstance(expr.rhs, (int, float)):
            raise TypeError(
                "in-memory comparison needs a numeric value; a string/bool "
                "comparison is filter-only (push it down with .filter()/.query())"
            )
        _emit_ast(expr.left, resolve, ast)
        ast.append((_AST_CMP, _CMP_CODES[expr.op], expr.rhs))
    elif isinstance(expr, _Logical):
        _emit_ast(expr.left, resolve, ast)
        if expr.op == "not" or expr.right is None:
            ast.append((_AST_NOT,))
        else:
            _emit_ast(expr.right, resolve, ast)
            ast.append((_AST_LOGICAL, _LOGICAL_CODES[expr.op]))
    elif isinstance(expr, (_Match, _Contains, _In)):
        raise TypeError(_PREDICATE_ONLY)
    else:
        raise TypeError(f"unsupported node {type(expr).__name__}")


class Agg:
    """An aggregate over a column expression - ``F.dur.mean()``,
    ``(F.a + F.b).sum()``, ``count()``. Rename the output with ``.alias(name)``.
    Feed to ``DataFrame.group_by(...).agg(...)``."""

    def __init__(self, op: str, value: "Expr | None", out: "str | None" = None) -> None:
        self.op = op
        self.value = value
        self._out = out

    def alias(self, name: str) -> "Agg":
        return Agg(self.op, self.value, name)

    @property
    def out(self) -> str:
        if self._out is not None:
            return self._out
        if self.value is None:
            return "count"
        if isinstance(self.value, _Col):
            return f"{self.op}_{self.value.name}"
        return self.op

    def _spec(self, names: List[str]) -> tuple:
        code = _AGG_CODE[self.op]
        if self.value is None:
            return (code, None, self.out)
        ast: List[tuple] = []
        _emit_ast(self.value, names.index, ast)
        return (code, ast, self.out)


def count() -> Agg:
    """The group row count (``count()`` -> column ``count``)."""
    return Agg("count", None)


def _make_agg_method(op: str) -> "Callable[[Expr], Agg]":
    def method(self: Expr) -> Agg:
        return Agg(op, self)

    method.__name__ = op
    method.__doc__ = f"The {op} of this expression per group (an :class:`Agg`)."
    return method


for _op in ("sum", "min", "max", "mean", "var", "std", "skew", "kurt"):
    setattr(Expr, _op, _make_agg_method(_op))


def _to_agg(spec: object) -> Agg:
    if isinstance(spec, Agg):
        return spec
    if isinstance(spec, str):
        if ":" in spec:
            op, column = spec.split(":", 1)
            return Agg(op, _Col(column), f"{op}_{column}")
        return Agg(spec, None, spec)  # e.g. "count"
    raise TypeError(f"agg spec must be an Agg or string, got {type(spec).__name__}")


class GroupBy:
    """A lazy group-by over a native ``DataFrame``; call ``.agg(*specs)`` with
    aggregate expressions (``F.x.sum()``, ``count()``) or legacy strings
    (``"sum:dur"``). Value expressions compile in one CSE'd, pruned pass."""

    def __init__(self, batch: "Union[DataFrame, _ext._DataFrame]", key: str) -> None:
        self._batch = batch
        self._key = key

    def agg(self, *specs: "Union[str, Agg]") -> DataFrame:
        if not specs:
            raise TypeError("agg() needs at least one aggregate")
        native = _unwrap(self._batch)
        names = list(native.column_names)
        ser = [_to_agg(s)._spec(names) for s in specs]
        return DataFrame(native._group_agg_expr(self._key, ser))


def _collect_columns(expr: Expr) -> List[str]:
    seen: List[str] = []

    def walk(e: Expr) -> None:
        if isinstance(e, _Col):
            if e.name not in seen:
                seen.append(e.name)
        elif isinstance(e, _Bin):
            walk(e.left)
            walk(e.right)
        elif isinstance(e, _Prim):
            walk(e.arg)
        elif isinstance(e, _Cmp):
            walk(e.left)
        elif isinstance(e, _Logical):
            walk(e.left)
            if e.right is not None:
                walk(e.right)
        elif isinstance(e, (_Match, _Contains, _In)):
            raise TypeError(_PREDICATE_ONLY)

    walk(expr)
    return seen


def _is_float(expr: Expr, cols: Dict[str, "_ext._Series"]) -> bool:
    if isinstance(expr, _Lit):
        return isinstance(expr.value, float)
    if isinstance(expr, _Col):
        return cols[expr.name].type in _FLOAT_TYPES
    if isinstance(expr, _Bin):
        if expr.op == "/":
            return True
        return _is_float(expr.left, cols) or _is_float(expr.right, cols)
    if isinstance(expr, (_Prim, _Cmp, _Logical)):
        return False  # primitives -> int64; comparisons/logical -> bool
    raise TypeError(f"unsupported column expression node {type(expr).__name__}")


# Discriminated result (isinstance-narrowable): a scalar or a native _Series.
class _ScalarVal(NamedTuple):
    value: _Scalar


class _ColVal(NamedTuple):
    value: "_ext._Series"


_EvalResult = Union[_ScalarVal, _ColVal]


def _need_col(r: _EvalResult, message: str) -> "_ext._Series":
    if isinstance(r, _ColVal):
        return r.value
    raise TypeError(message)


def _eval(expr: Expr, cols: Dict[str, "_ext._Series"], out_float: bool) -> _EvalResult:
    if isinstance(expr, _Lit):
        return _ScalarVal(float(expr.value) if out_float else int(expr.value))
    if isinstance(expr, _Col):
        vc = cols[expr.name]
        if out_float and vc.type not in _FLOAT_TYPES:
            vc = vc.cast(_FLOAT64)
        return _ColVal(vc)
    if isinstance(expr, _Bin):
        left = _eval(expr.left, cols, out_float)
        right = _eval(expr.right, cols, out_float)
        if isinstance(left, _ScalarVal) and isinstance(right, _ScalarVal):
            return _ScalarVal(_fold(expr.op, left.value, right.value))
        return _ColVal(_binop(expr.op, left, right))
    if isinstance(expr, _Prim):
        # Primitives run on a 64-bit integer column; evaluate the argument as an
        # int expression and cast if it is not already i64/u64.
        v = _need_col(
            _eval(expr.arg, cols, out_float=False),
            f"columnar {expr.name}() needs a column argument",
        )
        if v.type not in (_INT64, _UINT64):
            v = v.cast(_INT64)
        return _ColVal(v.prim(_PRIM_CODES[expr.name]))
    if isinstance(expr, _Cmp):
        if isinstance(expr.rhs, bool) or not isinstance(expr.rhs, (int, float)):
            raise TypeError(
                "in-memory comparison needs a numeric value; a string/bool "
                "comparison is filter-only (push it down with .filter()/.query())"
            )
        v = _need_col(
            _eval(expr.left, cols, out_float=_is_float(expr.left, cols)),
            "columnar comparison needs a column on the left",
        )
        return _ColVal(v.compare(_CMP_CODES[expr.op], expr.rhs))
    if isinstance(expr, _Logical):
        lv = _need_col(_eval(expr.left, cols, out_float=False), "columnar logical needs a column")
        if expr.op == "not" or expr.right is None:
            return _ColVal(lv.logical_not())
        rv = _need_col(_eval(expr.right, cols, out_float=False), "columnar logical needs a column")
        return _ColVal(lv.logical(_LOGICAL_CODES[expr.op], rv))
    if isinstance(expr, (_Match, _Contains, _In)):
        raise TypeError(_PREDICATE_ONLY)
    raise TypeError(f"unsupported column expression node {type(expr).__name__}")


def _fold(op: str, a: _Scalar, b: _Scalar) -> _Scalar:
    if op == "+":
        return a + b
    if op == "-":
        return a - b
    if op == "*":
        return a * b
    return a / b


_COL_OP = {"+": "add", "-": "sub", "*": "mul", "/": "div"}
_SCALAR_OP = {"+": "add_scalar", "-": "sub_scalar", "*": "mul_scalar", "/": "div_scalar"}


def _binop(op: str, left: _EvalResult, right: _EvalResult) -> "_ext._Series":
    if isinstance(left, _ColVal) and isinstance(right, _ColVal):
        return getattr(left.value, _COL_OP[op])(right.value)
    if isinstance(left, _ColVal):  # col op scalar
        return getattr(left.value, _SCALAR_OP[op])(right.value)
    if isinstance(right, _ColVal) and op in ("+", "*"):  # scalar op col (commutes)
        return getattr(right.value, _SCALAR_OP[op])(left.value)
    raise NotImplementedError(f"scalar {op} column needs a vec kernel (not yet available)")
