"""Author a DFTracer plugin in Python and get a native one.

Decorate a class with :func:`plugin`; its :func:`map` attributes and single
:func:`each_event` method are AST-compiled to a C plugin against the stable ABI,
built to a cached ``.so``, and loaded through
:class:`dftracer.utils.plugins.PluginHost` exactly like a hand-written plugin.

The ``each_event`` subset is deliberately small: an optional ``if`` guard (one
comparison on ``e.<field>``) wrapping ``self.<map>[(<key>)] += <1 | e.<field>>``.
Anything outside it raises :class:`JitError` at decoration time, pointing at the
raw-C++ escape hatch, never a silent miscompile.

Every :func:`map` is one named accumulator on the dataframe engine's ``AggState``,
reached through the host's ``DFTU_EXT_AGG`` service: the key tuple names its
grouping columns (an empty tuple makes it a whole-scan scalar) and each value
reduction is one aggregate. Per-event contributions buffer into per-batch columns
that are folded into that accumulator at the end of each batch; the host merges
same-named accumulators across worker slices and finalizes each to a frame.

Example::

    @jit.plugin
    class NameEdges:
        edges = jit.map(key=(jit.i64, jit.str_), value=jit.count())

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.edges[(e.pid, e.name)] += 1
"""

from __future__ import annotations

import ast
import builtins
import inspect
import textwrap
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Generic,
    List,
    NoReturn,
    Protocol,
    Tuple,
    TypeVar,
    cast,
    overload,
)

from .. import _plugin_build, jit_op
from ..jit_op import Op, compile_op, op, run_op
from . import ops as ops

# Python 3.8 wraps a subscript's slice in ast.Index; 3.9+ stores the value
# directly, and 3.14 may drop the class entirely, so resolve it defensively.
_AST_INDEX = getattr(ast, "Index", None)


def _unwrap_index(node: ast.expr) -> ast.expr:
    # _AST_INDEX is resolved at runtime, so the type checker cannot narrow node
    # to ast.Index here; read .value dynamically.
    if _AST_INDEX is not None and isinstance(node, _AST_INDEX):
        return getattr(node, "value", node)
    return node


__all__ = [
    "JitError",
    "Op",
    "op",
    "ops",
    "compile_op",
    "run_op",
    "plugin",
    "map",
    "publish",
    "consume",
    "Port",
    "count",
    "sum",
    "min",
    "max",
    "minf",
    "maxf",
    "distinct",
    "set",
    "list",
    "mean",
    "variance",
    "var",
    "stddev",
    "std",
    "quantiles",
    "skew",
    "kurt",
    "sumsq",
    "hist",
    "busy",
    "concurrency",
    "utilization",
    "active",
    "first",
    "last",
    "count_valid",
    "argmin",
    "argmax",
    "topk",
    "bottomk",
    "approx_topk",
    "sample",
    "bitset",
    "corr",
    "covar_pop",
    "covar_samp",
    "regr_slope",
    "regr_intercept",
    "regr_r2",
    "record",
    "config",
    "each_event",
    "vfold",
    "each_batch",
    "series",
    # numeric primitives
    "ilog2",
    "bit_width",
    "clz",
    "ctz",
    "popcount",
    "ceil_pow2",
    "floor_pow2",
    "rotl",
    "rotr",
    "abs",
    "clamp",
    "div_ceil",
    "div_round",
    "align_up",
    "align_down",
    "isqrt",
    "gcd",
    "mul_hi",
    "mix64",
    "fastrange",
    "fmin",
    "fmax",
    "clampf",
    "sqrt",
    "log2",
    "log",
    "exp",
    "fma",
    "lerp",
    "copysign",
    "Map",
    "Event",
    "Str",
    "Counter",
    "Sum",
    "Min",
    "Max",
    "Distinct",
    "SetV",
    "ListV",
    "Mean",
    "Variance",
    "Stddev",
    "Quantiles",
    "ArgMin",
    "ArgMax",
    "TopK",
    "BottomK",
    "ApproxTopK",
    "Sample",
    "Product",
    "i64",
    "str_",
    "bytes",
    "i8",
    "i16",
    "i32",
    "u8",
    "u16",
    "u32",
    "u64",
    "f32",
    "f64",
    "NONE",
    "NEED_ARGS",
    "NEED_FHASH",
    "NEED_HHASH",
]


class JitError(Exception):
    """A construct outside the supported ``@jit.each_event`` subset."""


T = TypeVar("T")
T_co = TypeVar("T_co", covariant=True)
K = TypeVar("K")
V = TypeVar("V")
K1 = TypeVar("K1")
K2 = TypeVar("K2")
K3 = TypeVar("K3")
K4 = TypeVar("K4")
_C = TypeVar("_C")


class Str(Protocol):
    """An interned-string id: usable as a STR key / SET_STR|LIST_STR element and
    comparable to a ``str`` literal. Authoring-only, never instantiated."""

    def __eq__(self, other: object) -> bool: ...
    def __ne__(self, other: object) -> bool: ...


class _Type(Generic[T_co]):
    """Phantom type marker: the parameter tracks the Python type a jit column
    maps to (u64 -> int, str_ -> str) and is never consumed, so it is
    covariant - _Type[int] is a _Type[object]."""

    __slots__ = ("dft",)

    def __init__(self, dft: str) -> None:
        self.dft = dft


i64: _Type[int] = _Type("DFTU_T_I64")
str_: _Type[Str] = _Type("DFTU_T_STR")
bytes: "_Type[builtins.bytes]" = _Type("DFTU_T_BYTES")

i8: _Type[int] = _Type("DFTU_T_I8")
i16: _Type[int] = _Type("DFTU_T_I16")
i32: _Type[int] = _Type("DFTU_T_I32")
u8: _Type[int] = _Type("DFTU_T_U8")
u16: _Type[int] = _Type("DFTU_T_U16")
u32: _Type[int] = _Type("DFTU_T_U32")
u64: _Type[int] = _Type("DFTU_T_U64")
f32: _Type[float] = _Type("DFTU_T_F32")
f64: _Type[float] = _Type("DFTU_T_F64")


class _Config(Generic[T_co]):
    """A runtime config field marker (jit.config). Read from the plugin's load
    config into a file-scope static and referenced in the body as self.<name>."""

    __slots__ = ("dft",)

    def __init__(self, dft: str) -> None:
        self.dft = dft


def config(of: "_Type[T_co]") -> T_co:
    """Declare a runtime config field bound at load from PluginHost config.

    ``threshold = jit.config(jit.i64)`` reads ``config["threshold"]`` into the
    plugin and the body reads it as ``self.threshold``. Any numeric type works
    (all integer/float widths ride the int64/double slot); a string config is
    not supported yet (interning it needs the host, absent at load time)."""
    if of.dft in ("DFTU_T_STR", "DFTU_T_BYTES"):
        raise JitError("jit.config supports a numeric type; a string config is not supported yet")
    return cast(T_co, _Config(of.dft))


class _Sentinel:
    __slots__ = ()


NONE = _Sentinel()


class _Need:
    __slots__ = ("dft",)

    def __init__(self, dft: str) -> None:
        self.dft = dft


NEED_ARGS = _Need("DFTU_NEED_ARGS")
NEED_FHASH = _Need("DFTU_NEED_FHASH")
NEED_HHASH = _Need("DFTU_NEED_HHASH")


class _Monoid:
    """One aggregate of a jit accumulator, lowered to a ``dftu_agg_col``.

    ``dft`` is the ``DFTU_AGG_*`` op code, ``param`` its scalar parameter (a
    quantile, a k, an occupancy cell tolerance), ``needs_by`` marks an op that
    reads a second (ordering / x) column, and ``elem`` is the value column's
    family: ``"none"`` (no value column), ``"i64"``, ``"f64"`` or ``"str"``."""

    __slots__ = ("dft", "param", "needs_by", "elem")

    def __init__(
        self, dft: str, param: float = 0.0, needs_by: bool = False, elem: str = "f64"
    ) -> None:
        self.dft = dft
        self.param = param
        self.needs_by = needs_by
        self.elem = elem


class Counter(_Monoid):
    __slots__ = ()

    def __iadd__(self, x: int) -> "Counter":
        raise NotImplementedError

    def observe(self, x: int) -> None: ...


class Sum(_Monoid):
    __slots__ = ()

    def __iadd__(self, x: float) -> "Sum":
        raise NotImplementedError

    def observe(self, x: float) -> None: ...


class Min(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, x: T) -> None: ...


class Max(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, x: T) -> None: ...


class Distinct(_Monoid):
    __slots__ = ()

    def observe(self, x: object) -> None: ...


class SetV(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, x: T) -> None: ...
    def add(self, x: T) -> None: ...


class ListV(_Monoid, Generic[T]):
    __slots__ = ()

    def append(self, x: T, *, order_by: int) -> None: ...


class Mean(_Monoid):
    __slots__ = ()

    def observe(self, v: float) -> None: ...


class Variance(_Monoid):
    __slots__ = ()

    def observe(self, v: float) -> None: ...


class Stddev(_Monoid):
    __slots__ = ()

    def observe(self, v: float) -> None: ...


class AggReduce(_Monoid):
    """A reduction with no ``+=`` / ``.observe`` shorthand of its own; it is
    declared as a map value and folded by the engine like any other aggregate."""

    __slots__ = ()


class Quantiles(_Monoid):
    """A DDSketch quantile aggregate; ``qs`` materializes one ``p<q*100>``
    column per quantile alongside a ``count`` column."""

    __slots__ = ("qs",)

    def __init__(self, qs: "Tuple[float, ...]") -> None:
        super().__init__("DFTU_AGG_PCT", param=qs[0], elem="f64")
        self.qs = qs

    def observe(self, v: float) -> None: ...


class ArgMin(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, payload: T, *, by: float) -> None: ...


class ArgMax(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, payload: T, *, by: float) -> None: ...


class TopK(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, payload: T, *, by: float) -> None: ...


class BottomK(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, payload: T, *, by: float) -> None: ...


class ApproxTopK(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, x: T) -> None: ...


class Sample(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, x: T) -> None: ...


class _Accum(_Monoid):
    """A :class:`Product` component: the permissive union of accumulator ops
    (``+= n`` and ``.observe(...)``), since dict/tuple products do not carry
    per-component monoid types."""

    __slots__ = ()

    def __iadd__(self, x: "int | float") -> "_Accum":
        raise NotImplementedError

    def observe(self, *args: object, **kwargs: object) -> None: ...


class Product(_Monoid):
    __slots__ = ()

    # Type-only (runtime access is AST-compiled; a real __setattr__ would break
    # _Monoid.__init__). __setattr__ lets `product.cnt += 1` type-check.
    if TYPE_CHECKING:

        def __getitem__(self, i: int) -> _Accum: ...
        def __getattr__(self, name: str) -> _Accum: ...
        def __setattr__(self, name: str, value: _Accum) -> None: ...


def count() -> Counter:
    """An integer counting value: ``+= 1`` counts, ``+= <expr>`` sums.

    Materializes to an int64 column."""
    return Counter("DFTU_AGG_SUM", elem="i64")


def sum() -> Sum:
    """A double-summing value; materializes to an f64 column."""
    return Sum("DFTU_AGG_SUM", elem="f64")


# ---- numeric primitives -----------------------------------------------------
# Authoring markers, not implementations. Inside @jit.each_event each call is
# AST-lowered to the matching prims.h helper (see _PRIM_C) - the C in prims.h is
# the single source of truth, so there is no Python reimplementation to drift
# from it. Calling one directly in Python raises, just like a monoid factory's
# value cannot be read at Python runtime.


def _prim(name: str):
    def marker(*args: int) -> int:
        raise JitError(
            f"jit.{name} is a plugin primitive; use it inside a @jit.each_event "
            f"body, where it lowers to dftu_{name}_* in C."
        )

    marker.__name__ = name
    return marker


ilog2 = _prim("ilog2")
"""Floor of log2(x): the highest set-bit index - the natural log-scale histogram
bucket. Lowers to ``dftu_ilog2_u64``."""

bit_width = _prim("bit_width")
"""Bits needed to represent x (``ilog2 + 1``); 0 for 0. ``dftu_bit_width_u64``."""

clz = _prim("clz")
"""Leading zero bits of a 64-bit x (64 for 0). ``dftu_clz_u64``."""

ctz = _prim("ctz")
"""Trailing zero bits of a 64-bit x (64 for 0). ``dftu_ctz_u64``."""

popcount = _prim("popcount")
"""Number of set bits in a 64-bit x. ``dftu_popcount_u64``."""

ceil_pow2 = _prim("ceil_pow2")
"""Smallest power of two >= x (1 for 0). ``dftu_ceil_pow2_u64``."""

floor_pow2 = _prim("floor_pow2")
"""Largest power of two <= x; 0 for 0. ``dftu_floor_pow2_u64``."""

rotl = _prim("rotl")
"""Rotate a 64-bit x left by r bits (mod 64). ``dftu_rotl_u64``."""

rotr = _prim("rotr")
"""Rotate a 64-bit x right by r bits (mod 64). ``dftu_rotr_u64``."""

abs = _prim("abs")
"""Absolute value. ``dftu_abs_i64``."""

clamp = _prim("clamp")
"""Clamp x to the inclusive range [lo, hi]. ``dftu_clamp_i64``."""

div_ceil = _prim("div_ceil")
"""Ceiling of a / b; 0 when b is 0. ``dftu_div_ceil_i64``."""

div_round = _prim("div_round")
"""a / b rounded to nearest, half away from zero; 0 when b is 0. ``dftu_div_round_i64``."""

align_up = _prim("align_up")
"""Round x up to the next multiple of a; x when a is 0. ``dftu_align_up_u64``."""

align_down = _prim("align_down")
"""Round x down to the previous multiple of a; x when a is 0. ``dftu_align_down_u64``."""

isqrt = _prim("isqrt")
"""Integer square root (largest r with r*r <= x). ``dftu_isqrt_u64``."""

gcd = _prim("gcd")
"""Greatest common divisor. ``dftu_gcd_u64``."""

mul_hi = _prim("mul_hi")
"""High 64 bits of the 128-bit product a*b. ``dftu_mul_hi_u64``."""

mix64 = _prim("mix64")
"""SplitMix64 finalizer: a fast, well-avalanched hash of x - good for
hash-bucketing or deterministic sampling. ``dftu_mix64``."""

fastrange = _prim("fastrange")
"""Map x uniformly into [0, n) without a modulo (Lemire's reduction); pair with
:func:`mix64` for hash bucketing. 0 when n is 0. ``dftu_fastrange_u64``."""


def _primf(name: str):
    def marker(*args: float) -> float:
        raise JitError(
            f"jit.{name} is a plugin primitive; use it inside a @jit.each_event "
            f"body, where it lowers to dftu_{name}_* in C."
        )

    marker.__name__ = name
    return marker


fmin = _primf("fmin")
"""Lesser of two doubles. ``dftu_min_f64``."""

fmax = _primf("fmax")
"""Greater of two doubles. ``dftu_max_f64``."""

clampf = _primf("clampf")
"""Clamp a double x to the inclusive range [lo, hi]. ``dftu_clamp_f64``."""

sqrt = _primf("sqrt")
"""Square root. ``dftu_sqrt_f64``."""

log2 = _primf("log2")
"""Base-2 logarithm (contrast :func:`ilog2`, the integer bit index). ``dftu_log2_f64``."""

log = _primf("log")
"""Natural logarithm. ``dftu_log_f64``."""

exp = _primf("exp")
"""Natural exponential. ``dftu_exp_f64``."""

fma = _primf("fma")
"""Fused multiply-add ``a*b + c`` with a single rounding. ``dftu_fma_f64``."""

lerp = _primf("lerp")
"""Linear interpolation from a to b by t (unclamped). ``dftu_lerp_f64``."""

copysign = _primf("copysign")
"""Magnitude of x with the sign of y. ``dftu_copysign_f64``."""


_MINMAX_WIDTHS = frozenset(
    {
        "DFTU_T_I8",
        "DFTU_T_I16",
        "DFTU_T_I32",
        "DFTU_T_I64",
        "DFTU_T_U8",
        "DFTU_T_U16",
        "DFTU_T_U32",
        "DFTU_T_U64",
        "DFTU_T_F32",
        "DFTU_T_F64",
    }
)

_FLOAT_TYPES = frozenset({"DFTU_T_F32", "DFTU_T_F64"})


def _minmax_elem(op: str, of: "_Type[object]") -> str:
    if of.dft not in _MINMAX_WIDTHS:
        raise JitError(f"jit.{op}(of=...) must be a fixed-width int or float type")
    return "f64" if of.dft in _FLOAT_TYPES else "i64"


@overload
def min() -> "Min[int]": ...
@overload
def min(of: "_Type[T]") -> "Min[T]": ...
def min(of: "_Type[object]" = u64) -> "Min[object]":
    """A typed min value (default u64); ``of=`` picks the int or float element.

    ``.observe(v)`` contributes."""
    return Min("DFTU_AGG_MIN", elem=_minmax_elem("min", of))


@overload
def max() -> "Max[int]": ...
@overload
def max(of: "_Type[T]") -> "Max[T]": ...
def max(of: "_Type[object]" = u64) -> "Max[object]":
    """A typed max value (default u64); ``of=`` picks the int or float element.

    ``.observe(v)`` contributes."""
    return Max("DFTU_AGG_MAX", elem=_minmax_elem("max", of))


def minf() -> "Min[float]":
    """A double min value; contribute with ``.observe(v)``."""
    return Min("DFTU_AGG_MIN", elem="f64")


def maxf() -> "Max[float]":
    """A double max value; contribute with ``.observe(v)``."""
    return Max("DFTU_AGG_MAX", elem="f64")


def distinct() -> Distinct:
    """An approximate distinct count (u64); contribute with ``.observe(v)``."""
    return Distinct("DFTU_AGG_DISTINCT", elem="i64")


def bitset() -> _Monoid:
    """The bitwise OR of the observed integers; contribute with ``.observe(v)``."""
    return _Monoid("DFTU_AGG_BIT_OR", elem="i64")


def _elem_of(fn: str, of: "_Type[object]") -> str:
    if of is i64:
        return "i64"
    if of is str_:
        return "str"
    raise JitError(f"jit.{fn}(of=...) must be jit.str_ or jit.i64")


@overload
def set() -> "SetV[Str]": ...
@overload
def set(of: "_Type[T]") -> "SetV[T]": ...
def set(of: "_Type[object]" = str_) -> "SetV[object]":
    """A set value collecting distinct elements added with ``.observe``.

    ``of=jit.str_`` (default) observes strings, ``of=jit.i64`` int64 values.
    The result is one String cell per group: the distinct values as reprs,
    sorted and joined by ``\\x1e``. Use ``jit.list`` for a list column."""
    return SetV("DFTU_AGG_SET_UNION", elem=_elem_of("set", of))


@overload
def list() -> "ListV[Str]": ...
@overload
def list(of: "_Type[T]") -> "ListV[T]": ...
def list(of: "_Type[object]" = str_) -> "ListV[object]":
    """An ordered-list value collecting elements added with ``.append``.

    ``of=jit.str_`` (default) collects strings, ``of=jit.i64`` int64 values;
    both materialize as a list<string> of reprs sorted by ``order_by``."""
    return ListV("DFTU_AGG_LIST_SORTED", needs_by=True, elem=_elem_of("list", of))


def mean() -> Mean:
    """A mean value; contribute one value with ``.observe(v)``."""
    return Mean("DFTU_AGG_MEAN", elem="f64")


def variance() -> Variance:
    """A sample (n-1) variance value; contribute with ``.observe(v)``."""
    return Variance("DFTU_AGG_VAR", elem="f64")


var = variance


def stddev() -> Stddev:
    """A sample (n-1) standard deviation value; contribute with ``.observe(v)``."""
    return Stddev("DFTU_AGG_STD", elem="f64")


std = stddev


def quantiles(qs: "Tuple[float, ...]" = (0.5, 0.9, 0.95, 0.99)) -> Quantiles:
    """A DDSketch quantile value; contribute with ``.observe(v)``.

    Materializes a ``count`` int64 column plus one f64 column per quantile in
    ``qs`` (each in [0, 1]), named ``p<q*100>`` (``p50``, ``p90``, ``p99``, ..).
    Quantiles are approximate (DDSketch, ~1% relative error). Top-level only:
    it cannot be a product/record component."""
    qs = tuple(float(q) for q in qs)
    if not qs:
        raise JitError("jit.quantiles needs at least one quantile")
    for q in qs:
        if not 0.0 <= q <= 1.0:
            raise JitError(f"jit.quantiles value {q} is outside [0, 1]")
    return Quantiles(qs)


def skew() -> AggReduce:
    """Population skewness of the observed values."""
    return AggReduce("DFTU_AGG_SKEW", elem="f64")


def kurt() -> AggReduce:
    """Excess (population) kurtosis of the observed values."""
    return AggReduce("DFTU_AGG_KURT", elem="f64")


def sumsq() -> AggReduce:
    """Sum of squares (Float64)."""
    return AggReduce("DFTU_AGG_SUMSQ", elem="f64")


def first() -> AggReduce:
    """First non-null value in row order."""
    return AggReduce("DFTU_AGG_FIRST", elem="f64")


def last() -> AggReduce:
    """Last non-null value in row order."""
    return AggReduce("DFTU_AGG_LAST", elem="f64")


def count_valid() -> AggReduce:
    """Count of non-null observed values."""
    return AggReduce("DFTU_AGG_COUNT_VALID", elem="f64")


def hist() -> AggReduce:
    """The DDSketch histogram of the observed values, a
    list<struct{lo, hi, count}> column (mergeable, relative-error buckets)."""
    return AggReduce("DFTU_AGG_HIST", elem="f64")


def busy(cell: float = 0.0) -> AggReduce:
    """Occupancy: interval-union length (us) where overlap depth > 0, over a
    (ts, dur) pair. ``cell`` is the endpoint-snap tolerance in us (0 = exact)."""
    return AggReduce("DFTU_AGG_BUSY", param=float(cell), needs_by=True, elem="f64")


def concurrency(cell: float = 0.0) -> AggReduce:
    """Occupancy: sum(dur) / busy over a (ts, dur) pair."""
    return AggReduce("DFTU_AGG_CONCURRENCY", param=float(cell), needs_by=True, elem="f64")


def utilization(cell: float = 0.0) -> AggReduce:
    """Occupancy: busy / (max_end - min_ts) over a (ts, dur) pair."""
    return AggReduce("DFTU_AGG_UTILIZATION", param=float(cell), needs_by=True, elem="f64")


def active(cell: float = 0.0) -> AggReduce:
    """Occupancy: peak overlap depth over a (ts, dur) pair."""
    return AggReduce("DFTU_AGG_ACTIVE", param=float(cell), needs_by=True, elem="f64")


def corr() -> AggReduce:
    """Pearson correlation of the (y, x) pair; y is the value, x the ``by``."""
    return AggReduce("DFTU_AGG_CORR", needs_by=True, elem="f64")


def covar_pop() -> AggReduce:
    """Population covariance of the (y, x) pair."""
    return AggReduce("DFTU_AGG_COVAR_POP", needs_by=True, elem="f64")


def covar_samp() -> AggReduce:
    """Sample covariance of the (y, x) pair."""
    return AggReduce("DFTU_AGG_COVAR_SAMP", needs_by=True, elem="f64")


def regr_slope() -> AggReduce:
    """Least-squares slope of y on x."""
    return AggReduce("DFTU_AGG_REGR_SLOPE", needs_by=True, elem="f64")


def regr_intercept() -> AggReduce:
    """Least-squares intercept of y on x."""
    return AggReduce("DFTU_AGG_REGR_INTERCEPT", needs_by=True, elem="f64")


def regr_r2() -> AggReduce:
    """Coefficient of determination of the least-squares fit of y on x."""
    return AggReduce("DFTU_AGG_REGR_R2", needs_by=True, elem="f64")


@overload
def argmin() -> "ArgMin[Str]": ...
@overload
def argmin(of: "_Type[T]") -> "ArgMin[T]": ...
def argmin(of: "_Type[object]" = str_) -> "ArgMin[object]":
    """An argmin value: keep the payload whose ``by=`` key is smallest.

    ``of=jit.str_`` (default) observes a string payload, ``of=jit.i64`` an
    int64 one; the result is the payload's String repr. Contribute with
    ``.observe(payload, by=<expr>)``."""
    return ArgMin("DFTU_AGG_ARGMIN", needs_by=True, elem=_elem_of("argmin", of))


@overload
def argmax() -> "ArgMax[Str]": ...
@overload
def argmax(of: "_Type[T]") -> "ArgMax[T]": ...
def argmax(of: "_Type[object]" = str_) -> "ArgMax[object]":
    """An argmax value: keep the payload whose ``by=`` key is largest.

    ``of=jit.str_`` (default) observes a string payload, ``of=jit.i64`` an
    int64 one; the result is the payload's String repr. Contribute with
    ``.observe(payload, by=<expr>)``."""
    return ArgMax("DFTU_AGG_ARGMAX", needs_by=True, elem=_elem_of("argmax", of))


def _check_k(op: str, k: int) -> float:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise JitError(f"jit.{op}(k, ...) k must be a positive integer")
    return float(k)


@overload
def topk(k: int) -> "TopK[Str]": ...
@overload
def topk(k: int, of: "_Type[T]") -> "TopK[T]": ...
def topk(k: int, of: "_Type[object]" = str_) -> "TopK[object]":
    """A bounded top-k value: keep the k payloads at the k largest ``by=`` keys.

    ``of=jit.str_`` (default) observes string payloads, ``of=jit.i64`` int64
    ones; the result is a list<string> of their reprs. Contribute with
    ``.observe(payload, by=<expr>)``."""
    return TopK(
        "DFTU_AGG_TOPK", param=_check_k("topk", k), needs_by=True, elem=_elem_of("topk", of)
    )


@overload
def bottomk(k: int) -> "BottomK[Str]": ...
@overload
def bottomk(k: int, of: "_Type[T]") -> "BottomK[T]": ...
def bottomk(k: int, of: "_Type[object]" = str_) -> "BottomK[object]":
    """A bounded bottom-k value: keep the k payloads at the k smallest ``by=``
    keys, mirroring :func:`topk`."""
    return BottomK(
        "DFTU_AGG_BOTTOMK",
        param=_check_k("bottomk", k),
        needs_by=True,
        elem=_elem_of("bottomk", of),
    )


@overload
def approx_topk(k: int) -> "ApproxTopK[Str]": ...
@overload
def approx_topk(k: int, of: "_Type[T]") -> "ApproxTopK[T]": ...
def approx_topk(k: int, of: "_Type[object]" = str_) -> "ApproxTopK[object]":
    """An approximate heavy-hitters value: the k most FREQUENT observed values
    (SpaceSaving), in bounded memory. Contribute with ``.observe(value)``."""
    return ApproxTopK(
        "DFTU_AGG_APPROX_TOPK",
        param=_check_k("approx_topk", k),
        elem=_elem_of("approx_topk", of),
    )


@overload
def sample(k: int) -> "Sample[Str]": ...
@overload
def sample(k: int, of: "_Type[T]") -> "Sample[T]": ...
def sample(k: int, of: "_Type[object]" = str_) -> "Sample[object]":
    """A deterministic mergeable sample of k items, a list<string> of their
    reprs; contribute with ``.observe(item)``."""
    return Sample("DFTU_AGG_SAMPLE", param=_check_k("sample", k), elem=_elem_of("sample", of))


class Map(Generic[K, V]):
    __slots__ = ("key_types", "values", "is_product", "value_names")

    def __init__(
        self,
        key_types: "Tuple[_Type[object], ...]",
        values: Tuple[_Monoid, ...],
        is_product: bool,
        value_names: Tuple[str, ...] | None = None,
    ) -> None:
        self.key_types = key_types
        self.values = values
        self.is_product = is_product
        self.value_names = value_names

    @overload
    def __getitem__(self, key: K) -> V: ...
    @overload
    def __getitem__(self, key: object) -> V: ...
    def __getitem__(self, key: object) -> V:
        raise NotImplementedError

    @overload
    def __setitem__(self, key: K, value: V) -> None: ...
    @overload
    def __setitem__(self, key: object, value: V) -> None: ...
    def __setitem__(self, key: object, value: V) -> None: ...


_MapDecl = Map


class Event(Protocol):
    """Typed per-event surface for a ``@jit.each_event`` method. Authoring-only:
    the method body is AST-compiled, never executed, so nothing is instantiated."""

    pid: int
    tid: int
    ts: int
    dur: int
    phase: int
    has_dur: bool
    cat: Str
    name: Str
    fhash: Str
    hhash: Str

    def arg_i64(self, name: str) -> int: ...
    def arg_f64(self, name: str) -> float: ...
    def arg_str(self, name: str) -> Str: ...


class _Record:
    __slots__ = ("names", "monoids")

    def __init__(self, names: Tuple[str, ...], monoids: Tuple[_Monoid, ...]) -> None:
        self.names = names
        self.monoids = monoids


def _as_monoid(v: object) -> _Monoid:
    if isinstance(v, _Monoid):
        return v
    if callable(v):
        r = cast(Callable[[], object], v)()
        if isinstance(r, _Monoid):
            return r
    raise JitError("record field must be a jit monoid such as jit.count or jit.sum")


def record(cls: type) -> type:
    """Class decorator: turn monoid-annotated fields into a named product value.

    Each annotation (``count: jit.count``) declares one value component in
    declaration order; pass the class as a :func:`map` ``value``."""
    anns = dict(getattr(cls, "__annotations__", {}))
    if not anns:
        raise JitError("@jit.record needs at least one monoid-annotated field")
    names = tuple(anns.keys())
    monoids = tuple(_as_monoid(v) for v in anns.values())
    setattr(cls, "_jit_record", _Record(names, monoids))
    return cls


def _value_spec(value: object) -> Tuple[Tuple[_Monoid, ...], bool, Tuple[str, ...] | None]:
    if isinstance(value, _Monoid):
        return (value,), False, None
    rec = getattr(value, "_jit_record", None)
    if isinstance(rec, _Record):
        return rec.monoids, True, rec.names
    if isinstance(value, dict):
        if not value or not all(isinstance(v, _Monoid) for v in value.values()):
            raise JitError("map value dict must map names to monoids such as jit.count()")
        return tuple(value.values()), True, tuple(value.keys())
    if isinstance(value, tuple) and value and all(isinstance(v, _Monoid) for v in value):
        return value, True, None
    raise JitError("map value must be a monoid, a tuple/dict of monoids, or a @jit.record")


# A dict/tuple of monoids is a product (value type Product); these precede the
# generic `value: V` forms so it is not typed as a bare dict/tuple.
@overload
def map(*, key: "_Type[K1]", value: "Dict[str, _Monoid]") -> "Map[Tuple[K1], Product]": ...
@overload
def map(*, key: "_Type[K1]", value: "Tuple[_Monoid, ...]") -> "Map[Tuple[K1], Product]": ...
@overload
def map(*, key: "Tuple[_Type[K1]]", value: "Dict[str, _Monoid]") -> "Map[Tuple[K1], Product]": ...
@overload
def map(*, key: "Tuple[_Type[K1]]", value: "Tuple[_Monoid, ...]") -> "Map[Tuple[K1], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2]]", value: "Dict[str, _Monoid]"
) -> "Map[Tuple[K1, K2], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2]]", value: "Tuple[_Monoid, ...]"
) -> "Map[Tuple[K1, K2], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]", value: "Dict[str, _Monoid]"
) -> "Map[Tuple[K1, K2, K3], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]", value: "Tuple[_Monoid, ...]"
) -> "Map[Tuple[K1, K2, K3], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]", value: "Dict[str, _Monoid]"
) -> "Map[Tuple[K1, K2, K3, K4], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]", value: "Tuple[_Monoid, ...]"
) -> "Map[Tuple[K1, K2, K3, K4], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[object], ...]", value: "Dict[str, _Monoid]"
) -> "Map[Tuple[object, ...], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[object], ...]", value: "Tuple[_Monoid, ...]"
) -> "Map[Tuple[object, ...], Product]": ...
@overload
def map(*, key: "_Type[K1]", value: V) -> "Map[Tuple[K1], V]": ...
@overload
def map(*, key: "Tuple[_Type[K1]]", value: V) -> "Map[Tuple[K1], V]": ...
@overload
def map(*, key: "Tuple[_Type[K1], _Type[K2]]", value: V) -> "Map[Tuple[K1, K2], V]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]", value: V
) -> "Map[Tuple[K1, K2, K3], V]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]", value: V
) -> "Map[Tuple[K1, K2, K3, K4], V]": ...
@overload
def map(*, key: "Tuple[_Type[object], ...]", value: V) -> "Map[Tuple[object, ...], V]": ...
def map(key: "Tuple[_Type[object], ...] | _Type[object]", value: object) -> "Map[object, object]":
    """Declare a keyed accumulator: a typed-tuple key, and either one value
    reduction, a tuple of reductions (a positional product), a dict of
    name->reduction, or a :func:`record` class (both named products).

    It lowers to the engine's ``AggState`` grouped by the key columns, one
    aggregate per value reduction; the host merges same-named accumulators
    across worker slices and finalizes each to a native DataFrame.

    A single-component key may be passed bare (``key=jit.i64``); it normalizes to
    the 1-tuple ``(jit.i64,)`` and the body may then use a bare subscript
    ``self.m[k]`` as well as ``self.m[(k,)]``. Multi-key maps stay tuple-only.
    An empty key tuple declares a scalar accumulator: one whole-scan row."""
    if isinstance(key, _Type):
        key = (key,)
    if not isinstance(key, tuple):
        raise JitError("map key must be a tuple of jit.i64 / jit.str_")
    for k in key:
        if not isinstance(k, _Type):
            raise JitError("map key components must be jit.i64 or jit.str_")
        if k.dft == "DFTU_T_BYTES":
            raise JitError(
                "a bytes map key has no column form; group by jit.str_ or jit.i64 instead"
            )
    values, is_product, value_names = _value_spec(value)
    if is_product and any(isinstance(v, Quantiles) for v in values):
        raise JitError("jit.quantiles is top-level only; it cannot be a product/record component")
    return _MapDecl(key, values, is_product, value_names)


class Port(Protocol):
    """Authoring-only handle for a batch-scoped inter-plugin port.

    A :func:`publish` port is written per event with ``self.<port> += <expr>`` and
    the host publishes the per-batch total for a later plugin's consume port of the
    same capability id; a :func:`consume` port reads that value as a plain scalar
    (``self.<port>``, 0 when no producer published this batch). Never instantiated."""

    def __iadd__(self, x: float) -> "Port": ...


_PORT_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


class _Port:
    __slots__ = ("is_f64", "name", "role")

    def __init__(self, name: str, is_f64: bool, role: str) -> None:
        self.name = name
        self.is_f64 = is_f64
        self.role = role


def _check_port_name(fn: str, name: object) -> str:
    if not isinstance(name, str) or not name:
        raise JitError(f"jit.{fn} needs a non-empty port name such as 'com.example.edges'")
    if name.startswith("dftu."):
        raise JitError(
            f"jit.{fn} port name '{name}' uses the reserved dftu. namespace; pick your own, "
            "e.g. 'com.example.edges'"
        )
    if any(c not in _PORT_ID_CHARS for c in name):
        raise JitError(f"jit.{fn} port name '{name}' must be ASCII [a-z0-9._-]")
    return name


def _port_is_f64(fn: str, of: "_Type[object]") -> bool:
    if of is f64:
        return True
    if of is u64 or of is i64:
        return False
    raise JitError(f"jit.{fn}(of=...) must be jit.u64, jit.i64, or jit.f64")


def publish(name: str, of: "_Type[object]" = u64) -> Port:
    """Declare a batch-scoped publish port called ``name``.

    A consumer wires to it by naming the same port. In ``each_event`` accumulate
    a per-batch total with ``self.<port> += <expr>`` (a u64 sum by default,
    ``of=jit.f64`` for a double sum). The total is published once per batch and
    reset for the next batch; ``of`` picks the wire width (u64/i64 as an 8-byte
    int, f64 as a double)."""
    _check_port_name("publish", name)
    return cast(Port, _Port(name, _port_is_f64("publish", of), "publish"))


def consume(name: str, of: "_Type[object]" = u64) -> Port:
    """Declare a batch-scoped consume port called ``name``.

    In ``each_event`` read the value a producer published for the current batch
    as a plain scalar ``self.<port>`` (0 when no producer published this batch).
    ``of`` must match the producer's width (u64/i64/f64). The producer must be
    registered before this plugin."""
    _check_port_name("consume", name)
    return cast(Port, _Port(name, _port_is_f64("consume", of), "consume"))


class _EachEvent:
    __slots__ = ("fn", "raw")

    def __init__(self, fn: Callable[..., object], raw: bool = False) -> None:
        self.fn = fn
        self.raw = raw


def each_event(
    fn: Callable[..., object] | None = None, *, raw: bool = False
) -> _EachEvent | Callable[[Callable[..., object]], _EachEvent]:
    """Mark the one per-event method to AST-compile.

    With ``raw=True`` the method instead returns a C++ string spliced verbatim
    into the per-event loop; available C names are ``host``, ``e``, ``b``, ``i``,
    and, for each declared accumulator ``m``, its row buffers ``_n_m``,
    ``_cap_m``, ``_k<i>_m``, ``_v<c>_m`` (plus ``_o<c>_m`` for a product and
    ``_b<c>_m`` for a by-reading reduction). A raw body appends at most
    ``_RAW_ROWS_PER_EVENT`` rows per event per accumulator."""
    if fn is None:
        return lambda f: _EachEvent(f, raw)
    return _EachEvent(fn, raw)


class _EachBatch:
    __slots__ = ("fn",)

    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn


def each_batch(fn: Callable[..., object]) -> _EachBatch:
    """Mark the one per-batch method of a :func:`vfold`.

    It takes ``(self, df)`` where ``df`` is the current scan batch as columns.
    Its body folds column reductions into the class's scalar accumulators, e.g.
    ``self.total += df["dur"].sum()``."""
    return _EachBatch(fn)


class JitPlugin:
    __slots__ = ("name", "source", "renames")

    def __init__(self, name: str, source: str, renames: Dict[str, List[str]]) -> None:
        self.name = name
        self.source = source
        self.renames = renames


def _c_str_literal(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


_FIELD_EXPR: Dict[str, str] = {
    "pid": "e->pid",
    "tid": "e->tid",
    "ts": "e->ts",
    "dur": "e->dur",
    "cat": "e->cat",
    "name": "e->name",
    "fhash": "e->fhash",
    "hhash": "e->hhash",
    "phase": "e->phase",
    "has_dur": "e->has_dur",
}

_FIELD_NEED: Dict[str, str] = {
    "fhash": "DFTU_NEED_FHASH",
    "hhash": "DFTU_NEED_HHASH",
}

_ARG_METHODS: Dict[str, Tuple[str, str]] = {
    "arg_i64": ("dftu_jit_arg_i64", "0"),
    "arg_f64": ("dftu_jit_arg_f64", "0.0"),
    "arg_str": ("dftu_jit_arg_str", "DFTU_STR_NONE"),
}

_ARG_HELPER_DEFS: Dict[str, List[str]] = {
    "arg_i64": [
        "static int64_t dftu_jit_arg_i64(const dftu_event* e, dftu_str k, int64_t d) {",
        "    for (uint32_t i = 0; i < e->arg_count; ++i)",
        "        if (e->args[i].key == k && e->args[i].kind == DFTU_ARG_I64)",
        "            return e->args[i].v.i64;",
        "    return d;",
        "}",
        "",
    ],
    "arg_f64": [
        "static double dftu_jit_arg_f64(const dftu_event* e, dftu_str k, double d) {",
        "    for (uint32_t i = 0; i < e->arg_count; ++i)",
        "        if (e->args[i].key == k && e->args[i].kind == DFTU_ARG_F64)",
        "            return e->args[i].v.f64;",
        "    return d;",
        "}",
        "",
    ],
    "arg_str": [
        "static dftu_str dftu_jit_arg_str(const dftu_event* e, dftu_str k, dftu_str d) {",
        "    for (uint32_t i = 0; i < e->arg_count; ++i)",
        "        if (e->args[i].key == k && e->args[i].kind == DFTU_ARG_STR)",
        "            return e->args[i].v.str;",
        "    return d;",
        "}",
        "",
    ],
}

_STR_FIELDS = frozenset({"cat", "name", "fhash", "hhash"})

# Reductions contributed with `+=` (everything else takes .observe / .append).
_ADDITIVE_OPS = frozenset({"DFTU_AGG_SUM"})

_ARGBY_OPS = frozenset({"DFTU_AGG_ARGMIN", "DFTU_AGG_ARGMAX"})

_TOPK_OPS = frozenset({"DFTU_AGG_TOPK", "DFTU_AGG_BOTTOMK"})

_SET_OPS = frozenset({"DFTU_AGG_SET_UNION"})

_LIST_OPS = frozenset({"DFTU_AGG_LIST_SORTED"})

# The C buffer element type and the batch column dtype for each value family.
_ELEM_CTYPE = {"i64": "int64_t", "f64": "double", "str": "dftu_str"}
_ELEM_DTYPE = {"i64": "DFTU_TYPE_INT64", "f64": "DFTU_TYPE_FLOAT64", "str": "DFTU_TYPE_STRING"}

# Key component type -> (C buffer element type, batch column dtype).
_KEY_CTYPE = {
    "DFTU_T_STR": ("dftu_str", "DFTU_TYPE_STRING"),
    "DFTU_T_F32": ("double", "DFTU_TYPE_FLOAT64"),
    "DFTU_T_F64": ("double", "DFTU_TYPE_FLOAT64"),
}

_CMP_OP: Dict[type, str] = {
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
}

_ARITH_OP: Dict[type, str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
}

# Numeric primitives usable in a key or value expression; each lowers to the
# matching static-inline helper in prims.h so a body reads jit.ilog2(e.dur)
# instead of a raw compiler builtin. Value is (c_function, arity, arg_cast); all
# return an integer.
_PRIM_C: Dict[str, Tuple[str, int, str]] = {
    # bit / log-scale bucketing (unsigned)
    "ilog2": ("dftu_ilog2_u64", 1, "uint64_t"),
    "clz": ("dftu_clz_u64", 1, "uint64_t"),
    "ctz": ("dftu_ctz_u64", 1, "uint64_t"),
    "popcount": ("dftu_popcount_u64", 1, "uint64_t"),
    "bit_width": ("dftu_bit_width_u64", 1, "uint64_t"),
    "ceil_pow2": ("dftu_ceil_pow2_u64", 1, "uint64_t"),
    "floor_pow2": ("dftu_floor_pow2_u64", 1, "uint64_t"),
    "rotl": ("dftu_rotl_u64", 2, "uint64_t"),
    "rotr": ("dftu_rotr_u64", 2, "uint64_t"),
    # integer math
    "abs": ("dftu_abs_i64", 1, "int64_t"),
    "clamp": ("dftu_clamp_i64", 3, "int64_t"),
    "div_ceil": ("dftu_div_ceil_i64", 2, "int64_t"),
    "div_round": ("dftu_div_round_i64", 2, "int64_t"),
    "align_up": ("dftu_align_up_u64", 2, "uint64_t"),
    "align_down": ("dftu_align_down_u64", 2, "uint64_t"),
    "isqrt": ("dftu_isqrt_u64", 1, "uint64_t"),
    "gcd": ("dftu_gcd_u64", 2, "uint64_t"),
    # fast integer hashing
    "mul_hi": ("dftu_mul_hi_u64", 2, "uint64_t"),
    "mix64": ("dftu_mix64", 1, "uint64_t"),
    "fastrange": ("dftu_fastrange_u64", 2, "uint64_t"),
    # floating point (double)
    "fmin": ("dftu_min_f64", 2, "double"),
    "fmax": ("dftu_max_f64", 2, "double"),
    "clampf": ("dftu_clamp_f64", 3, "double"),
    "sqrt": ("dftu_sqrt_f64", 1, "double"),
    "log2": ("dftu_log2_f64", 1, "double"),
    "log": ("dftu_log_f64", 1, "double"),
    "exp": ("dftu_exp_f64", 1, "double"),
    "fma": ("dftu_fma_f64", 3, "double"),
    "lerp": ("dftu_lerp_f64", 3, "double"),
    "copysign": ("dftu_copysign_f64", 2, "double"),
}

_CONSTRUCT = {
    ast.For: "for loop",
    ast.While: "while loop",
    ast.With: "with statement",
    ast.Call: "function call",
    ast.BinOp: "arithmetic expression",
    ast.BoolOp: "boolean expression",
    ast.Assign: "local assignment",
    ast.Return: "return statement",
    ast.IfExp: "conditional expression",
}


def _describe(node: ast.AST) -> str:
    return _CONSTRUCT.get(type(node), type(node).__name__)


def _reject(what: str) -> NoReturn:
    raise JitError(f"unsupported in @jit.each_event: {what}; use a raw C++ plugin")


_I64_MIN = -(1 << 63)
_I64_MAX = (1 << 63) - 1
_U64_MAX = (1 << 64) - 1


def _int_literal(v: int) -> str:
    """C text for an integer literal, rejecting one that would not fit in the
    64-bit key/value slot (Python ints are unbounded; the compiled body is not,
    so an out-of-range literal would silently wrap)."""
    if v < _I64_MIN or v > _U64_MAX:
        raise JitError(
            f"integer literal {v} does not fit in 64 bits; jit keys and values "
            "are 64-bit and it would silently wrap"
        )
    return f"{v}ULL" if v > _I64_MAX else str(v)


def _key_ctype(ktype: str) -> Tuple[str, str]:
    """(C buffer element type, batch column dtype) for a key component type."""
    return _KEY_CTYPE.get(ktype, ("int64_t", "DFTU_TYPE_INT64"))


def _out_name(decl: _MapDecl, comp: int) -> str:
    """Result column name for value component ``comp``."""
    if not decl.is_product:
        return "value"
    if decl.value_names is not None:
        return decl.value_names[comp]
    return f"v{comp}"


class _Compiler:
    def __init__(
        self,
        maps: Dict[str, _MapDecl],
        ops: Dict[str, Op] | None = None,
        ports: Dict[str, _Port] | None = None,
        config_fields: "Dict[str, bool] | None" = None,
    ) -> None:
        self.maps = maps
        self.ports = ports if ports is not None else {}
        # Config fields {name: is_f64}; each_event reads them as self.<name>,
        # lowering to the file-scope _cfg_<name> static set at load.
        self.config_fields = config_fields if config_fields is not None else {}
        # Referenced @jit.op transforms, inlined into the plugin as static C
        # functions the moment the body calls one.
        self.ops = ops if ops is not None else {}
        self.op_defs: List[str] = []
        self.op_emitted: builtins.set[str] = builtins.set()
        self.needs: builtins.set[str] = builtins.set()
        self.str_literals: List[str] = []
        self.arg_keys: List[str] = []
        self.arg_helpers: builtins.set[str] = builtins.set()
        self.self_name = "self"
        self.event_name = "e"
        # Contribution statements per accumulator; the emitted row buffers hold
        # this many rows per scanned event.
        self.rows_per_event: Dict[str, int] = {}
        # CSE: ast.dump(node) -> (local name, is_float) for a hoisted expression
        # shared within the current scope.
        self._cse: Dict[str, Tuple[str, bool]] = {}
        self._cse_n = 0

    def lower(self, fn: Callable[..., object]) -> List[str]:
        src = textwrap.dedent(inspect.getsource(fn))
        mod = ast.parse(src)
        funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
        if len(funcs) != 1:
            _reject("each_event must decorate a single method")
        fn_ast = funcs[0]
        params = fn_ast.args.args
        if len(params) != 2:
            _reject("each_event must take exactly (self, e)")
        self.self_name = params[0].arg
        self.event_name = params[1].arg
        defs, lines = self._compile_scope(fn_ast.body)
        return defs + lines

    def _slice_elts(self, sl: ast.expr) -> List[ast.expr]:
        sl = _unwrap_index(sl)
        return builtins.list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]

    # Hoist each expression used more than once in this straight-line scope into
    # a local computed once, then compile the scope's statements against those
    # locals. Nested guard bodies are their own scopes (a guarded expression is
    # not hoisted above its guard, preserving its execution condition).
    def _compile_scope(self, stmts: List[ast.stmt]) -> Tuple[List[str], List[str]]:
        outer = self._cse
        self._cse = dict(outer)
        counts: Dict[str, int] = {}
        reps: Dict[str, ast.expr] = {}
        for stmt in stmts:
            roots = [stmt.test] if isinstance(stmt, ast.If) else [stmt]
            for root in roots:
                for node in ast.walk(root):
                    if isinstance(node, ast.expr) and self._is_cse_candidate(node):
                        d = ast.dump(node)
                        counts[d] = counts.get(d, 0) + 1
                        reps.setdefault(d, node)
        cands = [reps[d] for d, c in counts.items() if c >= 2 and d not in self._cse]
        # Define inner (smaller) expressions first so an outer local can name
        # them.
        order = sorted(range(len(cands)), key=lambda i: builtins.sum(1 for _ in ast.walk(cands[i])))
        shared = [cands[i] for i in order]
        defs: List[str] = []
        for node in shared:
            d = ast.dump(node)
            if d in self._cse:
                continue
            expr, is_float = self._arith(node)
            name = f"_cse{self._cse_n}"
            self._cse_n += 1
            defs.append(f"{'double' if is_float else 'int64_t'} {name} = ({expr});")
            self._cse[d] = (name, is_float)
        lines: List[str] = []
        for stmt in stmts:
            lines.extend(self._stmt(stmt))
        self._cse = outer
        return defs, lines

    def _is_cse_candidate(self, node: ast.expr) -> bool:
        if isinstance(node, ast.BinOp) and type(node.op) in _ARITH_OP:
            return True
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            f = node.func
            if isinstance(f.value, ast.Name):
                if f.value.id == "jit" and f.attr in _PRIM_C:
                    return True
                if f.value.id == self.event_name and f.attr in _ARG_METHODS:
                    return True
        return False

    def _stmt(self, stmt: ast.stmt) -> List[str]:
        if isinstance(stmt, ast.If):
            if stmt.orelse:
                _reject("else / elif branch")
            guard = self._compare(stmt.test)
            defs, body = self._compile_scope(stmt.body)
            lines = [f"if ({guard}) {{"]
            lines.extend("    " + ln for ln in defs)
            lines.extend("    " + ln for ln in body)
            lines.append("}")
            return lines
        return self._add_stmt(stmt)

    def _add_stmt(self, stmt: ast.stmt) -> List[str]:
        if isinstance(stmt, ast.AugAssign):
            return self._aug(stmt)
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
            func = stmt.value.func
            if isinstance(func, ast.Attribute) and func.attr == "append":
                return self._append(stmt.value)
            return self._observe(stmt.value)
        _reject(_describe(stmt))

    def _compare(self, node: ast.expr) -> str:
        if not isinstance(node, ast.Compare) or len(node.ops) != 1:
            _reject("guard must be a single comparison on e.<field>")
        op_t = type(node.ops[0])
        op = _CMP_OP.get(op_t)
        if op is None:
            _reject("comparison operator")
        str_cmp = self._str_compare(node.left, node.comparators[0], op_t)
        if str_cmp is not None:
            return str_cmp
        left = self._value(node.left)
        right = self._value(node.comparators[0])
        return f"{left} {op} {right}"

    def _str_compare(self, a: ast.expr, b: ast.expr, op_t: type) -> str | None:
        a_lit = isinstance(a, ast.Constant) and isinstance(a.value, str)
        b_lit = isinstance(b, ast.Constant) and isinstance(b.value, str)
        if not a_lit and not b_lit:
            return None
        if op_t not in (ast.Eq, ast.NotEq):
            _reject("string comparison operator other than == / !=")
        lit_node, field_node = (a, b) if a_lit else (b, a)
        operand = self._str_operand(field_node)
        if operand is None:
            _reject("string literal compared to a non-string e.<field>")
        assert isinstance(lit_node, ast.Constant) and isinstance(lit_node.value, str)
        cached = self._str_literal(lit_node.value)
        op = _CMP_OP[op_t]
        return f"{operand} {op} {cached}"

    def _str_operand(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.event_name
            and node.attr in _STR_FIELDS
        ):
            return self._field(node.attr)
        arg = self._arg_call(node)
        if arg is not None and arg[0] == "arg_str":
            expr, _ = self._lower_arg(*arg)
            return expr
        return None

    def _arg_call(self, node: ast.expr) -> Tuple[str, str] | None:
        """(method, arg-name) for an ``e.arg_*("NAME")`` call; None if not an event
        method call, JitError for an unknown method or a non-literal name."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == self.event_name
        ):
            return None
        method = node.func.attr
        if method not in _ARG_METHODS:
            _reject(f"event method e.{method}(...); use arg_i64 / arg_f64 / arg_str")
        if len(node.args) != 1 or node.keywords:
            _reject(f"e.{method} takes exactly one string-literal arg name")
        name_node = node.args[0]
        if not (isinstance(name_node, ast.Constant) and isinstance(name_node.value, str)):
            _reject(f"e.{method} arg name must be a string literal")
        return method, name_node.value

    def _prim_call(self, node: ast.expr) -> Tuple[str, str, List[ast.expr]] | None:
        """(c-function, arg-cast, arg-nodes) for a ``jit.<prim>(...)`` numeric
        primitive; None when not a ``jit.`` call at all, JitError for a bad
        prim or arity."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "jit"
        ):
            return None
        prim = node.func.attr
        spec = _PRIM_C.get(prim)
        if spec is None:
            _reject(f"jit.{prim}(...) in an expression; expected a numeric primitive")
        c_fn, arity, cast = spec
        if len(node.args) != arity or node.keywords:
            _reject(f"jit.{prim}() takes exactly {arity} argument(s)")
        return c_fn, cast, builtins.list(node.args)

    def _op_call(self, node: ast.expr) -> Tuple[str, bool] | None:
        """(c-expr, is_float) for a call to a referenced ``@jit.op`` transform,
        inlining its C body on first use; None when not an op call."""
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in self.ops
        ):
            return None
        name = node.func.id
        op_obj = self.ops[name]
        if len(node.args) != 1 or node.keywords:
            _reject(f"jit.op '{name}' takes exactly one positional argument")
        c_name = f"dftu_jitop_{name}"
        if name not in self.op_emitted:
            self.op_defs.append(jit_op.emit_inline(op_obj, c_name))
            self.op_emitted.add(name)
        arg_expr, _ = self._arith(node.args[0])
        in_ct = jit_op.c_type(op_obj.in_type)
        return f"{c_name}(({in_ct})({arg_expr}))", jit_op.is_float_type(op_obj.out_type)

    def _lower_arg(self, method: str, name: str) -> Tuple[str, bool]:
        """Lower ``e.arg_*("NAME")`` to a scan-helper call, returning (expr, is_float);
        sets DFTU_NEED_ARGS and interns NAME once."""
        self.needs.add("DFTU_NEED_ARGS")
        self.arg_helpers.add(method)
        fn, dflt = _ARG_METHODS[method]
        if name not in self.arg_keys:
            self.arg_keys.append(name)
        key = f"argkey_{self.arg_keys.index(name)}"
        return f"{fn}(e, {key}, {dflt})", method == "arg_f64"

    def _str_literal(self, s: str) -> str:
        if s not in self.str_literals:
            self.str_literals.append(s)
        return f"lit_{self.str_literals.index(s)}"

    def _resolve_accum(self, target: ast.expr) -> Tuple[str, List[ast.expr], int, _Monoid]:
        comp_name: str | None = None
        node = target
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Subscript):
            comp_name = node.attr
            node = node.value
        if not isinstance(node, ast.Subscript):
            _reject("assignment target (expected self.<map>[(...)])")
        subs: List[ast.expr] = []
        cur: ast.expr = node
        while isinstance(cur, ast.Subscript):
            subs.append(cur.slice)
            cur = cur.value
        subs.reverse()
        base = cur
        if not (
            isinstance(base, ast.Attribute)
            and isinstance(base.value, ast.Name)
            and base.value.id == self.self_name
        ):
            _reject("assignment target (expected self.<map>[(...)])")
        decl = self.maps.get(base.attr)
        if decl is None:
            _reject(f"unknown map self.{base.attr}")
        return self._resolve_flat(base.attr, decl, subs, comp_name)

    def _resolve_flat(
        self, attr: str, decl: _MapDecl, subs: List[ast.expr], comp_name: str | None
    ) -> Tuple[str, List[ast.expr], int, _Monoid]:
        if decl.is_product:
            keys = self._keys(subs[0])
            comp_node = subs[1] if len(subs) > 1 else None
            if len(subs) > 2:
                _reject(f"too many subscripts for product map self.{attr}")
            if comp_name is not None:
                comp = self._named_comp(comp_name, decl, attr)
            elif comp_node is not None:
                comp = self._comp(comp_node, decl, attr)
            else:
                _reject(f"product map self.{attr} add without a component")
        else:
            if comp_name is not None:
                _reject(f"single-value map self.{attr} add with a component")
            keys = []
            for sl in subs:
                keys.extend(self._keys(sl))
            comp = 0
        if len(keys) != len(decl.key_types):
            _reject(f"self.{attr} takes {len(decl.key_types)} key components, got {len(keys)}")
        return attr, keys, comp, decl.values[comp]

    def _publish_target(self, target: ast.expr) -> str | None:
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == self.self_name
            and target.attr in self.ports
        ):
            return target.attr
        return None

    def _publish(self, name: str, value: ast.expr) -> List[str]:
        port = self.ports[name]
        if port.role != "publish":
            _reject(f"self.{name} is a jit.consume port; read its value, do not += into it")
        expr, saw_float = self._arith(value)
        if not port.is_f64 and saw_float:
            _reject("float value into a u64 publish port")
        cast = "double" if port.is_f64 else "uint64_t"
        return [f"_pub_{name} += ({cast})({expr});"]

    def _consume_ref(self, name: str) -> str:
        port = self.ports[name]
        if port.role != "consume":
            _reject(f"self.{name} is a jit.publish port; publish into it with +=, do not read it")
        return f"_sub_{name}"

    def _aug(self, stmt: ast.AugAssign) -> List[str]:
        if not isinstance(stmt.op, ast.Add):
            _reject("augmented operator other than +=")
        pub = self._publish_target(stmt.target)
        if pub is not None:
            return self._publish(pub, stmt.value)
        attr, keys, comp, monoid = self._resolve_accum(stmt.target)
        if monoid.dft in _LIST_OPS:
            _reject("+= on an ordered-list monoid; use .append(elem, order_by=<expr>)")
        if monoid.dft not in _ADDITIVE_OPS:
            _reject(f"+= on the non-additive monoid {monoid.dft}; use .observe(v) instead")
        return self._emit_row(attr, keys, comp, monoid, stmt.value, None)

    def _observe(self, call: ast.Call) -> List[str]:
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "observe"):
            _reject("method call other than .observe(v)")
        attr, keys, comp, monoid = self._resolve_accum(func.value)
        if monoid.dft in _LIST_OPS:
            _reject(".observe on an ordered-list monoid; use .append(elem, order_by=<expr>)")
        by: ast.expr | None = None
        if monoid.needs_by:
            if len(call.args) != 1:
                _reject(".observe takes one payload and by=<expr>")
            for kw in call.keywords:
                if kw.arg == "by":
                    by = kw.value
                else:
                    _reject(f".observe keyword other than by ({kw.arg})")
            if by is None:
                raise JitError(
                    f"self.{attr} .observe requires by=<expr>; the by key ranks the payloads"
                )
        elif len(call.args) != 1 or call.keywords:
            _reject(".observe expects exactly one value argument")
        self._check_element(call.args[0], monoid)
        return self._emit_row(attr, keys, comp, monoid, call.args[0], by)

    def _append(self, call: ast.Call) -> List[str]:
        func = call.func
        assert isinstance(func, ast.Attribute)
        if len(call.args) != 1:
            _reject(".append expects exactly one element argument")
        order: ast.expr | None = None
        for kw in call.keywords:
            if kw.arg == "order_by":
                order = kw.value
            else:
                _reject(f".append keyword other than order_by ({kw.arg})")
        attr, keys, comp, monoid = self._resolve_accum(func.value)
        if monoid.dft not in _LIST_OPS:
            _reject(f".append on the non-list monoid {monoid.dft}; use += or .observe(v)")
        if order is None:
            raise JitError(
                "ordered list requires order_by=<expr>; order-by-arrival is not parallel-safe"
            )
        self._check_element(call.args[0], monoid)
        return self._emit_row(attr, keys, comp, monoid, call.args[0], order)

    def _check_element(self, node: ast.expr, monoid: _Monoid) -> None:
        if monoid.elem not in ("str", "i64"):
            return
        if monoid.dft not in (_SET_OPS | _LIST_OPS | _ARGBY_OPS | _TOPK_OPS) and monoid.dft not in (
            "DFTU_AGG_APPROX_TOPK",
            "DFTU_AGG_SAMPLE",
        ):
            return
        is_str = self._is_str_element(node)
        if monoid.elem == "str" and not is_str:
            raise JitError(
                "a str-valued element must be a string field such as e.name; "
                "use of=jit.i64 for an int64 value"
            )
        if monoid.elem == "i64" and is_str:
            raise JitError(
                "an i64-valued element must be an int64 expr such as e.dur; "
                "use of=jit.str_ for a string value"
            )

    def _is_str_element(self, node: ast.expr) -> bool:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.event_name
            and node.attr in _STR_FIELDS
        ):
            return True
        arg = self._arg_call(node)
        return arg is not None and arg[0] == "arg_str"

    def _emit_row(
        self,
        attr: str,
        keys: List[ast.expr],
        comp: int,
        monoid: _Monoid,
        value: ast.expr,
        by: ast.expr | None,
    ) -> List[str]:
        """Append one contribution row to ``attr``'s per-batch column buffers;
        the buffers are folded into the DFTU_EXT_AGG accumulator after the loop."""
        decl = self.maps[attr]
        self.rows_per_event[attr] = self.rows_per_event.get(attr, 0) + 1
        lines = ["{", f"    uint32_t _r = _n_{attr}++;"]
        for idx, kn in enumerate(keys):
            ktype = decl.key_types[idx].dft
            if ktype == "DFTU_T_BYTES":
                _reject("a bytes key; e.<field> exposes no raw byte blob, author it raw")
            slot = f"_k{idx}_{attr}[_r]"
            if ktype == "DFTU_T_STR":
                lines.append(f"    {slot} = (dftu_str)({self._value(kn)});")
            elif ktype in _FLOAT_TYPES:
                expr, _ = self._arith(kn)
                lines.append(f"    {slot} = (double)({expr});")
            else:
                lines.append(f"    {slot} = (int64_t)({self._value(kn)});")
        if decl.is_product:
            for c, m in enumerate(decl.values):
                lines.append(f"    _v{c}_{attr}[_r] = ({_ELEM_CTYPE[m.elem]})0;")
                lines.append(f"    _o{c}_{attr}[_r] = 0;")
                if m.needs_by:
                    lines.append(f"    _b{c}_{attr}[_r] = 0.0;")
            lines.append(f"    _o{comp}_{attr}[_r] = 1;")
        lines.append(f"    _v{comp}_{attr}[_r] = {self._rhs(value, monoid.elem)};")
        if by is not None:
            by_expr, _ = self._arith(by)
            lines.append(f"    _b{comp}_{attr}[_r] = (double)({by_expr});")
        lines.append("}")
        return lines

    def _comp(self, node: ast.expr, decl: _MapDecl, attr: str) -> int:
        node = _unwrap_index(node)
        if (
            not isinstance(node, ast.Constant)
            or not isinstance(node.value, int)
            or isinstance(node.value, bool)
        ):
            _reject("component index (expected an integer literal)")
        comp = node.value
        if comp < 0 or comp >= len(decl.values):
            _reject(f"component index {comp} out of range for self.{attr}")
        return comp

    def _named_comp(self, name: str, decl: _MapDecl, attr: str) -> int:
        names = decl.value_names
        if names is None or name not in names:
            _reject(f"unknown value component .{name} for self.{attr}")
        return names.index(name)

    def _keys(self, sl: ast.expr) -> List[ast.expr]:
        sl = _unwrap_index(sl)
        if isinstance(sl, ast.Tuple):
            return builtins.list(sl.elts)
        return [sl]

    def _value(self, node: ast.expr) -> str:
        node = _unwrap_index(node)
        hoisted = self._cse.get(ast.dump(node))
        if hoisted is not None:
            return hoisted[0]
        opc = self._op_call(node)
        if opc is not None:
            return opc[0]
        prim = self._prim_call(node)
        if prim is not None:
            c_fn, cast, arg_nodes = prim
            parts = [f"({cast})({self._value(a)})" for a in arg_nodes]
            return f"{c_fn}({', '.join(parts)})"
        arg = self._arg_call(node)
        if arg is not None:
            expr, _ = self._lower_arg(*arg)
            return expr
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == self.self_name and node.attr in self.ports:
                return self._consume_ref(node.attr)
            if node.value.id == self.self_name and node.attr in self.config_fields:
                return f"_cfg_{node.attr}"
            if node.value.id == self.event_name:
                return self._field(node.attr)
            if node.attr == "NONE":
                return "DFTU_STR_NONE"
        if isinstance(node, ast.Constant):
            return self._const(node)
        _reject(_describe(node))

    def _rhs(self, node: ast.expr, elem: str) -> str:
        """Cast an expression into a value buffer of family ``elem``."""
        if elem == "str":
            return f"(dftu_str)({self._value(node)})"
        expr, saw_float = self._arith(node)
        if elem != "f64" and saw_float:
            _reject("float value into a u64 monoid component")
        return f"({'double' if elem == 'f64' else 'int64_t'})({expr})"

    def _arith(self, node: ast.expr) -> Tuple[str, bool]:
        node = _unwrap_index(node)
        hoisted = self._cse.get(ast.dump(node))
        if hoisted is not None:
            return hoisted[0], hoisted[1]
        if isinstance(node, ast.BinOp):
            sym = _ARITH_OP.get(type(node.op))
            if sym is None:
                _reject("arithmetic operator (only + - * / are allowed)")
            left, lf = self._arith(node.left)
            right, rf = self._arith(node.right)
            return f"({left} {sym} {right})", lf or rf or isinstance(node.op, ast.Div)
        opc = self._op_call(node)
        if opc is not None:
            return opc
        prim = self._prim_call(node)
        if prim is not None:
            c_fn, cast, arg_nodes = prim
            parts = [f"({cast})({self._arith(a)[0]})" for a in arg_nodes]
            return f"{c_fn}({', '.join(parts)})", False
        arg = self._arg_call(node)
        if arg is not None:
            return self._lower_arg(*arg)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id == self.self_name and node.attr in self.ports:
                port = self.ports[node.attr]
                return self._consume_ref(node.attr), port.is_f64
            if node.value.id == self.self_name and node.attr in self.config_fields:
                return f"_cfg_{node.attr}", self.config_fields[node.attr]
            if node.value.id == self.event_name:
                return self._field(node.attr), False
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool):
                _reject("boolean literal")
            if isinstance(v, int):
                return _int_literal(v), False
            if isinstance(v, float):
                return repr(v), True
            _reject("literal (only integer or float literals are allowed)")
        _reject(_describe(node))

    def _const(self, node: ast.Constant) -> str:
        v = node.value
        if isinstance(v, bool):
            _reject("boolean literal")
        if isinstance(v, int):
            return _int_literal(v)
        if isinstance(v, float):
            _reject("float literal")
        _reject("literal (only integer literals are allowed)")

    def _field(self, attr: str) -> str:
        expr = _FIELD_EXPR.get(attr)
        if expr is None:
            _reject(f"event field e.{attr}")
        need = _FIELD_NEED.get(attr)
        if need is not None:
            self.needs.add(need)
        return expr


def _port_ctype(port: _Port) -> Tuple[str, str]:
    return ("double", "0.0") if port.is_f64 else ("uint64_t", "0")


def _emit_name_lists(provides: List[str], consumes: List[str]) -> Tuple[List[str], List[str]]:
    """The dftu_plugin::provides / ::consumes definitions and the two factory
    assignment lines. Both lists are derived, never author-declared: a plugin
    produces the ports it publishes and the accumulators it creates, and reads
    the ports it consumes."""
    defs: List[str] = []
    assigns: List[str] = []
    for slot, names in (("provides", provides), ("consumes", consumes)):
        if not names:
            assigns.append(f"    g_plugin.{slot} = NULL;")
            continue
        rows = ", ".join(_c_str_literal(n) for n in names)
        defs += [
            f"static const char* const _{slot}_names[{len(names) + 1}] = {{{rows}, NULL}};",
            f"static const char* const* {slot}(void* self) {{",
            "    (void)self;",
            f"    return _{slot}_names;",
            "}",
            "",
        ]
        assigns.append(f"    g_plugin.{slot} = {slot};")
    return defs, assigns


def _emit_port_pre(
    pub_ports: List[Tuple[str, _Port]], sub_ports: List[Tuple[str, _Port]]
) -> List[str]:
    """Batch-scoped port setup emitted before the event loop: fetch the ports
    extension, read each consume port's value for this batch, and zero each
    publish port's per-batch accumulator."""
    out: List[str] = [
        "    const dftu_ext_ports* _ports =",
        "        (const dftu_ext_ports*)host->get_extension(host->h, DFTU_EXT_PORTS);",
        "    (void)_ports;",
    ]
    for name, port in sub_ports:
        ctype, zero = _port_ctype(port)
        cap = _c_str_literal(port.name)
        out += [
            f"    {ctype} _sub_{name} = {zero};",
            "    if (_ports && _ports->consume) {",
            "        uint32_t _sub_len = 0;",
            f"        const void* _sub_p = _ports->consume(host->h, _ports->port_key(host->h, {cap}), &_sub_len);",
            f"        if (_sub_p && _sub_len == 8u) std::memcpy(&_sub_{name}, _sub_p, 8);",
            "    }",
            f"    (void)_sub_{name};",
        ]
    for name, port in pub_ports:
        ctype, zero = _port_ctype(port)
        out.append(f"    {ctype} _pub_{name} = {zero};")
    return out


def _emit_port_flush(pub_ports: List[Tuple[str, _Port]]) -> List[str]:
    """Publish each publish port's per-batch accumulator after the event loop."""
    out: List[str] = []
    for name, port in pub_ports:
        cap = _c_str_literal(port.name)
        out += [
            "    if (_ports && _ports->publish) {",
            f"        _ports->publish(host->h, _ports->port_key(host->h, {cap}), &_pub_{name}, 8u);",
            "    }",
        ]
    return out


_STR_COL_HELPER = [
    "static dftu_series* dftu_jit_str_col(const dftu_host* host, const dftu_str* ids,",
    "                                     uint32_t n) {",
    "    int32_t* off = (int32_t*)malloc(sizeof(int32_t) * (size_t)(n + 1u));",
    "    if (!off) return NULL;",
    "    uint32_t tot = 0;",
    "    for (uint32_t i = 0; i < n; ++i) {",
    "        uint32_t len = 0;",
    "        const char* s = host->resolve(host->h, ids[i], &len);",
    "        if (!s) len = 0;",
    "        off[i] = (int32_t)tot;",
    "        tot += len;",
    "    }",
    "    off[n] = (int32_t)tot;",
    "    char* buf = (char*)malloc(tot ? tot : 1u);",
    "    if (!buf) { free(off); return NULL; }",
    "    for (uint32_t i = 0; i < n; ++i) {",
    "        uint32_t len = 0;",
    "        const char* s = host->resolve(host->h, ids[i], &len);",
    "        if (s && len) std::memcpy(buf + off[i], s, len);",
    "    }",
    "    dftu_series* col = dftu_series_new_string(DFTU_TYPE_STRING, off, buf, (int64_t)n, NULL);",
    "    free(buf);",
    "    free(off);",
    "    return col;",
    "}",
    "",
]

_BITMAP_HELPER = [
    "static uint8_t* dftu_jit_bitmap(const uint8_t* flags, uint32_t n) {",
    "    uint32_t nb = (n + 7u) / 8u;",
    "    uint8_t* bm = (uint8_t*)calloc(nb ? nb : 1u, 1);",
    "    if (!bm) return NULL;",
    "    for (uint32_t i = 0; i < n; ++i)",
    "        if (flags[i]) bm[i >> 3] |= (uint8_t)(1u << (i & 7u));",
    "    return bm;",
    "}",
    "",
]


def _quantile_name(q: float) -> str:
    return f"p{q * 100:g}"


def _agg_specs(decl: _MapDecl) -> List[Tuple[str, str, str, str, str]]:
    """The (op, value, out, param, by) dftu_agg_col rows for one accumulator."""
    specs: List[Tuple[str, str, str, str, str]] = []
    for c, m in enumerate(decl.values):
        if isinstance(m, Quantiles):
            specs.append(("DFTU_AGG_COUNT", "NULL", _c_str_literal("count"), "0.0", "NULL"))
            for q in m.qs:
                specs.append(
                    (
                        "DFTU_AGG_PCT",
                        _c_str_literal(f"v{c}"),
                        _c_str_literal(_quantile_name(q)),
                        repr(q),
                        "NULL",
                    )
                )
            continue
        specs.append(
            (
                m.dft,
                _c_str_literal(f"v{c}"),
                _c_str_literal(_out_name(decl, c)),
                repr(m.param),
                _c_str_literal(f"b{c}") if m.needs_by else "NULL",
            )
        )
    return specs


def _emit_buffers(attr: str, decl: _MapDecl, rows: int) -> List[str]:
    """Per-batch row buffers for one accumulator, sized to the batch."""
    out = [
        f"    uint32_t _cap_{attr} = b->count * {rows}u;",
        f"    uint32_t _n_{attr} = 0;",
    ]
    for idx, kt in enumerate(decl.key_types):
        ct, _ = _key_ctype(kt.dft)
        out.append(
            f"    {ct}* _k{idx}_{attr} = ({ct}*)malloc(sizeof({ct}) * "
            f"(size_t)(_cap_{attr} ? _cap_{attr} : 1u));"
        )
        out.append(f"    if (!_k{idx}_{attr}) _alloc_ok = 0;")
    for c, m in enumerate(decl.values):
        ct = _ELEM_CTYPE[m.elem]
        out.append(
            f"    {ct}* _v{c}_{attr} = ({ct}*)malloc(sizeof({ct}) * "
            f"(size_t)(_cap_{attr} ? _cap_{attr} : 1u));"
        )
        out.append(f"    if (!_v{c}_{attr}) _alloc_ok = 0;")
        if decl.is_product:
            out.append(
                f"    uint8_t* _o{c}_{attr} = (uint8_t*)malloc("
                f"(size_t)(_cap_{attr} ? _cap_{attr} : 1u));"
            )
            out.append(f"    if (!_o{c}_{attr}) _alloc_ok = 0;")
        if m.needs_by:
            out.append(
                f"    double* _b{c}_{attr} = (double*)malloc(sizeof(double) * "
                f"(size_t)(_cap_{attr} ? _cap_{attr} : 1u));"
            )
            out.append(f"    if (!_b{c}_{attr}) _alloc_ok = 0;")
    return out


def _emit_flush(attr: str, decl: _MapDecl) -> List[str]:
    """Build one batch frame from the row buffers and fold it into the named
    DFTU_EXT_AGG accumulator."""
    cols: List[Tuple[str, List[str]]] = []
    for idx, kt in enumerate(decl.key_types):
        _, dtype = _key_ctype(kt.dft)
        if kt.dft == "DFTU_T_STR":
            cols.append((f"k{idx}", [f"dftu_jit_str_col(host, _k{idx}_{attr}, _n_{attr})"]))
        else:
            cols.append(
                (
                    f"k{idx}",
                    [f"dftu_series_new_flat({dtype}, _k{idx}_{attr}, (int64_t)_n_{attr}, NULL)"],
                )
            )
    for c, m in enumerate(decl.values):
        dtype = _ELEM_DTYPE[m.elem]
        if m.elem == "str":
            cols.append((f"v{c}", [f"dftu_jit_str_col(host, _v{c}_{attr}, _n_{attr})"]))
        elif decl.is_product:
            cols.append(
                (
                    f"v{c}",
                    [
                        f"dftu_series_new_flat({dtype}, _v{c}_{attr}, (int64_t)_n_{attr}, "
                        f"_bm_{attr})"
                    ],
                )
            )
        else:
            cols.append(
                (
                    f"v{c}",
                    [f"dftu_series_new_flat({dtype}, _v{c}_{attr}, (int64_t)_n_{attr}, NULL)"],
                )
            )
        if m.needs_by:
            cols.append(
                (
                    f"b{c}",
                    [
                        f"dftu_series_new_flat(DFTU_TYPE_FLOAT64, _b{c}_{attr}, "
                        f"(int64_t)_n_{attr}, NULL)"
                    ],
                )
            )
    n_cols = len(cols)
    specs = _agg_specs(decl)
    n_keys = len(decl.key_types)
    # The accumulator is created for every batch, empty ones included, so a map
    # that never collected a row still finalizes to a zero-row result.
    out = ["    {"]
    if n_keys:
        key_list = ", ".join(_c_str_literal(f"k{i}") for i in range(n_keys))
        out.append(f"        const char* _keys[{n_keys}] = {{{key_list}}};")
        keys_arg = "_keys"
    else:
        keys_arg = "NULL"
    spec_rows = ", ".join("{" + ", ".join(s) + "}" for s in specs)
    out += [
        f"        const dftu_agg_col _specs[{len(specs)}] = {{{spec_rows}}};",
        f"        dftu_agg* _a = _agg->agg_new(host->h, {_c_str_literal(attr)}, "
        f"{keys_arg}, {n_keys}u, _specs, {len(specs)}u);",
        f"        if (_a && _n_{attr} > 0) {{",
        f"            dftu_series* _cols[{n_cols}];",
        f"            const char* _cnames[{n_cols}];",
    ]
    if decl.is_product:
        out.append(f"            uint8_t* _bm_{attr} = NULL;")
    ci = 0
    for cname, expr in cols:
        if decl.is_product and cname.startswith("v"):
            c = cname[1:]
            out.append(f"            _bm_{attr} = dftu_jit_bitmap(_o{c}_{attr}, _n_{attr});")
        out.append(f"            _cols[{ci}] = {expr[0]};")
        out.append(f"            _cnames[{ci}] = {_c_str_literal(cname)};")
        if decl.is_product and cname.startswith("v"):
            out.append(f"            free(_bm_{attr});")
            out.append(f"            _bm_{attr} = NULL;")
        ci += 1
    out += [
        "            int _cok = 1;",
        f"            for (int _ci = 0; _ci < {n_cols}; ++_ci)",
        "                if (!_cols[_ci]) _cok = 0;",
        "            if (_cok) {",
        f"                dftu_dataframe* _df = dftu_dataframe_new(_cnames, _cols, {n_cols});",
        "                if (_df) {",
        "                    _agg->agg_accumulate(host->h, _a, _df);",
        "                    dftu_dataframe_free(_df);",
        "                }",
        "            } else {",
        f"                for (int _ci = 0; _ci < {n_cols}; ++_ci)",
        "                    if (_cols[_ci]) dftu_series_free(_cols[_ci]);",
        "            }",
        "        }",
        "    }",
    ]
    return out


def _emit_frees(attr: str, decl: _MapDecl) -> List[str]:
    out = [f"    free(_k{idx}_{attr});" for idx in range(len(decl.key_types))]
    for c, m in enumerate(decl.values):
        out.append(f"    free(_v{c}_{attr});")
        if decl.is_product:
            out.append(f"    free(_o{c}_{attr});")
        if m.needs_by:
            out.append(f"    free(_b{c}_{attr});")
    return out


def _emit(
    maps: Dict[str, _MapDecl],
    body: List[str],
    needs: builtins.set[str],
    plan_query: str | None,
    str_literals: List[str],
    arg_keys: List[str],
    arg_helpers: builtins.set[str],
    rows_per_event: Dict[str, int],
    op_defs: List[str] | None = None,
    ports: Dict[str, _Port] | None = None,
    config_fields: "Dict[str, bool] | None" = None,
) -> str:
    op_defs = op_defs or []
    ports = ports or {}
    pub_ports = [(n, p) for n, p in ports.items() if p.role == "publish"]
    sub_ports = [(n, p) for n, p in ports.items() if p.role == "consume"]
    needs_expr = " | ".join(sorted(needs)) if needs else "0u"
    needs_str = any(
        kt.dft == "DFTU_T_STR" for decl in maps.values() for kt in decl.key_types
    ) or any(m.elem == "str" for decl in maps.values() for m in decl.values)
    needs_bitmap = any(decl.is_product for decl in maps.values())
    plan_query_field = "plan_query" if plan_query is not None else "NULL"
    out: List[str] = [
        "#include <dftracer/utils/plugins/abi.h>",
        "#include <dftracer/utils/plugins/prims.h>",
        "#include <dftracer/utils/dataframe/abi.h>",
        "",
        "#include <cstring>",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "",
        "static uint32_t needs(void* self) {",
        "    (void)self;",
        f"    return {needs_expr};",
        "}",
        "",
    ]
    for method in ("arg_i64", "arg_f64", "arg_str"):
        if method in arg_helpers:
            out += _ARG_HELPER_DEFS[method]
    if needs_str:
        out += _STR_COL_HELPER
    if needs_bitmap:
        out += _BITMAP_HELPER
    for fn_src in op_defs:
        out += [fn_src, ""]
    for idx, _ in enumerate(str_literals):
        out.append(f"static dftu_str lit_{idx};")
    if str_literals:
        out += ["static int lits_resolved = 0;", ""]
    for idx, _ in enumerate(arg_keys):
        out.append(f"static dftu_str argkey_{idx};")
    if arg_keys:
        out += ["static int args_resolved = 0;", ""]
    cfgs = config_fields if config_fields is not None else {}
    for cn, is_f in cfgs.items():
        out.append(f"static {'double' if is_f else 'int64_t'} _cfg_{cn} = 0;")
    if cfgs:
        out.append("")
    # Config reads for the factory: (void)config then one dftu_as_* per field.
    cfg_reads = "\n".join(
        ["    (void)config;"]
        + [
            f"    _cfg_{cn} = dftu_as_{'f64' if is_f else 'i64'}"
            f"(dftu_obj_get(config, {_c_str_literal(cn)}), {'0.0' if is_f else '0'});"
            for cn, is_f in cfgs.items()
        ]
    )
    if plan_query is not None:
        out += [
            "static const char* plan_query(void* self) {",
            "    (void)self;",
            f"    return {_c_str_literal(plan_query)};",
            "}",
            "",
        ]
    name_defs, name_assigns = _emit_name_lists(
        sorted([p.name for _, p in pub_ports] + [*maps]),
        sorted(p.name for _, p in sub_ports),
    )
    out += name_defs
    out += [
        "static void* make_slice(void* self) {",
        "    (void)self;",
        "    return calloc(1, 1);",
        "}",
        "",
        "static dftu_task* on_batch(void* slice, const dftu_batch* b,",
        "                          const dftu_host* host) {",
        "    (void)slice;",
        "    const dftu_ext_agg* _agg =",
        "        (const dftu_ext_agg*)host->get_extension(host->h, DFTU_EXT_AGG);",
    ]
    if maps:
        out.append("    if (!_agg || !_agg->agg_new || !_agg->agg_accumulate) return NULL;")
    else:
        out.append("    (void)_agg;")
    if str_literals:
        out.append("    if (!lits_resolved) {")
        for idx, lit in enumerate(str_literals):
            blen = len(lit.encode("utf-8"))
            out.append(f"        lit_{idx} = host->intern(host->h, {_c_str_literal(lit)}, {blen});")
        out.append("        lits_resolved = 1;")
        out.append("    }")
    if arg_keys:
        out.append("    if (!args_resolved) {")
        for idx, argname in enumerate(arg_keys):
            blen = len(argname.encode("utf-8"))
            out.append(
                f"        argkey_{idx} = host->intern(host->h, {_c_str_literal(argname)}, {blen});"
            )
        out.append("        args_resolved = 1;")
        out.append("    }")
    out.append("    int _alloc_ok = 1;")
    for attr, decl in maps.items():
        out += _emit_buffers(attr, decl, builtins.max(1, rows_per_event.get(attr, 1)))
    out.append("    if (_alloc_ok) {")
    inner: List[str] = []
    if ports:
        inner += _emit_port_pre(pub_ports, sub_ports)
    inner.append("    for (uint32_t i = 0; i < b->count; ++i) {")
    inner.append("        const dftu_event* e = &b->events[i];")
    for line in body:
        inner.append("        " + line)
    inner.append("    }")
    if ports:
        inner += _emit_port_flush(pub_ports)
    for attr, decl in maps.items():
        inner += _emit_flush(attr, decl)
    out += ["    " + ln for ln in inner]
    out.append("    }")
    for attr, decl in maps.items():
        out += _emit_frees(attr, decl)
    out.extend(
        [
            "    return NULL;",
            "}",
            "",
            "static void merge(void* into, void* other) {",
            "    (void)into;",
            "    (void)other;",
            "}",
            "",
            "static dftu_task* on_finalize(void* slice, const dftu_host* host) {",
            "    (void)slice;",
            "    (void)host;",
            "    return NULL;",
            "}",
            "",
            "static void destroy_slice(void* slice) { free(slice); }",
            "",
            "static void destroy(void* self) { (void)self; }",
            "",
            "static dftu_plugin g_plugin;",
            "",
            "#ifdef __cplusplus",
            'extern "C"',
            "#endif",
            "dftu_plugin* dftracer_plugin(const dftu_value* config) {",
            cfg_reads,
            "    g_plugin.abi_version = DFTRACER_PLUGIN_ABI_VERSION;",
            "    g_plugin.self = NULL;",
            "    g_plugin.needs = needs;",
            f"    g_plugin.plan_query = {plan_query_field};",
            "    g_plugin.make_slice = make_slice;",
            "    g_plugin.on_batch = on_batch;",
            "    g_plugin.merge = merge;",
            "    g_plugin.on_finalize = on_finalize;",
            "    g_plugin.destroy_slice = destroy_slice;",
            "    g_plugin.destroy = destroy;",
        ]
        + name_assigns
    )
    out.extend(
        [
            "    return &g_plugin;",
            "}",
            "",
        ]
    )
    return "\n".join(out)


def _resolve_needs(needs: Tuple[object, ...] | None) -> builtins.set[str]:
    if needs is None:
        return builtins.set()
    out: builtins.set[str] = builtins.set()
    for n in needs:
        if not isinstance(n, _Need):
            raise JitError("needs must be jit.NEED_ARGS / jit.NEED_FHASH / jit.NEED_HHASH flags")
        out.add(n.dft)
    return out


# Row budget per event per accumulator for a raw each_event body; the emitted
# buffers hold this many rows per scanned event and a raw body must not append
# more (there is no AST to count its contributions from).
_RAW_ROWS_PER_EVENT = 4


def _raw_body(fn: Callable[..., object]) -> List[str]:
    argc = getattr(fn, "__code__").co_argcount
    result = fn(*([None] * argc))
    if not isinstance(result, str):
        raise JitError("@jit.each_event(raw=True) method must return a C++ body string")
    return textwrap.dedent(result).splitlines()


def _referenced_ops(fn: Callable[..., object]) -> Dict[str, Op]:
    """The ``@jit.op`` transforms the each_event body can reach by name, from its
    globals and closure - so calling one inlines that op into the plugin."""
    try:
        cv = inspect.getclosurevars(fn)
    except TypeError:
        return {}
    scope = {**cv.globals, **cv.nonlocals}
    return {name: val for name, val in scope.items() if isinstance(val, Op)}


def _build_plugin(cls: type, needs: Tuple[object, ...] | None) -> type:
    maps: Dict[str, _MapDecl] = {}
    ports: Dict[str, _Port] = {}
    configs: Dict[str, _Config] = {}
    each: List[_EachEvent] = []
    for attr, val in vars(cls).items():
        if isinstance(val, _MapDecl):
            maps[attr] = val
        elif isinstance(val, _Port):
            ports[attr] = val
        elif isinstance(val, _Config):
            configs[attr] = val
        elif isinstance(val, _EachEvent):
            each.append(val)
    if not maps and not ports:
        raise JitError(
            "@jit.plugin needs at least one jit.map or jit.publish/jit.consume attribute"
        )
    for attr, decl in maps.items():
        if decl.is_product and any(m.elem == "str" for m in decl.values):
            raise JitError(
                f"@jit.plugin map '{attr}': a string-valued reduction "
                "(jit.set/list/argmin/argmax/topk/sample with of=jit.str_) cannot be a "
                "product component; declare it as its own jit.map"
            )
    if len(each) != 1:
        raise JitError("@jit.plugin needs exactly one @jit.each_event method")
    plan_query = getattr(cls, "plan_query", None)
    if plan_query is not None and not isinstance(plan_query, str):
        raise JitError("@jit.plugin plan_query must be a query DSL string")
    explicit_needs = _resolve_needs(needs)
    # {name: is_f64} for config fields, read into a body-visible static.
    config_f64 = {n: c.dft in ("DFTU_T_F64", "DFTU_T_F32") for n, c in configs.items()}
    op_defs: List[str] = []
    if each[0].raw:
        body = _raw_body(each[0].fn)
        inferred_needs = explicit_needs
        str_literals: List[str] = []
        arg_keys: List[str] = []
        arg_helpers: builtins.set[str] = builtins.set()
        # A raw body appends rows itself, so the row budget cannot be counted
        # from the AST; RAW_ROWS_PER_EVENT is the contract it must respect.
        rows_per_event = {attr: _RAW_ROWS_PER_EVENT for attr in maps}
    else:
        compiler = _Compiler(
            maps,
            _referenced_ops(each[0].fn),
            ports=ports,
            config_fields=config_f64,
        )
        body = compiler.lower(each[0].fn)
        inferred_needs = compiler.needs | explicit_needs
        str_literals = compiler.str_literals
        arg_keys = compiler.arg_keys
        arg_helpers = compiler.arg_helpers
        op_defs = compiler.op_defs
        rows_per_event = compiler.rows_per_event
    source = _emit(
        maps,
        body,
        inferred_needs,
        plan_query,
        str_literals,
        arg_keys,
        arg_helpers,
        rows_per_event,
        op_defs,
        ports,
        config_f64,
    )
    setattr(cls, "_jit_plugin", JitPlugin(cls.__name__, source, {}))
    return cls


# ---- vfold: a per-batch fold on columns -----------------------------------

# Reducer sugar accepted on a scalar accumulator; each must match the
# accumulator's own reduction.
_VFOLD_REDUCERS = {
    "sum": "DFTU_AGG_SUM",
    "min": "DFTU_AGG_MIN",
    "max": "DFTU_AGG_MAX",
}
_VFOLD_TOP_FIELDS = frozenset({"name", "cat", "pid", "tid", "ts", "dur", "ph", "fhash", "hhash"})
# Numeric top-level columns; an ordering (by) column must be one of these.
_VFOLD_NUMERIC = frozenset({"pid", "tid", "ts", "dur", "ph"})
_VFOLD_STRING_FIELDS = frozenset({"name", "cat", "fhash", "hhash"})


def _vfold_df_field(node: ast.expr) -> str:
    if not (
        isinstance(node, ast.Subscript)
        and isinstance(node.value, ast.Name)
        and node.value.id == "df"
    ):
        raise JitError('@jit.vfold body must be self.<acc> += df["<field>"].<reducer>()')
    key = _unwrap_index(node.slice)
    if not (isinstance(key, ast.Constant) and isinstance(key.value, str)):
        raise JitError('@jit.vfold: df subscript must be a string field name, df["dur"]')
    return key.value


def _vfold_specs(opd: Dict[str, object]) -> List[Tuple[str, str, str, str, str]]:
    """The (op, value, out, param, by) dftu_agg_col rows for one vfold op."""
    mon = cast(_Monoid, opd["mon"])
    vfield = opd["vfield"]
    byfield = opd["byfield"]
    value = _c_str_literal(cast(str, vfield)) if vfield is not None else "NULL"
    by = _c_str_literal(cast(str, byfield)) if byfield is not None else "NULL"
    if isinstance(mon, Quantiles):
        specs = [("DFTU_AGG_COUNT", "NULL", _c_str_literal("count"), "0.0", "NULL")]
        for q in mon.qs:
            specs.append(
                ("DFTU_AGG_PCT", value, _c_str_literal(_quantile_name(q)), repr(q), "NULL")
            )
        return specs
    if vfield is None:
        return [("DFTU_AGG_COUNT", "NULL", _c_str_literal("value"), "0.0", "NULL")]
    return [(mon.dft, value, _c_str_literal("value"), repr(mon.param), by)]


def _compile_vfold(
    fn: Callable[..., object],
    accums: Dict[str, _Monoid],
    maps: Dict[str, _MapDecl],
) -> Tuple[List[Dict[str, object]], "builtins.set[str]"]:
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef):
        raise JitError("@jit.each_batch must decorate a function")
    ops: List[Dict[str, object]] = []
    fields: "builtins.set[str]" = builtins.set()
    shape = (
        '@jit.vfold body must be self.<acc> += df["f"].<reducer>() '
        'or self.<map>[df["k"]] += df["v"]'
    )
    for stmt in func.body:
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            continue  # docstring
        if not (isinstance(stmt, ast.AugAssign) and isinstance(stmt.op, ast.Add)):
            raise JitError(shape)
        tgt = stmt.target
        rhs = stmt.value
        keyfields: List[str] = []
        if (
            isinstance(tgt, ast.Subscript)
            and isinstance(tgt.value, ast.Attribute)
            and isinstance(tgt.value.value, ast.Name)
            and tgt.value.value.id == "self"
        ):
            attr = tgt.value.attr
            if attr not in maps:
                raise JitError(f"@jit.vfold: '{attr}' is not a declared jit.map")
            decl = maps[attr]
            sl = _unwrap_index(tgt.slice)
            elts = builtins.list(sl.elts) if isinstance(sl, ast.Tuple) else [sl]
            keyfields = [_vfold_df_field(k) for k in elts]
            if len(keyfields) != len(decl.key_types):
                raise JitError(
                    f"@jit.vfold: '{attr}' takes {len(decl.key_types)} key columns, "
                    f"got {len(keyfields)}"
                )
            for kt, kf in zip(decl.key_types, keyfields):
                if kf not in _VFOLD_TOP_FIELDS:
                    raise JitError(f"@jit.vfold: key column '{kf}' is not a scan column")
                if (kf in _VFOLD_STRING_FIELDS) != (kt.dft == "DFTU_T_STR"):
                    raise JitError(
                        f"@jit.vfold: key column '{kf}' does not match the declared key type; "
                        "a string column needs jit.str_, a numeric one an integer type"
                    )
            mon = decl.values[0]
        elif (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ):
            attr = tgt.attr
            if attr not in accums:
                raise JitError(f"@jit.vfold: '{attr}' is not a declared accumulator")
            mon = accums[attr]
        else:
            raise JitError(shape)
        vfield: str | None = None
        byfield: str | None = None
        if isinstance(rhs, ast.Call):
            # Scalar sugar: df["f"].sum() / .min() / .max().
            if not (not rhs.args and isinstance(rhs.func, ast.Attribute)):
                raise JitError('@jit.vfold: scalar value must be df["<field>"].<reducer>()')
            reducer = rhs.func.attr
            if reducer not in _VFOLD_REDUCERS:
                raise JitError(f"@jit.vfold: unsupported reducer '.{reducer}()' (use sum/min/max)")
            if _VFOLD_REDUCERS[reducer] != mon.dft:
                raise JitError(
                    f"@jit.vfold: '.{reducer}()' does not match accumulator '{attr}' "
                    f"(a jit.{reducer}() accumulator)"
                )
            vfield = _vfold_df_field(rhs.func.value)
        elif mon.needs_by:
            if not (isinstance(rhs, ast.Tuple) and len(rhs.elts) == 2):
                raise JitError(
                    f"@jit.vfold: '{attr}' needs a (value, by) column pair, "
                    'e.g. self.m[df["pid"]] += df["ts"], df["dur"]'
                )
            vfield = _vfold_df_field(rhs.elts[0])
            byfield = _vfold_df_field(rhs.elts[1])
            if byfield not in _VFOLD_NUMERIC:
                raise JitError(
                    f"@jit.vfold: the by column '{byfield}' must be numeric (pid/tid/ts/dur)"
                )
        elif isinstance(rhs, ast.Constant) and isinstance(rhs.value, int):
            if not isinstance(mon, Counter) or rhs.value != 1:
                raise JitError(
                    f"@jit.vfold: '{attr}' reduces a value column; only a jit.count() "
                    "accumulator takes the constant 1"
                )
        elif isinstance(rhs, ast.Tuple):
            raise JitError(f"@jit.vfold: '{attr}' takes a single value column, not a pair")
        else:
            vfield = _vfold_df_field(rhs)
        if isinstance(mon, Counter) and vfield is not None:
            raise JitError(
                f"@jit.vfold: a jit.count() accumulator ('{attr}') counts rows; "
                "write += 1, not a column"
            )
        if vfield is not None:
            if vfield not in _VFOLD_TOP_FIELDS:
                raise JitError(f"@jit.vfold: column '{vfield}' is not a scan column")
            want_str = mon.elem == "str"
            if (vfield in _VFOLD_STRING_FIELDS) != want_str:
                raise JitError(
                    f"@jit.vfold: '{attr}' reduces a "
                    f"{'string' if want_str else 'numeric'} column, but '{vfield}' is not one"
                )
            fields.add(vfield)
        if byfield is not None:
            fields.add(byfield)
        fields.update(keyfields)
        ops.append(
            {
                "attr": attr,
                "mon": mon,
                "keyfields": keyfields,
                "vfield": vfield,
                "byfield": byfield,
            }
        )
    if not ops:
        raise JitError("@jit.vfold: the @jit.each_batch body is empty")
    return ops, fields


def _emit_vfold(
    ops: List[Dict[str, object]],
    needs_expr: str,
    plan_query: str | None,
) -> str:
    plan_query_field = "plan_query" if plan_query is not None else "NULL"
    out: List[str] = [
        "#include <dftracer/utils/plugins/abi.h>",
        "#include <dftracer/utils/dataframe/abi.h>",
        "",
        "#include <stdint.h>",
        "#include <stdlib.h>",
        "",
        "static uint32_t needs(void* self) {",
        "    (void)self;",
        f"    return {needs_expr};",
        "}",
        "",
    ]
    if plan_query is not None:
        out += [
            "static const char* plan_query(void* self) {",
            "    (void)self;",
            f"    return {_c_str_literal(plan_query)};",
            "}",
            "",
        ]
    name_defs, name_assigns = _emit_name_lists(sorted({cast(str, opd["attr"]) for opd in ops}), [])
    out += name_defs
    out += [
        "static void* make_slice(void* self) {",
        "    (void)self;",
        "    return calloc(1, 1);",
        "}",
        "",
        "static dftu_task* on_batch_columns(void* slice,",
        "                                   const dftu_dataframe* df,",
        "                                   const dftu_host* host) {",
        "    (void)slice;",
        "    const dftu_ext_agg* agg =",
        "        (const dftu_ext_agg*)host->get_extension(host->h, DFTU_EXT_AGG);",
        "    if (!agg || !agg->agg_new || !agg->agg_accumulate) return NULL;",
    ]
    for opd in ops:
        attr = cast(str, opd["attr"])
        keyfields = cast(List[str], opd["keyfields"])
        specs = _vfold_specs(opd)
        spec_rows = ", ".join("{" + ", ".join(s) + "}" for s in specs)
        out.append("    {")
        out.append(
            f"        const dftu_agg_col specs[{len(specs)}] = {{{spec_rows}}};",
        )
        if keyfields:
            key_list = ", ".join(_c_str_literal(k) for k in keyfields)
            out.append(f"        const char* keys[{len(keyfields)}] = {{{key_list}}};")
            keys_arg = "keys"
        else:
            keys_arg = "NULL"
        out += [
            f"        dftu_agg* a = agg->agg_new(host->h, {_c_str_literal(attr)}, "
            f"{keys_arg}, {len(keyfields)}u, specs, {len(specs)}u);",
            "        if (a) agg->agg_accumulate(host->h, a, df);",
            "    }",
        ]
    out += [
        "    return NULL;",
        "}",
        "",
        "static void merge(void* into, void* other) { (void)into; (void)other; }",
        "",
        "static dftu_task* on_finalize(void* slice, const dftu_host* host) {",
        "    (void)slice;",
        "    (void)host;",
        "    return NULL;",
        "}",
        "",
        "static void destroy_slice(void* slice) { free(slice); }",
        "static void destroy(void* self) { (void)self; }",
        "",
        "static dftu_plugin g_plugin;",
        "",
        "#ifdef __cplusplus",
        'extern "C"',
        "#endif",
        "dftu_plugin* dftracer_plugin(const dftu_value* config) {",
        "    (void)config;",
        "    g_plugin.abi_version = DFTRACER_PLUGIN_ABI_VERSION;",
        "    g_plugin.self = NULL;",
        "    g_plugin.needs = needs;",
        f"    g_plugin.plan_query = {plan_query_field};",
        "    g_plugin.make_slice = make_slice;",
        "    g_plugin.merge = merge;",
        "    g_plugin.on_finalize = on_finalize;",
        "    g_plugin.destroy_slice = destroy_slice;",
        "    g_plugin.destroy = destroy;",
        "    g_plugin.on_batch_columns = on_batch_columns;",
    ]
    out += name_assigns
    out += [
        "    return &g_plugin;",
        "}",
        "",
    ]
    return "\n".join(out)


def _build_vfold(cls: type) -> type:
    accums: Dict[str, _Monoid] = {}
    maps: Dict[str, _MapDecl] = {}
    batch: List[_EachBatch] = []
    for attr, val in vars(cls).items():
        if isinstance(val, _MapDecl):
            maps[attr] = val
        elif isinstance(val, _Monoid):
            accums[attr] = val
        elif isinstance(val, _EachBatch):
            batch.append(val)
    for attr, m in maps.items():
        if m.is_product:
            raise JitError(
                f"@jit.vfold: '{attr}' must be a simple jit.map(key=<type>, value=<reduction>)"
            )
        if not m.key_types:
            raise JitError(f"@jit.vfold: '{attr}' needs at least one key")
    if not accums and not maps:
        raise JitError("@jit.vfold needs an accumulator (jit.sum()/min()/max()) or a jit.map")
    if len(batch) != 1:
        raise JitError("@jit.vfold needs exactly one @jit.each_batch method")
    plan_query = getattr(cls, "plan_query", None)
    if plan_query is not None and not isinstance(plan_query, str):
        raise JitError("@jit.vfold plan_query must be a query DSL string")
    ops, fields = _compile_vfold(batch[0].fn, accums, maps)
    needs_expr = "DFTU_NEED_ARGS" if any(f not in _VFOLD_TOP_FIELDS for f in fields) else "0u"
    source = _emit_vfold(ops, needs_expr, plan_query)
    setattr(cls, "_jit_plugin", JitPlugin(cls.__name__, source, {}))
    return cls


def vfold(cls: type) -> type:
    """Author a vectorized fold: a per-batch fold whose body runs SIMD column
    ops on the batch and folds them into scalar accumulators or keyed maps.

    Declare scalar accumulators as a bare reduction (``jit.sum()``, ``jit.min()``,
    ...) and keyed accumulators as ``jit.map(key=<types>, value=<reduction>)``,
    plus one :func:`each_batch` method whose body is a sequence of
    ``self.<acc> += df["f"]`` (or the ``df["f"].sum()/min()/max()`` sugar) and
    ``self.<map>[df["k"]] += df["v"]`` (a multi-key map subscripts a tuple of
    columns, and ``+= 1`` counts rows into a ``jit.count()``). A reduction that
    reads a second column - argmin/argmax, top-k/bottom-k, ordered list, the
    occupancy ops and the co-moments - takes a ``(value, by)`` pair written
    ``self.<map>[df["k"]] += df["v"], df["by"]``.

    Every accumulator folds through the engine's DFTU_EXT_AGG AggState: a keyed
    map is one with key columns, a scalar accumulator one with none. The host
    merges same-named accumulators across worker slices and finalizes each to a
    native DataFrame; run it through
    :class:`dftracer.utils.plugins.PluginHost` like any other jit plugin."""
    return _build_vfold(cls)


# The op registry is shared with the engine's built-ins, whose names are bare
# (`add`, `sum`) or `dftu.`-prefixed; both namespaces are the host's, so a user
# op must be `<module>.<name>` (plugin ABI rule, see plugins/reserved_names.h).
_HOST_NAME_PREFIX = "dftu."


def _series_module(fn: Callable[..., object], module: "str | None") -> str:
    mod = module if module is not None else getattr(fn, "__module__", None)
    if not isinstance(mod, str) or not mod:
        raise JitError('@jit.series needs a module prefix; pass module="<name>"')
    if mod == _HOST_NAME_PREFIX[:-1] or mod.startswith(_HOST_NAME_PREFIX):
        raise JitError(
            f"module '{mod}' is reserved for host ops; the 'dftu.' namespace is not "
            "available to a user op"
        )
    return mod


def _series_impl(fn: Callable[..., object], module: "str | None") -> Callable[..., object]:
    from ..columnar import Expr, col
    from . import ops as _ops

    fname = getattr(fn, "__name__", None)
    if not isinstance(fname, str):
        raise JitError("@jit.series must decorate a named function")
    params = builtins.list(inspect.signature(fn).parameters.values())
    if any(p.kind not in (p.POSITIONAL_OR_KEYWORD, p.POSITIONAL_ONLY) for p in params):
        raise JitError("@jit.series takes only positional column arguments")
    n = len(params)
    if n == 0:
        raise JitError("@jit.series needs at least one column argument")
    built = fn(*[col(f"__x{i}__") for i in range(n)])
    if not isinstance(built, Expr):
        raise JitError("@jit.series body must return a column expression built from its arguments")
    name = f"{_series_module(fn, module)}.{fname}"
    _ops._register_user(name, n, built)
    return _ops.get(name)


def series(fn: "Callable[..., object] | None" = None, *, module: "str | None" = None) -> object:
    """Author a reusable column op from an expression over its arguments.

    The body builds a lazy column expression from its Series arguments using the
    columnar DSL (arithmetic, comparisons, clip/cast/fillna, prims), e.g.::

        @jit.series
        def doubled(dur):
            return dur * 2

    It returns a callable, so ``doubled(s)`` applies it via the engine's Expr
    evaluator - no compile step, and it fuses with built-in Expr math.

    It also registers in dftracer.utils.jit.ops under ``<module>.<name>``, where
    the module defaults to the defining ``fn.__module__``; pass
    ``module="stats"`` to choose it. The registry is shared with the engine's
    built-in ops, so a user op never takes a bare name, and the host's ``dftu.``
    namespace is refused. Reach it as ``ops.run("stats.<name>", s)``,
    ``ops.stats.<name>(s)`` and ``s.ops.stats.<name>()``. Arguments are
    positional column operands."""
    if fn is None:
        return lambda f: _series_impl(f, module)
    return _series_impl(fn, module)


@overload
def plugin(cls: type[_C]) -> type[_C]: ...
@overload
def plugin(
    cls: None = None, *, needs: Tuple[object, ...] | None = None
) -> Callable[[type[_C]], type[_C]]: ...
def plugin(
    cls: type | None = None, *, needs: Tuple[object, ...] | None = None
) -> type | Callable[[type], type]:
    """Class decorator: AST-compile the map decls + ``each_event`` to a C plugin.

    Attaches the emitted source under ``cls._jit_plugin``; the native ``.so`` is
    built lazily on first load. Pass ``needs=(jit.NEED_FHASH, ...)`` to declare
    scan extraction a raw ``each_event`` body cannot be inferred from. Raises
    :class:`JitError` for any construct outside the supported subset.
    """
    if cls is None:
        return lambda c: _build_plugin(c, needs)
    return _build_plugin(cls, needs)


def plugin_renames(obj: object) -> Dict[str, List[str]]:
    """Value-column rename map ``{result_name: [field, ...]}`` for a
    ``@jit.plugin`` class, or empty if it declares no named product."""
    spec = getattr(obj, "_jit_plugin", None)
    if isinstance(spec, JitPlugin):
        return spec.renames
    return {}


def compile_plugin(spec: JitPlugin) -> str:
    """Build ``spec`` to a cached native ``.so`` and return its path.

    Content-hashes the emitted source, include dir, and compiler so an unchanged
    plugin never rebuilds. Raises :class:`JitError` if the compile fails.
    """
    try:
        include = _plugin_build.include_dir()
        cxx = _plugin_build.compiler()
        digest = _plugin_build.source_digest(spec.source, include, cxx)
        out = _plugin_build.cache_dir() / f"{spec.name}.{digest}.so"
        if out.is_file():
            return str(out)
        src = out.with_suffix(".cpp")
        src.write_text(spec.source, encoding="utf-8")
        return _plugin_build.build_shared(str(src), out=str(out), name=spec.name)
    except _plugin_build.PluginBuildError as exc:
        raise JitError(f"jit compile failed:\n{exc}") from exc


def compile_class(obj: object) -> str:
    """Compile a ``@jit.plugin`` class to a native ``.so`` and return the path."""
    spec = getattr(obj, "_jit_plugin", None)
    if not isinstance(spec, JitPlugin):
        raise TypeError("expected a @jit.plugin-decorated class")
    return compile_plugin(spec)


def is_jit_plugin(obj: object) -> bool:
    """True if ``obj`` is a ``@jit.plugin``-decorated class."""
    return isinstance(getattr(obj, "_jit_plugin", None), JitPlugin)
