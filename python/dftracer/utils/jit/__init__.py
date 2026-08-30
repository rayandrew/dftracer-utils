"""Author a DFTracer plugin in Python and get a native one.

Decorate a class with :func:`plugin`; its :func:`map` attributes and single
:func:`each_event` method are AST-compiled to a C plugin against the stable ABI,
built to a cached ``.so``, and loaded through
:class:`dftracer.utils.plugins.PluginHost` exactly like a hand-written plugin.

The ``each_event`` subset is deliberately small: an optional ``if`` guard (one
comparison on ``e.<field>``) wrapping ``self.<map>[(<key>)] += <1 | e.<field>>``.
Anything outside it raises :class:`JitError` at decoration time, pointing at the
raw-C++ escape hatch, never a silent miscompile.

``jit.bytes`` exists for declaring and typing a ``DFTU_T_BYTES``-keyed map; the
per-event subset exposes no raw byte blob, so a bytes key is authored via the C
ABI / a ``raw=True`` body, not a bare ``each_event`` subscript.

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
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    NoReturn,
    Protocol,
    Tuple,
    TypedDict,
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
    "join",
    "nested",
    "publish",
    "consume",
    "shared",
    "Port",
    "Handle",
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
    "argmin",
    "argmax",
    "topk",
    "bottomk",
    "approx_topk",
    "sample",
    "argmin_row",
    "argmax_row",
    "record",
    "each_event",
    "vfold",
    "each_batch",
    "on_resolve",
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
    "JoinDecl",
    "Nested",
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
    "ArgMinRow",
    "ArgMaxRow",
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
V2 = TypeVar("V2")
K = TypeVar("K")
V = TypeVar("V")
K1 = TypeVar("K1")
K2 = TypeVar("K2")
K3 = TypeVar("K3")
K4 = TypeVar("K4")
IK = TypeVar("IK")
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
    __slots__ = ("dft",)

    def __init__(self, dft: str) -> None:
        self.dft = dft


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


class Quantiles(_Monoid):
    __slots__ = ("qs",)

    def __init__(self, dft: str, qs: "Tuple[float, ...]") -> None:
        super().__init__(dft)
        self.qs = qs

    def observe(self, v: float) -> None: ...


class ArgMin(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, payload: T, *, by: float) -> None: ...


class ArgMax(_Monoid, Generic[T]):
    __slots__ = ()

    def observe(self, payload: T, *, by: float) -> None: ...


class TopK(_Monoid, Generic[T]):
    __slots__ = ("k",)

    def __init__(self, dft: str, k: int) -> None:
        super().__init__(dft)
        self.k = k

    def observe(self, payload: T, *, by: float) -> None: ...


class BottomK(_Monoid, Generic[T]):
    __slots__ = ("k",)

    def __init__(self, dft: str, k: int) -> None:
        super().__init__(dft)
        self.k = k

    def observe(self, payload: T, *, by: float) -> None: ...


class ApproxTopK(_Monoid, Generic[T]):
    __slots__ = ("k",)

    def __init__(self, dft: str, k: int) -> None:
        super().__init__(dft)
        self.k = k

    def observe(self, x: T) -> None: ...


class Sample(_Monoid, Generic[T]):
    __slots__ = ("k",)

    def __init__(self, dft: str, k: int) -> None:
        super().__init__(dft)
        self.k = k

    def observe(self, x: T) -> None: ...


class ArgMinRow(_Monoid):
    __slots__ = ("payload_types",)
    is_max = False

    def __init__(self, payload_types: "Tuple[_Type[object], ...]") -> None:
        super().__init__("DFTU_MONOID_ARGMIN_ROW")
        self.payload_types = payload_types

    def observe(self, payload: "Tuple[object, ...]", *, by: float) -> None: ...


class ArgMaxRow(_Monoid):
    __slots__ = ("payload_types",)
    is_max = True

    def __init__(self, payload_types: "Tuple[_Type[object], ...]") -> None:
        super().__init__("DFTU_MONOID_ARGMAX_ROW")
        self.payload_types = payload_types

    def observe(self, payload: "Tuple[object, ...]", *, by: float) -> None: ...


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
    """A COUNTER value monoid (u64 sum)."""
    return Counter("DFTU_MONOID_COUNTER")


def sum() -> Sum:
    """A SUM_F64 value monoid (double sum)."""
    return Sum("DFTU_MONOID_SUM_F64")


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


_MINMAX_WIDTHS = {
    "DFTU_T_I8": "I8",
    "DFTU_T_I16": "I16",
    "DFTU_T_I32": "I32",
    "DFTU_T_I64": "I64",
    "DFTU_T_U8": "U8",
    "DFTU_T_U16": "U16",
    "DFTU_T_U32": "U32",
    "DFTU_T_U64": "U64",
    "DFTU_T_F32": "F32",
    "DFTU_T_F64": "F64",
}


def _minmax_dft(op: str, of: "_Type[object]") -> str:
    width = _MINMAX_WIDTHS.get(of.dft)
    if width is None:
        raise JitError(f"jit.{op.lower()}(of=...) must be a fixed-width int or float type")
    return f"DFTU_MONOID_{op}_{width}"


@overload
def min() -> "Min[int]": ...
@overload
def min(of: "_Type[T]") -> "Min[T]": ...
def min(of: "_Type[object]" = u64) -> "Min[object]":
    """A typed min value monoid (default u64); ``of=`` sets the element width.

    ``.observe(v)`` contributes; the result column materializes at ``of``'s exact width."""
    return Min(_minmax_dft("MIN", of))


@overload
def max() -> "Max[int]": ...
@overload
def max(of: "_Type[T]") -> "Max[T]": ...
def max(of: "_Type[object]" = u64) -> "Max[object]":
    """A typed max value monoid (default u64); ``of=`` sets the element width.

    ``.observe(v)`` contributes; the result column materializes at ``of``'s exact width."""
    return Max(_minmax_dft("MAX", of))


def minf() -> "Min[float]":
    """A MIN_F64 value monoid; contribute with ``.observe(v)``."""
    return Min("DFTU_MONOID_MIN_F64")


def maxf() -> "Max[float]":
    """A MAX_F64 value monoid; contribute with ``.observe(v)``."""
    return Max("DFTU_MONOID_MAX_F64")


def distinct() -> Distinct:
    """A DISTINCT value monoid (approx distinct count, u64); contribute with ``.observe(v)``."""
    return Distinct("DFTU_MONOID_DISTINCT")


@overload
def set() -> "SetV[Str]": ...
@overload
def set(of: "_Type[T]") -> "SetV[T]": ...
def set(of: "_Type[object]" = str_) -> "SetV[object]":
    """A set value monoid collecting distinct elements added with ``.observe``.

    ``of=jit.str_`` (default) collects interned strings into a list<string> column;
    ``of=jit.i64`` collects int64 values into a list<int64> column sorted ascending."""
    if of is i64:
        return SetV("DFTU_MONOID_SET_I64")
    if of is str_:
        return SetV("DFTU_MONOID_SET_STR")
    raise JitError("jit.set(of=...) must be jit.str_ or jit.i64")


@overload
def list() -> "ListV[Str]": ...
@overload
def list(of: "_Type[T]") -> "ListV[T]": ...
def list(of: "_Type[object]" = str_) -> "ListV[object]":
    """An ordered-list value monoid collecting elements added with ``.append``.

    ``of=jit.str_`` (default) collects interned strings into a list<string>
    column; ``of=jit.i64`` collects raw int64 values into a list<int64> column.
    Both materialize sorted by the ``order_by`` key."""
    if of is i64:
        return ListV("DFTU_MONOID_LIST_I64")
    if of is str_:
        return ListV("DFTU_MONOID_LIST_STR")
    raise JitError("jit.list(of=...) must be jit.str_ or jit.i64")


def mean() -> Mean:
    """A MEAN value monoid; contribute one value with ``.observe(v)``.

    Materializes to a double column (the running mean of the observed values)."""
    return Mean("DFTU_MONOID_MEAN")


def variance() -> Variance:
    """A VARIANCE value monoid (sample, n-1); contribute with ``.observe(v)``.

    Materializes to a double column; 0 for n<2, matching the aggregator."""
    return Variance("DFTU_MONOID_VARIANCE")


var = variance


def stddev() -> Stddev:
    """A STDDEV value monoid (sample, n-1); contribute with ``.observe(v)``.

    Materializes to a double column; 0 for n<2, matching the aggregator."""
    return Stddev("DFTU_MONOID_STDDEV")


std = stddev


def quantiles(qs: "Tuple[float, ...]" = (0.5, 0.9, 0.95, 0.99)) -> Quantiles:
    """A DDSketch quantile value monoid; contribute with ``.observe(v)``.

    Materializes to a ``count`` int64 column plus one f64 column per quantile in
    ``qs`` (each in [0, 1]), named ``p<q*100>`` (``p50``, ``p90``, ``p99``, ..).
    Quantiles are approximate (DDSketch, ~1% relative error). Top-level only:
    it cannot be a product/nested/record component."""
    qs = tuple(float(q) for q in qs)
    if not qs:
        raise JitError("jit.quantiles needs at least one quantile")
    for q in qs:
        if not 0.0 <= q <= 1.0:
            raise JitError(f"jit.quantiles value {q} is outside [0, 1]")
    return Quantiles("DFTU_MONOID_SKETCH", qs)


def _argby_dft(op: str, of: "_Type[object]") -> str:
    if of is str_:
        return f"DFTU_MONOID_{op}_STR"
    if of is i64:
        return f"DFTU_MONOID_{op}_I64"
    raise JitError(f"jit.{op.lower()}(of=...) must be jit.str_ or jit.i64")


@overload
def argmin() -> "ArgMin[Str]": ...
@overload
def argmin(of: "_Type[T]") -> "ArgMin[T]": ...
def argmin(of: "_Type[object]" = str_) -> "ArgMin[object]":
    """An argmin value: keep the payload whose ``by=`` key is smallest.

    ``of=jit.str_`` (default) keeps an interned-string payload (string column);
    ``of=jit.i64`` keeps a raw int64 payload. Contribute with ``.observe(payload, by=<expr>)``."""
    return ArgMin(_argby_dft("ARGMIN", of))


@overload
def argmax() -> "ArgMax[Str]": ...
@overload
def argmax(of: "_Type[T]") -> "ArgMax[T]": ...
def argmax(of: "_Type[object]" = str_) -> "ArgMax[object]":
    """An argmax value: keep the payload whose ``by=`` key is largest.

    ``of=jit.str_`` (default) keeps an interned-string payload (string column);
    ``of=jit.i64`` keeps a raw int64 payload. Contribute with ``.observe(payload, by=<expr>)``."""
    return ArgMax(_argby_dft("ARGMAX", of))


def _check_k(op: str, k: int) -> int:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise JitError(f"jit.{op}(k, ...) k must be a positive integer")
    return k


def _kv_dft(op: str, of: "_Type[object]") -> str:
    if of is str_:
        return f"DFTU_MONOID_{op}_STR"
    if of is i64:
        return f"DFTU_MONOID_{op}_I64"
    raise JitError(f"jit.{op.lower()}(of=...) must be jit.str_ or jit.i64")


@overload
def topk(k: int) -> "TopK[Str]": ...
@overload
def topk(k: int, of: "_Type[T]") -> "TopK[T]": ...
def topk(k: int, of: "_Type[object]" = str_) -> "TopK[object]":
    """A bounded top-k value: keep the k payloads at the k largest ``by=`` keys.

    ``of=jit.str_`` (default) keeps interned-string payloads into a list<string>
    column; ``of=jit.i64`` keeps raw int64 payloads into a list<int64>. Payloads
    emit in ``by`` order (descending). Contribute with
    ``.observe(payload, by=<expr>)``."""
    return TopK(_kv_dft("TOPK", of), _check_k("topk", k))


@overload
def bottomk(k: int) -> "BottomK[Str]": ...
@overload
def bottomk(k: int, of: "_Type[T]") -> "BottomK[T]": ...
def bottomk(k: int, of: "_Type[object]" = str_) -> "BottomK[object]":
    """A bounded bottom-k value: keep the k payloads at the k smallest ``by=``
    keys. ``of=`` selects a string (default) or int64 payload, mirroring
    :func:`topk`. Contribute with ``.observe(payload, by=<expr>)``."""
    return BottomK(_kv_dft("BOTTOMK", of), _check_k("bottomk", k))


@overload
def approx_topk(k: int) -> "ApproxTopK[Str]": ...
@overload
def approx_topk(k: int, of: "_Type[T]") -> "ApproxTopK[T]": ...
def approx_topk(k: int, of: "_Type[object]" = str_) -> "ApproxTopK[object]":
    """An approximate heavy-hitters value: the k most FREQUENT observed values
    (SpaceSaving), in bounded memory.

    ``of=jit.str_`` (default) counts interned string ids; ``of=jit.i64`` counts
    raw int64 values. Contribute with ``.observe(value)``; materializes to a
    list<struct<value, count>> column ordered by count descending."""
    return ApproxTopK(_kv_dft("APPROX_TOPK", of), _check_k("approx_topk", k))


@overload
def sample(k: int) -> "Sample[Str]": ...
@overload
def sample(k: int, of: "_Type[T]") -> "Sample[T]": ...
def sample(k: int, of: "_Type[object]" = str_) -> "Sample[object]":
    """A deterministic mergeable sample: keep k DISTINCT items by smallest
    hash(item) (bottom-k / KMV), not an Algorithm-R reservoir.

    ``of=jit.str_`` (default) samples interned string ids; ``of=jit.i64`` samples
    raw int64 items. Feeding a unique per-row item makes it a uniform row sample.
    Contribute with ``.observe(item)``; materializes to a sorted list column."""
    return Sample(_kv_dft("SAMPLE", of), _check_k("sample", k))


def _argrow_payload(op: str, of: object) -> "Tuple[_Type[object], ...]":
    if not isinstance(of, tuple) or not of:
        raise JitError(
            f"jit.{op}(of=...) must be a non-empty tuple of jit types such as "
            "(jit.str_, jit.i64, jit.f64)"
        )
    for t in of:
        if not isinstance(t, _Type):
            raise JitError(
                f"jit.{op}(of=...) components must be jit types (jit.str_, jit.i64, ...)"
            )
    return tuple(of)


def argmin_row(of: "Tuple[_Type[object], ...]") -> "ArgMinRow":
    """An argmin-row value (full-row min-by, DISTINCT ON): keep the ENTIRE payload
    row - a fixed tuple of typed components ``of=(jit.str_, jit.i64, ...)`` - from
    the contribution whose ``by=`` key is smallest. Materializes to one column per
    payload component (p0..p{n-1}). Contribute with
    ``.observe((p0, p1, ...), by=<expr>)``."""
    return ArgMinRow(_argrow_payload("argmin_row", of))


def argmax_row(of: "Tuple[_Type[object], ...]") -> "ArgMaxRow":
    """An argmax-row value (full-row max-by, DISTINCT ON): keep the ENTIRE payload
    row at the largest ``by=`` key, mirroring :func:`argmin_row`. Contribute with
    ``.observe((p0, p1, ...), by=<expr>)``."""
    return ArgMaxRow(_argrow_payload("argmax_row", of))


class Nested(Generic[IK, V]):
    """A nested-preserved map value: at each outer key, a sub-map inner_key -> V.

    Runtime decl for a :func:`nested` value and the authoring-only type of a
    nested outer-key lookup; ``[inner_key]`` yields the inner accumulator ``V``.
    Its inner values must be a scalar monoid or a product of them (collections
    and ordered lists are rejected, matching the engine)."""

    __slots__ = ("inner_key_types", "values", "is_product", "value_names")

    def __init__(
        self,
        inner_key_types: "Tuple[_Type[object], ...]",
        values: Tuple[_Monoid, ...],
        is_product: bool,
        value_names: Tuple[str, ...] | None = None,
    ) -> None:
        self.inner_key_types = inner_key_types
        self.values = values
        self.is_product = is_product
        self.value_names = value_names

    def __getitem__(self, inner_key: IK) -> V:
        raise NotImplementedError

    def __setitem__(self, inner_key: IK, value: V) -> None: ...


class Map(Generic[K, V]):
    __slots__ = ("key_types", "values", "is_product", "value_names", "ordered", "nested", "argrow")

    def __init__(
        self,
        key_types: "Tuple[_Type[object], ...]",
        values: Tuple[_Monoid, ...],
        is_product: bool,
        value_names: Tuple[str, ...] | None = None,
        ordered: bool = False,
        nested: "Nested[Any, Any] | None" = None,
        argrow: "ArgMinRow | ArgMaxRow | None" = None,
    ) -> None:
        self.key_types = key_types
        self.values = values
        self.is_product = is_product
        self.value_names = value_names
        self.ordered = ordered
        self.nested = nested
        self.argrow = argrow

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


@overload
def nested(*, key: "Tuple[_Type[K1]]", value: V) -> "Nested[Tuple[K1], V]": ...
@overload
def nested(*, key: "Tuple[_Type[K1], _Type[K2]]", value: V) -> "Nested[Tuple[K1, K2], V]": ...
@overload
def nested(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]", value: V
) -> "Nested[Tuple[K1, K2, K3], V]": ...
@overload
def nested(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]", value: V
) -> "Nested[Tuple[K1, K2, K3, K4], V]": ...
@overload
def nested(*, key: "Tuple[_Type[object], ...]", value: V) -> "Nested[Tuple[object, ...], V]": ...
def nested(key: "Tuple[_Type[object], ...]", value: object) -> "Nested[object, object]":
    """Declare a nested-preserved map value: an inner typed-tuple key and a scalar
    value monoid (or a tuple/dict/``@jit.record`` product of them).

    Pass the result as a :func:`map` ``value`` to keep the outer key's sub-map as
    one ``list<struct<inner_keys.., values..>>`` column instead of flattening it
    into the row key. Inner values must be materializable scalars; a set/list
    collection is rejected here, mirroring the engine."""
    if not isinstance(key, tuple) or not key:
        raise JitError("nested key must be a non-empty tuple of jit.i64 / jit.str_")
    for k in key:
        if not isinstance(k, _Type):
            raise JitError("nested key components must be jit.i64 or jit.str_")
    values, is_product, value_names = _value_spec(value)
    for m in values:
        if m.dft in _SET_MONOIDS or m.dft in _LIST_MONOIDS:
            raise JitError(
                "jit.nested value cannot be a set/list collection; inner values must be "
                "scalar (count/sum/min/max/distinct/mean/variance/stddev) or a product of them"
            )
        if m.dft in _ARGBY_MONOIDS:
            raise JitError(
                "jit.nested value cannot be an argmin/argmax; there is no nested argby add"
            )
        if m.dft in _TOPK_MONOIDS or m.dft in _APPROX_MONOIDS or m.dft in _SAMPLE_MONOIDS:
            raise JitError(
                "jit.nested value cannot be a top-k/approx_top-k/sample collection; inner "
                "values must be scalar or a product of them"
            )
        if m.dft in _ARGROW_MONOIDS:
            raise JitError(
                "jit.nested value cannot be an argmin_row/argmax_row; it is created only via "
                "map_new_argrow"
            )
    return Nested(key, values, is_product, value_names)


# A dict/tuple of monoids is a product (value type Product); these precede the
# generic `value: V` forms so it is not typed as a bare dict/tuple.
@overload
def map(
    *, key: "_Type[K1]", value: "Dict[str, _Monoid]", ordered: bool = False
) -> "Map[Tuple[K1], Product]": ...
@overload
def map(
    *, key: "_Type[K1]", value: "Tuple[_Monoid, ...]", ordered: bool = False
) -> "Map[Tuple[K1], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1]]", value: "Dict[str, _Monoid]", ordered: bool = False
) -> "Map[Tuple[K1], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1]]", value: "Tuple[_Monoid, ...]", ordered: bool = False
) -> "Map[Tuple[K1], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2]]", value: "Dict[str, _Monoid]", ordered: bool = False
) -> "Map[Tuple[K1, K2], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2]]", value: "Tuple[_Monoid, ...]", ordered: bool = False
) -> "Map[Tuple[K1, K2], Product]": ...
@overload
def map(
    *,
    key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]",
    value: "Dict[str, _Monoid]",
    ordered: bool = False,
) -> "Map[Tuple[K1, K2, K3], Product]": ...
@overload
def map(
    *,
    key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]",
    value: "Tuple[_Monoid, ...]",
    ordered: bool = False,
) -> "Map[Tuple[K1, K2, K3], Product]": ...
@overload
def map(
    *,
    key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]",
    value: "Dict[str, _Monoid]",
    ordered: bool = False,
) -> "Map[Tuple[K1, K2, K3, K4], Product]": ...
@overload
def map(
    *,
    key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]",
    value: "Tuple[_Monoid, ...]",
    ordered: bool = False,
) -> "Map[Tuple[K1, K2, K3, K4], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[object], ...]", value: "Dict[str, _Monoid]", ordered: bool = False
) -> "Map[Tuple[object, ...], Product]": ...
@overload
def map(
    *, key: "Tuple[_Type[object], ...]", value: "Tuple[_Monoid, ...]", ordered: bool = False
) -> "Map[Tuple[object, ...], Product]": ...
@overload
def map(*, key: "_Type[K1]", value: V, ordered: bool = False) -> "Map[Tuple[K1], V]": ...
@overload
def map(*, key: "Tuple[_Type[K1]]", value: V, ordered: bool = False) -> "Map[Tuple[K1], V]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2]]", value: V, ordered: bool = False
) -> "Map[Tuple[K1, K2], V]": ...
@overload
def map(
    *, key: "Tuple[_Type[K1], _Type[K2], _Type[K3]]", value: V, ordered: bool = False
) -> "Map[Tuple[K1, K2, K3], V]": ...
@overload
def map(
    *,
    key: "Tuple[_Type[K1], _Type[K2], _Type[K3], _Type[K4]]",
    value: V,
    ordered: bool = False,
) -> "Map[Tuple[K1, K2, K3, K4], V]": ...
@overload
def map(
    *, key: "Tuple[_Type[object], ...]", value: V, ordered: bool = False
) -> "Map[Tuple[object, ...], V]": ...
def map(
    key: "Tuple[_Type[object], ...] | _Type[object]", value: object, ordered: bool = False
) -> "Map[object, object]":
    """Declare a mergeable map class-attribute: a typed-tuple key, and either one
    value monoid, a tuple of monoids (a positional product), a dict of
    name->monoid, or a :func:`record` class (both named products).

    A single-component key may be passed bare (``key=jit.i64``); it normalizes to
    the 1-tuple ``(jit.i64,)`` and the body may then use a bare subscript
    ``self.m[k]`` as well as ``self.m[(k,)]``. Multi-key maps stay tuple-only.

    With ``ordered=True`` the map's result rows materialize sorted by key in the
    native engine (I64 components by value, STR components by resolved label);
    default is unordered."""
    if isinstance(key, _Type):
        key = (key,)
    if not isinstance(key, tuple) or not key:
        raise JitError("map key must be a non-empty tuple of jit.i64 / jit.str_")
    for k in key:
        if not isinstance(k, _Type):
            raise JitError("map key components must be jit.i64 or jit.str_")
    if isinstance(value, Nested):
        return _MapDecl(key, value.values, value.is_product, value.value_names, nested=value)
    if isinstance(value, (ArgMinRow, ArgMaxRow)):
        return _MapDecl(key, (value,), False, None, argrow=value)
    values, is_product, value_names = _value_spec(value)
    if is_product and any(v.dft == "DFTU_MONOID_SKETCH" for v in values):
        raise JitError("jit.quantiles is top-level only; it cannot be a product/record component")
    return _MapDecl(key, values, is_product, value_names, ordered=ordered)


_JOIN_TYPES: Dict[str, str] = {
    "inner": "DFTU_JOIN_INNER",
    "left": "DFTU_JOIN_LEFT",
    "right": "DFTU_JOIN_RIGHT",
    "full": "DFTU_JOIN_FULL",
}


class JoinDecl(Generic[K, V, K2, V2]):
    __slots__ = ("left", "right", "how")

    def __init__(self, left: "Map[K, V]", right: "Map[K2, V2]", how: str) -> None:
        self.left = left
        self.right = right
        self.how = how


def join(
    left: "Map[K, V]",
    right: "Map[K2, V2]",
    *,
    how: Literal["inner", "left", "right", "full"] = "inner",
) -> "JoinDecl[K, V, K2, V2]":
    """Declare a class-level equi-join of two of this plugin's maps on their shared
    key tuple; the host runs it at finalize and emits the result under this
    attribute's name. ``how`` is one of inner (default), left, right, full."""
    if not isinstance(left, _MapDecl) or not isinstance(right, _MapDecl):
        raise JitError("jit.join(left, right) operands must be jit.map declarations")
    if how not in _JOIN_TYPES:
        raise JitError("jit.join how must be one of inner / left / right / full")
    return JoinDecl(left, right, how)


class Port(Protocol):
    """Authoring-only handle for a batch-scoped inter-plugin port.

    A :func:`publish` port is written per event with ``self.<port> += <expr>`` and
    the host publishes the per-batch total for a later plugin's consume port of the
    same capability id; a :func:`consume` port reads that value as a plain scalar
    (``self.<port>``, 0 when no producer published this batch). Never instantiated."""

    def __iadd__(self, x: float) -> "Port": ...


_PORT_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._-")


class _Port:
    __slots__ = ("cap_id", "is_f64", "role", "required", "ver", "ver_op")

    def __init__(
        self,
        cap_id: str,
        is_f64: bool,
        role: str,
        required: bool,
        ver: Tuple[int, int, int] = (0, 0, 0),
        ver_op: str | None = None,
    ) -> None:
        self.cap_id = cap_id
        self.is_f64 = is_f64
        self.role = role
        self.required = required
        # publish: `ver` is the provided semantic version, `ver_op` is None.
        # consume: `ver`/`ver_op` are the version constraint on a provider.
        self.ver = ver
        self.ver_op = ver_op


def _check_cap_id(fn: str, cap_id: object) -> str:
    if not isinstance(cap_id, str) or not cap_id:
        raise JitError(
            f"jit.{fn} needs a non-empty capability id string such as 'com.example.edges'"
        )
    if cap_id.startswith("dftu."):
        raise JitError(
            f"jit.{fn} capability id '{cap_id}' uses the reserved dftu. namespace; pick your own, "
            "e.g. 'com.example.edges'"
        )
    if any(c not in _PORT_ID_CHARS for c in cap_id):
        raise JitError(f"jit.{fn} capability id '{cap_id}' must be ASCII [a-z0-9._-]")
    return cap_id


def _port_is_f64(fn: str, of: "_Type[object]") -> bool:
    if of is f64:
        return True
    if of is u64 or of is i64:
        return False
    raise JitError(f"jit.{fn}(of=...) must be jit.u64, jit.i64, or jit.f64")


# Version-constraint operator strings the require side accepts, mapped to the
# DFTU_VER_* enum the ABI defines in abi.h.
_VER_OP_ENUM: Dict[str, str] = {
    ">=": "DFTU_VER_GE",
    ">": "DFTU_VER_GT",
    "<=": "DFTU_VER_LE",
    "<": "DFTU_VER_LT",
    "=": "DFTU_VER_EQ",
    "^": "DFTU_VER_CARET",
    "~": "DFTU_VER_TILDE",
}


def _parse_version(fn: str, s: object) -> Tuple[int, int, int]:
    """Parse a ``MAJOR[.MINOR[.PATCH]]`` string into a 3-tuple, missing
    components defaulting to 0. Each component must be a non-negative integer at
    most 65535 (the ABI stores them as uint16)."""
    if not isinstance(s, str) or not s:
        raise JitError(f"jit.{fn} version must be a non-empty 'MAJOR.MINOR.PATCH' string")
    parts = s.split(".")
    if len(parts) > 3:
        raise JitError(f"jit.{fn} version '{s}' has too many components; use MAJOR[.MINOR[.PATCH]]")
    nums = [0, 0, 0]
    for i, p in enumerate(parts):
        if not p.isdigit():
            raise JitError(f"jit.{fn} version '{s}' component '{p}' is not a non-negative integer")
        v = int(p)
        if v > 0xFFFF:
            raise JitError(f"jit.{fn} version '{s}' component {v} exceeds 65535")
        nums[i] = v
    return (nums[0], nums[1], nums[2])


def publish(cap_id: str, of: "_Type[object]" = u64, *, version: str = "0.0.0") -> Port:
    """Declare a batch-scoped publish port under capability ``cap_id``.

    The plugin PROVIDES ``cap_id`` at semantic ``version`` (``MAJOR.MINOR.PATCH``,
    default ``0.0.0``); the host orders it before any consumer of the same id and
    exposes the version to a consumer's version constraint. In ``each_event``
    accumulate a per-batch total with ``self.<port> += <expr>`` (a u64 sum by
    default, ``of=jit.f64`` for a double sum). The total is published once per
    batch and reset for the next batch; ``of`` picks the wire width (u64/i64 as an
    8-byte int, f64 as a double)."""
    _check_cap_id("publish", cap_id)
    ver = _parse_version("publish", version)
    return cast(Port, _Port(cap_id, _port_is_f64("publish", of), "publish", False, ver, None))


def consume(
    cap_id: str,
    of: "_Type[object]" = u64,
    *,
    required: bool = False,
    min_version: str | None = None,
    version_op: str | None = None,
) -> Port:
    """Declare a batch-scoped consume port for capability ``cap_id``.

    The plugin REQUIRES ``cap_id``; the host wires a producer of it to run first.
    In ``each_event`` read the value a producer published for the current batch as
    a plain scalar ``self.<port>`` (0 when no producer published this batch). ``of``
    must match the producer's width (u64/i64/f64). With ``required=True`` a missing
    producer fails :meth:`resolve`; the default degrades to reading 0.

    ``min_version`` adds a version constraint on the provider (default: any
    version). It is a ``>=`` constraint unless ``version_op`` overrides the
    operator (one of ``>=`` ``>`` ``<=`` ``<`` ``=`` ``^`` ``~``, matching the
    ABI's ``DFTU_VER_*`` ops). A ``@jit.on_resolve`` method can inspect whether a
    provider satisfying this constraint was found and at what version."""
    _check_cap_id("consume", cap_id)
    if min_version is None:
        if version_op is not None:
            raise JitError("jit.consume version_op requires min_version")
        ver = (0, 0, 0)
        op = "DFTU_VER_GE"
    else:
        ver = _parse_version("consume", min_version)
        if version_op is None:
            op = "DFTU_VER_GE"
        else:
            op = _VER_OP_ENUM.get(version_op, "")
            if not op:
                raise JitError("jit.consume version_op must be one of >= > <= < = ^ ~")
    return cast(
        Port, _Port(cap_id, _port_is_f64("consume", of), "consume", bool(required), ver, op)
    )


class Handle(Protocol):
    """Authoring-only handle for a cross-worker mergeable shared value.

    A :func:`shared` handle accumulates one scalar monoid across every worker
    slice: contribute in ``each_event`` with ``self.<handle>.add(<expr>)`` (or
    ``self.<handle> += <expr>`` for the additive count/sum kinds). The host merges
    same-cap handles across all slices; at scan end the plugin's generated
    finalize reads the merged value and emits it under the attribute's name, so
    :meth:`PluginHost.run` returns it (8 raw little-endian bytes: a ``uint64`` for
    an integer kind, a ``double`` for a float kind). Never instantiated."""

    def add(self, x: float) -> None: ...
    def __iadd__(self, x: float) -> "Handle": ...


# Monoid kinds the DFTU_EXT_HANDLES ABI can accumulate and reduce to one 8-byte
# scalar: add_u64 kinds (result read as uint64) and add_f64 kinds (result read as
# double). SKETCH is a handle kind too but its result is a quantiles struct, not
# a scalar, so it is not offered here.
_HANDLE_U64_KINDS = frozenset(
    {
        "DFTU_MONOID_COUNTER",
        "DFTU_MONOID_MIN_U64",
        "DFTU_MONOID_MAX_U64",
        "DFTU_MONOID_DISTINCT",
    }
)

_HANDLE_F64_KINDS = frozenset(
    {
        "DFTU_MONOID_SUM_F64",
        "DFTU_MONOID_MIN_F64",
        "DFTU_MONOID_MAX_F64",
    }
)


class _Shared:
    __slots__ = ("cap_id", "kind", "is_f64")

    def __init__(self, cap_id: str, kind: str, is_f64: bool) -> None:
        self.cap_id = cap_id
        self.kind = kind
        self.is_f64 = is_f64


def shared(cap_id: str, monoid: object) -> Handle:
    """Declare a cross-worker mergeable shared handle under capability ``cap_id``.

    ``monoid`` is one of :func:`count`, :func:`sum`, :func:`distinct`,
    :func:`min` / :func:`max` (u64), or :func:`minf` / :func:`maxf` (f64) - the
    scalar monoids the host can merge across workers into one value. In
    ``each_event`` contribute with ``self.<handle>.add(<expr>)`` (count and sum
    also accept ``self.<handle> += <expr>``). The merged value is emitted under
    the attribute's name at finalize; ``run()`` returns it as 8 raw bytes (a
    little-endian ``uint64`` for an integer kind, a ``double`` for a float kind).
    """
    _check_cap_id("shared", cap_id)
    if callable(monoid) and not isinstance(monoid, _Monoid):
        monoid = cast(Callable[[], object], monoid)()
    if not isinstance(monoid, _Monoid):
        raise JitError(
            "jit.shared monoid must be a scalar monoid such as jit.count(), jit.sum(), "
            "jit.distinct(), jit.min()/jit.max() (u64), or jit.minf()/jit.maxf() (f64)"
        )
    dft = monoid.dft
    if dft in _HANDLE_U64_KINDS:
        is_f64 = False
    elif dft in _HANDLE_F64_KINDS:
        is_f64 = True
    else:
        raise JitError(
            f"jit.shared does not support the {dft} monoid; a shared handle merges one "
            "scalar across workers - use jit.count(), jit.sum(), jit.distinct(), "
            "jit.min()/jit.max() (u64 only), or jit.minf()/jit.maxf() (f64)"
        )
    return cast(Handle, _Shared(cap_id, dft, is_f64))


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
    into the per-event loop; available C names are ``host``, ``map``, each
    declared map handle by its attribute name, ``e``, ``b``, and ``i``."""
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


class _OnResolve:
    __slots__ = ("fn",)

    def __init__(self, fn: Callable[..., object]) -> None:
        self.fn = fn


def on_resolve(fn: Callable[..., object]) -> _OnResolve:
    """Mark the one resolve-time method to AST-compile into ``comms_resolve``.

    The method takes ``(self)`` and runs once, after every plugin has declared,
    to adapt behavior to whichever providers are present. Its body is a sequence
    of ``self.<flag> = <expr>`` assignments; each ``<flag>`` becomes a plugin
    state value a later ``each_event`` reads as ``self.<flag>``. Expressions may
    read, for each :func:`consume` port on the class:

    - ``self.<port>.resolved`` - 1 if a provider satisfying the port's version
      constraint was found at resolve time, else 0;
    - ``self.<port>.version`` - the found provider's version, comparable to a
      ``(major, minor, patch)`` tuple literal (0.0.0 when unresolved);

    combined with ``and`` / ``or`` / ``not``, integer comparisons, and integer
    literals. The host always populates each port's ``resolved``/``version``
    before this body runs, so an empty body still records provider presence."""
    return _OnResolve(fn)


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

_F64_MONOIDS = frozenset(
    {
        "DFTU_MONOID_SUM_F64",
        "DFTU_MONOID_MIN_F64",
        "DFTU_MONOID_MAX_F64",
        "DFTU_MONOID_MIN_F32",
        "DFTU_MONOID_MAX_F32",
        "DFTU_MONOID_MEAN",
        "DFTU_MONOID_VARIANCE",
        "DFTU_MONOID_STDDEV",
        "DFTU_MONOID_SKETCH",
    }
)

_ARGBY_MONOIDS = frozenset(
    {
        "DFTU_MONOID_ARGMIN_I64",
        "DFTU_MONOID_ARGMAX_I64",
        "DFTU_MONOID_ARGMIN_STR",
        "DFTU_MONOID_ARGMAX_STR",
    }
)

_ARGBY_STR = frozenset({"DFTU_MONOID_ARGMIN_STR", "DFTU_MONOID_ARGMAX_STR"})

_ARGBY_I64 = frozenset({"DFTU_MONOID_ARGMIN_I64", "DFTU_MONOID_ARGMAX_I64"})

_TOPK_MONOIDS = frozenset(
    {
        "DFTU_MONOID_TOPK_I64",
        "DFTU_MONOID_TOPK_STR",
        "DFTU_MONOID_BOTTOMK_I64",
        "DFTU_MONOID_BOTTOMK_STR",
    }
)

_APPROX_MONOIDS = frozenset({"DFTU_MONOID_APPROX_TOPK_I64", "DFTU_MONOID_APPROX_TOPK_STR"})

_SAMPLE_MONOIDS = frozenset({"DFTU_MONOID_SAMPLE_I64", "DFTU_MONOID_SAMPLE_STR"})

_ARGROW_MONOIDS = frozenset({"DFTU_MONOID_ARGMIN_ROW", "DFTU_MONOID_ARGMAX_ROW"})

_KV_STR = frozenset(
    {
        "DFTU_MONOID_TOPK_STR",
        "DFTU_MONOID_BOTTOMK_STR",
        "DFTU_MONOID_APPROX_TOPK_STR",
        "DFTU_MONOID_SAMPLE_STR",
    }
)

_KV_I64 = frozenset(
    {
        "DFTU_MONOID_TOPK_I64",
        "DFTU_MONOID_BOTTOMK_I64",
        "DFTU_MONOID_APPROX_TOPK_I64",
        "DFTU_MONOID_SAMPLE_I64",
    }
)

_ADDITIVE_MONOIDS = frozenset({"DFTU_MONOID_COUNTER", "DFTU_MONOID_SUM_F64"})

_SET_MONOIDS = frozenset({"DFTU_MONOID_SET_STR", "DFTU_MONOID_SET_I64"})

_LIST_MONOIDS = frozenset({"DFTU_MONOID_LIST_STR", "DFTU_MONOID_LIST_I64"})

_STR_COLLECTIONS = frozenset({"DFTU_MONOID_SET_STR", "DFTU_MONOID_LIST_STR"})

_I64_COLLECTIONS = frozenset({"DFTU_MONOID_SET_I64", "DFTU_MONOID_LIST_I64"})

_FLOAT_KEYS = frozenset({"DFTU_T_F32", "DFTU_T_F64"})

# Monoids whose contribution is not a single u64/f64 add, so a map holding one
# cannot be coalesced into a fused map (which drives components via map_add_row).
_NON_FUSABLE_MONOIDS = frozenset(
    _ARGBY_MONOIDS
    | _TOPK_MONOIDS
    | _APPROX_MONOIDS
    | _SAMPLE_MONOIDS
    | _ARGROW_MONOIDS
    | _SET_MONOIDS
    | _LIST_MONOIDS
    | {"DFTU_MONOID_SKETCH"}
)

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


class _FusedGroup(TypedDict):
    """One coalesced group of same-key maps fused into a single product."""

    cname: str
    key_elts: List[ast.expr]
    key_types: "Tuple[_Type[object], ...]"
    members: List[str]
    ordered: bool


class _Compiler:
    def __init__(
        self,
        maps: Dict[str, _MapDecl],
        joins: builtins.set[str] | None = None,
        no_fuse: builtins.set[str] | None = None,
        ops: Dict[str, Op] | None = None,
        ports: Dict[str, _Port] | None = None,
        shared: Dict[str, _Shared] | None = None,
        resolve_flags: builtins.set[str] | None = None,
    ) -> None:
        self.maps = maps
        self.joins = joins if joins is not None else builtins.set()
        self.no_fuse = no_fuse if no_fuse is not None else builtins.set()
        self.ports = ports if ports is not None else {}
        self.shared = shared if shared is not None else {}
        # Flags a @jit.on_resolve method sets; each_event reads them as
        # self.<flag>, lowering to the file-scope _rflag_<flag> static.
        self.resolve_flags = resolve_flags if resolve_flags is not None else builtins.set()
        # Referenced @jit.op transforms, inlined into the plugin as static C
        # functions the moment the body calls one.
        self.ops = ops if ops is not None else {}
        self.op_defs: List[str] = []
        self.op_emitted: builtins.set[str] = builtins.set()
        self.needs: builtins.set[str] = builtins.set()
        self.funcs: builtins.set[str] = builtins.set()
        self.str_literals: List[str] = []
        self.arg_keys: List[str] = []
        self.arg_helpers: builtins.set[str] = builtins.set()
        self.self_name = "self"
        self.event_name = "e"
        # CSE: ast.dump(node) -> (local name, is_float) for a hoisted expression
        # shared within the current scope.
        self._cse: Dict[str, Tuple[str, bool]] = {}
        self._cse_n = 0
        # Fusion plan: same-key maps coalesced into one product.
        # fused_of: map name -> (fused C name, component index, is_f64).
        self.fused_of: Dict[str, Tuple[str, int, bool]] = {}
        self.fused_groups: List[_FusedGroup] = []
        # Per-scope batched contributions: fused C name -> [(comp, is_f64, rhs)].
        self._fused_pending: Dict[str, List[Tuple[int, bool, str]]] = {}

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
        self._plan_fusion(fn_ast.body)
        defs, lines = self._compile_scope(fn_ast.body)
        return defs + lines

    # Coalesce plain single-value maps that are always subscripted at the same
    # key into one product, so their per-event updates share one hash lookup.
    # The product still materializes as the separate maps the user declared.
    def _plan_fusion(self, stmts: List[ast.stmt]) -> None:
        sigs: Dict[str, builtins.set[str]] = {}
        key_of: Dict[str, List[ast.expr]] = {}

        def visit(stmt_list: List[ast.stmt]) -> None:
            for st in stmt_list:
                if isinstance(st, ast.If):
                    visit(st.body)
                for node in ast.walk(st):
                    if not (
                        isinstance(node, ast.Subscript)
                        and isinstance(node.value, ast.Attribute)
                        and isinstance(node.value.value, ast.Name)
                        and node.value.value.id == self.self_name
                    ):
                        continue
                    attr = node.value.attr
                    decl = self.maps.get(attr)
                    if decl is None or attr in self.no_fuse:
                        continue
                    # Only plain single-value maps with a fusable monoid coalesce.
                    if decl.nested is not None or decl.argrow is not None or decl.is_product:
                        continue
                    if decl.values[0].dft in _NON_FUSABLE_MONOIDS:
                        continue
                    elts = self._slice_elts(node.slice)
                    sigs.setdefault(attr, builtins.set()).add("|".join(ast.dump(e) for e in elts))
                    key_of[attr] = elts

        visit(stmts)
        # Group maps with a single, identical key signature, key types, and
        # ordered flag (a differing flag would change one member's row order).
        groups: Dict[Tuple[str, Tuple[str, ...], bool], List[str]] = {}
        for attr, sset in sigs.items():
            if len(sset) != 1:
                continue  # a map keyed inconsistently cannot fuse
            decl = self.maps[attr]
            kt = tuple(k.dft for k in decl.key_types)
            groups.setdefault((next(iter(sset)), kt, decl.ordered), []).append(attr)
        for gi, ((_, _, ordered), members) in enumerate(sorted(groups.items())):
            if len(members) < 2:
                continue
            members.sort()  # deterministic component order
            cname = f"fused{gi}"
            self.fused_groups.append(
                {
                    "cname": cname,
                    "key_elts": key_of[members[0]],
                    "key_types": self.maps[members[0]].key_types,
                    "members": members,
                    "ordered": ordered,
                }
            )
            for ci, a in enumerate(members):
                is_f64 = self.maps[a].values[0].dft in _F64_MONOIDS
                self.fused_of[a] = (cname, ci, is_f64)

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
        # Fused contributions collected while compiling this scope's statements
        # flush to one map_add_row per group at the scope's end.
        outer_pending = self._fused_pending
        self._fused_pending = {}
        lines: List[str] = []
        for stmt in stmts:
            lines.extend(self._stmt(stmt))
        lines.extend(self._flush_fused_rows())
        self._fused_pending = outer_pending
        self._cse = outer
        return defs, lines

    def _flush_fused_rows(self) -> List[str]:
        out: List[str] = []
        for group in self.fused_groups:
            cname = group["cname"]
            pending = self._fused_pending.get(cname)
            if not pending:
                continue
            self.funcs.add("map_add_row")
            n = len(group["key_types"])
            block = ["{", f"    int64_t key[{n}];"]
            block.extend(self._key_assigns(group["key_types"], group["key_elts"]))
            k = len(pending)
            block.append(f"    dftu_row_val _row[{k}];")
            for i, (comp, is_f64, rhs) in enumerate(pending):
                block.append(f"    _row[{i}].comp = {comp}u;")
                block.append(f"    _row[{i}].is_f64 = {1 if is_f64 else 0};")
                slot = "f" if is_f64 else "u"
                block.append(f"    _row[{i}].value.{slot} = {rhs};")
            block.append(f"    map->map_add_row(host->h, m_{cname}, key, _row, {k}u);")
            block.append("}")
            out.extend(block)
        return out

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
            if isinstance(func, ast.Attribute) and func.attr == "add":
                sh = self._shared_target(func.value)
                if sh is not None:
                    call = stmt.value
                    if len(call.args) != 1 or call.keywords:
                        _reject(f"self.{sh}.add expects exactly one value argument")
                    return self._shared_add(sh, call.args[0])
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

    def _resolve_accum(
        self, target: ast.expr
    ) -> Tuple[str, List[ast.expr], int | None, _Monoid, List[ast.expr] | None]:
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
            if base.attr in self.joins:
                _reject(
                    f"self.{base.attr} is a jit.join result computed at finalize; "
                    "observe the input maps, not the join"
                )
            _reject(f"unknown map self.{base.attr}")
        if decl.nested is not None:
            return self._resolve_nested(base.attr, decl, subs, comp_name)
        return self._resolve_flat(base.attr, decl, subs, comp_name)

    def _resolve_flat(
        self, attr: str, decl: _MapDecl, subs: List[ast.expr], comp_name: str | None
    ) -> Tuple[str, List[ast.expr], int | None, _Monoid, None]:
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
            monoid = decl.values[comp]
        else:
            if comp_name is not None:
                _reject(f"single-value map self.{attr} add with a component")
            keys = []
            for sl in subs:
                keys.extend(self._keys(sl))
            comp = None
            monoid = decl.values[0]
        if len(keys) != len(decl.key_types):
            _reject(f"self.{attr} takes {len(decl.key_types)} key components, got {len(keys)}")
        return attr, keys, comp, monoid, None

    def _resolve_nested(
        self, attr: str, decl: _MapDecl, subs: List[ast.expr], comp_name: str | None
    ) -> Tuple[str, List[ast.expr], int | None, _Monoid, List[ast.expr]]:
        assert decl.nested is not None
        if len(subs) < 2:
            _reject(f"nested map self.{attr} needs [outer][inner] subscripts")
        outer_keys = self._keys(subs[0])
        inner_keys = self._keys(subs[1])
        comp_node = subs[2] if len(subs) > 2 else None
        if len(subs) > 3:
            _reject(f"too many subscripts for nested map self.{attr}")
        outer_n = len(decl.key_types)
        inner_n = len(decl.nested.inner_key_types)
        if len(outer_keys) != outer_n:
            _reject(f"self.{attr} takes {outer_n} outer key components, got {len(outer_keys)}")
        if len(inner_keys) != inner_n:
            _reject(f"self.{attr} takes {inner_n} inner key components, got {len(inner_keys)}")
        if decl.is_product:
            if comp_name is not None:
                comp = self._named_comp(comp_name, decl, attr)
            elif comp_node is not None:
                comp = self._comp(comp_node, decl, attr)
            else:
                _reject(f"nested product map self.{attr} add without a component")
            monoid = decl.values[comp]
        else:
            if comp_name is not None or comp_node is not None:
                _reject(f"single-value nested map self.{attr} add with a component")
            comp = 0
            monoid = decl.values[0]
        return attr, outer_keys, comp, monoid, inner_keys

    def _key_assigns(
        self,
        key_types: "Tuple[_Type[object], ...]",
        keys: List[ast.expr],
        var: str = "key",
    ) -> List[str]:
        # A float key component is bit-cast into its int64 key slot (a numeric
        # expr, so float arithmetic is allowed); other keys take an i64 expr.
        lines: List[str] = []
        for idx, kn in enumerate(keys):
            ktype = key_types[idx].dft
            if ktype == "DFTU_T_BYTES":
                _reject("a bytes key; e.<field> exposes no raw byte blob, author it raw")
            if ktype == "DFTU_T_F64":
                expr, _ = self._arith(kn)
                lines.append(
                    f"    {{ double _t = (double)({expr}); std::memcpy(&{var}[{idx}], &_t, 8); }}"
                )
            elif ktype == "DFTU_T_F32":
                expr, _ = self._arith(kn)
                lines.append(
                    f"    {{ float _t = (float)({expr}); uint32_t _u; "
                    f"std::memcpy(&_u, &_t, 4); {var}[{idx}] = (int64_t)_u; }}"
                )
            else:
                lines.append(f"    {var}[{idx}] = (int64_t)({self._value(kn)});")
        return lines

    def _emit_add(
        self,
        attr: str,
        keys: List[ast.expr],
        comp: int | None,
        monoid: _Monoid,
        value: ast.expr,
        inner_keys: List[ast.expr] | None = None,
    ) -> List[str]:
        float_target = monoid.dft in _F64_MONOIDS
        rhs = self._rhs(value, float_target)
        # A fused map's contribution is deferred and batched into one map_add_row
        # per scope (built in _compile_scope), sharing the group's single lookup.
        fused = self.fused_of.get(attr) if inner_keys is None else None
        if fused is not None:
            cname, comp_idx, is_f64 = fused
            self._fused_pending.setdefault(cname, []).append((comp_idx, is_f64, rhs))
            return []
        var = f"m_{attr}"
        decl = self.maps[attr]
        if inner_keys is not None:
            assert decl.nested is not None
            fn = "map_add_nested_f64" if float_target else "map_add_nested_u64"
            self.funcs.add(fn)
            lines = [
                "{",
                f"    int64_t okey[{len(keys)}];",
                f"    int64_t ikey[{len(inner_keys)}];",
            ]
            lines.extend(self._key_assigns(decl.key_types, keys, "okey"))
            lines.extend(self._key_assigns(decl.nested.inner_key_types, inner_keys, "ikey"))
            lines.append(f"    map->{fn}(host->h, {var}, okey, ikey, {comp}, {rhs});")
            lines.append("}")
            return lines
        lines = ["{", f"    int64_t key[{len(keys)}];"]
        lines.extend(self._key_assigns(decl.key_types, keys))
        if comp is None:
            fn = "map_add_f64" if float_target else "map_add_u64"
            self.funcs.add(fn)
            lines.append(f"    map->{fn}(host->h, {var}, key, {rhs});")
        else:
            fn = "map_add_f64_at" if float_target else "map_add_u64_at"
            self.funcs.add(fn)
            lines.append(f"    map->{fn}(host->h, {var}, key, {comp}, {rhs});")
        lines.append("}")
        return lines

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
        rhs = self._rhs(value, port.is_f64)
        return [f"_pub_{name} += {rhs};"]

    def _consume_ref(self, name: str) -> str:
        port = self.ports[name]
        if port.role != "consume":
            _reject(f"self.{name} is a jit.publish port; publish into it with +=, do not read it")
        return f"_sub_{name}"

    def _shared_target(self, target: ast.expr) -> str | None:
        if (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == self.self_name
            and target.attr in self.shared
        ):
            return target.attr
        return None

    def _shared_add(self, name: str, value: ast.expr) -> List[str]:
        sh = self.shared[name]
        rhs = self._rhs(value, sh.is_f64)
        if sh.is_f64:
            return [f"if (_hd_{name}) _handles->add_f64(host->h, _hd_{name}, {rhs}, 1.0);"]
        return [f"if (_hd_{name}) _handles->add_u64(host->h, _hd_{name}, {rhs});"]

    def _shared_aug(self, name: str, value: ast.expr) -> List[str]:
        sh = self.shared[name]
        if sh.kind not in _ADDITIVE_MONOIDS:
            _reject(
                f"+= on the non-additive shared handle self.{name} ({sh.kind}); "
                "use self." + name + ".add(<expr>)"
            )
        return self._shared_add(name, value)

    def _aug(self, stmt: ast.AugAssign) -> List[str]:
        if not isinstance(stmt.op, ast.Add):
            _reject("augmented operator other than +=")
        pub = self._publish_target(stmt.target)
        if pub is not None:
            return self._publish(pub, stmt.value)
        sh = self._shared_target(stmt.target)
        if sh is not None:
            return self._shared_aug(sh, stmt.value)
        attr, keys, comp, monoid, inner_keys = self._resolve_accum(stmt.target)
        if monoid.dft in _LIST_MONOIDS:
            _reject("+= on an ordered-list monoid; use .append(elem, order_by=<expr>)")
        if monoid.dft not in _ADDITIVE_MONOIDS:
            _reject(f"+= on the non-additive monoid {monoid.dft}; use .observe(v) instead")
        return self._emit_add(attr, keys, comp, monoid, stmt.value, inner_keys)

    def _observe(self, call: ast.Call) -> List[str]:
        func = call.func
        if not (isinstance(func, ast.Attribute) and func.attr == "observe"):
            _reject("method call other than .observe(v)")
        attr, keys, comp, monoid, inner_keys = self._resolve_accum(func.value)
        if monoid.dft in _ARGBY_MONOIDS:
            return self._observe_argby(call, attr, keys, comp, monoid)
        if monoid.dft in _TOPK_MONOIDS:
            return self._observe_topk(call, attr, keys, comp, monoid)
        if monoid.dft in _ARGROW_MONOIDS:
            return self._observe_argrow(call, attr, keys, monoid)
        if len(call.args) != 1 or call.keywords:
            _reject(".observe expects exactly one value argument")
        if monoid.dft in _LIST_MONOIDS:
            _reject(".observe on an ordered-list monoid; use .append(elem, order_by=<expr>)")
        if monoid.dft in _SET_MONOIDS:
            self._check_element(call.args[0], monoid)
        if monoid.dft in _APPROX_MONOIDS:
            self._check_kv_element(call.args[0], monoid)
            return self._emit_kv_add(
                "map_add_approx_topk_at", attr, keys, comp, monoid, call.args[0]
            )
        if monoid.dft in _SAMPLE_MONOIDS:
            self._check_kv_element(call.args[0], monoid)
            return self._emit_kv_add("map_add_sample_at", attr, keys, comp, monoid, call.args[0])
        return self._emit_add(attr, keys, comp, monoid, call.args[0], inner_keys)

    def _check_kv_element(self, node: ast.expr, monoid: _Monoid) -> None:
        is_str = self._is_str_element(node)
        if monoid.dft in _KV_STR and not is_str:
            raise JitError(
                "a str-valued top-k/sample element must be a string field such as e.fhash; "
                "use of=jit.i64 for an int64 value"
            )
        if monoid.dft in _KV_I64 and is_str:
            raise JitError(
                "an i64-valued top-k/sample element must be an int64 expr such as e.dur; "
                "use of=jit.str_ for a string value"
            )

    def _observe_topk(
        self,
        call: ast.Call,
        attr: str,
        keys: List[ast.expr],
        comp: int | None,
        monoid: _Monoid,
    ) -> List[str]:
        if len(call.args) != 1:
            _reject("top-k/bottom-k .observe takes one payload and by=<expr>")
        by: ast.expr | None = None
        for kw in call.keywords:
            if kw.arg == "by":
                by = kw.value
            else:
                _reject(f"top-k/bottom-k .observe keyword other than by ({kw.arg})")
        if by is None:
            raise JitError(
                "top-k/bottom-k .observe requires by=<expr>; the by key ranks the kept payloads"
            )
        payload = call.args[0]
        self._check_kv_element(payload, monoid)
        k = getattr(monoid, "k", 0)
        self.funcs.add("map_add_topk_at")
        by_expr, _ = self._arith(by)
        payload_expr = self._value(payload)
        var = f"m_{attr}"
        c = 0 if comp is None else comp
        lines = ["{", f"    int64_t key[{len(keys)}];"]
        lines.extend(self._key_assigns(self.maps[attr].key_types, keys))
        lines.append(
            f"    map->map_add_topk_at(host->h, {var}, key, {c}, {k}u, "
            f"(double)({by_expr}), (int64_t)({payload_expr}));"
        )
        lines.append("}")
        return lines

    def _emit_kv_add(
        self,
        fn: str,
        attr: str,
        keys: List[ast.expr],
        comp: int | None,
        monoid: _Monoid,
        value: ast.expr,
    ) -> List[str]:
        self.funcs.add(fn)
        k = getattr(monoid, "k", 0)
        val_expr = self._value(value)
        var = f"m_{attr}"
        c = 0 if comp is None else comp
        lines = ["{", f"    int64_t key[{len(keys)}];"]
        lines.extend(self._key_assigns(self.maps[attr].key_types, keys))
        lines.append(f"    map->{fn}(host->h, {var}, key, {c}, {k}u, (int64_t)({val_expr}));")
        lines.append("}")
        return lines

    def _observe_argrow(
        self,
        call: ast.Call,
        attr: str,
        keys: List[ast.expr],
        monoid: _Monoid,
    ) -> List[str]:
        if len(call.args) != 1:
            _reject("argmin_row/argmax_row .observe takes one payload tuple and by=<expr>")
        by: ast.expr | None = None
        for kw in call.keywords:
            if kw.arg == "by":
                by = kw.value
            else:
                _reject(f"argmin_row/argmax_row .observe keyword other than by ({kw.arg})")
        if by is None:
            raise JitError(
                "argmin_row/argmax_row .observe requires by=<expr>; the by key selects the row"
            )
        payload_node = call.args[0]
        if not isinstance(payload_node, ast.Tuple):
            raise JitError(
                "argmin_row/argmax_row .observe payload must be a tuple such as "
                "(e.fhash, e.tid, e.ts) matching the of= schema"
            )
        payload = builtins.list(payload_node.elts)
        decl = self.maps[attr]
        assert decl.argrow is not None
        ptypes = decl.argrow.payload_types
        if len(payload) != len(ptypes):
            raise JitError(
                f"self.{attr} arg-row payload takes {len(ptypes)} components, got {len(payload)}"
            )
        self.funcs.add("map_add_argrow")
        by_expr, _ = self._arith(by)
        var = f"m_{attr}"
        pn = len(payload)
        lines = ["{", f"    int64_t key[{len(keys)}];", f"    int64_t payload[{pn}];"]
        lines.extend(self._key_assigns(decl.key_types, keys))
        lines.extend(self._key_assigns(ptypes, payload, "payload"))
        lines.append(
            f"    map->map_add_argrow(host->h, {var}, key, (double)({by_expr}), payload, {pn}u);"
        )
        lines.append("}")
        return lines

    def _observe_argby(
        self,
        call: ast.Call,
        attr: str,
        keys: List[ast.expr],
        comp: int | None,
        monoid: _Monoid,
    ) -> List[str]:
        if len(call.args) != 1:
            _reject("argmin/argmax .observe takes one payload and by=<expr>")
        by: ast.expr | None = None
        for kw in call.keywords:
            if kw.arg == "by":
                by = kw.value
            else:
                _reject(f"argmin/argmax .observe keyword other than by ({kw.arg})")
        if by is None:
            raise JitError(
                "argmin/argmax .observe requires by=<expr>; the by key selects the extreme payload"
            )
        payload = call.args[0]
        is_str = self._is_str_element(payload)
        if monoid.dft in _ARGBY_STR and not is_str:
            raise JitError(
                "an argmin/argmax(of=jit.str_) payload must be a string field such as "
                "e.fhash; use jit.argmin/argmax(of=jit.i64) for an int64 payload"
            )
        if monoid.dft in _ARGBY_I64 and is_str:
            raise JitError(
                "an argmin/argmax(of=jit.i64) payload must be an int64 expr such as "
                "e.dur; use jit.argmin/argmax(of=jit.str_) for a string payload"
            )
        return self._emit_argby(attr, keys, comp, by, payload)

    def _emit_argby(
        self,
        attr: str,
        keys: List[ast.expr],
        comp: int | None,
        by: ast.expr,
        payload: ast.expr,
    ) -> List[str]:
        self.funcs.add("map_add_argby_at")
        by_expr, _ = self._arith(by)
        payload_expr = self._value(payload)
        var = f"m_{attr}"
        c = 0 if comp is None else comp
        lines = ["{", f"    int64_t key[{len(keys)}];"]
        lines.extend(self._key_assigns(self.maps[attr].key_types, keys))
        lines.append(
            f"    map->map_add_argby_at(host->h, {var}, key, {c}, "
            f"(double)({by_expr}), (int64_t)({payload_expr}));"
        )
        lines.append("}")
        return lines

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
        attr, keys, comp, monoid, _inner = self._resolve_accum(func.value)
        if monoid.dft not in _LIST_MONOIDS:
            _reject(f".append on the non-list monoid {monoid.dft}; use += or .observe(v)")
        if order is None:
            raise JitError(
                "ordered list requires order_by=<expr>; order-by-arrival is not parallel-safe"
            )
        self._check_element(call.args[0], monoid)
        return self._emit_ordered(attr, keys, comp, order, call.args[0])

    def _check_element(self, node: ast.expr, monoid: _Monoid) -> None:
        is_str_field = self._is_str_element(node)
        if monoid.dft in _STR_COLLECTIONS and not is_str_field:
            raise JitError(
                "a str collection element must be a string field such as "
                "e.name; use jit.set(of=jit.i64) / jit.list(of=jit.i64) for int64"
            )
        if monoid.dft in _I64_COLLECTIONS and is_str_field:
            raise JitError(
                "an i64 collection element must be an int64 expr such as e.dur; "
                "use jit.set() / jit.list() for string fields"
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

    def _emit_ordered(
        self,
        attr: str,
        keys: List[ast.expr],
        comp: int | None,
        order: ast.expr,
        elem: ast.expr,
    ) -> List[str]:
        self.funcs.add("map_add_ordered_at")
        order_expr = self._value(order)
        elem_expr = self._value(elem)
        var = f"m_{attr}"
        c = 0 if comp is None else comp
        lines = ["{", f"    int64_t key[{len(keys)}];"]
        lines.extend(self._key_assigns(self.maps[attr].key_types, keys))
        lines.append(
            f"    map->map_add_ordered_at(host->h, {var}, key, {c}, "
            f"(int64_t)({order_expr}), (uint64_t)({elem_expr}));"
        )
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
            if node.value.id == self.self_name and node.attr in self.resolve_flags:
                return f"_rflag_{node.attr}"
            if node.value.id == self.event_name:
                return self._field(node.attr)
            if node.attr == "NONE":
                return "DFTU_STR_NONE"
        if isinstance(node, ast.Constant):
            return self._const(node)
        _reject(_describe(node))

    def _rhs(self, node: ast.expr, float_target: bool) -> str:
        expr, saw_float = self._arith(node)
        if not float_target and saw_float:
            _reject("float value into a u64 monoid component")
        cast = "double" if float_target else "uint64_t"
        return f"({cast})({expr})"

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
            if node.value.id == self.self_name and node.attr in self.resolve_flags:
                return f"_rflag_{node.attr}", False
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


def _reject_resolve(what: str) -> NoReturn:
    raise JitError(f"unsupported in @jit.on_resolve: {what}; use a raw C++ plugin")


def _ver_pack(major: int, minor: int, patch: int) -> int:
    """Pack a version into one comparable uint64 (same layout the emitted
    ``_resolved_ver_*`` statics use), so a version compare is a plain integer
    compare in the generated C."""
    return (major << 32) | (minor << 16) | patch


class _ResolveCompiler:
    """Compiles a ``@jit.on_resolve`` body into ``comms_resolve`` C statements.

    The expressible surface is intentionally narrow (see :func:`on_resolve`):
    ``self.<flag> = <expr>`` assignments whose expressions read each consume
    port's ``.resolved`` / ``.version`` and combine them with boolean ops,
    comparisons, and integer literals. Everything else is rejected rather than
    silently ignored."""

    def __init__(self, self_name: str, ports: Dict[str, _Port]) -> None:
        self.self_name = self_name
        self.ports = ports
        self.flags: List[str] = []
        self.flag_set: builtins.set[str] = builtins.set()

    def compile(self, stmts: List[ast.stmt]) -> List[str]:
        lines: List[str] = []
        for stmt in stmts:
            lines.extend(self._stmt(stmt))
        return lines

    def _stmt(self, stmt: ast.stmt) -> List[str]:
        if not isinstance(stmt, ast.Assign) or len(stmt.targets) != 1:
            _reject_resolve("statement other than a single self.<flag> = <expr> assignment")
        target = stmt.targets[0]
        if not (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and target.value.id == self.self_name
        ):
            _reject_resolve("assignment target other than self.<flag>")
        name = target.attr
        if name in self.ports:
            _reject_resolve(f"self.{name} names a port; pick a new flag name")
        if name not in self.flag_set:
            self.flag_set.add(name)
            self.flags.append(name)
        return [f"_rflag_{name} = ({self._expr(stmt.value)});"]

    def _expr(self, node: ast.expr) -> str:
        if isinstance(node, ast.BoolOp):
            sym = "&&" if isinstance(node.op, ast.And) else "||"
            parts = [self._expr(v) for v in node.values]
            return "(" + f" {sym} ".join(parts) + ")"
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return f"(!({self._expr(node.operand)}))"
        if isinstance(node, ast.Compare):
            return self._compare(node)
        val = self._int_operand(node)
        if val is not None:
            return val
        _reject_resolve(_describe(node))

    def _compare(self, node: ast.Compare) -> str:
        if len(node.ops) != 1:
            _reject_resolve("chained comparison")
        op = _CMP_OP.get(type(node.ops[0]))
        if op is None:
            _reject_resolve("comparison operator")
        left, right = node.left, node.comparators[0]
        lv, rv = self._ver_operand(left), self._ver_operand(right)
        if lv is not None or rv is not None:
            if lv is None or rv is None:
                _reject_resolve("version compared to a non-version operand")
            return f"({lv} {op} {rv})"
        li, ri = self._int_operand(left), self._int_operand(right)
        if li is None or ri is None:
            _reject_resolve("comparison operand")
        return f"({li} {op} {ri})"

    def _ver_operand(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "version"
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == self.self_name
        ):
            return f"_resolved_ver_{self._consume_attr(node.value.attr)}"
        if isinstance(node, ast.Tuple):
            comps = [0, 0, 0]
            if len(node.elts) > 3:
                _reject_resolve("version tuple with more than three components")
            for i, e in enumerate(node.elts):
                if not (isinstance(e, ast.Constant) and isinstance(e.value, int)) or isinstance(
                    e.value, bool
                ):
                    _reject_resolve("version tuple component (expected an integer literal)")
                if e.value < 0 or e.value > 0xFFFF:
                    _reject_resolve("version tuple component out of range 0..65535")
                comps[i] = e.value
            return f"{_ver_pack(*comps)}ULL"
        return None

    def _int_operand(self, node: ast.expr) -> str | None:
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, int)
            and not isinstance(node.value, bool)
        ):
            return str(node.value)
        if (
            isinstance(node, ast.Attribute)
            and node.attr == "resolved"
            and isinstance(node.value, ast.Attribute)
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == self.self_name
        ):
            return f"_resolved_{self._consume_attr(node.value.attr)}"
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == self.self_name
            and node.attr in self.flag_set
        ):
            return f"_rflag_{node.attr}"
        return None

    def _consume_attr(self, attr: str) -> str:
        port = self.ports.get(attr)
        if port is None or port.role != "consume":
            _reject_resolve(
                f"self.{attr}.<field>; only a jit.consume port exposes .resolved/.version"
            )
        return attr


def _compile_resolve(
    fn: Callable[..., object], ports: Dict[str, _Port]
) -> Tuple[List[str], List[str]]:
    """Compile an ``@jit.on_resolve`` method to (flag names, C statements)."""
    src = textwrap.dedent(inspect.getsource(fn))
    mod = ast.parse(src)
    funcs = [n for n in mod.body if isinstance(n, ast.FunctionDef)]
    if len(funcs) != 1:
        _reject_resolve("on_resolve must decorate a single method")
    fn_ast = funcs[0]
    params = fn_ast.args.args
    if len(params) != 1:
        _reject_resolve("on_resolve must take exactly (self)")
    rc = _ResolveCompiler(params[0].arg, ports)
    lines = rc.compile(fn_ast.body)
    return rc.flags, lines


def _emit_comms(
    pub_ports: List[Tuple[str, _Port]],
    sub_ports: List[Tuple[str, _Port]],
    resolve_flags: List[str] | None = None,
    resolve_body: List[str] | None = None,
) -> List[str]:
    """C for the plugin-side dftu_plugin_comms: publish ports become provides,
    consume ports become requirements, exposed via get_extension(DFTU_EXT_COMMS)."""
    out: List[str] = [
        "static uint32_t comms_provides(void* self, dftu_capability* out, uint32_t max) {",
        "    (void)self;",
    ]
    n = len(pub_ports)
    if n:
        ids = ", ".join(_c_str_literal(p.cap_id) for _, p in pub_ports)
        vers = ", ".join(f"{{{p.ver[0]}, {p.ver[1]}, {p.ver[2]}}}" for _, p in pub_ports)
        out += [
            f"    static const char* ids[{n}] = {{{ids}}};",
            f"    static const dftu_version vers[{n}] = {{{vers}}};",
            f"    for (uint32_t i = 0; i < {n}u && i < max; ++i) {{",
            "        out[i].id = ids[i];",
            "        out[i].ver = vers[i];",
            "    }",
            f"    return {n}u;",
        ]
    else:
        out += ["    (void)out; (void)max;", "    return 0u;"]
    out += ["}", ""]

    out += [
        "static uint32_t comms_require(void* self, dftu_requirement* out, uint32_t max) {",
        "    (void)self;",
    ]
    m = len(sub_ports)
    if m:
        ids = ", ".join(_c_str_literal(p.cap_id) for _, p in sub_ports)
        ops = ", ".join((p.ver_op or "DFTU_VER_GE") for _, p in sub_ports)
        vers = ", ".join(f"{{{p.ver[0]}, {p.ver[1]}, {p.ver[2]}}}" for _, p in sub_ports)
        reqd = ", ".join("1" if p.required else "0" for _, p in sub_ports)
        out += [
            f"    static const char* ids[{m}] = {{{ids}}};",
            f"    static const dftu_ver_op ops[{m}] = {{{ops}}};",
            f"    static const dftu_version vers[{m}] = {{{vers}}};",
            f"    static const int reqd[{m}] = {{{reqd}}};",
            f"    for (uint32_t i = 0; i < {m}u && i < max; ++i) {{",
            "        out[i].id = ids[i];",
            "        out[i].op = ops[i];",
            "        out[i].ver = vers[i];",
            "        out[i].required = reqd[i];",
            "    }",
            f"    return {m}u;",
        ]
    else:
        out += ["    (void)out; (void)max;", "    return 0u;"]
    out += ["}", ""]

    resolve_flags = resolve_flags or []
    resolve_body = resolve_body or []
    # Per consume port: whether a satisfying provider was found and its version
    # (packed major<<32|minor<<16|patch). Written once in comms_resolve before
    # any slice runs, then read from on_batch - write-once-then-read, no race.
    for name, _ in sub_ports:
        out += [
            f"static int _resolved_{name} = 0;",
            f"static uint64_t _resolved_ver_{name} = 0;",
        ]
    for flag in resolve_flags:
        out.append(f"static int64_t _rflag_{flag} = 0;")
    if sub_ports or resolve_flags:
        out.append("")

    out.append("static void comms_resolve(void* self, const dftu_host* host) {")
    out.append("    (void)self;")
    if sub_ports:
        out += [
            "    const dftu_ext_comms* _c =",
            "        (const dftu_ext_comms*)host->get_extension(host->h, DFTU_EXT_COMMS);",
            "    if (_c && _c->provider_best) {",
            "        dftu_requirement _req;",
            "        dftu_version _bv;",
        ]
        for name, port in sub_ports:
            cap = _c_str_literal(port.cap_id)
            op = port.ver_op or "DFTU_VER_GE"
            mj, mn, pt = port.ver
            reqd = 1 if port.required else 0
            out += [
                f"        _req.id = {cap};",
                f"        _req.op = {op};",
                f"        _req.ver.major = {mj}; _req.ver.minor = {mn}; _req.ver.patch = {pt};",
                f"        _req.required = {reqd};",
                "        if (_c->provider_best(host->h, &_req, &_bv) == 0) {",
                f"            _resolved_{name} = 1;",
                f"            _resolved_ver_{name} = ((uint64_t)_bv.major << 32)"
                " | ((uint64_t)_bv.minor << 16) | (uint64_t)_bv.patch;",
                "        }",
            ]
        out.append("    }")
    else:
        out.append("    (void)host;")
    for line in resolve_body:
        out.append("    " + line)
    out.append("}")
    out += [
        "",
        "static const dftu_plugin_comms g_plugin_comms = {",
        "    comms_provides, comms_require, comms_resolve",
        "};",
        "",
        "static const void* plugin_get_extension(void* self, const char* ext_id) {",
        "    (void)self;",
        "    if (ext_id && strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_plugin_comms;",
        "    return NULL;",
        "}",
        "",
    ]
    return out


def _port_ctype(port: _Port) -> Tuple[str, str]:
    return ("double", "0.0") if port.is_f64 else ("uint64_t", "0")


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
        cap = _c_str_literal(port.cap_id)
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


def _emit_handles_pre(shared_decls: List[Tuple[str, _Shared]]) -> List[str]:
    """Batch-scoped handle setup emitted before the event loop: fetch the handles
    extension and get-or-create each shared handle for this worker slice; the host
    merges same-cap handles across slices."""
    out: List[str] = [
        "    const dftu_ext_handles* _handles =",
        "        (const dftu_ext_handles*)host->get_extension(host->h, DFTU_EXT_HANDLES);",
    ]
    for name, sh in shared_decls:
        cap = _c_str_literal(sh.cap_id)
        out += [
            f"    dftu_handle* _hd_{name} = (_handles && _handles->shared_get)",
            f"        ? _handles->shared_get(host->h, {cap}, {sh.kind}) : NULL;",
            f"    (void)_hd_{name};",
        ]
    return out


def _emit_handles_finalize(shared_decls: List[Tuple[str, _Shared]]) -> List[str]:
    """on_finalize body that reads each cross-worker-merged handle and emits its
    scalar value (8 raw bytes) over the result channel under the attribute name."""
    out: List[str] = [
        "    (void)slice;",
        "    const dftu_ext_handles* _handles =",
        "        (const dftu_ext_handles*)host->get_extension(host->h, DFTU_EXT_HANDLES);",
        "    const dftu_ext_result* _result =",
        "        (const dftu_ext_result*)host->get_extension(host->h, DFTU_EXT_RESULT);",
        "    if (_handles && _handles->result && _result && _result->emit) {",
        "        dftu_monoid_value _v;",
    ]
    for name, sh in shared_decls:
        cap = _c_str_literal(sh.cap_id)
        rname = _c_str_literal(name)
        out += [
            f"        if (_handles->result(host->h, {cap}, &_v) == 0)",
            f"            _result->emit(host->h, {rname}, &_v.as, 8u);",
        ]
    out += ["    }", "    return NULL;"]
    return out


def _emit_port_flush(pub_ports: List[Tuple[str, _Port]]) -> List[str]:
    """Publish each publish port's per-batch accumulator after the event loop."""
    out: List[str] = []
    for name, port in pub_ports:
        cap = _c_str_literal(port.cap_id)
        out += [
            "    if (_ports && _ports->publish) {",
            f"        _ports->publish(host->h, _ports->port_key(host->h, {cap}), &_pub_{name}, 8u);",
            "    }",
        ]
    return out


def _emit(
    name: str,
    maps: Dict[str, _MapDecl],
    body: List[str],
    needs: builtins.set[str],
    funcs: builtins.set[str],
    plan_query: str | None,
    str_literals: List[str],
    arg_keys: List[str],
    arg_helpers: builtins.set[str],
    raw: bool,
    joins: List[Tuple[str, str, str, str]],
    fused_groups: List[_FusedGroup] | None = None,
    fused_of: Dict[str, Tuple[str, int, bool]] | None = None,
    op_defs: List[str] | None = None,
    ports: Dict[str, _Port] | None = None,
    shared: Dict[str, _Shared] | None = None,
    resolve_flags: List[str] | None = None,
    resolve_body: List[str] | None = None,
) -> str:
    fused_groups = fused_groups or []
    fused_of = fused_of or {}
    op_defs = op_defs or []
    ports = ports or {}
    shared = shared or {}
    pub_ports = [(n, p) for n, p in ports.items() if p.role == "publish"]
    sub_ports = [(n, p) for n, p in ports.items() if p.role == "consume"]
    shared_decls = builtins.list(shared.items())
    needs_expr = " | ".join(sorted(needs)) if needs else "0u"
    used = builtins.set(funcs)
    if joins:
        used.add("map_declare_join")
    if fused_groups:
        used.add("map_new_fused")
    for attr, decl in maps.items():
        if attr in fused_of:
            continue  # a fused member is created via map_new_fused, not here
        if decl.nested is not None:
            used.add("map_new_nested")
        elif decl.argrow is not None:
            used.add("map_new_argrow")
        elif len(decl.values) == 1 and decl.values[0].dft == "DFTU_MONOID_SKETCH":
            used.add("map_new_sketch")
        else:
            used.add("map_new_product" if decl.is_product else "map_new")
    guard = " || ".join(["!map"] + [f"!map->{fn}" for fn in sorted(used)])
    plan_query_field = "plan_query" if plan_query is not None else "NULL"
    out: List[str] = [
        "#include <dftracer/utils/plugins/abi.h>",
        "#include <dftracer/utils/plugins/prims.h>",
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
    if plan_query is not None:
        out += [
            "static const char* plan_query(void* self) {",
            "    (void)self;",
            f"    return {_c_str_literal(plan_query)};",
            "}",
            "",
        ]
    if ports:
        out += _emit_comms(pub_ports, sub_ports, resolve_flags, resolve_body)
    out += [
        "static void* make_slice(void* self) {",
        "    (void)self;",
        "    return calloc(1, 1);",
        "}",
        "",
        "static dftu_task* on_batch(void* slice, const dftu_batch* b,",
        "                          const dftu_host* host) {",
        "    const dftu_ext_map* map =",
        "        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);",
        "    (void)slice;",
        f"    if ({guard}) return NULL;",
    ]
    if str_literals:
        out.append("    if (!lits_resolved) {")
        for idx, lit in enumerate(str_literals):
            blen = len(lit.encode("utf-8"))
            out.append(f"        lit_{idx} = host->intern(host->h, {_c_str_literal(lit)}, {blen});")
        out.append("        lits_resolved = 1;")
        out.append("    }")
    if arg_keys:
        out.append("    if (!args_resolved) {")
        for idx, name in enumerate(arg_keys):
            blen = len(name.encode("utf-8"))
            out.append(
                f"        argkey_{idx} = host->intern(host->h, {_c_str_literal(name)}, {blen});"
            )
        out.append("        args_resolved = 1;")
        out.append("    }")
    for group in fused_groups:
        cname = group["cname"]
        kts = group["key_types"]
        n = len(kts)
        types = ", ".join(k.dft for k in kts)
        names = ", ".join(f'"{m}"' for m in group["members"])
        vals = ", ".join(maps[m].values[0].dft for m in group["members"])
        vn = len(group["members"])
        out.append(f"    static const dftu_type kt_{cname}[{n}] = {{{types}}};")
        out.append(f"    static const char* on_{cname}[{vn}] = {{{names}}};")
        out.append(f"    static const dftu_monoid_kind vt_{cname}[{vn}] = {{{vals}}};")
        out.append(
            f'    dftu_map* m_{cname} = map->map_new_fused(host->h, "{cname}", '
            f"kt_{cname}, {n}, on_{cname}, vt_{cname}, {vn});"
        )
        out.append(f"    if (!m_{cname}) return NULL;")
        if group.get("ordered"):
            out.append("    if (map->map_set_ordered)")
            out.append(f"        map->map_set_ordered(host->h, m_{cname}, 1);")
    for map_name, decl in maps.items():
        if map_name in fused_of:
            continue  # created above via map_new_fused
        types = ", ".join(k.dft for k in decl.key_types)
        n = len(decl.key_types)
        out.append(f"    static const dftu_type kt_{map_name}[{n}] = {{{types}}};")
        if decl.nested is not None:
            ikt = decl.nested.inner_key_types
            ni = len(ikt)
            itypes = ", ".join(k.dft for k in ikt)
            out.append(f"    static const dftu_type ikt_{map_name}[{ni}] = {{{itypes}}};")
            vn = len(decl.values)
            vals = ", ".join(v.dft for v in decl.values)
            out.append(f"    static const dftu_monoid_kind vt_{map_name}[{vn}] = {{{vals}}};")
            out.append(
                f'    dftu_map* m_{map_name} = map->map_new_nested(host->h, "{map_name}", '
                f"kt_{map_name}, {n}, ikt_{map_name}, {ni}, vt_{map_name}, {vn});"
            )
            out.append(f"    if (!m_{map_name}) return NULL;")
            continue
        if decl.argrow is not None:
            pt = decl.argrow.payload_types
            pn = len(pt)
            ptypes = ", ".join(t.dft for t in pt)
            out.append(f"    static const dftu_type pt_{map_name}[{pn}] = {{{ptypes}}};")
            is_max = 1 if decl.argrow.is_max else 0
            out.append(
                f'    dftu_map* m_{map_name} = map->map_new_argrow(host->h, "{map_name}", '
                f"kt_{map_name}, {n}, {is_max}, pt_{map_name}, {pn});"
            )
            out.append(f"    if (!m_{map_name}) return NULL;")
            continue
        if len(decl.values) == 1 and isinstance(decl.values[0], Quantiles):
            qs = decl.values[0].qs
            nq = len(qs)
            qvals = ", ".join(repr(q) for q in qs)
            out.append(f"    static const double qs_{map_name}[{nq}] = {{{qvals}}};")
            out.append(
                f'    dftu_map* m_{map_name} = map->map_new_sketch(host->h, "{map_name}", '
                f"kt_{map_name}, {n}, qs_{map_name}, {nq});"
            )
            out.append(f"    if (!m_{map_name}) return NULL;")
            continue
        if decl.is_product:
            vn = len(decl.values)
            vals = ", ".join(v.dft for v in decl.values)
            out.append(f"    static const dftu_monoid_kind vt_{map_name}[{vn}] = {{{vals}}};")
            out.append(
                f'    dftu_map* m_{map_name} = map->map_new_product(host->h, "{map_name}", '
                f"kt_{map_name}, {n}, vt_{map_name}, {vn});"
            )
        else:
            out.append(
                f'    dftu_map* m_{map_name} = map->map_new(host->h, "{map_name}", '
                f"kt_{map_name}, {n}, {decl.values[0].dft});"
            )
        out.append(f"    if (!m_{map_name}) return NULL;")
        if decl.ordered:
            out.append("    if (map->map_set_ordered)")
            out.append(f"        map->map_set_ordered(host->h, m_{map_name}, 1);")
    for out_name, left_name, right_name, how_enum in joins:
        out.append(
            f'    map->map_declare_join(host->h, "{out_name}", "{left_name}", '
            f'"{right_name}", {how_enum});'
        )
    if raw:
        for map_name in maps:
            out.append(f"    dftu_map* {map_name} = m_{map_name};")
            out.append(f"    (void){map_name};")
    if ports:
        out += _emit_port_pre(pub_ports, sub_ports)
    if shared_decls:
        out += _emit_handles_pre(shared_decls)
    out.append("    for (uint32_t i = 0; i < b->count; ++i) {")
    out.append("        const dftu_event* e = &b->events[i];")
    for line in body:
        out.append("        " + line)
    out.append("    }")
    if ports:
        out += _emit_port_flush(pub_ports)
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
        ]
    )
    if shared_decls:
        out += _emit_handles_finalize(shared_decls)
    else:
        out += ["    (void)slice;", "    (void)host;", "    return NULL;"]
    out.extend(
        [
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
            "    (void)config;",
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
    )
    if ports:
        out.append("    g_plugin.get_extension = plugin_get_extension;")
    out.extend(
        [
            "    return &g_plugin;",
            "}",
            "",
        ]
    )
    return "\n".join(out)


_RAW_FUNCS = frozenset(
    {
        "map_add_u64",
        "map_add_f64",
        "map_add_u64_at",
        "map_add_f64_at",
        "map_add_ordered_at",
        "map_add_nested_u64",
        "map_add_nested_f64",
        "map_add_argby_at",
        "map_add_topk_at",
        "map_add_approx_topk_at",
        "map_add_sample_at",
        "map_add_argrow",
    }
)


def _resolve_needs(needs: Tuple[object, ...] | None) -> builtins.set[str]:
    if needs is None:
        return builtins.set()
    out: builtins.set[str] = builtins.set()
    for n in needs:
        if not isinstance(n, _Need):
            raise JitError("needs must be jit.NEED_ARGS / jit.NEED_FHASH / jit.NEED_HHASH flags")
        out.add(n.dft)
    return out


def _raw_body(fn: Callable[..., object]) -> List[str]:
    argc = getattr(fn, "__code__").co_argcount
    result = fn(*([None] * argc))
    if not isinstance(result, str):
        raise JitError("@jit.each_event(raw=True) method must return a C++ body string")
    return textwrap.dedent(result).splitlines()


def _resolve_joins(
    join_decls: Dict[str, JoinDecl], maps: Dict[str, _MapDecl]
) -> List[Tuple[str, str, str, str]]:
    by_id = {id(decl): attr for attr, decl in maps.items()}
    out: List[Tuple[str, str, str, str]] = []
    for out_name, jd in join_decls.items():
        left = by_id.get(id(jd.left))
        right = by_id.get(id(jd.right))
        if left is None or right is None:
            raise JitError(f"jit.join {out_name} references a map not declared on this @jit.plugin")
        out.append((out_name, left, right, _JOIN_TYPES[jd.how]))
    return out


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
    join_decls: Dict[str, JoinDecl] = {}
    ports: Dict[str, _Port] = {}
    shared: Dict[str, _Shared] = {}
    each: List[_EachEvent] = []
    resolves: List[_OnResolve] = []
    for attr, val in vars(cls).items():
        if isinstance(val, _MapDecl):
            maps[attr] = val
        elif isinstance(val, JoinDecl):
            join_decls[attr] = val
        elif isinstance(val, _Port):
            ports[attr] = val
        elif isinstance(val, _Shared):
            shared[attr] = val
        elif isinstance(val, _EachEvent):
            each.append(val)
        elif isinstance(val, _OnResolve):
            resolves.append(val)
    if not maps and not ports and not shared:
        raise JitError(
            "@jit.plugin needs at least one jit.map, jit.publish/jit.consume, or "
            "jit.shared attribute"
        )
    joins = _resolve_joins(join_decls, maps)
    if len(each) != 1:
        raise JitError("@jit.plugin needs exactly one @jit.each_event method")
    if len(resolves) > 1:
        raise JitError("@jit.plugin allows at most one @jit.on_resolve method")
    resolve_flags: List[str] = []
    resolve_body: List[str] = []
    if resolves:
        if not any(p.role == "consume" for p in ports.values()):
            raise JitError("@jit.on_resolve needs at least one jit.consume port to inspect")
        resolve_flags, resolve_body = _compile_resolve(resolves[0].fn, ports)
    plan_query = getattr(cls, "plan_query", None)
    if plan_query is not None and not isinstance(plan_query, str):
        raise JitError("@jit.plugin plan_query must be a query DSL string")
    explicit_needs = _resolve_needs(needs)
    fused_groups: List[_FusedGroup] = []
    fused_of: Dict[str, Tuple[str, int, bool]] = {}
    op_defs: List[str] = []
    if each[0].raw:
        body = _raw_body(each[0].fn)
        inferred_needs = explicit_needs
        funcs = builtins.set(_RAW_FUNCS)
        str_literals: List[str] = []
        arg_keys: List[str] = []
        arg_helpers: builtins.set[str] = builtins.set()
    else:
        # A map named by any join must keep its own table, so it cannot fuse.
        no_fuse = builtins.set()
        for _out, left, right, _how in joins:
            no_fuse.update((_out, left, right))
        compiler = _Compiler(
            maps,
            builtins.set(join_decls),
            no_fuse,
            _referenced_ops(each[0].fn),
            ports=ports,
            shared=shared,
            resolve_flags=builtins.set(resolve_flags),
        )
        body = compiler.lower(each[0].fn)
        inferred_needs = compiler.needs | explicit_needs
        funcs = compiler.funcs
        str_literals = compiler.str_literals
        arg_keys = compiler.arg_keys
        arg_helpers = compiler.arg_helpers
        fused_groups = compiler.fused_groups
        fused_of = compiler.fused_of
        op_defs = compiler.op_defs
    source = _emit(
        cls.__name__,
        maps,
        body,
        inferred_needs,
        funcs,
        plan_query,
        str_literals,
        arg_keys,
        arg_helpers,
        each[0].raw,
        joins,
        fused_groups,
        fused_of,
        op_defs,
        ports,
        shared,
        resolve_flags,
        resolve_body,
    )
    renames = {
        attr: builtins.list(decl.value_names)
        for attr, decl in maps.items()
        if decl.value_names is not None and decl.nested is None
    }
    setattr(cls, "_jit_plugin", JitPlugin(cls.__name__, source, renames))
    return cls


# ---- vfold: a per-batch fold on columns -----------------------------------

# Reducer method -> engine reduce op; each composes as a single per-batch value
# under its matching monoid (batch sums summed, batch maxes maxed, ...).
_VFOLD_REDUCERS = {
    "sum": ("DFTU_REDUCE_SUM", "SUM"),
    "min": ("DFTU_REDUCE_MIN", "MIN"),
    "max": ("DFTU_REDUCE_MAX", "MAX"),
}
_VFOLD_TOP_FIELDS = frozenset({"name", "cat", "pid", "tid", "ts", "dur", "ph", "fhash", "hhash"})
# Numeric top-level columns build_row_frame emits as flat UInt64, so a keyed
# fold can read them directly as a uint64 buffer.
_VFOLD_NUMERIC = frozenset({"pid", "tid", "ts", "dur"})


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
        # keyed: self.<map>[df["key"]] += df["val"] | <int>
        if (
            isinstance(tgt, ast.Subscript)
            and isinstance(tgt.value, ast.Attribute)
            and isinstance(tgt.value.value, ast.Name)
            and tgt.value.value.id == "self"
        ):
            attr = tgt.value.attr
            if attr not in maps:
                raise JitError(f"@jit.vfold: '{attr}' is not a declared jit.map")
            keyfield = _vfold_df_field(_unwrap_index(tgt.slice))
            if keyfield not in _VFOLD_NUMERIC:
                raise JitError(
                    f"@jit.vfold: key field '{keyfield}' must be numeric (pid/tid/ts/dur)"
                )
            rhs = stmt.value
            vfield: str | None = None
            const: int | None = None
            if isinstance(rhs, ast.Constant) and isinstance(rhs.value, int):
                const = int(rhs.value)
            else:
                vfield = _vfold_df_field(rhs)
                if vfield not in _VFOLD_NUMERIC:
                    raise JitError(
                        f"@jit.vfold: value field '{vfield}' must be numeric (pid/tid/ts/dur)"
                    )
                fields.add(vfield)
            fields.add(keyfield)
            ops.append(
                {
                    "kind": "keyed",
                    "attr": attr,
                    "mon": maps[attr].values[0],
                    "keytype": maps[attr].key_types[0].dft,
                    "keyfield": keyfield,
                    "vfield": vfield,
                    "const": const,
                }
            )
            continue
        # scalar: self.<acc> += df["field"].<reducer>()
        if not (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ):
            raise JitError(shape)
        attr = tgt.attr
        if attr not in accums:
            raise JitError(f"@jit.vfold: '{attr}' is not a declared accumulator")
        call = stmt.value
        if not (
            isinstance(call, ast.Call) and not call.args and isinstance(call.func, ast.Attribute)
        ):
            raise JitError('@jit.vfold: scalar value must be df["<field>"].<reducer>()')
        reducer = call.func.attr
        if reducer not in _VFOLD_REDUCERS:
            raise JitError(f"@jit.vfold: unsupported reducer '.{reducer}()' (use sum/min/max)")
        rop, family = _VFOLD_REDUCERS[reducer]
        mon = accums[attr]
        parts = mon.dft.split("_")  # DFTU_MONOID_SUM_F64 -> [...,'SUM','F64']
        mon_family = parts[2] if len(parts) > 2 else ""
        if mon_family != family:
            raise JitError(
                f"@jit.vfold: '.{reducer}()' does not match accumulator '{attr}' "
                f"(a jit.{reducer}() accumulator)"
            )
        field = _vfold_df_field(call.func.value)
        ops.append({"kind": "scalar", "attr": attr, "mon": mon, "field": field, "rop": rop})
        fields.add(field)
    if not ops:
        raise JitError("@jit.vfold: the @jit.each_batch body is empty")
    return ops, fields


def _emit_vfold(
    cls_name: str,
    ops: List[Dict[str, object]],
    needs_expr: str,
    plan_query: str | None,
) -> str:
    used = {"map_new"}
    for opd in ops:
        mon = cast(_Monoid, opd["mon"])
        used.add("map_add_f64" if mon.dft in _F64_MONOIDS else "map_add_u64")
    guard = " || ".join(["!map"] + [f"!map->{fn}" for fn in sorted(used)])
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
    out += [
        "static void* make_slice(void* self) {",
        "    (void)self;",
        "    return calloc(1, 1);",
        "}",
        "",
        "static dftu_task* on_batch_columns(void* slice,",
        "                                   const dftu_dataframe* df,",
        "                                   const dftu_host* host) {",
        "    const dftu_ext_map* map =",
        "        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);",
        "    (void)slice;",
        f"    if ({guard}) return NULL;",
    ]
    for opd in ops:
        mon = cast(_Monoid, opd["mon"])
        f64 = mon.dft in _F64_MONOIDS
        add_fn = "map_add_f64" if f64 else "map_add_u64"
        attr = cast(str, opd["attr"])
        if opd["kind"] == "scalar":
            field = cast(str, opd["field"])
            rop = cast(str, opd["rop"])
            if f64:
                valexpr = (
                    "s.kind == DFTU_SCALAR_TAG_F64 ? s.value.d : "
                    "(s.kind == DFTU_SCALAR_TAG_U64 ? (double)s.value.u "
                    ": (double)s.value.i)"
                )
            else:
                valexpr = (
                    "s.kind == DFTU_SCALAR_TAG_F64 ? (uint64_t)s.value.d : "
                    "(s.kind == DFTU_SCALAR_TAG_U64 ? s.value.u "
                    ": (uint64_t)s.value.i)"
                )
            # A scalar accumulator is a single-key map with a constant key 0 (the
            # map machinery needs a key); the result is a one-row table.
            out += [
                "    {",
                f"        dftu_series* col = dftu_dataframe_column(df, {_c_str_literal(field)});",
                "        if (col) {",
                f"            dftu_scalar s = dftu_series_reduce(col, {rop});",
                "            const dftu_type kt[1] = {DFTU_T_I64};",
                "            int64_t key[1] = {0};",
                f"            dftu_map* m = map->map_new(host->h, {_c_str_literal(attr)}, kt, 1, {mon.dft});",
                f"            map->{add_fn}(host->h, m, key, {valexpr});",
                "            dftu_series_free(col);",
                "        }",
                "    }",
            ]
            continue
        keyfield = cast(str, opd["keyfield"])
        keytype = cast(str, opd["keytype"])
        vfield = opd["vfield"]
        read_val = vfield is not None
        parts = mon.dft.split("_")
        fam = parts[2] if len(parts) > 2 else ""
        group_flag = {
            "SUM": "DFTU_REDUCE_SUM",
            "MIN": "DFTU_REDUCE_MIN",
            "MAX": "DFTU_REDUCE_MAX",
        }.get(fam)
        # SUM/MIN/MAX of a column compose as one per-key aggregate, so group the
        # batch with a single SIMD pass and fold each distinct key. COUNTER/MEAN
        # (and a constant value) do not compose that way - fold row by row and
        # let the monoid accumulate per sample.
        if read_val and group_flag is not None:
            perkey = "(double)gv[i]" if f64 else "(uint64_t)gv[i]"
            out += [
                "    {",
                f"        dftu_series* kc = dftu_dataframe_column(df, {_c_str_literal(keyfield)});",
                f"        dftu_series* vc = dftu_dataframe_column(df, {_c_str_literal(cast(str, vfield))});",
                "        if (kc && vc) {",
                "            dftu_series* ok = 0;",
                "            dftu_series* ov[1] = {0};",
                f"            int32_t nout = dftu_dataframe_group_by(kc, vc, {group_flag}, &ok, ov, 1);",
                "            if (nout >= 1 && ok && ov[0]) {",
                "                int64_t g = dftu_series_length(ok);",
                "                const uint64_t* gk = (const uint64_t*)dftu_series_data(ok);",
                "                const uint64_t* gv = (const uint64_t*)dftu_series_data(ov[0]);",
                f"                const dftu_type kt[1] = {{{keytype}}};",
                f"                dftu_map* m = map->map_new(host->h, {_c_str_literal(attr)}, kt, 1, {mon.dft});",
                "                if (gk && gv) {",
                "                    for (int64_t i = 0; i < g; i++) {",
                "                        int64_t key[1];",
                "                        key[0] = (int64_t)gk[i];",
                f"                        map->{add_fn}(host->h, m, key, {perkey});",
                "                    }",
                "                }",
                "            }",
                "            if (ok) dftu_series_free(ok);",
                "            if (ov[0]) dftu_series_free(ov[0]);",
                "            dftu_series_free(kc);",
                "            dftu_series_free(vc);",
                "        }",
                "    }",
            ]
            continue
        # Row-fold path: per-key counter/mean, or a constant value.
        perrow = (
            ("(double)vd[i]" if f64 else "(uint64_t)vd[i]")
            if read_val
            else (f"(double){opd['const']}" if f64 else f"(uint64_t){opd['const']}")
        )
        block = [
            "    {",
            f"        dftu_series* kc = dftu_dataframe_column(df, {_c_str_literal(keyfield)});",
        ]
        if read_val:
            block.append(
                f"        dftu_series* vc = dftu_dataframe_column(df, {_c_str_literal(cast(str, vfield))});"
            )
        cond = "kc && vc" if read_val else "kc"
        block += [
            f"        if ({cond}) {{",
            "            int64_t n = dftu_series_length(kc);",
            "            const uint64_t* kd = (const uint64_t*)dftu_series_data(kc);",
        ]
        if read_val:
            block.append("            const uint64_t* vd = (const uint64_t*)dftu_series_data(vc);")
        block += [
            f"            const dftu_type kt[1] = {{{keytype}}};",
            f"            dftu_map* m = map->map_new(host->h, {_c_str_literal(attr)}, kt, 1, {mon.dft});",
            ("            if (kd && vd) {" if read_val else "            if (kd) {"),
            "                for (int64_t i = 0; i < n; i++) {",
            "                    int64_t key[1];",
            "                    key[0] = (int64_t)kd[i];",
            f"                    map->{add_fn}(host->h, m, key, {perrow});",
            "                }",
            "            }",
            "            dftu_series_free(kc);",
        ]
        if read_val:
            block.append("            dftu_series_free(vc);")
        block += ["        }", "    }"]
        out += block
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
        if m.is_product or m.nested is not None or m.argrow is not None or m.ordered:
            raise JitError(
                f"@jit.vfold: '{attr}' must be a simple jit.map(key=<type>, value=<monoid>)"
            )
        if len(m.key_types) != 1:
            raise JitError(f"@jit.vfold: '{attr}' must have exactly one key")
        if len(m.values) != 1:
            raise JitError(f"@jit.vfold: '{attr}' must have exactly one value monoid")
    if not accums and not maps:
        raise JitError("@jit.vfold needs an accumulator (jit.sum()/min()/max()) or a jit.map")
    if len(batch) != 1:
        raise JitError("@jit.vfold needs exactly one @jit.each_batch method")
    plan_query = getattr(cls, "plan_query", None)
    if plan_query is not None and not isinstance(plan_query, str):
        raise JitError("@jit.vfold plan_query must be a query DSL string")
    ops, fields = _compile_vfold(batch[0].fn, accums, maps)
    needs_expr = "DFTU_NEED_ARGS" if any(f not in _VFOLD_TOP_FIELDS for f in fields) else "0u"
    source = _emit_vfold(cls.__name__, ops, needs_expr, plan_query)
    setattr(cls, "_jit_plugin", JitPlugin(cls.__name__, source, {}))
    return cls


def vfold(cls: type) -> type:
    """Author a vectorized fold: a per-batch fold whose body runs SIMD column
    ops on the batch and folds them into scalar accumulators or keyed maps.

    Declare scalar accumulators as jit.sum()/min()/max() and/or keyed maps as
    jit.map(key=<type>, value=<monoid>), plus one :func:`each_batch` method whose
    body is a sequence of ``self.<acc> += df["f"].<reducer>()`` (reducer matching
    the accumulator) and ``self.<map>[df["k"]] += df["v"]`` (numeric key/value
    columns, or ``+= 1`` for a counter). It compiles to a native plugin using the
    columnar on_batch seam, so each batch is folded in-scan; run it through
    :class:`dftracer.utils.plugins.PluginHost` like any other jit plugin."""
    return _build_vfold(cls)


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
