"""The ``Series`` Python wrapper over the native columnar engine.

The C extension exposes a low-level ``_Series`` handle carrying the SIMD ops and
the Arrow C Data Interface capsule, but no Arrow/pandas/polars conversion. That
conversion lives here: ``Series`` wraps the handle, forwards every native op
(re-wrapping the handles it hands back), and adds ``to_arrow`` / ``to_pandas`` /
``to_numpy`` / ``to_polars``, NumPy-style ``+ - * /`` and ``s[i]`` indexing, the
``np.asarray`` protocol, and pickling. ``dataframe.DataFrame`` is the frame
counterpart.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Tuple, Type, Union

from . import dftracer_utils_ext as _ext

if TYPE_CHECKING:
    from .dataframe import DataFrame

# (native handle type, Python wrapper) pairs, filled as each wrapper module
# imports. _wrap consults it so the base forwarder can wrap any handle a native
# op returns without this module importing the frame/viewer wrappers.
_WRAP: List[Tuple[type, Type["_Wrapper"]]] = []


def _register(native_type: type, wrapper: Type["_Wrapper"]) -> None:
    _WRAP.append((native_type, wrapper))


def _wrap(obj: Any) -> Any:
    for native_type, wrapper in _WRAP:
        if isinstance(obj, native_type):
            return wrapper(obj)
    return obj


def _unwrap(obj: Any) -> Any:
    return obj._native if isinstance(obj, _Wrapper) else obj


class _Wrapper:
    """Holds a native handle and forwards unknown attributes to it, wrapping the
    handles it hands back and unwrapping the ones passed in."""

    __slots__ = ("_native",)

    def __init__(self, native: Any) -> None:
        self._native = native

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._native, name)
        if not callable(attr):
            return _wrap(attr)

        def forward(*args: Any, **kwargs: Any) -> Any:
            uargs = [_unwrap(a) for a in args]
            ukwargs = {k: _unwrap(v) for k, v in kwargs.items()}
            return _wrap(attr(*uargs, **ukwargs))

        return forward

    def __arrow_c_array__(self, requested_schema: Any = None) -> Any:
        return self._native.__arrow_c_array__(requested_schema)


def _require_pyarrow() -> Any:
    try:
        import pyarrow as pa  # ty: ignore[unresolved-import]
    except ImportError:
        raise ImportError(
            "pyarrow is required for Arrow conversion. Install with: pip install pyarrow"
        ) from None
    return pa


# Comparison op codes, mirroring abi.h DFTU_CMP_* (passed to native compare()).
_CMP_GT = 0
_CMP_GE = 1
_CMP_LT = 2
_CMP_LE = 3
_CMP_EQ = 4
_CMP_NE = 5


class Series(_Wrapper):
    """A typed column: SIMD ops from the native engine plus Arrow/NumPy conversion.

    Supports NumPy-style ``+ - * /`` against another Series (elementwise) or a
    scalar, ``s[i]`` / ``s[a:b]`` indexing, and ``np.asarray(s)``. Reflected
    scalar division (``2 / s``) is not provided; use ``2 / s.to_numpy()``.
    Comparisons (``<``, ``<=``, ``>``, ``>=``, ``gt``/``ge``/``lt``/``le``/``eq``/
    ``ne``) return a boolean mask Series."""

    @classmethod
    def from_arrow(cls, arr: Any) -> "Series":
        """Import a ``pyarrow.Array`` (or any ``__arrow_c_array__`` provider),
        zero-copy via the Arrow C Data Interface."""
        return _series_from_arrow(arr)

    @classmethod
    def from_pandas(cls, s: Any) -> "Series":
        """Import a ``pandas.Series`` (zero-copy for numeric, via pyarrow)."""
        pa = _require_pyarrow()
        return _series_from_arrow(pa.Array.from_pandas(s))

    @classmethod
    def from_polars(cls, s: Any) -> "Series":
        """Import a ``polars.Series`` (zero-copy via its Arrow buffers)."""
        return _series_from_arrow(s.to_arrow())

    @classmethod
    def from_numpy(cls, a: Any) -> "Series":
        """Import a NumPy array.

        A 1-D C-contiguous fixed-width numeric array takes the native
        pyarrow-free path (zero-copy borrow of the array's buffer); any other
        case (non-contiguous, unsupported dtype, non-1-D) falls back to the
        Arrow path, which needs pyarrow."""
        try:
            return Series(_ext._series_from_numpy(a))
        except (TypeError, BufferError, RuntimeError):
            pa = _require_pyarrow()
            return _series_from_arrow(pa.array(a))

    @classmethod
    def from_list(cls, values: Any, dtype: Any = None) -> "Series":
        """Import a Python sequence, optionally typed by a pyarrow ``dtype``."""
        pa = _require_pyarrow()
        return _series_from_arrow(pa.array(values, type=dtype))

    def to_arrow(self) -> Any:
        """This column as a ``pyarrow.Array`` (zero-copy via the C Data
        Interface)."""
        return _require_pyarrow().array(self._native)

    def to_pandas(self) -> Any:
        """This column as a pandas Series."""
        return self.to_arrow().to_pandas()

    def to_numpy(self) -> Any:
        """This column as a NumPy array.

        A flat, non-null, fixed-width numeric column is read directly from the
        native buffer (no pyarrow, zero-copy view); Bool, strings, nulls, and
        non-flat encodings fall back to the Arrow path (which needs pyarrow)."""
        import numpy as np  # ty: ignore[unresolved-import]

        try:
            # memoryview forces the native buffer protocol (a plain
            # np.asarray(handle) can fall back to a 0-d object array).
            return np.asarray(memoryview(self._native))
        except (BufferError, TypeError):
            return self.to_arrow().to_numpy(zero_copy_only=False)

    def to_polars(self) -> Any:
        """This column as a polars Series."""
        try:
            import polars as pl  # ty: ignore[unresolved-import]
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). Install with: pip install polars"
            ) from None
        return pl.from_arrow(self.to_arrow())

    def __array__(self, dtype: Any = None, copy: Any = None) -> Any:
        """NumPy array protocol, so ``np.asarray(series)`` works."""
        import numpy as np  # ty: ignore[unresolved-import]

        return np.asarray(self.to_numpy(), dtype=dtype)

    def __len__(self) -> int:
        return self._native.length

    def __getitem__(self, key: Any) -> Any:
        """``s[i]`` returns the element as a Python scalar; ``s[a:b]`` returns a
        Series (step-1 slices only)."""
        arr = self.to_arrow()
        if isinstance(key, slice):
            return _series_from_arrow(arr[key])
        idx = key + len(self) if key < 0 else key
        return arr[idx].as_py()

    def _elementwise(self, other: Any, series_op: str, scalar_op: str) -> Any:
        if isinstance(other, Series):
            return getattr(self, series_op)(other)
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return getattr(self, scalar_op)(other)
        return NotImplemented

    def __add__(self, other: Any) -> Any:
        return self._elementwise(other, "add", "add_scalar")

    def __radd__(self, other: Any) -> Any:
        return self._elementwise(other, "add", "add_scalar")

    def __sub__(self, other: Any) -> Any:
        return self._elementwise(other, "sub", "sub_scalar")

    def __rsub__(self, other: Any) -> Any:
        # other - self; a Series left operand would take __sub__, so other is a scalar.
        if isinstance(other, (int, float)) and not isinstance(other, bool):
            return self.mul_scalar(-1).add_scalar(other)
        return NotImplemented

    def __mul__(self, other: Any) -> Any:
        return self._elementwise(other, "mul", "mul_scalar")

    def __rmul__(self, other: Any) -> Any:
        return self._elementwise(other, "mul", "mul_scalar")

    def __truediv__(self, other: Any) -> Any:
        return self._elementwise(other, "div", "div_scalar")

    # Comparisons return a boolean mask Series. __eq__/__ne__ are intentionally
    # not defined (they would break hashing / `in`); use eq()/ne() instead, as
    # the C++ Series has no operator==/!= either.
    def __lt__(self, other: Any) -> Any:
        return self.compare(_CMP_LT, other)

    def __le__(self, other: Any) -> Any:
        return self.compare(_CMP_LE, other)

    def __gt__(self, other: Any) -> Any:
        return self.compare(_CMP_GT, other)

    def __ge__(self, other: Any) -> Any:
        return self.compare(_CMP_GE, other)

    def gt(self, value: Any) -> Any:
        """Boolean mask where the value is greater than ``value``."""
        return self.compare(_CMP_GT, value)

    def ge(self, value: Any) -> Any:
        """Boolean mask where the value is greater than or equal to ``value``."""
        return self.compare(_CMP_GE, value)

    def lt(self, value: Any) -> Any:
        """Boolean mask where the value is less than ``value``."""
        return self.compare(_CMP_LT, value)

    def le(self, value: Any) -> Any:
        """Boolean mask where the value is less than or equal to ``value``."""
        return self.compare(_CMP_LE, value)

    def eq(self, value: Any) -> Any:
        """Boolean mask where the value equals ``value``."""
        return self.compare(_CMP_EQ, value)

    def ne(self, value: Any) -> Any:
        """Boolean mask where the value does not equal ``value``."""
        return self.compare(_CMP_NE, value)

    # -- native engine ops (SIMD kernels, wrapped from the C extension) --------
    def add(self, other: "Series") -> "Series":
        return _wrap(self._native.add(_unwrap(other)))

    def sub(self, other: "Series") -> "Series":
        return _wrap(self._native.sub(_unwrap(other)))

    def mul(self, other: "Series") -> "Series":
        return _wrap(self._native.mul(_unwrap(other)))

    def div(self, other: "Series") -> "Series":
        return _wrap(self._native.div(_unwrap(other)))

    def add_scalar(self, v: Union[int, float]) -> "Series":
        return _wrap(self._native.add_scalar(_unwrap(v)))

    def sub_scalar(self, v: Union[int, float]) -> "Series":
        return _wrap(self._native.sub_scalar(_unwrap(v)))

    def mul_scalar(self, v: Union[int, float]) -> "Series":
        return _wrap(self._native.mul_scalar(_unwrap(v)))

    def div_scalar(self, v: Union[int, float]) -> "Series":
        return _wrap(self._native.div_scalar(_unwrap(v)))

    def cast(self, type_id: int) -> "Series":
        return _wrap(self._native.cast(_unwrap(type_id)))

    def prim(self, op: int) -> "Series":
        return _wrap(self._native.prim(_unwrap(op)))

    def compare(self, op: int, scalar: Union[int, float]) -> "Series":
        return _wrap(self._native.compare(_unwrap(op), _unwrap(scalar)))

    def logical(self, op: int, other: "Series") -> "Series":
        return _wrap(self._native.logical(_unwrap(op), _unwrap(other)))

    def logical_not(self) -> "Series":
        return _wrap(self._native.logical_not())

    def quantile(self, q: float) -> float:
        return self._native.quantile(_unwrap(q))

    def median(self) -> float:
        return self._native.median()

    def variance(self, sample: bool = True) -> float:
        return self._native.variance(_unwrap(sample))

    def stddev(self, sample: bool = True) -> float:
        return self._native.stddev(_unwrap(sample))

    def skewness(self) -> float:
        return self._native.skewness()

    def kurtosis(self) -> float:
        return self._native.kurtosis()

    def nunique(self) -> int:
        return self._native.nunique()

    def unique(self) -> "Series":
        return _wrap(self._native.unique())

    def value_counts(self) -> "DataFrame":
        return _wrap(self._native.value_counts())

    def abs(self) -> "Series":
        return _wrap(self._native.abs())

    def clip(self, lo: Union[int, float], hi: Union[int, float]) -> "Series":
        return _wrap(self._native.clip(_unwrap(lo), _unwrap(hi)))

    def round(self) -> "Series":
        return _wrap(self._native.round())

    def fillna(self, value: Union[int, float]) -> "Series":
        return _wrap(self._native.fillna(_unwrap(value)))

    def cumsum(self) -> "Series":
        return _wrap(self._native.cumsum())

    def cummax(self) -> "Series":
        return _wrap(self._native.cummax())

    def cummin(self) -> "Series":
        return _wrap(self._native.cummin())

    def cum_prod(self) -> "Series":
        return _wrap(self._native.cum_prod())

    def cum_count(self) -> "Series":
        return _wrap(self._native.cum_count())

    def ceil(self) -> "Series":
        return _wrap(self._native.ceil())

    def floor(self) -> "Series":
        return _wrap(self._native.floor())

    def trunc(self) -> "Series":
        return _wrap(self._native.trunc())

    def sign(self) -> "Series":
        return _wrap(self._native.sign())

    def negate(self) -> "Series":
        return _wrap(self._native.negate())

    def diff(self) -> "Series":
        return _wrap(self._native.diff())

    def pct_change(self) -> "Series":
        return _wrap(self._native.pct_change())

    def sqrt(self) -> "Series":
        return _wrap(self._native.sqrt())

    def exp(self) -> "Series":
        return _wrap(self._native.exp())

    def log(self) -> "Series":
        return _wrap(self._native.log())

    def rank(self, method: str = "average", descending: bool = False) -> "Series":
        return _wrap(self._native.rank(_unwrap(method), _unwrap(descending)))

    def rolling(self, window: int, op: str = "sum") -> "Series":
        return _wrap(self._native.rolling(_unwrap(window), _unwrap(op)))

    def rolling_var(self, window: int) -> "Series":
        return _wrap(self._native.rolling_var(_unwrap(window)))

    def rolling_std(self, window: int) -> "Series":
        return _wrap(self._native.rolling_std(_unwrap(window)))

    def rolling_median(self, window: int) -> "Series":
        return _wrap(self._native.rolling_median(_unwrap(window)))

    def rolling_quantile(self, window: int, q: float) -> "Series":
        return _wrap(self._native.rolling_quantile(_unwrap(window), _unwrap(q)))

    def ewm_mean(self, alpha: float) -> "Series":
        return _wrap(self._native.ewm_mean(_unwrap(alpha)))

    def ewm_std(self, alpha: float) -> "Series":
        return _wrap(self._native.ewm_std(_unwrap(alpha)))

    def cut(self, breaks: "Series") -> "Series":
        return _wrap(self._native.cut(_unwrap(breaks)))

    def qcut(self, q: int) -> "Series":
        return _wrap(self._native.qcut(_unwrap(q)))

    def search_sorted(self, values: "Series") -> "Series":
        return _wrap(self._native.search_sorted(_unwrap(values)))

    def interpolate(self) -> "Series":
        return _wrap(self._native.interpolate())

    def is_between(self, lo: Union[int, float], hi: Union[int, float]) -> "Series":
        return _wrap(self._native.is_between(_unwrap(lo), _unwrap(hi)))

    def dot(self, other: "Series") -> Union[int, float]:
        return self._native.dot(_unwrap(other))

    def sum(self) -> Union[int, float]:
        return self._native.sum()

    def min(self) -> Union[int, float]:
        return self._native.min()

    def max(self) -> Union[int, float]:
        return self._native.max()

    def mean(self) -> float:
        return self._native.mean()

    def count(self) -> int:
        return self._native.count()

    def product(self) -> Union[int, float]:
        return self._native.product()

    def mode(self) -> Union[int, float]:
        return self._native.mode()

    def all(self) -> bool:
        return self._native.all()

    def any(self) -> bool:
        return self._native.any()

    def arg_min(self) -> int:
        return self._native.arg_min()

    def arg_max(self) -> int:
        return self._native.arg_max()

    def take(self, indices: "Series") -> "Series":
        return _wrap(self._native.take(_unwrap(indices)))

    def filter(self, mask: "Series") -> "Series":
        return _wrap(self._native.filter(_unwrap(mask)))

    def argsort(self, descending: bool = False) -> "Series":
        return _wrap(self._native.argsort(_unwrap(descending)))

    def dictionary_encode(self) -> "Series":
        return _wrap(self._native.dictionary_encode())

    def materialize(self) -> "Series":
        return _wrap(self._native.materialize())

    def share(self) -> "Series":
        return _wrap(self._native.share())

    def slice(self, offset: int, length: int) -> "Series":
        return _wrap(self._native.slice(_unwrap(offset), _unwrap(length)))

    def is_null(self, i: int) -> bool:
        return self._native.is_null(_unwrap(i))

    def num_children(self) -> int:
        return self._native.num_children()

    def child(self, i: int) -> "Series":
        return _wrap(self._native.child(_unwrap(i)))

    def is_nan(self) -> "Series":
        return _wrap(self._native.is_nan())

    def is_finite(self) -> "Series":
        return _wrap(self._native.is_finite())

    def is_infinite(self) -> "Series":
        return _wrap(self._native.is_infinite())

    def is_unique(self) -> "Series":
        return _wrap(self._native.is_unique())

    def is_duplicated(self) -> "Series":
        return _wrap(self._native.is_duplicated())

    def is_sorted(self, descending: bool = False) -> bool:
        return self._native.is_sorted(_unwrap(descending))

    def drop_nulls(self) -> "Series":
        return _wrap(self._native.drop_nulls())

    def is_in(self, values: "Series") -> "Series":
        return _wrap(self._native.is_in(_unwrap(values)))

    def sort(self, descending: bool = False) -> "Series":
        return _wrap(self._native.sort(_unwrap(descending)))

    def head(self, n: int) -> "Series":
        return _wrap(self._native.head(_unwrap(n)))

    def tail(self, n: int) -> "Series":
        return _wrap(self._native.tail(_unwrap(n)))

    def reverse(self) -> "Series":
        return _wrap(self._native.reverse())

    def shift(self, n: int) -> "Series":
        return _wrap(self._native.shift(_unwrap(n)))

    def top_k(self, k: int) -> "Series":
        return _wrap(self._native.top_k(_unwrap(k)))

    def bottom_k(self, k: int) -> "Series":
        return _wrap(self._native.bottom_k(_unwrap(k)))

    def sample(self, n: int, seed: int = 0) -> "Series":
        return _wrap(self._native.sample(_unwrap(n), _unwrap(seed)))

    def str_eq(self, rhs: str) -> "Series":
        return _wrap(self._native.str_eq(_unwrap(rhs)))

    def str_contains(self, needle: str) -> "Series":
        return _wrap(self._native.str_contains(_unwrap(needle)))

    def str_starts_with(self, prefix: str) -> "Series":
        return _wrap(self._native.str_starts_with(_unwrap(prefix)))

    def str_ends_with(self, suffix: str) -> "Series":
        return _wrap(self._native.str_ends_with(_unwrap(suffix)))

    def str_matches(self, pattern: str) -> "Series":
        return _wrap(self._native.str_matches(_unwrap(pattern)))

    def str_like(self, pattern: str) -> "Series":
        return _wrap(self._native.str_like(_unwrap(pattern)))

    def str_len_bytes(self) -> "Series":
        return _wrap(self._native.str_len_bytes())

    def str_len_chars(self) -> "Series":
        return _wrap(self._native.str_len_chars())

    def str_find(self, needle: str) -> "Series":
        return _wrap(self._native.str_find(_unwrap(needle)))

    def to_lowercase(self) -> "Series":
        return _wrap(self._native.to_lowercase())

    def to_uppercase(self) -> "Series":
        return _wrap(self._native.to_uppercase())

    def str_strip(self) -> "Series":
        return _wrap(self._native.str_strip())

    def str_lstrip(self) -> "Series":
        return _wrap(self._native.str_lstrip())

    def str_rstrip(self) -> "Series":
        return _wrap(self._native.str_rstrip())

    def str_replace(self, pat: str, repl: str) -> "Series":
        return _wrap(self._native.str_replace(_unwrap(pat), _unwrap(repl)))

    def str_replace_all(self, pat: str, repl: str) -> "Series":
        return _wrap(self._native.str_replace_all(_unwrap(pat), _unwrap(repl)))

    def str_slice(self, start: int, length: int = -1) -> "Series":
        return _wrap(self._native.str_slice(_unwrap(start), _unwrap(length)))

    def str_pad_start(self, width: int, fill: str = " ") -> "Series":
        return _wrap(self._native.str_pad_start(_unwrap(width), _unwrap(fill)))

    def str_pad_end(self, width: int, fill: str = " ") -> "Series":
        return _wrap(self._native.str_pad_end(_unwrap(width), _unwrap(fill)))

    def str_zfill(self, width: int) -> "Series":
        return _wrap(self._native.str_zfill(_unwrap(width)))

    def str_split(self, sep: str) -> "Series":
        return _wrap(self._native.str_split(_unwrap(sep)))

    # -- native property accessors --------------------------------------------
    @property
    def type(self) -> Any:
        """The native element type of the column."""
        return self._native.type

    @property
    def encoding(self) -> Any:
        """The native storage encoding (flat, dictionary, selection)."""
        return self._native.encoding

    @property
    def length(self) -> int:
        """Number of elements in the column."""
        return self._native.length

    @property
    def null_count(self) -> int:
        """Number of null elements."""
        return self._native.null_count

    def __reduce__(self) -> Any:
        return (_series_from_arrow, (self.to_arrow(),))


_register(_ext._Series, Series)


def _series_from_arrow(array: Any) -> Series:
    """Import a pyarrow Array into a native Series wrapper."""
    return Series(_ext._series_from_arrow(array))
