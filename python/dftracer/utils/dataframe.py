"""The ``DataFrame`` Python wrapper (and the trace-viewer wrappers).

``DataFrame`` wraps the native ``_DataFrame`` handle: native frame ops plus
Python-side Arrow/pandas/polars conversion and pickling. Every relational and
reshape primitive - ``join``/``asof``/``interval``, ``window``/``gap_fill``,
``unnest``/``explode``, ``melt``/``pivot``, ``concat``/``union``, ``distinct``,
``sample``, ``top_k``, ``sort`` - is a native method that takes and returns
``DataFrame``s, running the SIMD kernels on the native columns; Arrow is crossed
only at the edge (``to_arrow``/``from_arrow``), never as the operating boundary.
``TraceViewer`` / ``AggregatedTraceViewer`` wrap the native lazy viewers so their
terminals (``collect`` / ``collect_typed`` / ``join``) return a wrapped
``DataFrame``. See :mod:`dftracer.utils.series` for the column counterpart.
"""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

from . import dftracer_utils_ext as _ext
from ._units import coerce_bytes, coerce_duration
from .series import Series, _register, _require_pyarrow, _unwrap, _wrap, _Wrapper

if TYPE_CHECKING:
    import numpy as np  # ty: ignore[unresolved-import]
    import pandas as pd  # ty: ignore[unresolved-import]
    import polars as pl  # ty: ignore[unresolved-import]
    import pyarrow as pa  # ty: ignore[unresolved-import]

    from .columnar import Agg, ColumnExpr, Expr, GroupBy
    from .runtime import Runtime

# Matches WINDOW_UNBOUNDED (int64 max): a frame bound of None means that side of
# the ROWS frame runs to the partition edge.
_WINDOW_UNBOUNDED = (1 << 63) - 1

_WINDOW_NULLARY = ("row_number", "rank", "dense_rank")
_WINDOW_VALUE_ONLY = (
    "running_sum",
    "running_min",
    "running_max",
    "running_count",
    "delta",
    "first_value",
    "last_value",
)
_WINDOW_FRAME = ("frame_sum", "frame_min", "frame_max", "frame_count", "frame_mean")

# One window spec per appended output column; the func literal selects the shape.
RankSpec = Tuple[Literal["row_number", "rank", "dense_rank"], str]
OffsetSpec = Tuple[Literal["lag", "lead"], str, int, str]
RunSpec = Tuple[Literal["running_sum", "running_min", "running_max", "running_count"], str, str]
DeltaSpec = Tuple[Literal["delta"], str, str]
RateSpec = Tuple[Literal["rate"], str, str, str]
RateCounterSpec = Tuple[Literal["rate"], str, str, str, bool]
SessSpec = Tuple[Literal["sessionize"], str, float, str]
FrameSpec = Tuple[
    Literal["frame_sum", "frame_min", "frame_max", "frame_count", "frame_mean"],
    str,
    Optional[int],
    Optional[int],
    str,
]
NtileSpec = Tuple[Literal["ntile"], int, str]
PosSpec = Tuple[Literal["first_value", "last_value"], str, str]
NthSpec = Tuple[Literal["nth_value"], str, int, str]
WindowSpec = Union[
    RankSpec,
    OffsetSpec,
    RunSpec,
    DeltaSpec,
    RateSpec,
    RateCounterSpec,
    SessSpec,
    FrameSpec,
    NtileSpec,
    PosSpec,
    NthSpec,
]


def _names(cols: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if cols is None:
        return []
    if isinstance(cols, str):
        return [cols]
    return list(cols)


def _window_bound(v: Optional[int]) -> int:
    return _WINDOW_UNBOUNDED if v is None else int(v)


def _window_arity(spec: Sequence[object], n: int) -> None:
    if len(spec) != n:
        raise ValueError(
            f"window: {spec[0]!r} spec expects {n} elements, got {len(spec)}: {spec!r}"
        )


# A spec is a heterogeneous positional tuple (str, col names, ints, None) read
# by index, so its elements are genuinely Any; the output 9-tuple is mixed too.
def _norm_window_spec(spec: Sequence[Any]) -> Tuple[object, ...]:
    # Normalize to the fixed 9-tuple the native kernel reads: (func, value|None,
    # offset, name, time|None, threshold, counter, frame_pre, frame_post).
    if not isinstance(spec, (tuple, list)) or not spec:
        raise ValueError(f"window: bad spec {spec!r}")
    func = spec[0]
    value: Optional[str] = None
    offset = 0
    name: Optional[str] = None
    time: Optional[str] = None
    threshold = 0.0
    counter = False
    pre = 0
    post = 0
    if func in _WINDOW_NULLARY:
        _window_arity(spec, 2)
        name = spec[1]
    elif func in ("lag", "lead"):
        _window_arity(spec, 4)
        value, offset, name = spec[1], int(spec[2]), spec[3]
    elif func in _WINDOW_VALUE_ONLY:
        _window_arity(spec, 3)
        value, name = spec[1], spec[2]
    elif func == "rate":
        if len(spec) not in (4, 5):
            raise ValueError(f"window: 'rate' spec expects 4 or 5 elements: {spec!r}")
        value, time, name = spec[1], spec[2], spec[3]
        counter = bool(spec[4]) if len(spec) == 5 else False
    elif func == "sessionize":
        _window_arity(spec, 4)
        time, threshold, name = spec[1], float(spec[2]), spec[3]
    elif func in _WINDOW_FRAME:
        _window_arity(spec, 5)
        value, pre, post, name = spec[1], _window_bound(spec[2]), _window_bound(spec[3]), spec[4]
    elif func == "ntile":
        _window_arity(spec, 3)
        offset, name = int(spec[1]), spec[2]
    elif func == "nth_value":
        _window_arity(spec, 4)
        value, offset, name = spec[1], int(spec[2]), spec[3]
    else:
        raise ValueError(f"window: unknown function {func!r}")
    if not isinstance(name, str):
        raise ValueError(f"window: output name must be a str: {spec!r}")
    return (func, value, offset, name, time, threshold, counter, pre, post)


class DataFrame(_Wrapper["_ext._DataFrame"]):
    """A named set of columns: native frame ops plus Arrow conversion."""

    @classmethod
    def from_arrow(cls, table: "pa.Table") -> "DataFrame":
        """Import a ``pyarrow.Table`` (or any ``__arrow_c_stream__`` provider),
        zero-copy via the Arrow C Data Interface."""
        return _dataframe_from_arrow(table)

    @classmethod
    def from_pandas(cls, df: "pd.DataFrame") -> "DataFrame":
        """Import a ``pandas.DataFrame`` (via pyarrow)."""
        pa = _require_pyarrow()
        return _dataframe_from_arrow(pa.Table.from_pandas(df))

    @classmethod
    def from_polars(cls, df: "pl.DataFrame") -> "DataFrame":
        """Import a ``polars.DataFrame`` (zero-copy via its Arrow buffers)."""
        return _dataframe_from_arrow(df.to_arrow())

    @classmethod
    def from_parquet(cls, path: str, columns: Optional[Sequence[str]] = None) -> "DataFrame":
        """Read a Parquet file into a frame (needs pyarrow)."""
        _require_pyarrow()
        import pyarrow.parquet as pq  # ty: ignore[unresolved-import]

        return _dataframe_from_arrow(pq.read_table(path, columns=columns))

    @classmethod
    def from_dict(cls, mapping: Mapping[str, object]) -> "DataFrame":
        """Import a ``{name: array-like}`` mapping (via pyarrow)."""
        pa = _require_pyarrow()
        return _dataframe_from_arrow(pa.table(mapping))

    @classmethod
    def from_numpy(
        cls,
        arr: "Union[np.ndarray, Dict[str, np.ndarray]]",
        columns: Sequence[str],
    ) -> "DataFrame":
        """Import NumPy columns into a frame.

        ``arr`` is either a 2-D array (one column per column index) or a
        ``{name: 1-D array}`` mapping; ``columns`` names the columns of a 2-D
        array (ignored for a mapping)."""
        pa = _require_pyarrow()
        if isinstance(arr, dict):
            data = {name: Series.from_numpy(col).to_arrow() for name, col in arr.items()}
        else:
            data = {name: Series.from_numpy(arr[:, i]).to_arrow() for i, name in enumerate(columns)}
        return _dataframe_from_arrow(pa.table(data))

    def to_arrow(self) -> "pa.Table":
        """This frame as a ``pyarrow.Table`` (zero-copy via the C Data Interface
        stream)."""
        return _require_pyarrow().table(self._native)

    def to_pandas(self) -> "pd.DataFrame":
        """This frame as a pandas DataFrame."""
        return self.to_arrow().to_pandas()

    def to_polars(self) -> "pl.DataFrame":
        """This frame as a polars DataFrame."""
        try:
            import polars as pl  # ty: ignore[unresolved-import]
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). Install with: pip install polars"
            ) from None
        table = self.to_arrow()
        return pl.DataFrame() if table.num_rows == 0 else pl.from_arrow(table)

    def __getitem__(self, name: str) -> Series:
        return Series(self._native[name])

    def __contains__(self, name: str) -> bool:
        return name in self._native

    def __len__(self) -> int:
        return self._native.num_rows

    # Opaque Arrow C Data Interface capsule; Python has no capsule type.
    def __arrow_c_stream__(self, requested_schema: Optional[object] = None) -> object:
        return self._native.__arrow_c_stream__(requested_schema)

    def __reduce__(self) -> "Tuple[Callable[[object], DataFrame], Tuple[object, ...]]":
        return (_dataframe_from_arrow, (self.to_arrow(),))

    def apply(self, expr: "ColumnExpr") -> Series:
        """Evaluate a columnar expression against this frame and return the
        resulting :class:`~dftracer.utils.Series`.

        The frame-first spelling of :meth:`ColumnExpr.apply`:
        ``df.apply(F.sum_dur / F.count)`` equals
        ``(F.sum_dur / F.count).apply(df)``. ``expr`` is a
        :class:`~dftracer.utils.columnar.ColumnExpr` (``col()`` / ``F.`` / ``lit()``
        and arithmetic)."""
        from .columnar import ColumnExpr

        if not isinstance(expr, ColumnExpr):
            raise TypeError(
                "DataFrame.apply expects a column expression (col()/F./lit()), "
                f"got {type(expr).__name__}"
            )
        return expr.apply(self)

    def hash_partition(self, keys: Union[str, List[str]], n_parts: int) -> "list[DataFrame]":
        """Hash-partition the rows into ``n_parts`` frames (the shuffle
        primitive); each part is wrapped."""
        return [DataFrame(p) for p in self._native.hash_partition(keys, n_parts)]

    def window(
        self,
        partition_by: Optional[Sequence[str]] = None,
        order_by: Optional[Sequence[str]] = None,
        specs: Optional[Sequence[WindowSpec]] = None,
    ) -> "DataFrame":
        """SQL window functions, PARTITION BY ``partition_by`` and ORDER BY
        ``order_by``.

        ``specs`` is a list of tuples, one per appended output column:

        - ``("row_number"|"rank"|"dense_rank", out)``
        - ``("lag"|"lead", value_col, offset, out)``
        - ``("running_sum"|"running_min"|"running_max"|"running_count", value_col, out)``
        - ``("delta", value_col, out)``
        - ``("rate", value_col, time_col, out[, counter])``
        - ``("sessionize", time_col, threshold, out)``
        - ``("frame_sum"|"frame_min"|"frame_max"|"frame_count"|"frame_mean",
          value_col, preceding, following, out)`` (a bound of ``None`` is unbounded)
        - ``("ntile", n, out)``
        - ``("first_value"|"last_value", value_col, out)``
        - ``("nth_value", value_col, k, out)``

        All input columns pass through, then one column per spec, in sorted
        (partition, order) row order."""
        from .dftracer_utils_ext import window as _window

        norm = [_norm_window_spec(s) for s in (specs or [])]
        return _wrap(_window(self._native, _names(partition_by), _names(order_by), norm))

    def gap_fill(
        self,
        partition_by: Optional[Sequence[str]],
        time: str,
        bucket: int,
        values: Union[str, Sequence[str]],
        mode: Literal["none", "locf", "linear"] = "none",
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> "DataFrame":
        """Materialize a regular time grid of width ``bucket``, PARTITION BY
        ``partition_by``, keyed on the integer ``time`` column.

        ``values`` (a name or list of names) are filled on generated rows per
        ``mode``: ``"none"`` leaves them null, ``"locf"`` carries the last real
        value forward, ``"linear"`` interpolates (emitting the value columns as
        double). ``start``/``end`` set an explicit grid range for every
        partition; pass both or neither."""
        from .dftracer_utils_ext import gap_fill as _gap_fill

        if (start is None) != (end is None):
            raise ValueError("gap_fill: pass both start and end, or neither")
        return _wrap(
            _gap_fill(
                self._native,
                _names(partition_by),
                time,
                int(bucket),
                _names(values),
                mode,
                start,
                end,
            )
        )

    def join(
        self,
        other: "DataFrame",
        on: Union[int, str, Sequence[str]] = 1,
        how: Literal["inner", "left", "right", "full", "semi", "anti"] = "inner",
    ) -> "DataFrame":
        """Equi-join with ``other``.

        ``on`` is either the shared key column name(s) (a str or list of names),
        or an int count of the leading key columns both frames share. ``how`` is
        ``"inner"``, ``"left"``, ``"right"``, ``"full"``, ``"semi"``, or
        ``"anti"``. Output is [key columns, left value columns, right value
        columns]; colliding right names are suffixed ``_right``. Semi/anti emit a
        left-only schema."""
        if isinstance(on, bool):
            raise TypeError("join: 'on' must be a key name/list or an int count")
        if isinstance(on, int):
            names = list(self._native.column_names)
            if on < 1 or on > len(names):
                raise ValueError(
                    f"join: 'on' count {on} is out of range for a {len(names)}-column frame"
                )
            keys = names[:on]
        else:
            keys = _names(on)
            if not keys:
                raise ValueError("join: 'on' must name at least one key column")
        from .dftracer_utils_ext import join as _join

        return _wrap(_join(self._native, _unwrap(other), keys, how))

    def compare_agg(
        self, variant: "DataFrame", on: Union[int, str, Sequence[str]] = 1
    ) -> "DataFrame":
        """Compare two aggregation results: FULL-join on the shared leading key
        columns, then append ``delta_<m>``/``pct_<m>`` for each numeric
        ``l_``/``r_`` metric pair.

        ``on`` is an int count of the leading key columns both frames share, or
        the shared key column name(s). Output is [key columns, ``l_``/``r_`` per
        metric, ``delta_``/``pct_`` per metric]. Both frames must group and
        aggregate the same way."""
        if isinstance(on, bool):
            raise TypeError("compare_agg: 'on' must be a key name/list or an int count")
        if isinstance(on, int):
            names = list(self._native.column_names)
            if on < 1 or on > len(names):
                raise ValueError(
                    f"compare_agg: 'on' count {on} is out of range for a {len(names)}-column frame"
                )
            n_key = on
        else:
            keys = _names(on)
            if not keys:
                raise ValueError("compare_agg: 'on' must name at least one key column")
            n_key = len(keys)
        return _wrap(self._native.compare_agg(_unwrap(variant), n_key))

    def asof(
        self,
        other: "DataFrame",
        on: str,
        by: Optional[Union[str, Sequence[str]]] = None,
        direction: Literal["backward", "forward", "nearest"] = "backward",
        tolerance: Optional[int] = None,
    ) -> "DataFrame":
        """Temporal (as-of) join: match each row to the nearest ``other`` row by
        the time column ``on``, within optional equi-key partition ``by``.

        ``direction`` is ``"backward"`` (largest ts <= self.ts), ``"forward"``
        (smallest ts >= self.ts), or ``"nearest"``. ``tolerance`` (if given)
        bounds the allowed time distance. Output is all left columns then the
        right value columns; unmatched left rows get null right values."""
        from .dftracer_utils_ext import asof as _asof

        return _wrap(_asof(self._native, _unwrap(other), on, _names(by), direction, tolerance))

    def interval(
        self,
        other: "DataFrame",
        point: str,
        lo: str,
        hi: str,
        by: Optional[Union[str, Sequence[str]]] = None,
        outer: bool = False,
    ) -> "DataFrame":
        """Point-in-range join: match each row (the ``point`` column) to every
        ``other`` row whose closed span ``[lo, hi]`` contains it, within optional
        equi-key partition ``by``.

        One output row per (left, matching right); ``outer=True`` also emits an
        unmatched left row once with null right values. Output is all left
        columns then the right value columns."""
        from .dftracer_utils_ext import interval as _interval

        return _wrap(
            _interval(self._native, _unwrap(other), point, lo, hi, _names(by), bool(outer))
        )

    def unnest(self, column: str, keep_empty: bool = False) -> "DataFrame":
        """UNNEST / EXPLODE the list-typed ``column`` into one row per element,
        repeating every other column.

        ``column`` names a ``list<utf8>``, ``list<int64>``, or
        ``list<struct<...>>`` column; a ``list<struct>`` flattens its fields into
        columns (named by the struct fields). Other columns pass through by
        value. An empty or null list drops the row unless ``keep_empty=True``
        (then one row with the exploded column(s) null). See :meth:`explode` for
        the native single-list variant."""
        from .dftracer_utils_ext import unnest as _unnest

        return _wrap(_unnest(self._native, column, bool(keep_empty)))

    def top_k(self, name: str, k: int, largest: bool = True) -> "DataFrame":
        """The ``k`` best rows by column ``name`` (alias of the native
        :meth:`topk`)."""
        return _wrap(self._native.topk(name, k, largest))

    def distinct(self) -> "DataFrame":
        """Drop duplicate rows, keeping the first (alias of :meth:`unique`)."""
        return _wrap(self._native.unique())

    def sort(self, by: Union[str, Sequence[str]], descending: bool = False) -> "DataFrame":
        """Order rows by one column (``by`` a name) or lexicographically by
        several (``by`` a list); the fluent spelling over the native
        :meth:`sort_by` / :meth:`sort_by_multi`."""
        if isinstance(by, str):
            return _wrap(self._native.sort_by(by, descending))
        return _wrap(self._native.sort_by_multi(list(by), descending))

    def union(self, *others: "DataFrame") -> "DataFrame":
        """Vertically concatenate with ``others`` and drop duplicate rows (SQL
        UNION); use :meth:`concat` for UNION ALL (keep duplicates)."""
        cat = self._native.concat(*[_unwrap(o) for o in others])
        return _wrap(cat.unique())

    # -- native engine ops (SIMD/relational kernels, wrapped from the C extension) --
    def keys(self) -> List[str]:
        return self._native.keys()

    def filter(self, mask: "Series") -> "DataFrame":
        return _wrap(self._native.filter(_unwrap(mask)))

    def select(self, *names: str) -> "DataFrame":
        return _wrap(self._native.select(*[_unwrap(x) for x in names]))

    def rename(self, mapping: "dict[str, str]") -> "DataFrame":
        return _wrap(self._native.rename(_unwrap(mapping)))

    def with_column(self, name: str, col: "Series") -> "DataFrame":
        return _wrap(self._native.with_column(_unwrap(name), _unwrap(col)))

    def take(self, indices: "Series") -> "DataFrame":
        return _wrap(self._native.take(_unwrap(indices)))

    def column_index(self, name: str) -> int:
        return self._native.column_index(_unwrap(name))

    def head(self, n: int) -> "DataFrame":
        return _wrap(self._native.head(_unwrap(n)))

    def tail(self, n: int) -> "DataFrame":
        return _wrap(self._native.tail(_unwrap(n)))

    def reverse(self) -> "DataFrame":
        return _wrap(self._native.reverse())

    def drop_nulls(self) -> "DataFrame":
        return _wrap(self._native.drop_nulls())

    def fill_null(self, value: Union[int, float]) -> "DataFrame":
        return _wrap(self._native.fill_null(_unwrap(value)))

    def unique(self) -> "DataFrame":
        return _wrap(self._native.unique())

    def drop_duplicates(self) -> "DataFrame":
        return _wrap(self._native.drop_duplicates())

    def sort_by_multi(self, names: "list[str]", descending: bool = False) -> "DataFrame":
        return _wrap(self._native.sort_by_multi(_unwrap(names), _unwrap(descending)))

    def sample(self, n: int, seed: int = 0) -> "DataFrame":
        return _wrap(self._native.sample(_unwrap(n), _unwrap(seed)))

    def with_row_index(self, name: str) -> "DataFrame":
        return _wrap(self._native.with_row_index(_unwrap(name)))

    def describe(self) -> "DataFrame":
        return _wrap(self._native.describe())

    def null_count(self) -> "DataFrame":
        return _wrap(self._native.null_count())

    def is_duplicated(self) -> "Series":
        return _wrap(self._native.is_duplicated())

    def is_unique(self) -> "Series":
        return _wrap(self._native.is_unique())

    def slice(self, offset: int, length: int) -> "DataFrame":
        return _wrap(self._native.slice(_unwrap(offset), _unwrap(length)))

    def sort_by(self, name: str, descending: bool = False) -> "DataFrame":
        return _wrap(self._native.sort_by(_unwrap(name), _unwrap(descending)))

    def topk(self, name: str, k: int, largest: bool = True) -> "DataFrame":
        return _wrap(self._native.topk(_unwrap(name), _unwrap(k), _unwrap(largest)))

    def concat(self, *others: "DataFrame") -> "DataFrame":
        return _wrap(self._native.concat(*[_unwrap(x) for x in others]))

    def unpivot(self, id_vars: "str | list[str]", value_vars: "str | list[str]") -> "DataFrame":
        return _wrap(self._native.unpivot(_unwrap(id_vars), _unwrap(value_vars)))

    def melt(self, id_vars: "str | list[str]", value_vars: "str | list[str]") -> "DataFrame":
        return _wrap(self._native.melt(_unwrap(id_vars), _unwrap(value_vars)))

    def explode(self, column: str) -> "DataFrame":
        return _wrap(self._native.explode(_unwrap(column)))

    def to_dummies(self, column: str) -> "DataFrame":
        return _wrap(self._native.to_dummies(_unwrap(column)))

    def pivot(self, index: str, columns: str, values: str, agg: str = "first") -> "DataFrame":
        return _wrap(
            self._native.pivot(_unwrap(index), _unwrap(columns), _unwrap(values), _unwrap(agg))
        )

    def group_by_dynamic(
        self,
        time_col: str,
        every: int,
        period: "int | None" = None,
        aggs: "list[str] | None" = None,
    ) -> "DataFrame":
        return _wrap(
            self._native.group_by_dynamic(
                _unwrap(time_col), _unwrap(every), _unwrap(period), _unwrap(aggs)
            )
        )

    def query(
        self,
        dsl: "str | None" = None,
        *,
        group_by: "str | None" = None,
        aggs: "list[str] | None" = None,
        select: "list[str] | None" = None,
        order_by: "str | None" = None,
        descending: bool = False,
        limit: "int | None" = None,
    ) -> "DataFrame":
        return _wrap(
            self._native.query(
                dsl,
                group_by=group_by,
                aggs=aggs,
                select=select,
                order_by=order_by,
                descending=descending,
                limit=limit,
            )
        )

    def group_by(self, key: str, *aggs: "Union[str, Agg]") -> "Union[DataFrame, GroupBy]":
        return _wrap(self._native.group_by(_unwrap(key), *[_unwrap(x) for x in aggs]))


def _to_filter_dsl(predicate: "Union[str, Expr]") -> str:
    """Coerce a filter argument to a query-DSL string for pushdown.

    Accepts a raw DSL string unchanged, or an ``Expr`` (``F.field ...``) whose
    pure-predicate subset serializes to the DSL. Raises ``TypeError`` if the
    ``Expr`` is not index-pushable (mixes value ops into the predicate); compute
    those with ``.apply()`` on a materialized frame instead."""
    if isinstance(predicate, str):
        return predicate
    from .columnar import Expr

    if isinstance(predicate, Expr):
        return predicate.to_query()
    raise TypeError(
        "filter/query expects a query-DSL string or an Expr (F.field ...), "
        f"got {type(predicate).__name__}"
    )


def _agg_spec_string(agg: "Agg") -> str:
    """Lower a bare-field :class:`~dftracer.utils.columnar.Agg` to a viewer agg
    spec string (``"op:field"``, or ``"count"``). Raises for a compound value
    expression, which the scan-time aggregator cannot express."""
    from .columnar import _Col

    if agg.value is None:
        return agg.op
    if not isinstance(agg.value, _Col):
        raise TypeError(
            "TraceViewer.agg() needs a bare-field aggregate like F.dur.sum(); "
            "compound expressions are not scan-time aggregable (collect(), then "
            "DataFrame.group_by)"
        )
    return f"{agg.op}:{agg.value.name}"


def _viewer_agg(
    native: "_ext._TraceViewer", specs: "Sequence[Union[str, Agg]]"
) -> "_ext._TraceViewer":
    """Apply the aggregates to `native`. Accepts legacy spec strings and unified
    ``Agg`` expressions (``F.dur.sum()``, ``F.any.mean()``) side by side. ``F.any``
    (mean) maps to the numeric-args path; other ``F.any`` reductions raise."""
    from .columnar import Agg, _Wildcard

    str_specs = []
    wildcard_mean = False
    for s in specs:
        if isinstance(s, str):
            str_specs.append(s)
        elif isinstance(s, Agg):
            if isinstance(s.value, _Wildcard):
                if s.op == "mean":
                    wildcard_mean = True
                elif s.op == "count":
                    str_specs.append("count")
                else:
                    raise ValueError(
                        "F.any supports only .mean() (per numeric arg) and "
                        ".count(); name a field for other reductions, e.g. "
                        "F('args.level').sum()"
                    )
            else:
                str_specs.append(_agg_spec_string(s))
        else:
            raise TypeError(
                "agg() expects a spec string, an Agg (F.dur.sum()), or "
                f"F.any.<op>(); got {type(s).__name__}"
            )
    result = native
    if str_specs:
        result = result.agg(*str_specs)
    if wildcard_mean:
        result = result.agg_numeric_args()
    return result


_ViewerT = TypeVar("_ViewerT", bound="_ViewerFilters")


class _ViewerFilters:
    """Shared filter/query/agg wiring for the lazy viewers: accept a DSL string
    or a unified ``Expr`` predicate for filters, and spec strings or unified
    ``Agg`` expressions for :meth:`agg`."""

    __slots__ = ()
    _native: "_ext._TraceViewer"

    # Wrap a native builder result back into a viewer. Overridden by SessionView
    # so every inherited builder returns a SessionView instead of a TraceViewer,
    # giving the session branch full builder parity with no duplicated methods.
    def _rewrap(self, native: object) -> Any:
        return _wrap(native)

    def filter(self: _ViewerT, predicate: "Union[str, Expr]") -> _ViewerT:
        return self._rewrap(self._native.filter(_to_filter_dsl(predicate)))

    def query(self: _ViewerT, predicate: "Union[str, Expr]") -> _ViewerT:
        return self._rewrap(self._native.query(_to_filter_dsl(predicate)))

    def agg(self, *specs: "Union[str, Agg]") -> "AggregatedTraceViewer":
        if not specs:
            raise TypeError("agg() needs at least one aggregate")
        return self._rewrap(_viewer_agg(self._native, specs))

    # Builder ops forwarded to the native viewer, wrapped back. Self-typed ones
    # preserve the concrete viewer (plain or aggregated) through the chain.
    def phase(self: _ViewerT, phase: str) -> _ViewerT:
        return self._rewrap(self._native.phase(phase))

    def time_range(self: _ViewerT, begin: float, end: float) -> _ViewerT:
        return self._rewrap(self._native.time_range(begin, end))

    def time_unit(self: _ViewerT, unit: str) -> _ViewerT:
        return self._rewrap(self._native.time_unit(unit))

    def time_scale(self: _ViewerT, ns_ratio: float) -> _ViewerT:
        return self._rewrap(self._native.time_scale(ns_ratio))

    def select(self: _ViewerT, *cols: str) -> _ViewerT:
        return self._rewrap(self._native.select(*cols))

    def limit(self: _ViewerT, n: int) -> _ViewerT:
        return self._rewrap(self._native.limit(n))

    def offset(self: _ViewerT, n: int) -> _ViewerT:
        return self._rewrap(self._native.offset(n))

    def auto_spill(self: _ViewerT) -> _ViewerT:
        return self._rewrap(self._native.auto_spill())

    def rollup_root(self: _ViewerT, path: str) -> _ViewerT:
        return self._rewrap(self._native.rollup_root(path))

    def views_root(self: _ViewerT, path: str) -> _ViewerT:
        return self._rewrap(self._native.views_root(path))

    def sort_by(self: _ViewerT, name: str, descending: bool = False) -> _ViewerT:
        return self._rewrap(self._native.sort_by(name, descending))

    def topk(self: _ViewerT, name: str, k: int, largest: bool = True) -> _ViewerT:
        return self._rewrap(self._native.topk(name, k, largest))

    def agg_numeric_args(self) -> "AggregatedTraceViewer":
        return self._rewrap(self._native.agg_numeric_args())

    def collect(self) -> "DataFrame":
        return _wrap(self._native.collect())

    def stream(
        self,
        batch_size: int = 65536,
        workers: int = 0,
        normalize: bool = False,
        dict: bool = True,
    ) -> "Iterator[object]":
        """Iterate matching events as pyarrow record batches (parallel, bounded
        memory). Yields raw ``pyarrow.RecordBatch`` objects, not wrapped
        DataFrames. ``workers=0`` uses the runtime's worker count."""
        return iter(
            self._native.stream(
                batch_size=batch_size,
                workers=workers,
                normalize=normalize,
                dict=dict,
            )
        )

    def time_bucket(self: _ViewerT, interval_us: Union[int, float, str]) -> _ViewerT:
        """Bucket width; a bare number is microseconds, a string ("1ms") is
        converted."""
        us = int(round(coerce_duration(interval_us, 1e6, "interval_us")))
        return self._rewrap(self._native.time_bucket(us))

    def occ_cell(self: _ViewerT, cell_us: Union[int, float, str]) -> _ViewerT:
        """Occupancy cell size (busy quantum); a bare number is microseconds, a
        string ("1ms") is converted. 0 = default. Finer resolves overlap on
        short events; honored with time_range."""
        us = int(round(coerce_duration(cell_us, 1e6, "cell_us")))
        return self._rewrap(self._native.occ_cell(us))

    def memory_budget(self: _ViewerT, nbytes: Union[int, str]) -> _ViewerT:
        """Spill budget; accepts a byte count or a unit string ("512MB")."""
        return self._rewrap(self._native.memory_budget(coerce_bytes(nbytes, "nbytes")))

    def materialize(
        self,
        checkpoint_size: Union[int, str] = 0,
        part_size: Union[int, str] = 0,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> None:
        """Persist this query as a materialized view. checkpoint_size/part_size
        accept a byte count or a unit string ("4MB")."""
        self._native.materialize(
            checkpoint_size=coerce_bytes(checkpoint_size, "checkpoint_size"),
            part_size=coerce_bytes(part_size, "part_size"),
            progress=progress,
        )

    def export_trace(
        self,
        path: str,
        compress: Optional[bool] = None,
        index: Optional[bool] = None,
        member_size: Optional[Union[int, str]] = None,
        level: Optional[int] = None,
        part_size: Optional[Union[int, str]] = None,
    ) -> None:
        """Export to a trace file. member_size/part_size accept a byte count or
        a unit string ("4MB"); unset args keep the native defaults."""
        kwargs: Dict[str, Union[bool, int]] = {}
        if compress is not None:
            kwargs["compress"] = compress
        if index is not None:
            kwargs["index"] = index
        if level is not None:
            kwargs["level"] = level
        if member_size is not None:
            kwargs["member_size"] = coerce_bytes(member_size, "member_size")
        if part_size is not None:
            kwargs["part_size"] = coerce_bytes(part_size, "part_size")
        # Unpacking a runtime-built kwargs dict onto the native method's
        # individually-typed parameters is a known checker gap.
        self._native.export_trace(path, **kwargs)  # ty: ignore[invalid-argument-type]

    def statistics(self) -> Dict[str, object]:
        """One-row summary of the matching events: count, mean/stddev dur,
        min/max ts."""
        return self._native.statistics()

    def aggregate_partial(self) -> bytes:
        """A combinable aggregation partial (opaque bytes) for a distributed
        merge; combine several with merge_partials_to_table."""
        return self._native.aggregate_partial()

    def merge_partials_to_table(self, partials: List[bytes]) -> "DataFrame":
        """Merge aggregate_partial() bytes from several ranks into one
        DataFrame."""
        return _wrap(self._native.merge_partials_to_table(partials))

    def mv_source(self) -> List[str]:
        """The materialized-view trace file(s) that would serve this query, or an
        empty list if a read would scan the base."""
        return self._native.mv_source()

    def materialize_dir(self) -> str:
        """Distributed row-MV coordinator: create and return the shared MV dir
        (call on a view over the full file set)."""
        return self._native.materialize_dir()

    def register_materialized(self, dir: str) -> None:
        """Write the MV manifest at ``dir`` over this view's base set."""
        self._native.register_materialized(dir)


class AggregatedTraceViewer(_ViewerFilters, _Wrapper["_ext._TraceViewer"]):
    """The aggregated form of a TraceViewer (after group_by/agg); terminals
    return a wrapped DataFrame."""


class TraceViewer(_ViewerFilters, _Wrapper["_ext._TraceViewer"]):
    """A lazy view over trace files. Builder methods chain; terminals
    (``collect`` / ``collect_typed`` / ``join``) return a wrapped DataFrame."""

    @overload
    def __init__(self, native: "_ext._TraceViewer", /) -> None: ...
    @overload
    def __init__(
        self,
        files: Union[str, Sequence[str]],
        index_path: Optional[str] = ...,
        runtime: "Optional[Runtime]" = ...,
    ) -> None: ...

    # Erased dispatcher (overloads carry the contract): wrap a native handle, or
    # build one from real args.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if len(args) == 1 and not kwargs and isinstance(args[0], _ext._TraceViewer):
            super().__init__(args[0])
        else:
            super().__init__(_ext._TraceViewer(*args, **kwargs))

    # group_by / join are TraceViewer-only (not on the aggregated form).
    def group_by(self, *keys: str) -> "AggregatedTraceViewer":
        return _wrap(self._native.group_by(*keys))

    def join(self, other: "TraceViewer", how: str = "inner") -> "DataFrame":
        return _wrap(self._native.join(_unwrap(other), how))

    def compare(self, other: "TraceViewer") -> "DataFrame":
        """Aggregate both viewers and compare on the shared group key, returning
        the difference DataFrame."""
        return _wrap(self._native.compare(_unwrap(other)))

    def collect_typed(
        self,
        shard_begin: int = 0,
        shard_end: int = 0,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> Dict[str, DataFrame]:
        result = self._native.collect_typed(shard_begin, shard_end, progress)
        return {k: _wrap(v) for k, v in result.items()}

    def session(self) -> "Session":
        """Open a Session that fuses several branch views over one shared scan of
        this view. ``s.view()`` starts a branch (a full lazy TraceViewer); its
        terminal (``collect``/``export``/``materialize``) returns a Handle. The
        trace is decompressed once and every branch reads it. See
        :class:`Session`."""
        return Session(self)


def _stats_to_dict(df: "DataFrame") -> Dict[str, object]:
    """Shape a no-group (count, mean/std dur, min/max ts) collect into the same
    dict TraceViewer.statistics returns."""
    tbl = _require_pyarrow().table(df)
    row = {c: tbl.column(c)[0].as_py() for c in tbl.column_names} if tbl.num_rows else {}
    return {
        "duration_count": row.get("count", 0),
        "duration_mean_us": row.get("mean_dur", 0.0),
        "duration_stddev_us": row.get("std_dur", 0.0),
        "min_timestamp_us": row.get("min_ts", 0),
        "max_timestamp_us": row.get("max_ts", 0),
    }


class Handle:
    """A deferred result from a :class:`Session` branch. ``result()`` returns the
    branch's value (a DataFrame for ``collect``, a stats dict for ``export``)
    once the session has executed; reading it before then triggers execute."""

    __slots__ = ("_session", "_value", "_resolved", "_transform", "_index")

    def __init__(
        self,
        session: "Session",
        index: int,
        transform: "Optional[Callable[[Any], Any]]" = None,
    ) -> None:
        self._session = session
        self._value: object = None
        self._resolved = False
        # Post-execute shaping of the native result (e.g. a 1-row DataFrame to a
        # stats dict). Applied once in execute().
        self._transform = transform
        # This branch's position in the session, so join/compare can reference it.
        self._index = index

    def result(self) -> object:
        if not self._resolved:
            self._session.execute()
        return self._value


class SessionView(_ViewerFilters):
    """A branch of a :class:`Session`: a full lazy view (the whole TraceViewer
    builder API, via ``_ViewerFilters``) whose terminals register the branch on
    the shared scan instead of running at once. Builder ops chain and stay a
    SessionView; a terminal (``collect``/``export``/``materialize``) returns a
    :class:`Handle`. Scan-wide settings (phase/time_range/time_scale/
    memory_budget) come from the base view; setting them per-branch is honored
    only where it does not change what the shared scan reads."""

    _session: "Session"

    def __init__(self, session: "Session", native: "_ext._TraceViewer") -> None:
        self._native = native
        self._session = session

    def _rewrap(self, native: object) -> "SessionView":
        return SessionView(self._session, native)  # ty: ignore[invalid-argument-type]

    def group_by(self, *keys: str) -> "SessionView":
        return self._rewrap(self._native.group_by(*keys))

    def agg(self, *specs: "Union[str, Agg]") -> "SessionView":  # ty: ignore[invalid-method-override]
        return self._rewrap(_viewer_agg(self._native, specs))

    def agg_numeric_args(self) -> "SessionView":  # ty: ignore[invalid-method-override]
        return self._rewrap(self._native.agg_numeric_args())

    def collect(self) -> Handle:  # ty: ignore[invalid-method-override]
        """Register an aggregation branch; ``result()`` is a DataFrame."""
        return self._session._register("collect", self, None)

    def export(self, sink: str) -> Handle:
        """Register a raw-event export branch to ``sink`` (NDJSON); ``result()``
        is a stats dict."""
        return self._session._register("export", self, sink)

    def materialize(self) -> None:  # ty: ignore[invalid-method-override]
        """Register a build-only rollup branch (no Handle)."""
        self._session._register("materialize", self, None)

    def statistics(self) -> Handle:  # ty: ignore[invalid-method-override]
        """Register a summary-statistics branch (count, mean/stddev dur, min/max
        ts) over the shared scan; ``result()`` is a dict, matching
        :meth:`TraceViewer.statistics`. Reuses the collect aggregation."""
        view = self.group_by().agg("count", "mean:dur", "std:dur", "min:ts", "max:ts")
        return self._session._register("collect", view, None, transform=_stats_to_dict)

    def events(self) -> Handle:
        """Register a raw-events branch: the matching events (this branch's
        filter/select) materialized into one DataFrame over the shared scan.
        ``result()`` is a DataFrame. Holds the whole matching set, so filter it."""
        return self._session._register("events", self, None)

    def stream(  # ty: ignore[invalid-method-override]
        self, batch_size: int = 65536
    ) -> Handle:
        """Register a raw-events branch yielding the matching events as DataFrame
        chunks of ~``batch_size`` rows over the shared scan. ``result()`` is an
        iterator of DataFrames. Fused, so the events are buffered during execute;
        use the standalone ``TraceViewer.stream`` for bounded-memory streaming."""

        def to_chunks(df: "DataFrame") -> "Iterator[DataFrame]":
            n = df.num_rows
            return (df.slice(i, min(batch_size, n - i)) for i in range(0, n, batch_size))

        return self._session._register("events", self, None, transform=to_chunks)

    def aggregate_partial(self) -> Handle:  # ty: ignore[invalid-method-override]
        """Register a distributed-partial branch: aggregate this branch into an
        opaque serialized partial (bytes) over the shared scan, for a distributed
        merge (combine several with ``DataFrame.merge_partials_to_table``)."""
        return self._session._register("partial", self, None)

    def plugin(
        self, plugin: "Union[str, type]", config: "Optional[Dict[str, Any]]" = None
    ) -> Handle:
        """Register a plugin/JIT branch fused into the shared scan. ``plugin`` is
        a compiled ``.so`` path or a ``@jit.plugin`` class. ``result()`` is the
        plugin's DataFrame result (a ``{name: DataFrame}`` dict when the plugin
        emits several named results). The branch scans over the same pass as the
        session's aggregations. Scan-wide settings come from the base; a plugin
        does its own per-event filtering, so this branch's builder ops (filter/
        select/...) do not narrow it."""
        return self._session._register_plugin(plugin, config)


class Session:
    """A batch of branch views over one shared scan of a base TraceViewer.

    ``s.view()`` starts a branch - a full lazy TraceViewer - and its terminal
    (``collect``/``export``/``materialize``) registers it and returns a Handle.
    :meth:`execute` (or a ``with`` block, or the first ``Handle.result()``) runs
    every branch over one scan, so several views of the same trace cost one
    decompression, not one per view. Branches are independent: each has its own
    schema.

    ::

        with tv.session() as s:
            a = s.view().group_by("cat").agg("count", "mean:dur").collect()
            b = s.view().group_by("rank").agg("count").sort_by("count").collect()
            c = s.view().filter('cat == "POSIX"').export("posix.pfw")
        da, db = a.result(), b.result()
    """

    def __init__(self, viewer: "TraceViewer") -> None:
        self._viewer = viewer
        # (kind, branch-object, sink); branch-object is a SessionView, or a
        # PluginHost for a plugin branch (both expose ._native to the native
        # session executor).
        self._branches: List[Tuple[str, Any, Optional[str]]] = []
        self._handles: List[Optional[Handle]] = []
        self._executed = False

    def view(self) -> SessionView:
        """Start a new branch view off the session's base (shares its files and
        scan settings). Chain the per-branch builder API, then a terminal."""
        return SessionView(self, self._viewer._native)

    def _register(
        self,
        kind: str,
        viewer: SessionView,
        sink: Optional[str],
        transform: "Optional[Callable[[Any], Any]]" = None,
    ) -> Handle:
        if self._executed:
            raise RuntimeError("cannot add a branch after the session has executed")
        handle = Handle(self, len(self._branches), transform)
        self._branches.append((kind, viewer, sink))
        self._handles.append(None if kind == "materialize" else handle)
        return handle

    def _register_plugin(
        self, plugin: "Union[str, type]", config: "Optional[Dict[str, Any]]"
    ) -> Handle:
        if self._executed:
            raise RuntimeError("cannot add a branch after the session has executed")
        # plugins imports dataframe, so import lazily to avoid a cycle.
        from .plugins import PluginHost

        host = PluginHost()
        host.load(plugin, config)
        host.resolve()

        def shape(raw: "Dict[str, Any]") -> object:
            shaped = {name: host._shape(name, val) for name, val in raw.items()}
            return next(iter(shaped.values())) if len(shaped) == 1 else shaped

        handle = Handle(self, len(self._branches), shape)
        self._branches.append(("plugin", host, None))
        self._handles.append(handle)
        return handle

    def _register_combine(
        self, kind: str, left: Handle, right: Handle, spec: "Tuple[Any, ...]"
    ) -> Handle:
        if self._executed:
            raise RuntimeError("cannot add a branch after the session has executed")
        if left._session is not self or right._session is not self:
            raise ValueError(f"{kind} handles must come from this session")
        handle = Handle(self, len(self._branches))
        self._branches.append((kind, spec, None))
        self._handles.append(handle)
        return handle

    def join(
        self,
        left: Handle,
        right: Handle,
        how: "Literal['inner', 'left', 'right', 'full', 'semi', 'anti']" = "inner",
    ) -> Handle:
        """Equi-join two collect branches on their shared group key after the one
        scan, the same join View.join uses (key columns, then ``l_``/``r_`` value
        columns). ``result()`` is a DataFrame. Both branches must group the same
        way; the shared key width is inferred natively."""
        return self._register_combine("join", left, right, (left._index, right._index, how))

    def compare(self, baseline: Handle, variant: Handle) -> Handle:
        """Compare two collect branches on the shared group key after the one
        scan: key columns, ``l_``/``r_`` per metric, plus ``delta_``/``pct_``,
        the same result View.compare produces. Both branches must group and
        aggregate the same way."""
        return self._register_combine(
            "compare", baseline, variant, (baseline._index, variant._index)
        )

    def execute(self) -> None:
        """Run every registered branch over one scan and resolve the Handles.
        Idempotent: a second call is a no-op."""
        if self._executed:
            return
        spec = [
            (kind, v if kind in ("join", "compare") else v._native, sink)
            for kind, v, sink in self._branches
        ]
        results = self._viewer._native._session_execute(spec)
        for handle, native in zip(self._handles, results):
            if handle is not None:
                value = _wrap(native)
                handle._value = handle._transform(value) if handle._transform else value
                handle._resolved = True
        self._executed = True

    def __enter__(self) -> "Session":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if exc_type is None:
            self.execute()


_register(_ext._DataFrame, DataFrame)
# _AggregatedTraceViewer is a native subclass of _TraceViewer, so it must be
# registered first: _wrap returns the first isinstance match.
_register(_ext._AggregatedTraceViewer, AggregatedTraceViewer)
_register(_ext._TraceViewer, TraceViewer)


def _dataframe_from_arrow(table: "pa.Table") -> DataFrame:
    """Import a pyarrow Table into a native DataFrame wrapper."""
    return DataFrame(_ext._dataframe_from_arrow(table))
