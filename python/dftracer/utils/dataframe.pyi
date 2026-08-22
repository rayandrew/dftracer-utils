"""Type stubs for the DataFrame and trace-viewer wrappers. Declaring them as
subclasses of the native handles gives editors autocomplete for every forwarded
op; the builder methods are overridden to return the wrapped types, and the
terminals to return the wrapped DataFrame."""

from typing import Any, Literal, Optional, Sequence, Tuple, Union

from .columnar import Agg, ColumnExpr, Expr
from .dftracer_utils_ext import _AggregatedTraceViewer, _DataFrame, _Series, _TraceViewer
from .series import Series

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

class DataFrame(_DataFrame):
    def __init__(self, native: _DataFrame) -> None: ...
    @classmethod
    def from_arrow(cls, table: Any) -> "DataFrame": ...
    @classmethod
    def from_pandas(cls, df: Any) -> "DataFrame": ...
    @classmethod
    def from_polars(cls, df: Any) -> "DataFrame": ...
    @classmethod
    def from_parquet(cls, path: Any, columns: Any = None) -> "DataFrame": ...
    @classmethod
    def from_dict(cls, mapping: Any) -> "DataFrame": ...
    @classmethod
    def from_numpy(cls, arr: Any, columns: Any) -> "DataFrame": ...
    def to_arrow(self) -> Any:
        """This frame as a pyarrow.Table (zero-copy via the C Data Interface)."""
        ...

    def to_pandas(self) -> Any:
        """This frame as a pandas DataFrame."""
        ...

    def to_polars(self) -> Any:
        """This frame as a polars DataFrame."""
        ...

    # Frame ops (forwarded to the native handle, re-wrapped as DataFrame/Series).
    def keys(self) -> list[str]:
        """Series names, in order."""
        ...

    def filter(self, mask: _Series) -> "DataFrame":
        """Keep rows where the Bool mask Series is true."""
        ...

    def select(self, *names: str) -> "DataFrame":
        """Project the named columns (zero-copy)."""
        ...

    def rename(self, mapping: dict[str, str]) -> "DataFrame":
        """Rename columns via a {old: new} mapping (zero-copy)."""
        ...

    def with_column(self, name: str, col: _Series) -> "DataFrame":
        """Add or replace a column."""
        ...

    def take(self, indices: _Series) -> "DataFrame":
        """Gather rows at the given Int64 row indices."""
        ...

    def column_index(self, name: str) -> int:
        """Index of a column, or -1 if absent."""
        ...

    def head(self, n: int) -> "DataFrame":
        """The first n rows."""
        ...

    def tail(self, n: int) -> "DataFrame":
        """The last n rows."""
        ...

    def reverse(self) -> "DataFrame":
        """Rows in reverse order."""
        ...

    def slice(self, offset: int, length: int) -> "DataFrame":
        """A contiguous row range."""
        ...

    def sort_by(self, name: str, descending: bool = False) -> "DataFrame":
        """Order rows by a single column."""
        ...

    def sort_by_multi(self, names: list[str], descending: bool = False) -> "DataFrame":
        """Stable lexicographic sort by several key columns."""
        ...

    def topk(self, name: str, k: int, largest: bool = True) -> "DataFrame":
        """The k best rows by a column."""
        ...

    def sample(self, n: int, seed: int = 0) -> "DataFrame":
        """Deterministic n-row sample."""
        ...

    def unique(self) -> "DataFrame":
        """Drop duplicate rows (keep first)."""
        ...

    def drop_duplicates(self) -> "DataFrame":
        """Drop duplicate rows (keep first); alias of unique()."""
        ...

    def drop_nulls(self) -> "DataFrame":
        """Drop rows null in any column."""
        ...

    def fill_null(self, value: int | float) -> "DataFrame":
        """Fill nulls in every column with a value."""
        ...

    def with_row_index(self, name: str) -> "DataFrame":
        """Prepend an Int64 row-index column."""
        ...

    def describe(self) -> "DataFrame":
        """Per-column summary statistics."""
        ...

    def null_count(self) -> "DataFrame":
        """A 1-row frame of each column's null count."""
        ...

    def is_duplicated(self) -> Series:
        """Bool Series, true where the whole row is duplicated."""
        ...

    def is_unique(self) -> Series:
        """Bool Series, true where the whole row is unique."""
        ...

    def concat(self, *others: _DataFrame) -> "DataFrame":
        """Vertically concatenate frames."""
        ...

    def group_by(self, key: str, *aggs: Any) -> Any:
        """Group by a key; aggs are 'count', '<op>:<column>', or aggregate
        expressions (F.x.sum(), ...). No aggs returns a GroupBy for .agg(...)."""
        ...

    def group_by_dynamic(
        self,
        time_col: str,
        every: int,
        period: int | None = None,
        aggs: list[str] | None = None,
    ) -> "DataFrame":
        """Tumbling/sliding time-window aggregation over an ascending Int64
        time column; aggs are 'count' / '<op>:<column>'."""
        ...

    def query(
        self,
        dsl: str | None = None,
        *,
        group_by: str | None = None,
        aggs: list[str] | None = None,
        select: list[str] | None = None,
        order_by: str | None = None,
        descending: bool = False,
        limit: int | None = None,
    ) -> "DataFrame":
        """A query plan (filter/group-by/select/sort/limit) over the frame; the
        DSL predicate is evaluated as a SIMD mask."""
        ...

    def join(  # type: ignore[override]  # ty: ignore[invalid-method-override]
        self,
        other: _DataFrame,
        on: Union[int, str, Sequence[str]] = 1,
        how: Literal["inner", "left", "right", "full", "semi", "anti"] = "inner",
    ) -> "DataFrame":
        """Equi-join with `other`; `on` is the shared key name(s) or an int count
        of leading key columns. how is inner|left|right|full|semi|anti."""
        ...

    def asof(
        self,
        other: _DataFrame,
        on: str,
        by: Optional[Union[str, Sequence[str]]] = None,
        direction: Literal["backward", "forward", "nearest"] = "backward",
        tolerance: Optional[int] = None,
    ) -> "DataFrame":
        """Temporal (as-of) join on time column `on`, within optional equi-key
        partition `by`; direction is backward|forward|nearest."""
        ...

    def interval(
        self,
        other: _DataFrame,
        point: str,
        lo: str,
        hi: str,
        by: Optional[Union[str, Sequence[str]]] = None,
        outer: bool = False,
    ) -> "DataFrame":
        """Point-in-range join: match each row's `point` to every `other` row
        whose closed span [lo, hi] contains it. outer keeps unmatched rows."""
        ...

    def window(
        self,
        partition_by: Optional[Sequence[str]] = None,
        order_by: Optional[Sequence[str]] = None,
        specs: Optional[Sequence[WindowSpec]] = None,
    ) -> "DataFrame":
        """SQL window functions, PARTITION BY `partition_by` ORDER BY `order_by`;
        one appended column per spec tuple."""
        ...

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
        """Materialize a regular time grid of width `bucket` keyed on `time`,
        filling `values` per mode none|locf|linear."""
        ...

    def unnest(self, column: str, keep_empty: bool = False) -> "DataFrame":
        """Explode a list-typed column into one row per element; a list<struct>
        flattens its fields. keep_empty keeps empty/null lists as a null row."""
        ...

    def top_k(self, name: str, k: int, largest: bool = True) -> "DataFrame":
        """The k best rows by a column (alias of topk)."""
        ...

    def distinct(self) -> "DataFrame":
        """Drop duplicate rows, keeping the first (alias of unique)."""
        ...

    def sort(self, by: Union[str, Sequence[str]], descending: bool = False) -> "DataFrame":
        """Order rows by one column (str) or lexicographically by several
        (list); the fluent spelling over sort_by / sort_by_multi."""
        ...

    def union(self, *others: _DataFrame) -> "DataFrame":
        """Vertically concatenate with `others` and drop duplicate rows (SQL
        UNION); use concat for UNION ALL."""
        ...

    def pivot(self, index: str, columns: str, values: str, agg: str = "first") -> "DataFrame":
        """Reshape long->wide; agg is first|last|sum|min|max|mean."""
        ...

    def unpivot(self, id_vars: list[str] | str, value_vars: list[str] | str) -> "DataFrame":
        """Reshape wide->long: keep id_vars, stack value_vars into
        'variable'/'value' columns."""
        ...

    def melt(self, id_vars: list[str] | str, value_vars: list[str] | str) -> "DataFrame":
        """Alias of unpivot()."""
        ...

    def explode(self, column: str) -> "DataFrame":
        """Expand a List column: one row per element (empty/null -> one null row)."""
        ...

    def to_dummies(self, column: str) -> "DataFrame":
        """One-hot encode a column into one Int8 column per distinct value."""
        ...

    def apply(self, expr: ColumnExpr) -> Series:
        """Evaluate a columnar expression against this frame and return a Series;
        the frame-first spelling of ``expr.apply(df)``."""
        ...

    def __getitem__(self, name: str) -> Series: ...
    def __contains__(self, name: str) -> bool: ...
    def __len__(self) -> int: ...

class TraceViewer(_TraceViewer):
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    # Builder ops return the wrapped viewer (group_by/agg move to aggregated).
    def filter(self, dsl: "str | Expr") -> "TraceViewer": ...
    def query(self, dsl: "str | Expr") -> "TraceViewer": ...
    def phase(self, phase: str) -> "TraceViewer": ...
    def time_bucket(self, interval_us: int) -> "TraceViewer": ...
    def time_unit(self, unit: str) -> "TraceViewer": ...
    def time_scale(self, ns_ratio: float) -> "TraceViewer": ...
    def time_range(self, begin: float, end: float) -> "TraceViewer": ...
    def select(self, *cols: str) -> "TraceViewer": ...
    def memory_budget(self, nbytes: int) -> "TraceViewer": ...
    def auto_spill(self) -> "TraceViewer": ...
    def limit(self, n: int) -> "TraceViewer": ...
    def offset(self, n: int) -> "TraceViewer": ...
    def sort_by(self, name: str, descending: bool = False) -> "TraceViewer": ...
    def topk(self, name: str, k: int, largest: bool = True) -> "TraceViewer": ...
    def rollup_root(self, path: str) -> "TraceViewer": ...
    def views_root(self, path: str) -> "TraceViewer": ...
    def group_by(self, *keys: str) -> "AggregatedTraceViewer": ...
    def agg(self, *specs: "str | Agg") -> "AggregatedTraceViewer": ...
    def agg_numeric_args(self) -> "AggregatedTraceViewer": ...
    def collect(self) -> DataFrame: ...
    def join(self, other: _TraceViewer, how: str = "inner") -> DataFrame: ...

class AggregatedTraceViewer(_AggregatedTraceViewer):
    # The aggregated viewer keeps its type through further builder ops.
    def filter(self, dsl: "str | Expr") -> "AggregatedTraceViewer": ...
    def query(self, dsl: "str | Expr") -> "AggregatedTraceViewer": ...
    def agg(self, *specs: "str | Agg") -> "AggregatedTraceViewer": ...
    def select(self, *cols: str) -> "AggregatedTraceViewer": ...
    def limit(self, n: int) -> "AggregatedTraceViewer": ...
    def offset(self, n: int) -> "AggregatedTraceViewer": ...
    def sort_by(self, name: str, descending: bool = False) -> "AggregatedTraceViewer": ...
    def topk(self, name: str, k: int, largest: bool = True) -> "AggregatedTraceViewer": ...
    def memory_budget(self, nbytes: int) -> "AggregatedTraceViewer": ...
    def auto_spill(self) -> "AggregatedTraceViewer": ...
    def rollup_root(self, path: str) -> "AggregatedTraceViewer": ...
    def views_root(self, path: str) -> "AggregatedTraceViewer": ...

def _dataframe_from_arrow(table: Any) -> DataFrame: ...
