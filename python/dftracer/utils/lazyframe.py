"""The ``LazyFrame`` Python wrapper: a deferred query over a native batch.

``LazyFrame`` wraps the native ``_LazyFrame`` handle. Builder methods record ops
and return a new ``LazyFrame``; nothing runs until :meth:`collect`, which
materializes a :class:`~dftracer.utils.DataFrame`. Start one with
``df.lazy()`` or ``lazy(df)``. A ``filter``/``with_column`` expression is a
:mod:`~dftracer.utils.columnar` ``Expr`` (``col("dur") > 100``); its column
references resolve by position against the frame's schema at that point, so a
``filter``/``with_column`` after a data-dependent op (``pivot`` / ``to_dummies``
/ ``describe``, whose schema is known only after running) is rejected.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, List, Sequence, Union

from .columnar import Expr, _as_expr, _emit_ast
from .dataframe import DataFrame
from .series import _wrap

if TYPE_CHECKING:
    from . import dftracer_utils_ext as _ext

Value = Union[int, float]


class LazyFrame:
    """A deferred query over a native batch; :meth:`collect` runs it."""

    __slots__ = ("_native",)

    def __init__(self, native: "_ext._LazyFrame") -> None:
        self._native = native

    def schema(self) -> List[str]:
        """Output column names without running, or ``[]`` when the plan ends in
        a data-dependent op (``pivot`` / ``to_dummies`` / ``describe``)."""
        return self._native.schema()

    def explain(self) -> str:
        """The optimized plan as text, for introspection and tests."""
        return self._native.explain()

    def _resolve(self, op: str) -> Callable[[str], int]:
        names = self._native.schema()
        if not names:
            raise ValueError(
                op + "(): the frame's schema is data-dependent here (after a "
                "pivot/to_dummies/describe), so an expression's columns cannot "
                "be resolved by name; apply it before that op"
            )
        index = {name: i for i, name in enumerate(names)}
        return index.__getitem__

    def filter(self, predicate: Expr) -> "LazyFrame":
        """Keep rows where ``predicate`` holds."""
        ast: List[tuple] = []
        _emit_ast(_as_expr(predicate), self._resolve("filter"), ast)
        return LazyFrame(self._native.filter(ast))

    def with_column(self, name: str, expr: Expr) -> "LazyFrame":
        """Add or replace column ``name`` with ``expr``."""
        ast: List[tuple] = []
        _emit_ast(_as_expr(expr), self._resolve("with_column"), ast)
        return LazyFrame(self._native.with_column(name, ast))

    def select(self, *names: str) -> "LazyFrame":
        """Project to ``names``."""
        return LazyFrame(self._native.select(list(names)))

    def rename(self, names: List[str]) -> "LazyFrame":
        """Rename columns positionally to ``names``."""
        return LazyFrame(self._native.rename(list(names)))

    def slice(self, offset: int, length: int) -> "LazyFrame":
        """The row window ``[offset, offset + length)``."""
        return LazyFrame(self._native.slice(offset, length))

    def head(self, n: int) -> "LazyFrame":
        """The first ``n`` rows."""
        return LazyFrame(self._native.head(n))

    def tail(self, n: int) -> "LazyFrame":
        """The last ``n`` rows."""
        return LazyFrame(self._native.tail(n))

    def drop_nulls(self) -> "LazyFrame":
        """Drop rows holding any null."""
        return LazyFrame(self._native.drop_nulls())

    def fill_null(self, value: Value) -> "LazyFrame":
        """Fill nulls in every column with ``value``."""
        return LazyFrame(self._native.fill_null(value))

    def with_row_index(self, name: str = "index") -> "LazyFrame":
        """Prepend an Int64 row-index column."""
        return LazyFrame(self._native.with_row_index(name))

    def null_count(self) -> "LazyFrame":
        """A one-row frame of each column's null count."""
        return LazyFrame(self._native.null_count())

    def explode(self, column: str) -> "LazyFrame":
        """Expand a List ``column``: each element becomes its own row."""
        return LazyFrame(self._native.explode(column))

    def unpivot(self, id_vars: List[str], value_vars: List[str]) -> "LazyFrame":
        """Reshape wide to long."""
        return LazyFrame(self._native.unpivot(list(id_vars), list(value_vars)))

    def melt(self, id_vars: List[str], value_vars: List[str]) -> "LazyFrame":
        """Alias of :meth:`unpivot`."""
        return LazyFrame(self._native.melt(list(id_vars), list(value_vars)))

    def topk(self, name: str, k: int, largest: bool = True) -> "LazyFrame":
        """The ``k`` rows with the largest (or smallest) ``name`` values."""
        return LazyFrame(self._native.topk(name, k, largest))

    def group_by(self, key: Union[str, Sequence[str]], *aggs: str) -> "LazyFrame":
        """Group by ``key`` (one column, or several for a composite key) with
        ``"op[:column]"`` aggregate specs."""
        keys = [key] if isinstance(key, str) else list(key)
        return LazyFrame(self._native.group_by(keys, list(aggs)))

    def sort_by(self, name: str, descending: bool = False) -> "LazyFrame":
        """Sort by one column (external merge sort, bounded memory)."""
        return LazyFrame(self._native.sort_by(name, descending))

    def unique(self) -> "LazyFrame":
        """Distinct rows, first occurrence, in original order."""
        return LazyFrame(self._native.unique())

    def drop_duplicates(self) -> "LazyFrame":
        """Alias of :meth:`unique`."""
        return LazyFrame(self._native.drop_duplicates())

    def sample(self, n: int, seed: int = 0) -> "LazyFrame":
        """A deterministic ``n``-row sample."""
        return LazyFrame(self._native.sample(n, seed))

    def is_duplicated(self) -> "LazyFrame":
        """One Bool column: true where the whole row is duplicated."""
        return LazyFrame(self._native.is_duplicated())

    def is_unique(self) -> "LazyFrame":
        """One Bool column: true where the whole row is unique."""
        return LazyFrame(self._native.is_unique())

    def group_by_dynamic(
        self,
        time_col: str,
        every: int,
        period: Union[int, None] = None,
        aggs: Union[List[str], None] = None,
        origin: int = 0,
        origin_min: bool = False,
    ) -> "LazyFrame":
        """Tumbling/sliding time-window aggregation over an ascending Int64
        time column. ``period`` defaults to ``every`` (tumbling)."""
        return LazyFrame(
            self._native.group_by_dynamic(
                time_col, every, period, list(aggs or []), origin, origin_min
            )
        )

    def pivot(self, index: str, on: str, values: str, agg: str = "first") -> "LazyFrame":
        """Reshape long to wide (schema is data-dependent until collected)."""
        return LazyFrame(self._native.pivot(index, on, values, agg))

    def to_dummies(self, column: str) -> "LazyFrame":
        """One-hot encode ``column`` (schema is data-dependent)."""
        return LazyFrame(self._native.to_dummies(column))

    def describe(self) -> "LazyFrame":
        """Per-column summary statistics (schema is data-dependent)."""
        return LazyFrame(self._native.describe())

    def memory_budget(self, nbytes: int) -> "LazyFrame":
        """Out-of-core spill budget for the pipeline breakers (bytes). 0 means
        auto (~1/3 of memory); a huge value disables spilling."""
        return LazyFrame(self._native.memory_budget(nbytes))

    def auto_spill(self) -> "LazyFrame":
        """Set the spill budget to ~1/3 of available memory."""
        return LazyFrame(self._native.auto_spill())

    def collect(self, morsel_rows: int = 65536) -> DataFrame:
        """Run the pipeline and materialize a :class:`DataFrame`."""
        return _wrap(self._native.collect(morsel_rows))

    def __repr__(self) -> str:
        try:
            plan = self._native.explain()
        except Exception:  # noqa: BLE001 - repr must never raise
            return "<LazyFrame>"
        return "<LazyFrame>\n" + plan


def lazy(frame: DataFrame) -> LazyFrame:
    """Free-function form of :meth:`DataFrame.lazy`."""
    return frame.lazy()
