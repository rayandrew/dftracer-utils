"""Arrow data interchange and I/O for DFTracer.

Provides:
- ArrowBatch and ArrowTable classes that wrap Arrow C Data Interface
  objects (PyCapsules) with convenience methods for conversion to pandas
  and polars DataFrames.
- write_arrow() and read_arrow() for Arrow IPC file I/O with Runtime
  parallelization.

These wrappers are pure Python. The actual Arrow data is produced by the
C extension (TraceViewer.stream/collect, utility to_arrow methods). Conversion
to pandas requires pyarrow; conversion to polars requires polars. Neither
is a required dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Optional

if TYPE_CHECKING:
    import pyarrow as pa

_HAS_PYARROW = False

try:
    import pyarrow as pa

    _HAS_PYARROW = True
except ImportError:
    pass


def _require_pyarrow() -> None:
    """Raise a clear, consistent error when pyarrow is unavailable."""
    if not _HAS_PYARROW:
        raise ImportError("pyarrow is required. Install with: pip install pyarrow")


def ipc_to_table(ipc_bytes: bytes) -> Any:
    """Deserialize Arrow IPC stream bytes into a pyarrow Table."""
    _require_pyarrow()
    return pa.ipc.open_stream(pa.BufferReader(ipc_bytes)).read_all()


def decode_dictionary_columns(table: Any) -> Any:
    """Cast dictionary-encoded columns to plain strings.

    Workers may dictionary-encode independently, so unify before concatenation.
    """
    for i, field in enumerate(table.schema):
        if pa.types.is_dictionary(field.type):
            table = table.set_column(i, field.name, table.column(i).cast(pa.string()))
    return table


class ArrowTable:
    """Wrapper around a collection of Arrow RecordBatches.

    Returned by read_arrow() and utility process() methods. Supports
    the Arrow PyCapsule stream protocol (__arrow_c_stream__) for
    zero-copy interchange.

    Accepts either a pre-built list of batches or a lazy iterator.
    When constructed from an iterator, batches are not materialized
    until data access (to_pandas, to_polars, batches, etc.).

    ``num_rows`` is special: if the iterator has not been consumed yet,
    it streams through counting rows without retaining batches (O(1)
    memory). After a streaming ``num_rows``, data access methods will
    return empty results -- use ``iter_arrow`` directly if you need
    both count and data for very large datasets.
    """

    def __init__(
        self,
        batches: Any,
        schema_capsule: Optional[Any] = None,
    ) -> None:
        self._stream: Any = None
        if isinstance(batches, list):
            self._batches: Optional[list[Any]] = batches
            self._iter: Optional[Iterator[Any]] = None
        elif hasattr(batches, "__arrow_c_stream__"):
            self._batches = None
            self._iter = None
            self._stream = batches
        else:
            self._batches = None
            self._iter = iter(batches)
        self._schema_capsule = schema_capsule
        self._pa_table: Any = None

    def _materialize(self) -> list[Any]:
        if self._batches is not None:
            return self._batches
        if self._stream is not None:
            self._to_pa_table()
            if self._pa_table is not None:
                self._batches = list(self._pa_table.to_batches())
                return self._batches
        if self._iter is not None:
            self._batches = list(self._iter)
            self._iter = None
            return self._batches
        self._batches = []
        return self._batches

    def _to_pa_table(self) -> Any:
        """Convert to pyarrow Table, caching the result.

        Arrow C Data Interface export is single-use (ownership transfer),
        so we cache the pyarrow table on first conversion.  After
        conversion the batch capsule references are cleared since pyarrow
        now owns the underlying buffers.

        Returns:
            pyarrow.Table: The converted table.

        Raises:
            ImportError: If pyarrow is not installed.
        """
        if self._pa_table is not None:
            return self._pa_table
        _require_pyarrow()
        import pyarrow as pa

        if self._stream is not None:
            self._pa_table = pa.table(self._stream)
            self._stream = None
            return self._pa_table
        batches = self._materialize()
        pa_batches = [pa.record_batch(b) for b in batches]
        self._batches = None
        if not pa_batches:
            schema = pa.schema([])
            if self._schema_capsule is not None:
                schema = pa.Schema.from_arrow(self._schema_capsule)
            self._pa_table = pa.table({}, schema=schema)
        else:
            self._pa_table = pa.Table.from_batches(pa_batches)
        return self._pa_table

    def __arrow_c_stream__(self, requested_schema: Any = None) -> Any:
        """Arrow C Stream Interface -- yields batches to consumers."""
        return self._to_pa_table().__arrow_c_stream__(requested_schema)

    @property
    def num_batches(self) -> int:
        """Number of batches."""
        return len(self._materialize())

    @property
    def num_rows(self) -> int:
        """Total number of rows across all batches."""
        if self._pa_table is not None:
            return self._pa_table.num_rows
        return sum(b.num_rows for b in self._materialize())

    @property
    def empty(self) -> bool:
        """True if there are no batches."""
        if self._pa_table is not None:
            return self._pa_table.num_rows == 0
        return len(self._materialize()) == 0

    def batch(self, i: int) -> Any:
        """Get the i-th batch."""
        return self._materialize()[i]

    def batches(self) -> Iterator[Any]:
        """Iterate over batches."""
        return iter(self._materialize())

    def to_pandas(self) -> Any:
        """Convert all batches to a single pandas DataFrame.

        Returns:
            pandas.DataFrame: The converted DataFrame.

        Raises:
            ImportError: If pyarrow is not installed.
        """
        return self._to_pa_table().to_pandas()

    def to_polars(self) -> Any:
        """Convert all batches to a single polars DataFrame.

        Returns:
            polars.DataFrame: The converted DataFrame.

        Raises:
            ImportError: If polars is not installed.
        """
        try:
            import polars as pl  # type: ignore[import-not-found]  # ty: ignore[unresolved-import]
        except ImportError:
            raise ImportError(
                "polars is required for to_polars(). Install with: pip install polars"
            ) from None
        table = self._to_pa_table()
        if table.num_rows == 0:
            return pl.DataFrame()
        return pl.from_arrow(table)
