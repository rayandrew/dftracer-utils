"""Load and run compiled DFTracer analysis plugins from Python.

A plugin is a compiled shared library exporting the ``dftracer_plugin`` ABI
symbol. :class:`PluginHost` loads one or more and runs them as folds over a
single fused parallel scan of a trace set.

Example::

    from dftracer.utils.plugins import PluginHost

    host = PluginHost()
    host.load("process_counts.so")     # a compiled .so path
    host.load(MyPlugin)                 # or a @jit.plugin class (compiled + cached)
    results = host.run("./traces")
    print(results["process_counts"])          # the plugin's emitted bytes
    print(host.stats["events_scanned"])
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .dataframe import DataFrame
from .dftracer_utils_ext import PluginHost as _NativePluginHost
from .dftracer_utils_ext import _DataFrame
from .lazyframe import LazyFrame

if TYPE_CHECKING:
    import pyarrow as pa  # ty: ignore[unresolved-import]

    from .runtime import Runtime

__all__ = ["PluginHost", "unnest"]

# A JSON-serializable value tree (config is handed straight to json.dumps).
JSONValue = Union[str, int, float, bool, None, List["JSONValue"], Dict[str, "JSONValue"]]

# One run() result value per emit kind: bytes, a native frame, a lazy plan, or
# an Arrow table.
_RunResult = Union[bytes, "DataFrame", "LazyFrame", "pa.Table"]


def unnest(
    result: "Union[pa.Table, pa.RecordBatchReader, pa.RecordBatch]",
    column: str,
    keep_empty: bool = False,
) -> "pa.Table":
    """Explode a list-typed ``column`` of a ``run()`` map result into one row per
    element, repeating the other columns; the inverse of the set/list/top-k
    aggregates.

    ``result`` is any ``run()`` map value (an eager Arrow table, a
    ``pyarrow.RecordBatchReader``, or a ``pyarrow.RecordBatch``). ``column`` names
    a ``list<utf8>``, ``list<int64>``, or ``list<struct<...>>`` column; a
    ``list<struct>`` flattens its fields into columns (named by the struct
    fields). Returns a ``pyarrow.Table``.

    An empty or null list drops the row (SQL inner unnest). With
    ``keep_empty=True`` it emits one row whose exploded column(s) are null (SQL
    outer/left unnest). Row order is input order, then element order within a row.
    """
    import pyarrow as pa

    from .dftracer_utils_ext import unnest as _unnest

    if isinstance(result, pa.RecordBatchReader):
        result = result.read_all()
    if isinstance(result, pa.RecordBatch):
        batch = result
    else:
        tbl = pa.table(result).combine_chunks()
        batches = tbl.to_batches()
        batch = (
            batches[0]
            if batches
            else pa.RecordBatch.from_arrays(
                [pa.array([], type=f.type) for f in tbl.schema], schema=tbl.schema
            )
        )
    return pa.table(_unnest(batch, column, keep_empty))


class PluginHost:
    """Load and run compiled DFTracer plugins over trace files."""

    __slots__ = ("_native", "_renames", "_result_names")

    def __init__(self, runtime: "Optional[Runtime]" = None) -> None:
        """Create a host bound to ``runtime`` (a ``Runtime`` or None for the
        module default)."""
        self._native = _NativePluginHost(runtime)
        self._renames: Dict[str, List[str]] = {}
        self._result_names: Dict[str, str] = {}

    # config values are an arbitrary JSON-serializable object tree (json.dumps'd).
    def load(self, path: Union[str, type], config: Optional[Dict[str, JSONValue]] = None) -> None:
        """Queue the compiled plugin at ``path``.

        ``path`` is either a compiled ``.so`` path or a ``@jit.plugin`` class,
        which is AST-compiled to a cached native ``.so`` first. ``config`` is an
        optional JSON-serializable dict handed to the plugin factory. The dlopen,
        the ABI check, and capability resolution all run when the plugin set is
        first used, so :meth:`run` is what raises ImportError on a
        load/symbol/ABI/capability failure. Raises DFTUtilsValueError on a bad
        config, or jit.JitError on a compile failure.
        """
        if not isinstance(path, str):
            from . import jit

            self._renames.update(jit.plugin_renames(path))
            self._result_names.update(jit.plugin_result_names(path))
            path = jit.compile_class(path)
        self._native.load(path, json.dumps(config) if config is not None else None)

    def run(
        self,
        traces: Union[str, List[str]],
        index_dir: Optional[str] = None,
        auto_index: bool = True,
    ) -> "Dict[str, _RunResult]":
        """Fold every loaded plugin over one fused scan of ``traces`` (a file, a
        directory, or a list of paths).

        With ``auto_index`` the traces are normalized and indexed first. Returns
        the emitted named results as ``{name: value}``; the scan counters are on
        :attr:`stats`. Raises ImportError if a loaded plugin fails to load, fails
        the ABI check, or has an unmet or reserved capability.

        A result's type depends on what the plugin emitted:

        The shape follows the plugin's emit call, never whether a conversion
        happens to succeed:

        - ``emit_frame`` -> our :class:`~dftracer.utils.DataFrame`, zero-copy.
          Call ``.to_arrow()`` / ``.to_pandas()`` for other shapes.
        - ``emit_arrow`` -> a ``pyarrow.Table`` as emitted, nested types
          included.
        - ``emit_lazyframe`` -> a :class:`~dftracer.utils.LazyFrame`.
        - ``emit_result`` bytes -> ``bytes``.
        """
        raw = self._native.run(traces, index_dir, auto_index)
        out: "Dict[str, _RunResult]" = {}
        for wire_name, obj in raw.items():
            name = self._result_names.get(wire_name, wire_name)
            out[name] = self._shape(name, obj)
        return out

    def _shape(self, name: str, obj: object) -> "_RunResult":
        """Return the result in the shape its emit kind implies.

        The native layer already hands back one Python type per emit kind, so
        this only renames a jit result's positional v0.. value columns. It must
        never re-derive the kind by trying a conversion: coercing an Arrow
        result into a DataFrame made the return type depend on whether the
        engine could import the schema, so the same plugin returned a DataFrame
        for a flat result and a pyarrow.Table for a nested one.
        """
        if isinstance(obj, bytes):
            return obj
        if isinstance(obj, _DataFrame):
            return DataFrame(obj)
        fields = self._renames.get(name)
        if not fields:
            return obj

        import pyarrow as pa

        if not isinstance(obj, pa.Table):
            return obj
        cols = list(obj.column_names)
        for i, field in enumerate(fields):
            v = f"v{i}"
            if v in cols:
                cols[cols.index(v)] = field
        return obj.rename_columns(cols)

    @property
    def stats(self) -> Optional[Dict[str, int]]:
        """The last run's scan counters ``{"events_scanned", "events_matched"}``,
        or ``None`` before the first :meth:`run`."""
        return self._native.stats
