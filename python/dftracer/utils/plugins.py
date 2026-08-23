"""Load and run compiled DFTracer analysis plugins from Python.

A plugin is a compiled shared library exporting the ``dftracer_plugin`` ABI
symbol. :class:`PluginHost` loads one or more, resolves their capabilities, and
runs them as folds over a single fused parallel scan of a trace set.

Example::

    from dftracer.utils.plugins import PluginHost

    host = PluginHost()
    host.load("process_counts.so")     # a compiled .so path
    host.load(MyPlugin)                 # or a @jit.plugin class (compiled + cached)
    assert host.resolve()
    results = host.run("./traces")
    print(results["process_counts"])          # the plugin's emitted bytes
    print(host.stats["events_scanned"])
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Dict, List, Optional, Union

from .dftracer_utils_ext import PluginHost as _NativePluginHost

if TYPE_CHECKING:
    import pyarrow as pa  # ty: ignore[unresolved-import]

    from .runtime import Runtime

__all__ = ["PluginHost", "unnest"]

# A JSON-serializable value tree (config is handed straight to json.dumps).
JSONValue = Union[str, int, float, bool, None, List["JSONValue"], Dict[str, "JSONValue"]]

# A run() map result value: emitted bytes, an eager Arrow table, or a pull-based
# reader when the map streamed to multiple batches.
_RunResult = Union[bytes, "pa.Table", "pa.RecordBatchReader"]


def unnest(
    result: "Union[pa.Table, pa.RecordBatchReader, pa.RecordBatch]",
    column: str,
    keep_empty: bool = False,
) -> "pa.Table":
    """Explode a list-typed ``column`` of a ``run()`` map result into one row per
    element, repeating the other columns; the inverse of the set/list/top-k
    monoids.

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

    __slots__ = ("_native", "_renames")

    def __init__(self, runtime: "Optional[Runtime]" = None) -> None:
        """Create a host bound to ``runtime`` (a ``Runtime`` or None for the
        module default)."""
        self._native = _NativePluginHost(runtime)
        self._renames: Dict[str, List[str]] = {}

    # config values are an arbitrary JSON-serializable object tree (json.dumps'd).
    def load(self, path: Union[str, type], config: Optional[Dict[str, JSONValue]] = None) -> None:
        """dlopen the compiled plugin at ``path``.

        ``path`` is either a compiled ``.so`` path or a ``@jit.plugin`` class,
        which is AST-compiled to a cached native ``.so`` first. ``config`` is an
        optional JSON-serializable dict handed to the plugin factory. Raises
        ImportError on load/symbol/ABI failure, DFTUtilsValueError on a bad
        config, or jit.JitError on a compile failure.
        """
        if not isinstance(path, str):
            from . import jit

            self._renames.update(jit.plugin_renames(path))
            path = jit.compile_class(path)
        self._native.load(path, json.dumps(config) if config is not None else None)

    def resolve(self) -> bool:
        """Wire plugin capabilities. False on an unmet or reserved capability
        (the reason is already logged)."""
        return bool(self._native.resolve())

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
        :attr:`stats`.

        A result's type depends on what the plugin emitted:

        - ``emit_result`` bytes -> ``bytes``.
        - A map that materialized to a single Arrow batch (the default, and any
          map that fit in memory) -> an eager Arrow table (``pa.table(result)``
          works, as do ``.column`` / ``.num_rows``).
        - A map materialized as multiple batches (streamed: the runtime has map
          streaming enabled via ``DFTRACER_PLUGIN_MAP_STREAM=1`` and the map
          spilled/partitioned into more than one batch) -> a pull-based
          ``pyarrow.RecordBatchReader``. Call ``.read_all()`` for a table. This
          keeps peak memory near one partition instead of the whole result.
        """
        return self._rename_columns(self._native.run(traces, index_dir, auto_index))

    def _rename_columns(self, results: "Dict[str, _RunResult]") -> "Dict[str, _RunResult]":
        """Rename jit product value columns v0.. to the field names a @jit.plugin
        declared; non-jit results are untouched."""
        if not self._renames:
            return results
        import pyarrow as pa

        for name, fields in self._renames.items():
            obj = results.get(name)
            if obj is None:
                continue
            table = pa.table(obj)
            cols = list(table.column_names)
            for i, field in enumerate(fields):
                v = f"v{i}"
                if v in cols:
                    cols[cols.index(v)] = field
            results[name] = table.rename_columns(cols)
        return results

    @property
    def stats(self) -> Optional[Dict[str, int]]:
        """The last run's scan counters ``{"events_scanned", "events_matched"}``,
        or ``None`` before the first :meth:`run`."""
        return self._native.stats
