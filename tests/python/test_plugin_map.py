#!/usr/bin/env python3
"""End-to-end test for the mergeable-map graph workflow.

Runs the compiled process_file_edges plugin (pure C ABI) over a generated trace
through PluginHost; the host merges a {pid, fhash} -> COUNTER map across workers
and materializes it to an Arrow table surfaced to Python as a pyarrow.Table. The
graph is then built in bulk from the columns (numpy + scipy), no per-edge Python.
"""

import glob
import gzip
import os
from pathlib import Path

import pytest

from dftracer.utils.plugins import PluginHost, unnest

_REPO_ROOT = Path(__file__).resolve().parents[2]

np = pytest.importorskip("numpy")
sparse = pytest.importorskip("scipy.sparse")
pa = pytest.importorskip("pyarrow")


def _find_plugin(name: str, env_var: str) -> str:
    env = os.environ.get(env_var)
    if env and os.access(env, os.R_OK):
        return env
    for suffix in ("so", "dylib"):
        hits = glob.glob(
            str(_REPO_ROOT / "**" / "examples" / "plugins" / f"{name}.{suffix}"),
            recursive=True,
        )
        if hits:
            return hits[0]
    return ""


_PLUGIN = _find_plugin("process_file_edges", "DFTRACER_PROCESS_FILE_EDGES_PLUGIN_PATH")
_PLUGIN_WIDE = _find_plugin(
    "process_file_edges_wide", "DFTRACER_PROCESS_FILE_EDGES_WIDE_PLUGIN_PATH"
)
_PLUGIN_NAMES = _find_plugin("name_edges", "DFTRACER_NAME_EDGES_PLUGIN_PATH")
_PLUGIN_SET = _find_plugin("process_file_set", "DFTRACER_PROCESS_FILE_SET_PLUGIN_PATH")
_PLUGIN_SEQ = _find_plugin("process_event_seq", "DFTRACER_PROCESS_EVENT_SEQ_PLUGIN_PATH")
_PLUGIN_DURS = _find_plugin("process_pid_durs", "DFTRACER_PROCESS_PID_DURS_PLUGIN_PATH")


def _write_trace(path: str, n: int, pids, files) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            pid = pids[i % len(pids)]
            fhash = files[i % len(files)]
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":{10 + i},"ph":"X",'
                f'"args":{{"fhash":"{fhash}","ret":{i}}}}}\n'
            )


@pytest.mark.skipif(
    not _PLUGIN, reason="no compiled process_file_edges plugin found in any build tree"
)
def test_process_file_edges_builds_adjacency(tmp_path):
    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(_PLUGIN)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "process_file_edges" in results
    # The host materialized the map to an Arrow table (zero new Python needed).
    tbl = pa.table(results["process_file_edges"])
    assert tbl.column_names == ["k0", "k1", "value"]

    k0 = tbl.column("k0").to_numpy(zero_copy_only=False)
    k1 = tbl.column("k1").to_numpy(zero_copy_only=False)
    val = tbl.column("value").to_numpy(zero_copy_only=False)

    # Every file event contributed 1, so the weights sum to the event count.
    assert int(val.sum()) == n

    # Build the weighted process<->file adjacency from bulk columns.
    procs, pi = np.unique(k0, return_inverse=True)
    fhs, fi = np.unique(k1, return_inverse=True)
    adj = sparse.coo_matrix((val, (pi, fi)), shape=(procs.size, fhs.size))
    assert adj.shape == (len(pids), len(files))
    assert int(adj.sum()) == n


@pytest.mark.skipif(
    not _PLUGIN_WIDE,
    reason="no compiled process_file_edges_wide plugin found in any build tree",
)
def test_process_file_edges_wide_builds_product_columns(tmp_path):
    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(_PLUGIN_WIDE)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "process_file_edges_wide" in results
    tbl = pa.table(results["process_file_edges_wide"])
    assert tbl.column_names == ["k0", "k1", "v0", "v1"]

    v0 = tbl.column("v0").to_numpy(zero_copy_only=False)
    v1 = tbl.column("v1").to_numpy(zero_copy_only=False)

    # v0 is the per-edge event count; v1 the per-edge total duration.
    assert int(v0.sum()) == n
    assert float(v1.sum()) == float(sum(10 + i for i in range(n)))


@pytest.mark.skipif(
    not _PLUGIN_NAMES, reason="no compiled name_edges plugin found in any build tree"
)
def test_name_edges_resolves_str_key_column(tmp_path):
    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(_PLUGIN_NAMES)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "name_edges" in results
    tbl = pa.table(results["name_edges"])
    assert tbl.column_names == ["k0", "k1", "value"]

    # k1 is the STR key component resolved to a string column, not raw ids.
    assert pa.types.is_string(tbl.schema.field("k1").type)
    names = set(tbl.column("k1").to_pylist())
    assert names == {"read"}

    val = tbl.column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == n


@pytest.mark.skipif(
    not _PLUGIN_SET,
    reason="no compiled process_file_set plugin found in any build tree",
)
def test_process_file_set_builds_list_string_column(tmp_path):
    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(_PLUGIN_SET)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "process_file_set" in results
    tbl = pa.table(results["process_file_set"])
    assert tbl.column_names == ["k0", "value"]

    # value is a list<string> column: the set of files each pid touched.
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_string(vtype.value_type)

    by_pid = {
        k: set(v) for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())
    }
    assert by_pid[1] == set(files)
    assert by_pid[2] == set(files)


def _write_seq_trace(path: str, events) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for pid, name, ts in events:
            f.write(
                f'{{"name":"{name}","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{ts},"dur":1,"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(
    not _PLUGIN_SEQ,
    reason="no compiled process_event_seq plugin found in any build tree",
)
def test_process_event_seq_builds_ts_ordered_list(tmp_path):
    # File order differs from ts order: pid 1 -> [a,b,c] by ts.
    events = [
        (1, "c", 30),
        (1, "a", 10),
        (1, "b", 20),
        (2, "y", 5),
        (2, "x", 1),
    ]
    _write_seq_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(_PLUGIN_SEQ)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "process_event_seq" in results
    tbl = pa.table(results["process_event_seq"])
    assert tbl.column_names == ["k0", "value"]

    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_string(vtype.value_type)

    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    assert by_pid[1] == ["a", "b", "c"]
    assert by_pid[2] == ["x", "y"]


@pytest.mark.skipif(
    not _PLUGIN, reason="no compiled process_file_edges plugin found in any build tree"
)
def test_streamed_map_returns_record_batch_reader(tmp_path, monkeypatch):
    # Many distinct (pid, file) keys + a tiny budget force the map to spill and
    # partition (K=256); with streaming enabled it materializes as several
    # batches and surfaces as a pull-based RecordBatchReader.
    keys = 300
    with gzip.open(str(tmp_path / "trace.pfw.gz"), "wt", encoding="utf-8") as f:
        for i in range(keys):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{i},"tid":1,'
                f'"ts":{1000 + i},"dur":{10 + i},"ph":"X",'
                f'"args":{{"fhash":"file{i}","ret":{i}}}}}\n'
            )

    spill_dir = tmp_path / "spill"
    spill_dir.mkdir()
    monkeypatch.setenv("DFTRACER_PLUGIN_MAP_STREAM", "1")
    # Global budget / 16 workers => a ~256 B share, far below the key count.
    monkeypatch.setenv("DFTRACER_PLUGIN_MAP_MEM_BUDGET", "4096")
    monkeypatch.setenv("DFTRACER_PLUGIN_MAP_SPILL_DIR", str(spill_dir))

    host = PluginHost()
    host.load(_PLUGIN)
    host.resolve()
    results = host.run(str(tmp_path))

    reader = results["process_file_edges"]
    assert isinstance(reader, pa.RecordBatchReader)

    # The streamed batches are self-contained: spill runs/temp dirs are cleaned
    # during the run's finalize, before the reader is ever pulled. So no
    # per-run dir lingers even before read_all, and (below) dropping the reader
    # unconsumed leaks nothing either.
    def _run_dirs():
        return [p for p in spill_dir.iterdir() if p.is_dir()]

    assert _run_dirs() == []

    tbl = reader.read_all()
    assert tbl.column_names == ["k0", "k1", "value"]
    # Every distinct key contributed exactly 1; concatenated batches recover all.
    assert tbl.num_rows == keys
    assert int(tbl.column("value").to_numpy(zero_copy_only=False).sum()) == keys
    k0 = tbl.column("k0").to_numpy(zero_copy_only=False)
    assert len(set(int(x) for x in k0)) == keys
    assert _run_dirs() == []  # still clean after full consumption

    # Drop a fresh reader unconsumed: no temp dir reappears (RAII), and the
    # unpulled Arrow batches are released without error.
    reader2 = host.run(str(tmp_path))["process_file_edges"]
    assert isinstance(reader2, pa.RecordBatchReader)
    del reader2
    assert _run_dirs() == []


@pytest.mark.skipif(
    not _PLUGIN, reason="no compiled process_file_edges plugin found in any build tree"
)
def test_small_map_stays_eager_table(tmp_path):
    # Without the streaming env, a small map stays a single eager Arrow table,
    # not a RecordBatchReader: pre-existing ergonomics are unaffected.
    _write_trace(str(tmp_path / "trace.pfw.gz"), 30, [1, 2], ["fileA", "fileB"])

    host = PluginHost()
    host.load(_PLUGIN)
    host.resolve()
    results = host.run(str(tmp_path))

    obj = results["process_file_edges"]
    assert not isinstance(obj, pa.RecordBatchReader)
    tbl = pa.table(obj)
    assert tbl.column_names == ["k0", "k1", "value"]


def _write_dur_trace(path: str, events) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for pid, dur, ts in events:
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{ts},"dur":{dur},"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(
    not _PLUGIN_DURS,
    reason="no compiled process_pid_durs plugin found in any build tree",
)
def test_process_pid_durs_builds_ts_ordered_int64_list(tmp_path):
    # File order differs from ts order: pid 1 durations by ts -> [20,5,20,8].
    events = [
        (1, 20, 10),
        (1, 5, 20),
        (1, 20, 30),
        (1, 8, 40),
        (2, 3, 1),
        (2, 7, 5),
    ]
    _write_dur_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(_PLUGIN_DURS)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "process_pid_durs" in results
    tbl = pa.table(results["process_pid_durs"])
    assert tbl.column_names == ["k0", "value"]

    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_integer(vtype.value_type)

    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    assert by_pid[1] == [20, 5, 20, 8]
    assert by_pid[2] == [3, 7]


def test_unnest_string_list():
    tbl = pa.table(
        {"pid": [1, 2], "files": [["a", "b", "c"], ["x"]]},
        schema=pa.schema([("pid", pa.int64()), ("files", pa.list_(pa.string()))]),
    )
    out = unnest(tbl, "files")
    assert out.column_names == ["pid", "files"]
    assert out.num_rows == 4
    assert out.column("pid").to_pylist() == [1, 1, 1, 2]
    assert out.column("files").to_pylist() == ["a", "b", "c", "x"]


def test_unnest_int64_list():
    tbl = pa.table(
        {"pid": [1, 2], "durs": [[20, 5, 8], [3]]},
        schema=pa.schema([("pid", pa.int64()), ("durs", pa.list_(pa.int64()))]),
    )
    out = unnest(tbl, "durs")
    assert out.num_rows == 4
    assert out.column("pid").to_pylist() == [1, 1, 1, 2]
    assert out.column("durs").to_pylist() == [20, 5, 8, 3]


def test_unnest_struct_list_flattens_fields():
    struct_ty = pa.struct([("value", pa.string()), ("count", pa.int64())])
    tbl = pa.table(
        {
            "pid": [1, 2],
            "tk": [
                [{"value": "read", "count": 5}, {"value": "write", "count": 2}],
                [{"value": "open", "count": 9}],
            ],
        },
        schema=pa.schema([("pid", pa.int64()), ("tk", pa.list_(struct_ty))]),
    )
    out = unnest(tbl, "tk")
    assert out.column_names == ["pid", "value", "count"]
    assert out.num_rows == 3
    assert out.column("pid").to_pylist() == [1, 1, 2]
    assert out.column("value").to_pylist() == ["read", "write", "open"]
    assert out.column("count").to_pylist() == [5, 2, 9]


def test_unnest_empty_list_drop_vs_keep_empty():
    tbl = pa.table(
        {"pid": [1, 2, 3], "files": [["a"], [], ["b"]]},
        schema=pa.schema([("pid", pa.int64()), ("files", pa.list_(pa.string()))]),
    )
    dropped = unnest(tbl, "files")
    assert dropped.column("pid").to_pylist() == [1, 3]
    assert dropped.column("files").to_pylist() == ["a", "b"]

    kept = unnest(tbl, "files", keep_empty=True)
    assert kept.column("pid").to_pylist() == [1, 2, 3]
    assert kept.column("files").to_pylist() == ["a", None, "b"]


@pytest.mark.skipif(
    not _PLUGIN_SET,
    reason="no compiled process_file_set plugin found in any build tree",
)
def test_unnest_aggregated_set_recovers_edges(tmp_path):
    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(_PLUGIN_SET)
    host.resolve()
    results = host.run(str(tmp_path))

    # The aggregated per-pid file set explodes back into scannable (pid, file)
    # rows: the inverse of the SET_STR monoid.
    out = unnest(results["process_file_set"], "value")
    assert out.column_names == ["k0", "value"]
    edges = set(zip(out.column("k0").to_pylist(), out.column("value").to_pylist()))
    assert edges == {(p, f) for p in pids for f in files}
