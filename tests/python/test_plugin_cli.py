#!/usr/bin/env python3
"""End-to-end test for the dftracer_plugin CLI.

`new` scaffolds a template plugin, `build` compiles it against the bundled ABI
headers to a loadable .so, and PluginHost loads and runs it over a fake trace.
The C template folds each batch into a per-process count accumulator whose
value column sums to the scanned event count.
"""

import ctypes
import gzip
import shutil

import pytest

from dftracer.utils import plugin_cli
from dftracer.utils.plugins import PluginHost

_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))


def _write_trace(path: str, n: int) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{1 + i % 2},"tid":1,'
                f'"ts":{1000 + i},"dur":{10 + i},"ph":"X","args":{{}}}}\n'
            )


def test_cflags_prints_include_dir(capsys):
    assert plugin_cli.main(["cflags"]) == 0
    out = capsys.readouterr().out.strip()
    assert "-std=c++20" in out
    assert "-shared" in out
    assert "-I" in out
    assert "include" in out


def test_new_scaffolds_c_source(tmp_path):
    assert plugin_cli.main(["new", "mycount", "-o", str(tmp_path)]) == 0
    src = tmp_path / "mycount.c"
    assert src.is_file()
    assert "dftracer_plugin" in src.read_text()


def test_new_scaffolds_cpp_source(tmp_path):
    assert plugin_cli.main(["new", "mycount", "--cpp", "-o", str(tmp_path)]) == 0
    src = tmp_path / "mycount.cpp"
    assert src.is_file()
    assert "make_plugin" in src.read_text()


def test_build_missing_source_errors(tmp_path):
    assert plugin_cli.main(["build", str(tmp_path / "nope.c")]) == 1


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available")
def test_new_build_load_run_c(tmp_path):
    pa = pytest.importorskip("pyarrow")
    assert plugin_cli.main(["new", "mycount", "-o", str(tmp_path)]) == 0
    src = tmp_path / "mycount.c"
    so = tmp_path / "mycount.so"
    assert plugin_cli.main(["build", str(src), "-o", str(so)]) == 0
    assert so.is_file()

    # It is a real loadable shared object with the ABI entry symbol.
    lib = ctypes.CDLL(str(so))
    assert hasattr(lib, "dftracer_plugin")

    n = 40
    _write_trace(str(tmp_path / "trace.pfw.gz"), n)

    host = PluginHost()
    host.load(str(so))
    results = host.run(str(tmp_path))

    assert "mycount" in results
    tbl = pa.table(results["mycount"])
    assert tbl.column_names == ["pid", "value"]
    val = tbl.column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == n
    assert host.stats["events_scanned"] == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available")
def test_new_build_load_run_cpp(tmp_path):
    pa = pytest.importorskip("pyarrow")
    assert plugin_cli.main(["new", "mycount", "--cpp", "-o", str(tmp_path)]) == 0
    src = tmp_path / "mycount.cpp"
    so = tmp_path / "mycount_cpp.so"
    assert plugin_cli.main(["build", str(src), "-o", str(so)]) == 0
    assert so.is_file()

    n = 40
    _write_trace(str(tmp_path / "trace.pfw.gz"), n)

    host = PluginHost()
    host.load(str(so))
    results = host.run(str(tmp_path))

    tbl = pa.table(results["mycount"])
    val = tbl.column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == n
