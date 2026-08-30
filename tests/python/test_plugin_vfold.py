#!/usr/bin/env python3
"""End-to-end @jit.vfold: a per-batch column reduction, folded to a scalar and
materialized like any other jit plugin result."""

import gzip
import shutil

import pytest

from dftracer.utils import jit
from dftracer.utils.plugins import PluginHost

pa = pytest.importorskip("pyarrow")
_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))
_needs_cxx = pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler for the jit backend")


def _write_trace(path, durs, pids=None):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, d in enumerate(durs):
            pid = 1 if pids is None else pids[i]
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":{d},"ph":"X","args":{{}}}}\n'
            )


def _value(result, name):
    return pa.table(result[name]).column("value").to_numpy(zero_copy_only=False)


@_needs_cxx
def test_vfold_scalar_sum(tmp_path):
    @jit.vfold
    class Busy:
        total = jit.sum()

        @jit.each_batch
        def step(self, df):
            self.total += df["dur"].sum()

    durs = [10, 20, 30, 40, 50]
    _write_trace(str(tmp_path / "t.pfw.gz"), durs)
    host = PluginHost()
    host.load(Busy)
    host.resolve()
    result = host.run(str(tmp_path))
    assert float(_value(result, "total").sum()) == float(sum(durs))  # 150


@_needs_cxx
def test_vfold_scalar_min_max(tmp_path):
    @jit.vfold
    class Bounds:
        lo = jit.min()
        hi = jit.max()

        @jit.each_batch
        def step(self, df):
            self.lo += df["dur"].min()
            self.hi += df["dur"].max()

    durs = [30, 10, 50, 20, 40]
    _write_trace(str(tmp_path / "t.pfw.gz"), durs)
    host = PluginHost()
    host.load(Bounds)
    host.resolve()
    result = host.run(str(tmp_path))
    assert int(_value(result, "lo").sum()) == 10
    assert int(_value(result, "hi").sum()) == 50


@_needs_cxx
def test_vfold_keyed_sum_per_pid(tmp_path):
    @jit.vfold
    class PerPid:
        busy = jit.map(key=jit.i64, value=jit.sum())

        @jit.each_batch
        def step(self, df):
            self.busy[df["pid"]] += df["dur"]

    durs = [10, 20, 30, 40]
    pids = [1, 2, 1, 2]
    _write_trace(str(tmp_path / "t.pfw.gz"), durs, pids)
    host = PluginHost()
    host.load(PerPid)
    host.resolve()
    tbl = pa.table(host.run(str(tmp_path))["busy"])
    got = dict(zip(tbl.column(0).to_pylist(), tbl.column("value").to_pylist()))
    assert got == {1: 40.0, 2: 60.0}  # pid1: 10+30, pid2: 20+40


@_needs_cxx
def test_vfold_keyed_sum_per_name(tmp_path):
    @jit.vfold
    class ByName:
        dur = jit.map(key=jit.str_, value=jit.sum())

        @jit.each_batch
        def step(self, df):
            self.dur[df["name"]] += df["dur"]

    names = ["read", "write", "read", "write", "read"]
    durs = [10, 100, 20, 200, 30]
    with gzip.open(str(tmp_path / "t.pfw.gz"), "wt", encoding="utf-8") as f:
        for i, (nm, d) in enumerate(zip(names, durs)):
            f.write(
                f'{{"name":"{nm}","cat":"POSIX","pid":1,"tid":1,'
                f'"ts":{1000 + i},"dur":{d},"ph":"X","args":{{}}}}\n'
            )
    host = PluginHost()
    host.load(ByName)
    host.resolve()
    tbl = pa.table(host.run(str(tmp_path))["dur"])
    got = dict(zip(tbl.column(0).to_pylist(), tbl.column("value").to_pylist()))
    assert got == {"read": 60.0, "write": 300.0}


@_needs_cxx
def test_vfold_keyed_max_per_pid(tmp_path):
    @jit.vfold
    class Peak:
        hi = jit.map(key=jit.i64, value=jit.max())

        @jit.each_batch
        def step(self, df):
            self.hi[df["pid"]] += df["dur"]

    _write_trace(str(tmp_path / "t.pfw.gz"), [10, 90, 30, 40], [1, 2, 1, 2])
    host = PluginHost()
    host.load(Peak)
    host.resolve()
    tbl = pa.table(host.run(str(tmp_path))["hi"])
    got = dict(zip(tbl.column(0).to_pylist(), tbl.column("value").to_pylist()))
    assert got == {1: 30, 2: 90}  # per-pid max via SIMD group-by


@_needs_cxx
def test_vfold_keyed_count_per_pid(tmp_path):
    @jit.vfold
    class Hits:
        n = jit.map(key=jit.i64, value=jit.count())

        @jit.each_batch
        def step(self, df):
            self.n[df["pid"]] += 1

    _write_trace(str(tmp_path / "t.pfw.gz"), [1, 1, 1, 1, 1], [1, 1, 2, 2, 2])
    host = PluginHost()
    host.load(Hits)
    host.resolve()
    tbl = pa.table(host.run(str(tmp_path))["n"])
    got = dict(zip(tbl.column(0).to_pylist(), tbl.column("value").to_pylist()))
    assert got == {1: 2, 2: 3}


class TestAuthoring:
    def test_reducer_must_match_accumulator(self):
        with pytest.raises(jit.JitError):

            @jit.vfold
            class Bad:
                total = jit.sum()

                @jit.each_batch
                def step(self, df):
                    self.total += df["dur"].max()  # max into a sum accumulator

    def test_keyed_field_must_be_numeric(self):
        with pytest.raises(jit.JitError):

            @jit.vfold
            class BadKey:
                by_name = jit.map(key=jit.i64, value=jit.sum())

                @jit.each_batch
                def step(self, df):
                    self.by_name[df["name"]] += df["dur"]  # string key not supported

    def test_needs_one_each_batch(self):
        with pytest.raises(jit.JitError):

            @jit.vfold
            class NoBatch:
                total = jit.sum()
