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


def _write_trace(path, durs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, d in enumerate(durs):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":1,"tid":1,'
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


class TestAuthoring:
    def test_reducer_must_match_accumulator(self):
        with pytest.raises(jit.JitError):

            @jit.vfold
            class Bad:
                total = jit.sum()

                @jit.each_batch
                def step(self, df):
                    self.total += df["dur"].max()  # max into a sum accumulator

    def test_keyed_maps_not_supported_yet(self):
        with pytest.raises(jit.JitError):

            @jit.vfold
            class Keyed:
                busy = jit.map(key=jit.i64, value=jit.sum())

                @jit.each_batch
                def step(self, df):
                    self.busy[df["pid"]] += df["dur"]

    def test_needs_one_each_batch(self):
        with pytest.raises(jit.JitError):

            @jit.vfold
            class NoBatch:
                total = jit.sum()
