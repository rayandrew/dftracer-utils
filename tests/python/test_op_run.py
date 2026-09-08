#!/usr/bin/env python3
"""@jit.op: author a compose op in Python and run it two ways - standalone in
plain Python (JIT-compiled, driven on a standalone compose host) and inlined
into a @jit.each_event plugin body."""

import collections
import gzip
import os
import shutil

import pytest

from dftracer.utils import jit
from dftracer.utils.jit_op import JitOpError
from dftracer.utils.plugins import Plugins

_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))
_needs_cxx = pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler for the jit backend")


@jit.op
def double(x: jit.i64) -> jit.i64:
    return x * 2


@jit.op
def plus10(x: jit.i64) -> jit.i64:
    return x + 10


@jit.op
def half(x: jit.f64) -> jit.f64:
    return x / 2


@_needs_cxx
class TestStandalone:
    def test_single_op(self):
        assert double(5) == 10

    def test_piped_ops(self):
        assert (double | plus10)(5) == 20

    def test_nested_call(self):
        assert plus10(double(5)) == 20

    def test_float_op(self):
        assert half(9.0) == 4.5


class TestAuthoring:
    def test_pipe_type_mismatch_raises(self):
        with pytest.raises(JitOpError):
            _ = double | half  # i64 -> f64 does not chain

    def test_unknown_name_in_body_raises(self):
        with pytest.raises(JitOpError):

            @jit.op
            def bad(x: jit.i64) -> jit.i64:
                return x + y  # noqa: F821 - y is not the input

    def test_non_return_body_raises(self):
        with pytest.raises(JitOpError):

            @jit.op
            def bad(x: jit.i64) -> jit.i64:
                x += 1
                return x


@_needs_cxx
class TestInlineInPlugin:
    def test_op_inlined_into_each_event(self, tmp_path):
        tr = os.path.join(str(tmp_path), "t.pfw.gz")
        n = 250
        with gzip.open(tr, "wt", encoding="utf-8") as f:
            for i in range(n):
                f.write(
                    f'{{"name":"read","cat":"POSIX","pid":1,"tid":1,'
                    f'"ts":{1000 + i},"dur":{i},"ph":"X","args":{{}}}}\n'
                )

        pa = pytest.importorskip("pyarrow")

        @jit.op
        def bucket(x: jit.i64) -> jit.i64:
            return x / 100 * 100

        @jit.plugin
        class WithOp:
            counts = jit.map(key=jit.i64, value=jit.count())

            @jit.each_event
            def step(self, e):
                self.counts[(bucket(e.dur),)] += 1

        # The op lands as a static C function and is called from on_batch.
        src = WithOp._jit_plugin.source
        assert "static int64_t dftu_jitop_bucket" in src
        assert "dftu_jitop_bucket(" in src.split("on_batch")[1]

        plugins = Plugins([WithOp])
        table = pa.table(plugins.run([tr]).results["counts"])
        got = dict(zip(table.column("k0").to_pylist(), table.column("value").to_pylist()))
        ref = collections.Counter((i // 100) * 100 for i in range(n))
        assert got == dict(ref)
