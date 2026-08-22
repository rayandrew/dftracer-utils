#!/usr/bin/env python3
"""Cross-worker shared handles for @jit.plugin authoring.

A jit plugin declares a jit.shared handle of a scalar monoid; each event
contributes with self.<handle>.add(<expr>) (or += for the additive count/sum
kinds). The host merges same-cap handles across every worker slice, and the
plugin's generated finalize reads the merged value and emits it under the
attribute name. These tests cover the generated C without a compiler and the
end-to-end cross-worker reduction with one.
"""

import gzip
import shutil
import struct

import pytest

from dftracer.utils import jit
from dftracer.utils.plugins import PluginHost

pa = pytest.importorskip("pyarrow")

_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))


def _write_trace(path: str, durs) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, d in enumerate(durs):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{1 + i % 3},"tid":1,'
                f'"ts":{1000 + i},"dur":{d},"ph":"X","args":{{}}}}\n'
            )


def _u64(b) -> int:
    return struct.unpack("<Q", bytes(b))[0]


def _f64(b) -> float:
    return struct.unpack("<d", bytes(b))[0]


# --- codegen-only checks (no C++ compiler needed) ---------------------------


def test_jit_shared_emits_get_add_and_finalize_emit():
    @jit.plugin
    class Stats:
        per_pid = jit.map(key=(jit.i64,), value=jit.count())
        total_dur = jit.shared("com.example.dur_total", jit.sum())
        n_events = jit.shared("com.example.n", jit.count())
        max_dur = jit.shared("com.example.maxdur", jit.max())

        @jit.each_event
        def step(self, e):
            self.per_pid[(e.pid,)] += 1
            self.total_dur += e.dur
            self.n_events.add(1)
            self.max_dur.add(e.dur)

    src = Stats._jit_plugin.source
    assert "DFTU_EXT_HANDLES" in src and "DFTU_EXT_RESULT" in src
    assert '_handles->shared_get(host->h, "com.example.dur_total", DFTU_MONOID_SUM_F64)' in src
    assert '_handles->shared_get(host->h, "com.example.maxdur", DFTU_MONOID_MAX_U64)' in src
    assert "_handles->add_f64(host->h, _hd_total_dur, (double)(e->dur), 1.0);" in src
    assert "_handles->add_u64(host->h, _hd_n_events, (uint64_t)(1));" in src
    # The generated finalize reads each merged handle and emits it by attr name.
    assert '_handles->result(host->h, "com.example.dur_total", &_v)' in src
    assert '_result->emit(host->h, "total_dur", &_v.as, 8u);' in src


def test_jit_shared_rejects_reserved_namespace():
    with pytest.raises(jit.JitError, match="reserved dftu. namespace"):
        jit.shared("dftu.cap.events", jit.count())


def test_jit_shared_rejects_bad_charset():
    with pytest.raises(jit.JitError, match="ASCII"):
        jit.shared("Bad Id!", jit.count())


def test_jit_shared_rejects_unsupported_monoid():
    with pytest.raises(jit.JitError, match="does not support"):
        jit.shared("ok.id", jit.mean())


def test_jit_shared_rejects_aug_on_non_additive():
    with pytest.raises(jit.JitError, match="non-additive"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.count())
            hi = jit.shared("a.b", jit.max())

            @jit.each_event
            def step(self, e):
                self.hi += e.dur


# --- end-to-end cross-worker reduction (needs a C++ compiler) ---------------


def _stats_plugin():
    @jit.plugin
    class Stats:
        per_pid = jit.map(key=(jit.i64,), value=jit.count())
        total_dur = jit.shared("com.example.dur_total", jit.sum())
        n_events = jit.shared("com.example.n", jit.count())
        max_dur = jit.shared("com.example.maxdur", jit.max())
        min_dur = jit.shared("com.example.mindur", jit.min())

        @jit.each_event
        def step(self, e):
            self.per_pid[(e.pid,)] += 1
            self.total_dur += e.dur
            self.n_events.add(1)
            self.max_dur.add(e.dur)
            self.min_dur.add(e.dur)

    return Stats


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_shared_handles_merge_across_scan(tmp_path):
    durs = [10 + i for i in range(200)]
    _write_trace(str(tmp_path / "trace.pfw.gz"), durs)

    host = PluginHost()
    host.load(_stats_plugin())
    assert host.resolve()
    results = host.run(str(tmp_path))

    # The per-pid map still materializes normally.
    per_pid = pa.table(results["per_pid"])
    assert per_pid.num_rows == 3
    assert sum(per_pid.column("value").to_pylist()) == len(durs)

    # The shared handles merged across every worker slice to one scalar each.
    assert _u64(results["n_events"]) == len(durs)
    assert _f64(results["total_dur"]) == pytest.approx(float(sum(durs)))
    assert _u64(results["max_dur"]) == max(durs)
    assert _u64(results["min_dur"]) == min(durs)


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_shared_only_plugin(tmp_path):
    # A plugin that declares only a shared handle (no result map) still emits its
    # merged value.
    @jit.plugin
    class CountOnly:
        n = jit.shared("com.example.count_only", jit.count())

        @jit.each_event
        def step(self, e):
            self.n.add(1)

    durs = [5 + i for i in range(50)]
    _write_trace(str(tmp_path / "trace.pfw.gz"), durs)

    host = PluginHost()
    host.load(CountOnly)
    assert host.resolve()
    results = host.run(str(tmp_path))

    assert _u64(results["n"]) == len(durs)
