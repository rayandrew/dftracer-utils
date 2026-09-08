#!/usr/bin/env python3
"""Cross-worker scalar accumulators for @jit.plugin authoring.

A scalar accumulator is a jit.map with an empty key tuple: it groups nothing, so
it folds to one whole-scan row. Each event contributes with += or .observe(); the
host merges the same-named accumulator across every worker slice and finalizes it
to a one-row frame surfaced under the attribute name. These tests cover the
generated C without a compiler and the end-to-end cross-worker reduction with one.
"""

import gzip
import shutil

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


def _scalar(results, name):
    tbl = pa.table(results[name])
    assert tbl.column_names == ["value"]
    assert tbl.num_rows == 1
    return tbl.column("value")[0].as_py()


# --- codegen-only checks (no C++ compiler needed) ---------------------------


def test_jit_scalar_accumulator_emits_keyless_agg():
    @jit.plugin
    class Stats:
        per_pid = jit.map(key=(jit.i64,), value=jit.count())
        total_dur = jit.map(key=(), value=jit.sum())
        n_events = jit.map(key=(), value=jit.count())
        max_dur = jit.map(key=(), value=jit.max())

        @jit.each_event
        def step(self, e):
            self.per_pid[(e.pid,)] += 1
            self.total_dur[()] += e.dur
            self.n_events[()] += 1
            self.max_dur[()].observe(e.dur)

    src = Stats._jit_plugin.source
    assert "DFTU_EXT_AGG" in src
    # A keyed accumulator passes its key columns; a scalar one passes none.
    assert 'agg_new(host->h, "per_pid", _keys, 1u,' in src
    assert 'agg_new(host->h, "total_dur", NULL, 0u,' in src
    assert 'agg_new(host->h, "n_events", NULL, 0u,' in src
    assert '{DFTU_AGG_MAX, "v0", "value", 0.0, NULL}' in src
    # The per-event contribution lands in that accumulator's row buffer.
    assert "_v0_total_dur[_r] = (double)(e->dur);" in src
    assert "_v0_n_events[_r] = (int64_t)(1);" in src


def test_jit_rejects_observe_only_reduction_with_aug():
    with pytest.raises(jit.JitError, match="non-additive"):

        @jit.plugin
        class Bad:
            hi = jit.map(key=(), value=jit.max())

            @jit.each_event
            def step(self, e):
                self.hi[()] += e.dur


def test_jit_rejects_unknown_accumulator():
    with pytest.raises(jit.JitError, match="unknown map"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(), value=jit.count())

            @jit.each_event
            def step(self, e):
                self.nope[()] += 1


# --- end-to-end cross-worker reduction (needs a C++ compiler) ---------------


def _stats_plugin():
    @jit.plugin
    class Stats:
        per_pid = jit.map(key=(jit.i64,), value=jit.count())
        total_dur = jit.map(key=(), value=jit.sum())
        n_events = jit.map(key=(), value=jit.count())
        max_dur = jit.map(key=(), value=jit.max())
        min_dur = jit.map(key=(), value=jit.min())

        @jit.each_event
        def step(self, e):
            self.per_pid[(e.pid,)] += 1
            self.total_dur[()] += e.dur
            self.n_events[()] += 1
            self.max_dur[()].observe(e.dur)
            self.min_dur[()].observe(e.dur)

    return Stats


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_scalar_accumulators_merge_across_scan(tmp_path):
    durs = [10 + i for i in range(200)]
    _write_trace(str(tmp_path / "trace.pfw.gz"), durs)

    host = PluginHost()
    host.load(_stats_plugin())
    results = host.run(str(tmp_path))

    # The per-pid accumulator still materializes normally.
    per_pid = pa.table(results["per_pid"])
    assert per_pid.num_rows == 3
    assert sum(per_pid.column("value").to_pylist()) == len(durs)

    # The scalar accumulators merged across every worker slice to one row each.
    assert _scalar(results, "n_events") == len(durs)
    assert _scalar(results, "total_dur") == pytest.approx(float(sum(durs)))
    assert _scalar(results, "max_dur") == max(durs)
    assert _scalar(results, "min_dur") == min(durs)


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_scalar_only_plugin(tmp_path):
    # A plugin that declares only a scalar accumulator (no keyed result) still
    # emits its merged value.
    @jit.plugin
    class CountOnly:
        n = jit.map(key=(), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.n[()] += 1

    durs = [5 + i for i in range(50)]
    _write_trace(str(tmp_path / "trace.pfw.gz"), durs)

    host = PluginHost()
    host.load(CountOnly)
    results = host.run(str(tmp_path))

    assert _scalar(results, "n") == len(durs)


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_same_named_accumulator_is_shared_across_plugins(tmp_path):
    # An accumulator is host-merged across a plugin's worker slices, so each
    # plugin folds the whole scan into its own. A name two plugins share is one
    # result name, not a summed one: it holds a single whole-scan fold.
    @jit.plugin
    class A:
        shared_total = jit.map(key=(), value=jit.sum())
        a_total = jit.map(key=(), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.shared_total[()] += e.dur
            self.a_total[()] += e.dur

    @jit.plugin
    class B:
        shared_total = jit.map(key=(), value=jit.sum())
        b_total = jit.map(key=(), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.shared_total[()] += e.dur
            self.b_total[()] += e.dur

    durs = [3 + i for i in range(40)]
    _write_trace(str(tmp_path / "trace.pfw.gz"), durs)

    host = PluginHost()
    host.load(A)
    host.load(B)
    results = host.run(str(tmp_path))

    # Both plugins ran and each folded every event's dur.
    assert _scalar(results, "a_total") == pytest.approx(float(sum(durs)))
    assert _scalar(results, "b_total") == pytest.approx(float(sum(durs)))
    # The shared name carries one whole-scan fold, not the two added together.
    assert _scalar(results, "shared_total") == pytest.approx(float(sum(durs)))
