#!/usr/bin/env python3
"""Inter-plugin communication for @jit.plugin authoring.

A jit plugin declares a batch-scoped publish port with jit.publish and another a
consume port with jit.consume under the same port name; the host hands the
per-batch published value across via DFTU_EXT_PORTS, running the producer first
from the provides/consumes the JIT derives from the two bodies. These tests
cover the generated C (publish/consume scaffolding and the derived name lists)
without a compiler, and the end-to-end data crossing - in either declaration
order, plus the absent-producer build error - with one.
"""

import gzip
import shutil

import pytest

from dftracer.utils import jit
from dftracer.utils.plugins import PluginHost

pa = pytest.importorskip("pyarrow")

_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))


def _write_trace(path: str, n: int, pids) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            pid = pids[i % len(pids)]
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":{10 + i},"ph":"X","args":{{}}}}\n'
            )


# --- codegen-only checks (no C++ compiler needed) ---------------------------


def test_jit_publish_emits_flush():
    @jit.plugin
    class Producer:
        counts = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.publish("com.example.batchsig", of=jit.u64)

        @jit.each_event
        def step(self, e):
            self.counts[(e.pid,)] += 1
            self.sig += 1

    src = Producer._jit_plugin.source
    assert "com.example.batchsig" in src
    assert "_pub_sig += (uint64_t)" in src
    assert "_ports->publish(" in src
    assert "DFTU_EXT_PORTS" in src


def test_jit_consume_emits_read():
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.batchsig", of=jit.u64)

        @jit.each_event
        def step(self, e):
            if self.sig > 0:
                self.got[(e.pid,)] += 1

    src = Consumer._jit_plugin.source
    assert "com.example.batchsig" in src
    assert "_sub_sig" in src
    assert "_ports->consume(" in src


def test_jit_f64_publish_uses_double():
    @jit.plugin
    class P:
        m = jit.map(key=(jit.i64,), value=jit.count())
        tot = jit.publish("com.example.durtot", of=jit.f64)

        @jit.each_event
        def step(self, e):
            self.m[(e.pid,)] += 1
            self.tot += e.dur

    src = P._jit_plugin.source
    assert "double _pub_tot = 0.0;" in src
    assert "_pub_tot += (double)" in src


def test_jit_derives_provides_and_consumes():
    @jit.plugin
    class Both:
        counts = jit.map(key=(jit.i64,), value=jit.count())
        out = jit.publish("com.example.out", of=jit.u64)
        inp = jit.consume("com.example.in", of=jit.u64)

        @jit.each_event
        def step(self, e):
            if self.inp > 0:
                self.counts[(e.pid,)] += 1
            self.out += 1

    src = Both._jit_plugin.source
    # Produced: the published port and the accumulator this plugin creates.
    assert (
        'static const char* const _provides_names[3] = {"com.example.out", "counts", NULL};' in src
    )
    assert 'static const char* const _consumes_names[2] = {"com.example.in", NULL};' in src
    assert "g_plugin.provides = provides;" in src
    assert "g_plugin.consumes = consumes;" in src


def test_jit_derives_no_consumes_without_a_consume_port():
    @jit.plugin
    class Solo:
        counts = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.counts[(e.pid,)] += 1

    src = Solo._jit_plugin.source
    assert "g_plugin.consumes = NULL;" in src
    assert 'static const char* const _provides_names[2] = {"counts", NULL};' in src


def test_jit_port_rejects_reserved_namespace():
    with pytest.raises(jit.JitError, match="reserved dftu. namespace"):
        jit.publish("dftu.cap.events")


def test_jit_port_rejects_bad_charset():
    with pytest.raises(jit.JitError, match="ASCII"):
        jit.consume("Bad Id!")


def test_jit_port_rejects_bad_width():
    with pytest.raises(jit.JitError, match="must be jit.u64"):
        jit.publish("ok.id", of=jit.str_)


def test_jit_rejects_reading_a_publish_port():
    with pytest.raises(jit.JitError, match="publish port"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.sum())
            p = jit.publish("a.b")

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += self.p


def test_jit_rejects_writing_a_consume_port():
    with pytest.raises(jit.JitError, match="consume port"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.sum())
            c = jit.consume("a.b")

            @jit.each_event
            def step(self, e):
                self.c += 1


# --- end-to-end data-crossing checks (need a C++ compiler) ------------------


def _producer():
    @jit.plugin
    class Producer:
        # A results map keeps the producer's own output; the port carries the
        # per-batch event count (>= 1 for any non-empty batch) to a consumer.
        seen = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.publish("com.example.batchsig", of=jit.u64)

        @jit.each_event
        def step(self, e):
            self.seen[(e.pid,)] += 1
            self.sig += 1

    return Producer


def _consumer():
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.batchsig", of=jit.u64)

        @jit.each_event
        def step(self, e):
            # Counts an event only when a producer published a value this batch.
            if self.sig > 0:
                self.got[(e.pid,)] += 1

    return Consumer


def _pid_counts(tbl):
    return {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_ports_data_crosses_producer_to_consumer(tmp_path):
    n = 80
    pids = [1, 2, 3]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids)

    host = PluginHost()
    host.load(_producer())
    host.load(_consumer())
    results = host.run(str(tmp_path))

    seen = _pid_counts(pa.table(results["seen"]))
    got = _pid_counts(pa.table(results["got"]))
    # The producer published a nonzero value every batch, so the consumer counted
    # every event: its per-pid distribution matches the producer's own.
    assert got == seen
    assert sum(got.values()) == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_ports_cross_in_either_declaration_order(tmp_path):
    # The consumer is loaded first, which before the derived ordering left it
    # reading nothing. The publish/consume names now decide the fold order.
    n = 80
    pids = [1, 2, 3]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids)

    host = PluginHost()
    host.load(_consumer())
    host.load(_producer())
    results = host.run(str(tmp_path))

    seen = _pid_counts(pa.table(results["seen"]))
    got = _pid_counts(pa.table(results["got"]))
    assert got == seen
    assert sum(got.values()) == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_consume_without_a_producer_fails_the_build(tmp_path):
    # A consume port no loaded plugin publishes is a load error, not a whole
    # scan that quietly reads zero.
    _write_trace(str(tmp_path / "trace.pfw.gz"), 40, [1, 2])

    host = PluginHost()
    host.load(_consumer())
    with pytest.raises(Exception, match="com.example.batchsig"):
        host.run(str(tmp_path))
