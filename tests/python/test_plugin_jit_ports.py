#!/usr/bin/env python3
"""Inter-plugin communication for @jit.plugin authoring.

A jit plugin declares a batch-scoped publish port with jit.publish and another a
consume port with jit.consume under the same capability id; the host orders the
producer before the consumer and hands the per-batch published value across via
the DFTU_EXT_PORTS / DFTU_EXT_COMMS machinery. These tests cover the generated C
(provides/requires + publish/consume scaffolding) without a compiler, and the
end-to-end data crossing (plus the absent-producer fallback) with one.
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


def test_jit_publish_emits_provides_and_flush():
    @jit.plugin
    class Producer:
        counts = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.publish("com.example.batchsig", of=jit.u64)

        @jit.each_event
        def step(self, e):
            self.counts[(e.pid,)] += 1
            self.sig += 1

    src = Producer._jit_plugin.source
    assert "comms_provides" in src
    assert "com.example.batchsig" in src
    assert "g_plugin.get_extension = plugin_get_extension;" in src
    assert "_pub_sig += (uint64_t)" in src
    assert "_ports->publish(" in src
    assert "DFTU_EXT_PORTS" in src and "DFTU_EXT_COMMS" in src


def test_jit_consume_emits_requires_and_read():
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.batchsig", of=jit.u64, required=True)

        @jit.each_event
        def step(self, e):
            if self.sig > 0:
                self.got[(e.pid,)] += 1

    src = Consumer._jit_plugin.source
    assert "comms_require" in src
    assert "out[i].required = reqd[i];" in src
    assert "reqd[1] = {1}" in src  # required=True
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


# --- versioned capability negotiation codegen -------------------------------


def test_jit_publish_emits_declared_version():
    @jit.plugin
    class Producer:
        m = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.publish("com.example.tag", of=jit.u64, version="2.1.3")

        @jit.each_event
        def step(self, e):
            self.m[(e.pid,)] += 1
            self.sig += 1

    src = Producer._jit_plugin.source
    assert "static const dftu_version vers[1] = {{2, 1, 3}};" in src
    assert "out[i].ver = vers[i];" in src


def test_jit_consume_emits_version_constraint():
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume(
            "com.example.tag", of=jit.u64, required=True, min_version="1.2.0", version_op="^"
        )

        @jit.each_event
        def step(self, e):
            if self.sig > 0:
                self.got[(e.pid,)] += 1

    src = Consumer._jit_plugin.source
    assert "static const dftu_ver_op ops[1] = {DFTU_VER_CARET};" in src
    assert "static const dftu_version vers[1] = {{1, 2, 0}};" in src
    assert "out[i].op = ops[i];" in src


def test_jit_consume_min_version_defaults_to_ge():
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.tag", min_version="1.0.0")

        @jit.each_event
        def step(self, e):
            if self.sig > 0:
                self.got[(e.pid,)] += 1

    src = Consumer._jit_plugin.source
    assert "static const dftu_ver_op ops[1] = {DFTU_VER_GE};" in src
    assert "static const dftu_version vers[1] = {{1, 0, 0}};" in src


def test_jit_publish_rejects_bad_version():
    with pytest.raises(jit.JitError, match="version"):
        jit.publish("ok.id", version="1.x")


def test_jit_consume_rejects_version_op_without_min_version():
    with pytest.raises(jit.JitError, match="version_op requires min_version"):
        jit.consume("ok.id", version_op=">=")


def test_jit_consume_rejects_unknown_version_op():
    with pytest.raises(jit.JitError, match="version_op must be one of"):
        jit.consume("ok.id", min_version="1.0.0", version_op="!!")


# --- @jit.on_resolve codegen ------------------------------------------------


def test_jit_on_resolve_emits_provider_best_and_flags():
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.tag", of=jit.u64, min_version="2.0.0")

        @jit.on_resolve
        def on_resolve(self):
            self.wired = self.sig.resolved
            self.modern = self.sig.resolved and (self.sig.version >= (2, 1, 0))

        @jit.each_event
        def step(self, e):
            if self.wired > 0:
                self.got[(e.pid,)] += 1
            if self.modern > 0:
                self.got[(e.pid,)] += 1

    src = Consumer._jit_plugin.source
    # resolve queries provider_best against the requirement and records the result
    assert "DFTU_EXT_COMMS" in src
    assert "_c->provider_best(host->h, &_req, &_bv)" in src
    assert "static int _resolved_sig = 0;" in src
    assert "static uint64_t _resolved_ver_sig = 0;" in src
    # the author flags are set from the resolved state and read in each_event
    assert "_rflag_wired = (_resolved_sig);" in src
    # 2.1.0 packs to (2<<32)|(1<<16) = 8590000128
    assert "_resolved_ver_sig >= 8590000128ULL" in src
    assert "_rflag_wired" in src and "_rflag_modern" in src
    comms_resolve = src[
        src.find("static void comms_resolve") : src.find("static const dftu_plugin_comms")
    ]
    assert comms_resolve.strip() and "provider_best" in comms_resolve


def test_jit_on_resolve_requires_a_consume_port():
    with pytest.raises(jit.JitError, match="needs at least one jit.consume port"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.count())
            sig = jit.publish("com.example.tag")

            @jit.on_resolve
            def on_resolve(self):
                self.wired = 1

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += 1
                self.sig += 1


def test_jit_on_resolve_rejects_arbitrary_body():
    with pytest.raises(jit.JitError, match="unsupported in @jit.on_resolve"):

        @jit.plugin
        class Bad:
            got = jit.map(key=(jit.i64,), value=jit.count())
            sig = jit.consume("com.example.tag")

            @jit.on_resolve
            def on_resolve(self):
                for _ in range(3):
                    self.wired = 1

            @jit.each_event
            def step(self, e):
                self.got[(e.pid,)] += 1


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


def _consumer(required=False):
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.batchsig", of=jit.u64, required=required)

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
def test_jit_ports_resolve_orders_producer_first(tmp_path):
    # Load the consumer BEFORE the producer; the build must still order the
    # producer first so the value is present when the consumer reads it.
    n = 60
    pids = [1, 2]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids)

    host = PluginHost()
    host.load(_consumer())
    host.load(_producer())
    results = host.run(str(tmp_path))

    assert sum(_pid_counts(pa.table(results["got"])).values()) == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_ports_absent_producer_reads_zero(tmp_path):
    # An optional consume port with no producer degrades to reading 0, so the
    # guard never fires and nothing is counted.
    n = 40
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1, 2])

    host = PluginHost()
    host.load(_consumer(required=False))
    results = host.run(str(tmp_path))

    got = pa.table(results["got"])
    assert got.num_rows == 0


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_ports_required_missing_producer_fails_the_build(tmp_path):
    host = PluginHost()
    host.load(_consumer(required=True))
    # A required capability with no provider must fail the plugin-set build.
    with pytest.raises(ImportError):
        host.run(str(tmp_path))


# --- end-to-end resolve-time adaptation (need a C++ compiler) ----------------


def _versioned_producer(version):
    @jit.plugin
    class Producer:
        seen = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.publish("com.example.tag", of=jit.u64, version=version)

        @jit.each_event
        def step(self, e):
            self.seen[(e.pid,)] += 1
            self.sig += 1

    return Producer


def _adaptive_consumer(min_version):
    @jit.plugin
    class Consumer:
        got = jit.map(key=(jit.i64,), value=jit.count())
        sig = jit.consume("com.example.tag", of=jit.u64, min_version=min_version)

        @jit.on_resolve
        def on_resolve(self):
            self.wired = self.sig.resolved

        @jit.each_event
        def step(self, e):
            # Adapts to resolve: only counts when a compatible producer exists.
            if self.wired > 0:
                self.got[(e.pid,)] += 1

    return Consumer


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_on_resolve_adapts_to_compatible_producer(tmp_path):
    n = 60
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1, 2])

    host = PluginHost()
    host.load(_versioned_producer("2.1.0"))
    host.load(_adaptive_consumer(min_version="2.0.0"))
    results = host.run(str(tmp_path))

    # The provider satisfies >=2.0.0, so on_resolve wires the consumer and it
    # counts every event.
    assert sum(_pid_counts(pa.table(results["got"])).values()) == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_on_resolve_stands_down_for_incompatible_version(tmp_path):
    n = 40
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1, 2])

    host = PluginHost()
    host.load(_versioned_producer("1.0.0"))
    host.load(_adaptive_consumer(min_version="2.0.0"))
    results = host.run(str(tmp_path))

    # The only provider is version 1.0.0, below the >=2.0.0 constraint, so
    # on_resolve leaves the consumer unwired and it counts nothing.
    assert pa.table(results["got"]).num_rows == 0
