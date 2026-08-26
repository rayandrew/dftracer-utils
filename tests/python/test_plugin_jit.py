#!/usr/bin/env python3
"""End-to-end test for @jit.plugin authoring.

A plugin authored in Python is AST-compiled to a native C plugin, built to a
cached .so, and run through PluginHost. The result must be identical to the
hand-written name_edges plugin: a {pid, event-name} COUNTER map materialized to
an Arrow table [k0:int64, k1:string, value:int64] whose value column sums to the
event count.
"""

import builtins
import gzip
import shutil
import statistics

import pytest

from dftracer.utils import jit
from dftracer.utils.plugins import PluginHost

pa = pytest.importorskip("pyarrow")

_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))


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


def _write_homog_trace(path: str, n: int, name: str, cat: str) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            f.write(
                f'{{"name":"{name}","cat":"{cat}","pid":1,"tid":1,'
                f'"ts":{1000 + i * 100},"dur":{10 + i},"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_plan_query_narrows_scan(tmp_path):
    posix_dir = tmp_path / "posix"
    stdio_dir = tmp_path / "stdio"
    posix_dir.mkdir()
    stdio_dir.mkdir()
    n = 40
    _write_homog_trace(str(posix_dir / "trace.pfw.gz"), n, "read", "POSIX")
    _write_homog_trace(str(stdio_dir / "trace.pfw.gz"), n, "fwrite", "STDIO")
    files = [str(posix_dir / "trace.pfw.gz"), str(stdio_dir / "trace.pfw.gz")]

    @jit.plugin
    class Pruned:
        plan_query = 'cat == "POSIX"'
        hits = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.hits[(e.pid,)] += 1

    @jit.plugin
    class Full:
        hits = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.hits[(e.pid,)] += 1

    pruned_host = PluginHost()
    pruned_host.load(Pruned)
    pruned_host.resolve()
    pruned_res = pruned_host.run(files)
    pruned_scanned = pruned_host.stats["events_scanned"]

    full_host = PluginHost()
    full_host.load(Full)
    full_host.resolve()
    full_res = full_host.run(files)
    full_scanned = full_host.stats["events_scanned"]

    assert full_scanned == 2 * n
    assert pruned_scanned == n
    assert pruned_scanned < full_scanned

    pruned_val = pa.table(pruned_res["hits"]).column("value").to_numpy(zero_copy_only=False)
    full_val = pa.table(full_res["hits"]).column("value").to_numpy(zero_copy_only=False)
    assert int(pruned_val.sum()) == n
    assert int(full_val.sum()) == 2 * n


def _write_mixed_trace(path: str, n: int, cats) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            cat = cats[i % len(cats)]
            f.write(
                f'{{"name":"read","cat":"{cat}","pid":1,"tid":1,'
                f'"ts":{1000 + i * 100},"dur":{10 + i},"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_string_literal_guard_counts_matching(tmp_path):
    @jit.plugin
    class OnlyPosix:
        hits = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            if e.cat == "POSIX":
                self.hits[(e.pid,)] += 1

    n = 60
    cats = ["POSIX", "STDIO", "OTHER"]
    _write_mixed_trace(str(tmp_path / "trace.pfw.gz"), n, cats)
    expected = sum(1 for i in range(n) if cats[i % len(cats)] == "POSIX")

    host = PluginHost()
    host.load(OnlyPosix)
    host.resolve()
    results = host.run(str(tmp_path))

    val = pa.table(results["hits"]).column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == expected


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_string_literal_ne_guard(tmp_path):
    @jit.plugin
    class NotPosix:
        hits = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            if e.cat != "POSIX":
                self.hits[(e.pid,)] += 1

    n = 60
    cats = ["POSIX", "STDIO", "OTHER"]
    _write_mixed_trace(str(tmp_path / "trace.pfw.gz"), n, cats)
    expected = sum(1 for i in range(n) if cats[i % len(cats)] != "POSIX")

    host = PluginHost()
    host.load(NotPosix)
    host.resolve()
    results = host.run(str(tmp_path))

    val = pa.table(results["hits"]).column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == expected


def test_jit_rejects_string_ordering_op():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class Ordered:
            hits = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                if e.cat < "POSIX":
                    self.hits[(e.pid,)] += 1


def test_jit_rejects_string_literal_on_numeric_field():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class BadField:
            hits = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                if e.pid == "POSIX":
                    self.hits[(e.pid,)] += 1


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_name_edges_matches_handwritten(tmp_path):
    @jit.plugin
    class NameEdges:
        edges = jit.map(key=(jit.i64, jit.str_), value=jit.count())

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.edges[(e.pid, e.name)] += 1

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(NameEdges)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "edges" in results
    tbl = pa.table(results["edges"])
    assert tbl.column_names == ["k0", "k1", "value"]
    assert pa.types.is_string(tbl.schema.field("k1").type)

    names = set(tbl.column("k1").to_pylist())
    assert names == {"read"}

    val = tbl.column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_wide_edges_matches_handwritten(tmp_path):
    @jit.plugin
    class WideEdges:
        edges = jit.map(key=(jit.i64, jit.str_), value=(jit.count(), jit.sum()))

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.edges[(e.pid, e.name)][0] += 1
                self.edges[(e.pid, e.name)][1] += e.dur

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(WideEdges)
    host.resolve()
    results = host.run(str(tmp_path))

    assert "edges" in results
    tbl = pa.table(results["edges"])
    assert tbl.column_names == ["k0", "k1", "v0", "v1"]
    assert pa.types.is_int64(tbl.schema.field("k0").type)
    assert pa.types.is_string(tbl.schema.field("k1").type)
    assert pa.types.is_int64(tbl.schema.field("v0").type)
    assert pa.types.is_float64(tbl.schema.field("v1").type)

    v0 = tbl.column("v0").to_numpy(zero_copy_only=False)
    v1 = tbl.column("v1").to_numpy(zero_copy_only=False)
    assert int(v0.sum()) == n
    assert float(v1.sum()) == float(sum(10 + i for i in range(n)))


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_dict_value_names_columns(tmp_path):
    @jit.plugin
    class Wide:
        edges = jit.map(
            key=(jit.i64, jit.str_),
            value=dict(count=jit.count(), dur=jit.sum()),
        )

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.edges[(e.pid, e.name)].count += 1
                self.edges[(e.pid, e.name)].dur += e.dur

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(Wide)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["edges"])
    assert tbl.column_names == ["k0", "k1", "count", "dur"]
    assert pa.types.is_int64(tbl.schema.field("count").type)
    assert pa.types.is_float64(tbl.schema.field("dur").type)
    assert int(tbl.column("count").to_numpy(zero_copy_only=False).sum()) == n
    assert float(tbl.column("dur").to_numpy(zero_copy_only=False).sum()) == float(
        sum(10 + i for i in range(n))
    )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_dict_value_positional_and_named(tmp_path):
    @jit.plugin
    class Wide:
        edges = jit.map(
            key=(jit.i64, jit.str_),
            value=dict(count=jit.count(), dur=jit.sum()),
        )

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.edges[(e.pid, e.name)][0] += 1
                self.edges[(e.pid, e.name)].dur += e.dur

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(Wide)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["edges"])
    assert tbl.column_names == ["k0", "k1", "count", "dur"]
    assert int(tbl.column("count").to_numpy(zero_copy_only=False).sum()) == n
    assert float(tbl.column("dur").to_numpy(zero_copy_only=False).sum()) == float(
        sum(10 + i for i in range(n))
    )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_record_value_form(tmp_path):
    @jit.record
    class Stats:
        count: jit.count
        dur: jit.sum

    @jit.plugin
    class Wide:
        edges = jit.map(key=(jit.i64, jit.str_), value=Stats)

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.edges[(e.pid, e.name)].count += 1
                self.edges[(e.pid, e.name)].dur += e.dur

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(Wide)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["edges"])
    assert tbl.column_names == ["k0", "k1", "count", "dur"]
    assert int(tbl.column("count").to_numpy(zero_copy_only=False).sum()) == n
    assert float(tbl.column("dur").to_numpy(zero_copy_only=False).sum()) == float(
        sum(10 + i for i in range(n))
    )


def test_jit_rejects_unknown_named_component():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class Bad:
            edges = jit.map(key=(jit.i64,), value=dict(count=jit.count()))

            @jit.each_event
            def step(self, e):
                self.edges[(e.pid,)].nope += 1


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_single_sum_map(tmp_path):
    @jit.plugin
    class DurSum:
        tot = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.tot[(e.pid,)] += e.dur

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(DurSum)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["tot"])
    assert tbl.column_names == ["k0", "value"]
    assert pa.types.is_float64(tbl.schema.field("value").type)
    val = tbl.column("value").to_numpy(zero_copy_only=False)
    assert float(val.sum()) == float(sum(10 + i for i in range(n)))


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_arithmetic_doubles_sum(tmp_path):
    @jit.plugin
    class DurSumX2:
        tot = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.tot[(e.pid,)] += e.dur * 2

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(DurSumX2)
    host.resolve()
    results = host.run(str(tmp_path))

    val = pa.table(results["tot"]).column("value").to_numpy(zero_copy_only=False)
    assert float(val.sum()) == float(2 * sum(10 + i for i in range(n)))


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_primitive_ilog2_buckets_duration(tmp_path):
    @jit.plugin
    class DurHist:
        hist = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.hist[(jit.ilog2(e.dur),)] += 1

    n = 60
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1, 2], ["fileA", "fileB"])

    host = PluginHost()
    host.load(DurHist)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["hist"])
    got = dict(zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist()))

    expected: dict = {}
    for i in range(n):
        bucket = (10 + i).bit_length() - 1
        expected[bucket] = expected.get(bucket, 0) + 1
    assert got == expected


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_float_primitives_sum(tmp_path):
    @jit.plugin
    class SqrtSum:
        tot = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.tot[(0,)] += jit.sqrt(jit.log2(jit.fma(e.dur, 1.0, 0.0)))

    n = 60
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1, 2], ["fileA", "fileB"])

    host = PluginHost()
    host.load(SqrtSum)
    host.resolve()
    results = host.run(str(tmp_path))

    val = pa.table(results["tot"]).column("value").to_numpy(zero_copy_only=False)
    import math

    expected = sum(math.sqrt(math.log2(10 + i)) for i in range(n))
    assert float(val.sum()) == pytest.approx(expected, rel=1e-9)


def test_jit_primitive_marker_not_callable_in_python():
    with pytest.raises(jit.JitError, match="plugin primitive"):
        jit.ilog2(1024)
    with pytest.raises(jit.JitError, match="plugin primitive"):
        jit.sqrt(2.0)


def test_jit_cse_hoists_repeated_subexpression():
    @jit.plugin
    class Hoist:
        a = jit.map(key=(jit.i64,), value=jit.count())
        b = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.a[(jit.ilog2(e.dur),)] += 1
            self.b[(jit.ilog2(e.dur),)] += e.dur

    src = Hoist._jit_plugin.source
    # ilog2(e.dur) is used by both maps, so it is computed once into a local.
    assert src.count("dftu_ilog2_u64(") == 1
    assert "_cse0" in src


def test_jit_cse_applies_across_all_op_kinds():
    # The planner is expression-level, not += specific: a shared subexpression
    # is hoisted whether it feeds observe, append, an argmax by=, or quantiles.
    @jit.plugin
    class Mixed:
        avg = jit.map(key=(jit.i64,), value=jit.mean())
        seq = jit.map(key=(jit.i64,), value=jit.list(of=jit.i64))
        worst = jit.map(key=(jit.i64,), value=jit.argmax(of=jit.str_))
        lat = jit.map(key=(jit.i64,), value=jit.quantiles(qs=(0.5, 0.9)))

        @jit.each_event
        def step(self, e):
            self.avg[(jit.mix64(e.hhash),)].observe(e.dur)
            self.seq[(jit.mix64(e.hhash),)].append(e.dur, order_by=e.ts)
            self.worst[(jit.mix64(e.hhash),)].observe(e.name, by=e.dur)
            self.lat[(jit.mix64(e.hhash),)].observe(e.dur)

    src = Mixed._jit_plugin.source
    # mix64(e.hhash) keys all four different op kinds; computed exactly once.
    assert src.count("dftu_mix64(") == 1
    assert "_cse0" in src


def _body(cls):
    s = cls._jit_plugin.source
    i = s.index("for (uint32_t i = 0; i < b->count;")
    return s[i : s.index("return NULL;", i)]


def test_jit_cse_shares_a_value_expression():
    @jit.plugin
    class V:
        a = jit.map(key=(jit.i64,), value=jit.sum())
        b = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.a[(e.pid,)] += e.dur * e.dur
            self.b[(e.pid,)] += e.dur * e.dur

    assert _body(V).count("e->dur * e->dur") == 1


def test_jit_cse_shares_nested_outer_and_inner_keys():
    @jit.plugin
    class N:
        m = jit.map(
            key=(jit.i64,),
            value=jit.nested(key=(jit.i64,), value=dict(cnt=jit.count(), dur=jit.sum())),
        )

        @jit.each_event
        def step(self, e):
            self.m[(jit.mix64(e.pid),)][(jit.ilog2(e.dur),)].cnt += 1
            self.m[(jit.mix64(e.pid),)][(jit.ilog2(e.dur),)].dur += e.dur

    b = _body(N)
    assert b.count("dftu_mix64(") == 1
    assert b.count("dftu_ilog2_u64(") == 1


def test_jit_cse_shares_arg_and_float_prim():
    @jit.plugin
    class A:
        a = jit.map(key=(jit.i64,), value=jit.count())
        d = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.a[(e.arg_i64("rank"),)] += 1
            self.d[(e.arg_i64("rank"),)] += jit.sqrt(e.dur)

    b = _body(A)
    assert b.count("dftu_jit_arg_i64(") == 1  # arg call hoisted


def test_jit_cse_shares_a_guard_test_with_a_statement():
    @jit.plugin
    class G:
        a = jit.map(key=(jit.i64,), value=jit.count())
        c = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.a[(jit.ilog2(e.dur),)] += 1
            if jit.ilog2(e.dur) > 3:
                self.c[(e.pid,)] += 1

    # The guard test shares ilog2(e.dur) with the statement: computed once at the
    # loop top, above the guard.
    b = _body(G)
    assert b.count("dftu_ilog2_u64(") == 1


def test_jit_cse_stays_inside_a_guard():
    @jit.plugin
    class Guarded:
        a = jit.map(key=(jit.i64,), value=jit.count())
        b = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            if e.dur > 0:
                self.a[(jit.ilog2(e.dur),)] += 1
                self.b[(jit.ilog2(e.dur),)] += e.dur

    src = Guarded._jit_plugin.source
    # The shared expression is guarded, so its local must be defined after the
    # guard opens, never hoisted above it.
    assert src.count("dftu_ilog2_u64(") == 1
    guard = src.index("if (")
    cse = src.index("_cse0 =")
    assert cse > guard


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_cse_result_matches_unfused(tmp_path):
    @jit.plugin
    class Buckets:
        n = jit.map(key=(jit.i64,), value=jit.count())
        tot = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.n[(jit.ilog2(e.dur),)] += 1
            self.tot[(jit.ilog2(e.dur),)] += e.dur

    n = 60
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1, 2], ["fileA"])
    host = PluginHost()
    host.load(Buckets)
    host.resolve()
    results = host.run(str(tmp_path))

    counts = dict(
        zip(
            pa.table(results["n"]).column("k0").to_pylist(),
            pa.table(results["n"]).column("value").to_pylist(),
        )
    )
    exp: dict = {}
    for i in range(n):
        b = (10 + i).bit_length() - 1
        exp[b] = exp.get(b, 0) + 1
    assert counts == exp


def test_jit_fusion_coalesces_same_key_maps():
    @jit.plugin
    class Stats:
        n = jit.map(key=(jit.i64,), value=jit.count())
        tot = jit.map(key=(jit.i64,), value=jit.sum())
        avg = jit.map(key=(jit.i64,), value=jit.mean())

        @jit.each_event
        def step(self, e):
            self.n[(e.pid,)] += 1
            self.tot[(e.pid,)] += e.dur
            self.avg[(e.pid,)].observe(e.dur)

    src = Stats._jit_plugin.source
    b = _body(Stats)
    # Three same-key maps share one fused product and one lookup per event.
    assert src.count("map_new_fused(") == 1
    assert b.count("map_add_row(") == 1
    assert b.count("map_add_u64(") == 0 and b.count("map_add_f64(") == 0


def test_jit_fusion_skips_different_keys():
    @jit.plugin
    class NoFuse:
        by_pid = jit.map(key=(jit.i64,), value=jit.count())
        by_tid = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.by_pid[(e.pid,)] += 1
            self.by_tid[(e.tid,)] += 1

    src = NoFuse._jit_plugin.source
    # Different keys cannot share a lookup, so no fusion.
    assert "map_new_fused(" not in src
    assert _body(NoFuse).count("map_add_u64(") == 2


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_fusion_result_matches_unfused(tmp_path):
    @jit.plugin
    class Stats:
        n = jit.map(key=(jit.i64,), value=jit.count())
        tot = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.n[(e.pid,)] += 1
            self.tot[(e.pid,)] += e.dur

    m = 60
    _write_trace(str(tmp_path / "trace.pfw.gz"), m, [1, 2], ["fileA"])
    host = PluginHost()
    host.load(Stats)
    host.resolve()
    results = host.run(str(tmp_path))

    # The fused maps still surface as the two separate declared tables.
    ncount = dict(
        zip(
            pa.table(results["n"]).column("k0").to_pylist(),
            pa.table(results["n"]).column("value").to_pylist(),
        )
    )
    tot = dict(
        zip(
            pa.table(results["tot"]).column("k0").to_pylist(),
            pa.table(results["tot"]).column("value").to_pylist(),
        )
    )
    assert sum(ncount.values()) == m
    assert sum(tot.values()) == sum(10 + i for i in range(m))


def test_jit_rejects_out_of_range_integer_literal():
    # 18446744073709551616 == 2**64, one past UINT64_MAX; a bare literal (not a
    # ** power expression, which the grammar rejects earlier).
    with pytest.raises(jit.JitError, match="does not fit in 64 bits"):

        @jit.plugin
        class TooBig:
            m = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                self.m[(18446744073709551616,)] += 1


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_accepts_large_in_range_literal(tmp_path):
    # 1099511627776 == 2**40 fits in 64 bits and routes as an ordinary key.
    @jit.plugin
    class BigKey:
        m = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.m[(1099511627776,)] += 1

    n = 5
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, [1], ["fileA"])
    host = PluginHost()
    host.load(BigKey)
    host.resolve()
    results = host.run(str(tmp_path))
    tbl = pa.table(results["m"])
    assert tbl.column("k0").to_pylist() == [1099511627776]
    assert int(tbl.column("value")[0].as_py()) == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_raw_body_reproduces_name_edges(tmp_path):
    @jit.plugin(needs=(jit.NEED_FHASH,))
    class RawEdges:
        edges = jit.map(key=(jit.i64, jit.str_), value=jit.count())

        @jit.each_event(raw=True)
        def step(self):
            return """
            if (e->fhash == DFTU_STR_NONE) continue;
            int64_t key[2];
            key[0] = (int64_t)e->pid;
            key[1] = (int64_t)e->name;
            map->map_add_u64(host->h, edges, key, 1);
            """

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(RawEdges)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["edges"])
    assert tbl.column_names == ["k0", "k1", "value"]
    assert set(tbl.column("k1").to_pylist()) == {"read"}
    assert int(tbl.column("value").to_numpy(zero_copy_only=False).sum()) == n


def test_jit_raw_body_must_return_string():
    with pytest.raises(jit.JitError, match="must return a C\\+\\+ body string"):

        @jit.plugin
        class BadRaw:
            edges = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event(raw=True)
            def step(self):
                return 123


def test_jit_non_raw_return_still_rejects():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class NotRaw:
            edges = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                return "map->map_add_u64(host->h, edges, key, 1);"


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_quantiles_per_key(tmp_path):
    @jit.plugin
    class Lat:
        lat = jit.map(key=(jit.i64,), value=jit.quantiles(qs=(0.5, 0.9, 0.99)))

        @jit.each_event
        def step(self, e):
            self.lat[(e.pid,)].observe(e.dur)

    # dur = 1..1000 all on pid 1, so quantiles are the percentiles of 1..1000.
    with gzip.open(str(tmp_path / "trace.pfw.gz"), "wt", encoding="utf-8") as f:
        for i in range(1, 1001):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":1,"tid":1,'
                f'"ts":{1000 + i},"dur":{i},"ph":"X","args":{{}}}}\n'
            )

    host = PluginHost()
    host.load(Lat)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["lat"])
    assert tbl.column_names == ["k0", "count", "p50", "p90", "p99"]
    assert int(tbl.column("count")[0].as_py()) == 1000
    # DDSketch is ~1% relative error.
    for col, exp in [("p50", 500), ("p90", 900), ("p99", 990)]:
        got = tbl.column(col)[0].as_py()
        assert abs(got - exp) <= 0.02 * exp, f"{col}={got} vs {exp}"


def test_jit_quantiles_rejected_as_product_component():
    with pytest.raises(jit.JitError, match="top-level only"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=(jit.count(), jit.quantiles()))

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.dur)


def _kv(tbl, value_col="value"):
    keys = tbl.column("k0").to_pylist()
    vals = tbl.column(value_col).to_numpy(zero_copy_only=False)
    return {k: v for k, v in zip(keys, vals)}


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_distinct_count_per_key(tmp_path):
    @jit.plugin
    class OutDeg:
        outdeg = jit.map(key=(jit.i64,), value=jit.distinct())

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.outdeg[(e.pid,)].observe(e.fhash)

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(OutDeg)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["outdeg"])
    assert tbl.column_names == ["k0", "value"]
    assert pa.types.is_int64(tbl.schema.field("value").type)
    kv = _kv(tbl)
    assert kv[1] == 3
    assert kv[2] == 3


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_min_max_observe(tmp_path):
    @jit.plugin
    class Lat:
        lo = jit.map(key=(jit.i64,), value=jit.min())
        hi = jit.map(key=(jit.i64,), value=jit.max())

        @jit.each_event
        def step(self, e):
            self.lo[(e.pid,)].observe(e.dur)
            self.hi[(e.pid,)].observe(e.dur)

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(Lat)
    host.resolve()
    results = host.run(str(tmp_path))

    exp_min: dict = {}
    exp_max: dict = {}
    for i in range(n):
        pid = pids[i % len(pids)]
        dur = 10 + i
        exp_min[pid] = builtins.min(exp_min.get(pid, dur), dur)
        exp_max[pid] = builtins.max(exp_max.get(pid, dur), dur)

    lo = _kv(pa.table(results["lo"]))
    hi = _kv(pa.table(results["hi"]))
    assert {k: int(v) for k, v in lo.items()} == exp_min
    assert {k: int(v) for k, v in hi.items()} == exp_max


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_product_mixed_add_and_observe(tmp_path):
    @jit.plugin
    class Mixed:
        m = jit.map(key=(jit.i64,), value=dict(cnt=jit.count(), lo=jit.min()))

        @jit.each_event
        def step(self, e):
            self.m[(e.pid,)].cnt += 1
            self.m[(e.pid,)].lo.observe(e.dur)

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(Mixed)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["m"])
    assert tbl.column_names == ["k0", "cnt", "lo"]
    assert pa.types.is_int64(tbl.schema.field("cnt").type)
    assert pa.types.is_int64(tbl.schema.field("lo").type)
    assert int(tbl.column("cnt").to_numpy(zero_copy_only=False).sum()) == n

    exp_min: dict = {}
    for i in range(n):
        pid = pids[i % len(pids)]
        dur = 10 + i
        exp_min[pid] = builtins.min(exp_min.get(pid, dur), dur)
    lo = _kv(tbl, "lo")
    assert {k: int(v) for k, v in lo.items()} == exp_min


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_minf_maxf_observe(tmp_path):
    @jit.plugin
    class LatF:
        lo = jit.map(key=(jit.i64,), value=jit.minf())
        hi = jit.map(key=(jit.i64,), value=jit.maxf())

        @jit.each_event
        def step(self, e):
            self.lo[(e.pid,)].observe(e.dur)
            self.hi[(e.pid,)].observe(e.dur)

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(LatF)
    host.resolve()
    results = host.run(str(tmp_path))

    exp_min: dict = {}
    exp_max: dict = {}
    for i in range(n):
        pid = pids[i % len(pids)]
        dur = 10 + i
        exp_min[pid] = builtins.min(exp_min.get(pid, dur), dur)
        exp_max[pid] = builtins.max(exp_max.get(pid, dur), dur)

    assert pa.types.is_float64(pa.table(results["lo"]).schema.field("value").type)
    lo = _kv(pa.table(results["lo"]))
    hi = _kv(pa.table(results["hi"]))
    assert {k: float(v) for k, v in lo.items()} == {k: float(v) for k, v in exp_min.items()}
    assert {k: float(v) for k, v in hi.items()} == {k: float(v) for k, v in exp_max.items()}


def _write_named_trace(path: str, n: int, pids, names) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            pid = pids[i % len(pids)]
            name = names[i % len(names)]
            f.write(
                f'{{"name":"{name}","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":{10 + i},"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_set_collects_distinct_names_per_pid(tmp_path):
    @jit.plugin
    class Files:
        files = jit.map(key=(jit.i64,), value=jit.set())

        @jit.each_event
        def step(self, e):
            self.files[(e.pid,)].observe(e.name)

    n = 60
    pids = [1, 2]
    names = ["read", "write", "open"]
    _write_named_trace(str(tmp_path / "trace.pfw.gz"), n, pids, names)

    host = PluginHost()
    host.load(Files)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["files"])
    assert tbl.column_names == ["k0", "value"]
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_string(vtype.value_type)

    by_pid = {
        k: set(v) for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())
    }
    assert by_pid[1] == set(names)
    assert by_pid[2] == set(names)


def _write_seq_trace(path: str, events) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for pid, name, ts in events:
            f.write(
                f'{{"name":"{name}","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{ts},"dur":1,"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_list_collects_ts_ordered_names_per_pid(tmp_path):
    @jit.plugin
    class Seq:
        seq = jit.map(key=(jit.i64,), value=jit.list())

        @jit.each_event
        def step(self, e):
            self.seq[(e.pid,)].append(e.name, order_by=e.ts)

    # File (insertion) order differs from ts order: pid 1 -> [a,b,c] by ts.
    events = [
        (1, "c", 30),
        (1, "a", 10),
        (1, "b", 20),
        (2, "y", 5),
        (2, "x", 1),
    ]
    _write_seq_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(Seq)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["seq"])
    assert tbl.column_names == ["k0", "value"]
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_string(vtype.value_type)

    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    assert by_pid[1] == ["a", "b", "c"]
    assert by_pid[2] == ["x", "y"]


def test_jit_rejects_append_without_order_by():
    with pytest.raises(jit.JitError, match=r"order_by"):

        @jit.plugin
        class NoOrder:
            m = jit.map(key=(jit.i64,), value=jit.list())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].append(e.name)


def test_jit_rejects_add_on_list():
    with pytest.raises(jit.JitError, match=r"\.append"):

        @jit.plugin
        class ListAdd:
            m = jit.map(key=(jit.i64,), value=jit.list())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.name


def test_jit_rejects_observe_on_list():
    with pytest.raises(jit.JitError, match=r"\.append"):

        @jit.plugin
        class ListObserve:
            m = jit.map(key=(jit.i64,), value=jit.list())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.name)


def test_jit_rejects_add_on_set():
    with pytest.raises(jit.JitError, match=r"\.observe"):

        @jit.plugin
        class SetAdd:
            m = jit.map(key=(jit.i64,), value=jit.set())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.name


def test_jit_rejects_add_on_distinct():
    with pytest.raises(jit.JitError, match=r"\.observe"):

        @jit.plugin
        class DistAdd:
            m = jit.map(key=(jit.i64,), value=jit.distinct())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.fhash


def test_jit_rejects_observe_float_into_u64():
    with pytest.raises(jit.JitError, match="u64 monoid"):

        @jit.plugin
        class BadF:
            m = jit.map(key=(jit.i64,), value=jit.min())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.dur * 2.0)


def test_jit_rejects_min_monoid():
    with pytest.raises(jit.JitError, match=r"\.observe"):

        @jit.plugin
        class MinMap:
            m = jit.map(key=(jit.i64,), value=jit.min())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.dur


def test_jit_rejects_arithmetic_in_key():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class KeyMath:
            m = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid + 1,)] += 1


def test_jit_rejects_bare_add_on_product():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class Prod:
            m = jit.map(key=(jit.i64,), value=(jit.count(), jit.sum()))

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += 1


def test_jit_rejects_for_loop():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class Loopy:
            edges = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                for _ in range(3):
                    self.edges[(e.pid,)] += 1


def test_jit_rejects_float_arithmetic():
    with pytest.raises(jit.JitError, match="unsupported in @jit.each_event"):

        @jit.plugin
        class Floaty:
            edges = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                self.edges[(e.pid,)] += e.dur * 2.0


def _write_dur_trace(path: str, events) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for pid, dur, ts in events:
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{ts},"dur":{dur},"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_set_i64_collects_distinct_durs_sorted(tmp_path):
    @jit.plugin
    class Durs:
        durs = jit.map(key=(jit.i64,), value=jit.set(of=jit.i64))

        @jit.each_event
        def step(self, e):
            self.durs[(e.pid,)].observe(e.dur)

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
    host.load(Durs)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["durs"])
    assert tbl.column_names == ["k0", "value"]
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_integer(vtype.value_type)

    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    assert by_pid[1] == [5, 8, 20]
    assert by_pid[2] == [3, 7]


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_list_i64_collects_ts_ordered_durs(tmp_path):
    @jit.plugin
    class Durs:
        durs = jit.map(key=(jit.i64,), value=jit.list(of=jit.i64))

        @jit.each_event
        def step(self, e):
            self.durs[(e.pid,)].append(e.dur, order_by=e.ts)

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
    host.load(Durs)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["durs"])
    assert tbl.column_names == ["k0", "value"]
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_integer(vtype.value_type)

    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    assert by_pid[1] == [20, 5, 20, 8]
    assert by_pid[2] == [3, 7]


def test_jit_rejects_str_field_into_i64_set():
    with pytest.raises(jit.JitError, match=r"int64"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.set(of=jit.i64))

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.name)


def test_jit_rejects_i64_expr_into_str_set():
    with pytest.raises(jit.JitError, match=r"string field"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.set())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.dur)


def _write_pid_tid_trace(path: str, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, (pid, tid) in enumerate(rows):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":{tid},'
                f'"ts":{1000 + i},"dur":1,"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_ordered_map_sorts_rows_by_key(tmp_path):
    @jit.plugin
    class Two:
        asc = jit.map(key=(jit.i64, jit.i64), value=jit.count(), ordered=True)
        plain = jit.map(key=(jit.i64, jit.i64), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.asc[(e.pid, e.tid)] += 1
            self.plain[(e.pid, e.tid)] += 1

    # File order is deliberately not ascending by (pid, tid).
    rows = [(3, 7), (1, 2), (3, 1), (1, 9), (2, 5), (1, 2), (3, 7), (2, 5)]
    _write_pid_tid_trace(str(tmp_path / "trace.pfw.gz"), rows)

    host = PluginHost()
    host.load(Two)
    host.resolve()
    results = host.run(str(tmp_path))

    asc = pa.table(results["asc"])
    plain = pa.table(results["plain"])

    asc_keys = list(zip(asc.column("k0").to_pylist(), asc.column("k1").to_pylist()))
    asc_vals = dict(zip(asc_keys, asc.column("value").to_pylist()))
    plain_keys = list(zip(plain.column("k0").to_pylist(), plain.column("k1").to_pylist()))
    plain_vals = dict(zip(plain_keys, plain.column("value").to_pylist()))

    # The ordered map's rows are sorted ascending by (k0, k1) by the engine.
    assert asc_keys == sorted(asc_keys)
    # Same content either way: only the row order differs.
    assert asc_vals == plain_vals
    assert sorted(plain_keys) == asc_keys


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_typed_integer_key_columns(tmp_path):
    @jit.plugin
    class Typed:
        m = jit.map(key=(jit.u16, jit.i32), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.m[(e.pid, e.tid)] += 1

    rows = [(1000, 7), (1000, 7), (2000, 9), (2000, 9), (2000, 9)]
    _write_pid_tid_trace(str(tmp_path / "trace.pfw.gz"), rows)

    host = PluginHost()
    host.load(Typed)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["m"])
    assert tbl.column_names == ["k0", "k1", "value"]
    assert pa.types.is_uint16(tbl.schema.field("k0").type)
    assert pa.types.is_int32(tbl.schema.field("k1").type)

    kv = {
        (k0, k1): v
        for k0, k1, v in zip(
            tbl.column("k0").to_pylist(),
            tbl.column("k1").to_pylist(),
            tbl.column("value").to_pylist(),
        )
    }
    assert kv[(1000, 7)] == 2
    assert kv[(2000, 9)] == 3


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_f64_key_column(tmp_path):
    @jit.plugin
    class ByDur:
        m = jit.map(key=(jit.f64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.m[(e.dur,)] += 1

    # dur repeats so bit-identical float keys merge: 10.0 -> 2, 20.0 -> 3.
    events = [(1, 10, 1), (1, 10, 2), (1, 20, 3), (1, 20, 4), (1, 20, 5)]
    _write_dur_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(ByDur)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["m"])
    assert tbl.column_names == ["k0", "value"]
    assert pa.types.is_float64(tbl.schema.field("k0").type)

    kv = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    assert kv[10.0] == 2
    assert kv[20.0] == 3


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_typed_min_max_value_columns(tmp_path):
    @jit.plugin
    class Vals:
        mn = jit.map(key=(jit.i64,), value=jit.min(of=jit.i32))
        mx = jit.map(key=(jit.i64,), value=jit.max(of=jit.u32))
        mf = jit.map(key=(jit.i64,), value=jit.min(of=jit.f32))

        @jit.each_event
        def step(self, e):
            self.mn[(e.pid,)].observe(e.dur - 8)
            self.mx[(e.pid,)].observe(e.dur)
            self.mf[(e.pid,)].observe(e.dur * 0.5)

    # dur-8 spans a negative i32 min (3-8=-5); dur holds a u32 max above
    # INT32_MAX; dur*0.5 a float32 min of 1.5.
    events = [(1, 3, 10), (1, 11, 20), (1, 4000000000, 30)]
    _write_dur_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(Vals)
    host.resolve()
    results = host.run(str(tmp_path))

    mn = pa.table(results["mn"])
    assert pa.types.is_int32(mn.schema.field("value").type)
    assert mn.column("value").to_pylist() == [-5]

    mx = pa.table(results["mx"])
    assert pa.types.is_uint32(mx.schema.field("value").type)
    assert mx.column("value").to_pylist() == [4000000000]

    mf = pa.table(results["mf"])
    assert pa.types.is_float32(mf.schema.field("value").type)
    assert mf.column("value").to_pylist() == [1.5]


def _write_name_only_trace(path: str, names) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, name in enumerate(names):
            f.write(
                f'{{"name":"{name}","cat":"POSIX","pid":1,"tid":1,'
                f'"ts":{1000 + i},"dur":1,"ph":"X","args":{{}}}}\n'
            )


def _write_arg_trace(path: str, n: int, pids, tags) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i in range(n):
            pid = pids[i % len(pids)]
            tag = tags[i % len(tags)]
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":1,"ph":"X",'
                f'"args":{{"size":{100 + i},"bw":{i * 0.5},'
                f'"rank":{i % 3},"tag":"{tag}"}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_arg_f64_as_value_sums(tmp_path):
    @jit.plugin
    class Bw:
        bw = jit.map(key=(jit.i64,), value=jit.sum())

        @jit.each_event
        def step(self, e):
            self.bw[(e.pid,)] += e.arg_f64("bw")

    # The compiled plugin must request args and intern the arg key.
    src = Bw._jit_plugin.source
    assert "DFTU_NEED_ARGS" in src
    assert "static dftu_str argkey_0;" in src
    assert "dftu_jit_arg_f64(e, argkey_0, 0.0)" in src

    n = 60
    pids = [1, 2]
    tags = ["ta", "tb", "tc"]
    _write_arg_trace(str(tmp_path / "trace.pfw.gz"), n, pids, tags)

    host = PluginHost()
    host.load(Bw)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["bw"])
    assert pa.types.is_float64(tbl.schema.field("value").type)
    val = tbl.column("value").to_numpy(zero_copy_only=False)
    assert float(val.sum()) == float(builtins.sum(i * 0.5 for i in range(n)))


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_arg_i64_into_counter_sums(tmp_path):
    @jit.plugin
    class Sizes:
        tot = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.tot[(e.pid,)] += e.arg_i64("size")

    n = 60
    pids = [1, 2]
    tags = ["ta", "tb", "tc"]
    _write_arg_trace(str(tmp_path / "trace.pfw.gz"), n, pids, tags)

    host = PluginHost()
    host.load(Sizes)
    host.resolve()
    results = host.run(str(tmp_path))

    val = pa.table(results["tot"]).column("value").to_numpy(zero_copy_only=False)
    assert int(val.sum()) == builtins.sum(100 + i for i in range(n))


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_arg_i64_as_key(tmp_path):
    @jit.plugin
    class ByRank:
        cnt = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.cnt[(e.arg_i64("rank"),)] += 1

    n = 60
    pids = [1, 2]
    tags = ["ta", "tb", "tc"]
    _write_arg_trace(str(tmp_path / "trace.pfw.gz"), n, pids, tags)

    host = PluginHost()
    host.load(ByRank)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["cnt"])
    kv = _kv(tbl)
    expected: dict = {}
    for i in range(n):
        expected[i % 3] = expected.get(i % 3, 0) + 1
    assert {int(k): int(v) for k, v in kv.items()} == expected


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_arg_str_set_element(tmp_path):
    @jit.plugin
    class Tags:
        tags = jit.map(key=(jit.i64,), value=jit.set())

        @jit.each_event
        def step(self, e):
            self.tags[(e.pid,)].observe(e.arg_str("tag"))

    n = 60
    pids = [1, 2]
    tags = ["ta", "tb", "tc"]
    _write_arg_trace(str(tmp_path / "trace.pfw.gz"), n, pids, tags)

    host = PluginHost()
    host.load(Tags)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["tags"])
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_string(vtype.value_type)
    by_pid = {
        k: set(v) for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())
    }
    assert by_pid[1] == set(tags)
    assert by_pid[2] == set(tags)


def test_jit_rejects_non_literal_arg_name():
    with pytest.raises(jit.JitError, match="string literal"):

        @jit.plugin
        class BadName:
            m = jit.map(key=(jit.i64,), value=jit.sum())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.arg_f64(some_name)  # noqa: F821


def test_jit_rejects_unknown_arg_method():
    with pytest.raises(jit.JitError, match="arg_i64 / arg_f64 / arg_str"):

        @jit.plugin
        class BadMethod:
            m = jit.map(key=(jit.i64,), value=jit.sum())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.arg_bytes("size")


def test_jit_rejects_arg_f64_into_counter():
    with pytest.raises(jit.JitError, match="u64 monoid"):

        @jit.plugin
        class BadCounter:
            m = jit.map(key=(jit.i64,), value=jit.count())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += e.arg_f64("bw")


def _nested_entries(tbl):
    """{outer_key: {inner_key: entry_dict}} for a nested list<struct> result."""
    out: dict = {}
    for k0, entries in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist()):
        inner = {}
        for entry in entries:
            fname = builtins.next(v for v in entry.values() if isinstance(v, str))
            inner[fname] = entry
        out[k0] = inner
    return out


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_nested_counter_per_pid_file(tmp_path):
    @jit.plugin
    class PidFileCounts:
        m = jit.map(key=(jit.i64,), value=jit.nested(key=(jit.str_,), value=jit.count()))

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.m[(e.pid,)][(e.fhash,)] += 1

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(PidFileCounts)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["m"])
    assert tbl.column_names == ["k0", "value"]
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_struct(vtype.value_type)
    assert [f.name for f in vtype.value_type] == ["ik0", "value"]
    assert pa.types.is_string(vtype.value_type.field("ik0").type)
    assert pa.types.is_int64(vtype.value_type.field("value").type)

    # One row per pid; each carries every file it touched with its own count.
    expected: dict = {}
    for i in range(n):
        pid = pids[i % len(pids)]
        f = files[i % len(files)]
        expected.setdefault(pid, {})
        expected[pid][f] = expected[pid].get(f, 0) + 1

    by_pid = _nested_entries(tbl)
    assert builtins.set(by_pid) == builtins.set(expected)
    total = 0
    for pid, inner in by_pid.items():
        assert builtins.set(inner) == builtins.set(expected[pid])
        for fname, entry in inner.items():
            assert entry["value"] == expected[pid][fname]
            total += entry["value"]
    assert total == n


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_nested_product_count_and_sum(tmp_path):
    @jit.plugin
    class PidFileStats:
        m = jit.map(
            key=(jit.i64,),
            value=jit.nested(key=(jit.str_,), value=dict(cnt=jit.count(), dur=jit.sum())),
        )

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.m[(e.pid,)][(e.fhash,)].cnt += 1
                self.m[(e.pid,)][(e.fhash,)].dur += e.dur

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    host = PluginHost()
    host.load(PidFileStats)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["m"])
    assert tbl.column_names == ["k0", "value"]
    st = tbl.schema.field("value").type.value_type
    assert [f.name for f in st] == ["ik0", "v0", "v1"]
    assert pa.types.is_int64(st.field("v0").type)
    assert pa.types.is_float64(st.field("v1").type)

    exp_cnt: dict = {}
    exp_dur: dict = {}
    for i in range(n):
        pid = pids[i % len(pids)]
        f = files[i % len(files)]
        exp_cnt[(pid, f)] = exp_cnt.get((pid, f), 0) + 1
        exp_dur[(pid, f)] = exp_dur.get((pid, f), 0.0) + (10 + i)

    by_pid = _nested_entries(tbl)
    tot_cnt = 0
    tot_dur = 0.0
    for pid, inner in by_pid.items():
        for fname, entry in inner.items():
            assert entry["v0"] == exp_cnt[(pid, fname)]
            assert entry["v1"] == exp_dur[(pid, fname)]
            tot_cnt += entry["v0"]
            tot_dur += entry["v1"]
    assert tot_cnt == n
    assert tot_dur == float(sum(10 + i for i in range(n)))


def test_jit_rejects_single_subscript_on_nested():
    with pytest.raises(jit.JitError, match="needs .outer..inner."):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.nested(key=(jit.i64,), value=jit.count()))

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)] += 1


def test_jit_rejects_collection_in_nested_value():
    with pytest.raises(jit.JitError, match="set/list collection"):
        jit.nested(key=(jit.i64,), value=jit.set())


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_flat_multikey_chained_subscript_sugar(tmp_path):
    @jit.plugin
    class Chained:
        m = jit.map(key=(jit.i64, jit.i64), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.m[e.pid][e.tid] += 1

    rows = [(1, 7), (1, 7), (2, 9), (2, 9), (2, 9)]
    _write_pid_tid_trace(str(tmp_path / "trace.pfw.gz"), rows)

    host = PluginHost()
    host.load(Chained)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["m"])
    assert tbl.column_names == ["k0", "k1", "value"]
    kv = {
        (k0, k1): v
        for k0, k1, v in zip(
            tbl.column("k0").to_pylist(),
            tbl.column("k1").to_pylist(),
            tbl.column("value").to_pylist(),
        )
    }
    assert kv[(1, 7)] == 2
    assert kv[(2, 9)] == 3


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_bare_single_key_matches_tuple_form(tmp_path):
    @jit.plugin
    class Bare:
        m = jit.map(key=jit.i64, value=jit.count())

        @jit.each_event
        def step(self, e):
            self.m[e.pid] += 1

    @jit.plugin
    class Tupled:
        m = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            self.m[(e.pid,)] += 1

    # Bare key normalizes to the identical 1-tuple decl the compiler reads.
    assert "kt_m[1]" in Bare._jit_plugin.source
    assert "DFTU_T_I64" in Bare._jit_plugin.source

    n = 60
    pids = [1, 2]
    files = ["fileA", "fileB", "fileC"]
    _write_trace(str(tmp_path / "trace.pfw.gz"), n, pids, files)

    bare_host = PluginHost()
    bare_host.load(Bare)
    bare_host.resolve()
    bare = _kv(pa.table(bare_host.run(str(tmp_path))["m"]))

    tup_host = PluginHost()
    tup_host.load(Tupled)
    tup_host.resolve()
    tup = _kv(pa.table(tup_host.run(str(tmp_path))["m"]))

    assert {int(k): int(v) for k, v in bare.items()} == {int(k): int(v) for k, v in tup.items()}
    assert int(sum(bare.values())) == n


def test_jit_rejects_bare_subscript_on_multikey():
    with pytest.raises(jit.JitError, match="key components"):

        @jit.plugin
        class Multi:
            m = jit.map(key=(jit.i64, jit.i64), value=jit.count())

            @jit.each_event
            def step(self, e):
                self.m[e.pid] += 1


def test_jit_bytes_key_declares_but_rejects_each_event_source():
    @jit.plugin(needs=(jit.NEED_FHASH,))
    class RawBytes:
        m = jit.map(key=jit.bytes, value=jit.count())

        @jit.each_event(raw=True)
        def step(self):
            return ""

    # The bare bytes marker normalizes and emits a DFTU_T_BYTES-keyed decl.
    src = RawBytes._jit_plugin.source
    assert "DFTU_T_BYTES" in src
    assert "kt_m[1]" in src

    # A non-raw each_event has no honest per-event bytes source, so it is rejected.
    with pytest.raises(jit.JitError, match="bytes key"):

        @jit.plugin
        class BadBytes:
            m = jit.map(key=jit.bytes, value=jit.count())

            @jit.each_event
            def step(self, e):
                self.m[e.pid] += 1


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_ordered_str_key_map_sorts_by_label(tmp_path):
    @jit.plugin
    class ByName:
        by_name = jit.map(key=(jit.str_,), value=jit.count(), ordered=True)

        @jit.each_event
        def step(self, e):
            self.by_name[(e.name,)] += 1

    # Names emitted out of alphabetical order: apple x2, banana x1, cat x3.
    names = ["cat", "apple", "cat", "banana", "apple", "cat"]
    _write_name_only_trace(str(tmp_path / "trace.pfw.gz"), names)

    host = PluginHost()
    host.load(ByName)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["by_name"])
    keys = tbl.column("k0").to_pylist()
    vals = tbl.column("value").to_pylist()

    # STR key rows come out sorted by resolved label (alphabetical).
    assert keys == ["apple", "banana", "cat"]
    assert dict(zip(keys, vals)) == {"apple": 2, "banana": 1, "cat": 3}


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_variance_matches_handcomputed(tmp_path):
    @jit.plugin
    class DurVar:
        dv = jit.map(key=(jit.i64,), value=jit.variance())

        @jit.each_event
        def step(self, e):
            self.dv[(e.pid,)].observe(e.dur)

    events = [(1, 20, 10), (1, 5, 20), (1, 20, 30), (1, 8, 40), (2, 3, 1), (2, 7, 5)]
    _write_dur_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(DurVar)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["dv"])
    assert tbl.column_names == ["k0", "value"]
    assert pa.types.is_float64(tbl.schema.field("value").type)

    durs: dict = {}
    for pid, dur, _ts in events:
        durs.setdefault(pid, []).append(dur)
    got = _kv(tbl)
    assert float(got[1]) == pytest.approx(statistics.variance(durs[1]))
    assert float(got[2]) == pytest.approx(statistics.variance(durs[2]))


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_mean_and_stddev_columns(tmp_path):
    @jit.plugin
    class DurStats:
        mu = jit.map(key=(jit.i64,), value=jit.mean())
        sd = jit.map(key=(jit.i64,), value=jit.stddev())

        @jit.each_event
        def step(self, e):
            self.mu[(e.pid,)].observe(e.dur)
            self.sd[(e.pid,)].observe(e.dur)

    events = [(1, 20, 10), (1, 5, 20), (1, 20, 30), (1, 8, 40), (2, 3, 1), (2, 7, 5)]
    _write_dur_trace(str(tmp_path / "trace.pfw.gz"), events)

    host = PluginHost()
    host.load(DurStats)
    host.resolve()
    results = host.run(str(tmp_path))

    assert pa.types.is_float64(pa.table(results["mu"]).schema.field("value").type)
    assert pa.types.is_float64(pa.table(results["sd"]).schema.field("value").type)

    durs: dict = {}
    for pid, dur, _ts in events:
        durs.setdefault(pid, []).append(dur)
    mu = _kv(pa.table(results["mu"]))
    sd = _kv(pa.table(results["sd"]))
    for pid, vs in durs.items():
        assert float(mu[pid]) == pytest.approx(statistics.mean(vs))
        assert float(sd[pid]) == pytest.approx(statistics.stdev(vs))


def _write_file_dur_trace(path: str, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, (pid, fhash, dur) in enumerate(rows):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":{dur},"ph":"X",'
                f'"args":{{"fhash":"{fhash}"}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_argmax_argmin_str_payload(tmp_path):
    @jit.plugin(needs=(jit.NEED_FHASH,))
    class HotCold:
        hot = jit.map(key=(jit.i64,), value=jit.argmax(of=jit.str_))
        cold = jit.map(key=(jit.i64,), value=jit.argmin(of=jit.str_))

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.hot[(e.pid,)].observe(e.fhash, by=e.dur)
                self.cold[(e.pid,)].observe(e.fhash, by=e.dur)

    # pid 1: max dur 30 -> "b", min dur 5 -> "a"; pid 2: max 7 -> "x", min 2 -> "y".
    rows = [(1, "a", 5), (1, "b", 30), (1, "c", 10), (2, "x", 7), (2, "y", 2)]
    _write_file_dur_trace(str(tmp_path / "trace.pfw.gz"), rows)

    host = PluginHost()
    host.load(HotCold)
    host.resolve()
    results = host.run(str(tmp_path))

    hot = pa.table(results["hot"])
    cold = pa.table(results["cold"])
    assert pa.types.is_string(hot.schema.field("value").type)
    hot_by_pid = {
        k: v for k, v in zip(hot.column("k0").to_pylist(), hot.column("value").to_pylist())
    }
    cold_by_pid = {
        k: v for k, v in zip(cold.column("k0").to_pylist(), cold.column("value").to_pylist())
    }
    assert hot_by_pid == {1: "b", 2: "x"}
    assert cold_by_pid == {1: "a", 2: "y"}


def _write_tid_dur_trace(path: str, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, (pid, tid, dur) in enumerate(rows):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":{tid},'
                f'"ts":{1000 + i},"dur":{dur},"ph":"X","args":{{}}}}\n'
            )


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_argmax_i64_payload(tmp_path):
    @jit.plugin
    class SlowTid:
        tid_at_max = jit.map(key=(jit.i64,), value=jit.argmax(of=jit.i64))

        @jit.each_event
        def step(self, e):
            self.tid_at_max[(e.pid,)].observe(e.tid, by=e.dur)

    # pid 1: max dur 30 at tid 8; pid 2: max dur 40 at tid 5.
    rows = [(1, 7, 10), (1, 8, 30), (1, 9, 20), (2, 5, 40), (2, 6, 15)]
    _write_tid_dur_trace(str(tmp_path / "trace.pfw.gz"), rows)

    host = PluginHost()
    host.load(SlowTid)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["tid_at_max"])
    assert pa.types.is_int64(tbl.schema.field("value").type)
    got = {k: int(v) for k, v in _kv(tbl).items()}
    assert got == {1: 8, 2: 5}


def test_jit_rejects_argby_observe_without_by():
    with pytest.raises(jit.JitError, match="requires by="):

        @jit.plugin
        class NoBy:
            m = jit.map(key=(jit.i64,), value=jit.argmax(of=jit.str_))

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.fhash)


def test_jit_rejects_moment_observe_with_by():
    with pytest.raises(jit.JitError, match="one value argument"):

        @jit.plugin
        class MomentBy:
            m = jit.map(key=(jit.i64,), value=jit.variance())

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.dur, by=e.ts)


def test_jit_rejects_str_payload_into_i64_argmax():
    with pytest.raises(jit.JitError, match="int64 expr"):

        @jit.plugin
        class Bad:
            m = jit.map(key=(jit.i64,), value=jit.argmax(of=jit.i64))

            @jit.each_event
            def step(self, e):
                self.m[(e.pid,)].observe(e.name, by=e.dur)


def test_jit_rejects_argby_in_nested():
    with pytest.raises(jit.JitError, match="nested argby"):
        jit.nested(key=(jit.str_,), value=jit.argmax(of=jit.str_))


def _write_row_trace(path: str, rows) -> None:
    """rows: (pid, fhash, tid, dur, ts)."""
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for pid, fhash, tid, dur, ts in rows:
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":{tid},'
                f'"ts":{ts},"dur":{dur},"ph":"X",'
                f'"args":{{"fhash":"{fhash}"}}}}\n'
            )


# pid 1: a@100, b@300, c@200, a@50; pid 2: x@40, y@70. dur is the by/rank key.
_ROWS = [
    (1, "a", 7, 100, 1000),
    (1, "b", 8, 300, 1001),
    (1, "c", 9, 200, 1002),
    (1, "a", 7, 50, 1003),
    (2, "x", 5, 40, 2000),
    (2, "y", 6, 70, 2001),
]


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_topk_keeps_extreme_payloads_in_order(tmp_path):
    @jit.plugin(needs=(jit.NEED_FHASH,))
    class Hot:
        hot = jit.map(key=jit.i64, value=jit.topk(2, of=jit.str_))
        low = jit.map(key=jit.i64, value=jit.bottomk(2, of=jit.i64))

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.hot[e.pid].observe(e.fhash, by=e.dur)
                self.low[e.pid].observe(e.dur, by=e.dur)

    _write_row_trace(str(tmp_path / "trace.pfw.gz"), _ROWS)

    host = PluginHost()
    host.load(Hot)
    host.resolve()
    results = host.run(str(tmp_path))

    hot = pa.table(results["hot"])
    vtype = hot.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_string(vtype.value_type)
    hot_by_pid = {
        k: v for k, v in zip(hot.column("k0").to_pylist(), hot.column("value").to_pylist())
    }
    # Top 2 payloads by dur, largest first.
    assert hot_by_pid[1] == ["b", "c"]
    assert hot_by_pid[2] == ["y", "x"]

    low = pa.table(results["low"])
    assert pa.types.is_integer(low.schema.field("value").type.value_type)
    low_by_pid = {
        k: v for k, v in zip(low.column("k0").to_pylist(), low.column("value").to_pylist())
    }
    # Bottom 2 dur payloads, smallest first.
    assert low_by_pid[1] == [50, 100]
    assert low_by_pid[2] == [40, 70]


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_approx_topk_exact_when_k_ge_distinct(tmp_path):
    @jit.plugin(needs=(jit.NEED_FHASH,))
    class Freq:
        freq = jit.map(key=jit.i64, value=jit.approx_topk(8, of=jit.str_))

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.freq[e.pid].observe(e.fhash)

    _write_row_trace(str(tmp_path / "trace.pfw.gz"), _ROWS)

    host = PluginHost()
    host.load(Freq)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["freq"])
    vtype = tbl.schema.field("value").type
    assert pa.types.is_list(vtype)
    assert pa.types.is_struct(vtype.value_type)
    assert [f.name for f in vtype.value_type] == ["value", "count"]

    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    # k >= distinct so counts are exact: pid 1 sees a twice, b and c once.
    counts1 = {e["value"]: e["count"] for e in by_pid[1]}
    assert counts1 == {"a": 2, "b": 1, "c": 1}
    counts2 = {e["value"]: e["count"] for e in by_pid[2]}
    assert counts2 == {"x": 1, "y": 1}


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_sample_keeps_items_from_input(tmp_path):
    @jit.plugin
    class Samp:
        samp = jit.map(key=jit.i64, value=jit.sample(10, of=jit.i64))
        small = jit.map(key=jit.i64, value=jit.sample(2, of=jit.i64))

        @jit.each_event
        def step(self, e):
            self.samp[e.pid].observe(e.ts)
            self.small[e.pid].observe(e.ts)

    _write_row_trace(str(tmp_path / "trace.pfw.gz"), _ROWS)

    host = PluginHost()
    host.load(Samp)
    host.resolve()
    results = host.run(str(tmp_path))

    ts_by_pid = {1: {1000, 1001, 1002, 1003}, 2: {2000, 2001}}

    tbl = pa.table(results["samp"])
    assert pa.types.is_integer(tbl.schema.field("value").type.value_type)
    by_pid = {k: v for k, v in zip(tbl.column("k0").to_pylist(), tbl.column("value").to_pylist())}
    # k >= distinct: the full distinct set, sorted.
    assert by_pid[1] == sorted(ts_by_pid[1])
    assert by_pid[2] == sorted(ts_by_pid[2])

    small = pa.table(results["small"])
    sm_by_pid = {
        k: v for k, v in zip(small.column("k0").to_pylist(), small.column("value").to_pylist())
    }
    for pid, items in sm_by_pid.items():
        assert len(items) <= 2
        assert set(items) <= ts_by_pid[pid]


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_argmax_row_keeps_whole_winning_row(tmp_path):
    @jit.plugin(needs=(jit.NEED_FHASH,))
    class Slow:
        slow = jit.map(key=jit.i64, value=jit.argmax_row(of=(jit.str_, jit.i64, jit.i64)))

        @jit.each_event
        def step(self, e):
            if e.fhash != jit.NONE:
                self.slow[e.pid].observe((e.fhash, e.tid, e.ts), by=e.dur)

    _write_row_trace(str(tmp_path / "trace.pfw.gz"), _ROWS)

    host = PluginHost()
    host.load(Slow)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["slow"])
    assert tbl.column_names == ["k0", "p0", "p1", "p2"]
    assert pa.types.is_string(tbl.schema.field("p0").type)
    assert pa.types.is_int64(tbl.schema.field("p1").type)
    assert pa.types.is_int64(tbl.schema.field("p2").type)

    row = {
        k: (p0, p1, p2)
        for k, p0, p1, p2 in zip(
            tbl.column("k0").to_pylist(),
            tbl.column("p0").to_pylist(),
            tbl.column("p1").to_pylist(),
            tbl.column("p2").to_pylist(),
        )
    }
    # Row-consistent: the whole (fhash, tid, ts) at the max dur.
    assert row[1] == ("b", 8, 1001)
    assert row[2] == ("y", 6, 2001)


def test_jit_rejects_topk_observe_without_by():
    with pytest.raises(jit.JitError, match="requires by="):

        @jit.plugin
        class NoBy:
            m = jit.map(key=jit.i64, value=jit.topk(3, of=jit.str_))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe(e.fhash)


def test_jit_rejects_approx_topk_observe_with_by():
    with pytest.raises(jit.JitError, match="one value argument"):

        @jit.plugin
        class WithBy:
            m = jit.map(key=jit.i64, value=jit.approx_topk(3, of=jit.str_))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe(e.fhash, by=e.dur)


def test_jit_rejects_sample_observe_with_by():
    with pytest.raises(jit.JitError, match="one value argument"):

        @jit.plugin
        class WithBy:
            m = jit.map(key=jit.i64, value=jit.sample(3, of=jit.i64))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe(e.ts, by=e.dur)


def test_jit_rejects_sample_str_into_i64():
    with pytest.raises(jit.JitError, match="int64 expr"):

        @jit.plugin
        class Bad:
            m = jit.map(key=jit.i64, value=jit.sample(3, of=jit.i64))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe(e.name)


def test_jit_rejects_argrow_observe_without_tuple():
    with pytest.raises(jit.JitError, match="must be a tuple"):

        @jit.plugin
        class Bad:
            m = jit.map(key=jit.i64, value=jit.argmax_row(of=(jit.str_,)))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe(e.fhash, by=e.dur)


def test_jit_rejects_argrow_observe_without_by():
    with pytest.raises(jit.JitError, match="requires by="):

        @jit.plugin
        class Bad:
            m = jit.map(key=jit.i64, value=jit.argmax_row(of=(jit.str_, jit.i64)))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe((e.fhash, e.tid))


def test_jit_rejects_argrow_wrong_payload_arity():
    with pytest.raises(jit.JitError, match="payload takes 2 components"):

        @jit.plugin
        class Bad:
            m = jit.map(key=jit.i64, value=jit.argmax_row(of=(jit.str_, jit.i64)))

            @jit.each_event
            def step(self, e):
                self.m[e.pid].observe((e.fhash,), by=e.dur)


def test_jit_rejects_argrow_bad_of_type():
    with pytest.raises(jit.JitError, match="non-empty tuple"):
        jit.argmax_row(of=jit.str_)


def _write_join_trace(path: str, rows) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, (pid, cat, dur) in enumerate(rows):
            f.write(
                f'{{"name":"read","cat":"{cat}","pid":{pid},"tid":1,'
                f'"ts":{1000 + i},"dur":{dur},"ph":"X","args":{{}}}}\n'
            )


# pid 1 is always POSIX (present in both maps); pid 2 is never POSIX, so it lands
# in counts but not durs: a left join keeps it with a null right value.
_JOIN_ROWS = [
    (1, "POSIX", 10),
    (1, "POSIX", 20),
    (2, "STDIO", 30),
    (2, "STDIO", 40),
    (2, "STDIO", 50),
]


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_join_left_null_pads_unmatched(tmp_path):
    @jit.plugin
    class PidJoin:
        counts = jit.map(key=jit.i64, value=jit.count())
        durs = jit.map(key=jit.i64, value=jit.sum())
        joined = jit.join(counts, durs, how="left")

        @jit.each_event
        def step(self, e):
            self.counts[e.pid] += 1
            if e.cat == "POSIX":
                self.durs[e.pid] += e.dur

    # The declaration lowers to a map_declare_join alongside the two map_new calls.
    src = PidJoin._jit_plugin.source
    assert 'map->map_declare_join(host->h, "joined", "counts", "durs", DFTU_JOIN_LEFT);' in src

    _write_join_trace(str(tmp_path / "trace.pfw.gz"), _JOIN_ROWS)

    host = PluginHost()
    host.load(PidJoin)
    host.resolve()
    results = host.run(str(tmp_path))

    # Input maps still surface additively.
    assert "counts" in results
    assert "durs" in results
    assert "joined" in results
    assert _kv(pa.table(results["counts"])) == {1: 2, 2: 3}

    tbl = pa.table(results["joined"])
    assert tbl.column_names == ["k0", "l_value", "r_value"]
    rows = {
        k0: (lv, rv)
        for k0, lv, rv in zip(
            tbl.column("k0").to_pylist(),
            tbl.column("l_value").to_pylist(),
            tbl.column("r_value").to_pylist(),
        )
    }
    assert set(rows) == {1, 2}
    assert rows[1] == (2, 30.0)
    # pid 2 matched no right row: left join null-pads the right value.
    assert rows[2][0] == 3
    assert rows[2][1] is None


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_join_inner_keeps_only_matched(tmp_path):
    @jit.plugin
    class PidJoin:
        counts = jit.map(key=jit.i64, value=jit.count())
        durs = jit.map(key=jit.i64, value=jit.sum())
        joined = jit.join(counts, durs)

        @jit.each_event
        def step(self, e):
            self.counts[e.pid] += 1
            if e.cat == "POSIX":
                self.durs[e.pid] += e.dur

    assert 'map->map_declare_join(host->h, "joined", "counts", "durs", DFTU_JOIN_INNER);' in (
        PidJoin._jit_plugin.source
    )

    _write_join_trace(str(tmp_path / "trace.pfw.gz"), _JOIN_ROWS)

    host = PluginHost()
    host.load(PidJoin)
    host.resolve()
    results = host.run(str(tmp_path))

    tbl = pa.table(results["joined"])
    rows = {
        k0: (lv, rv)
        for k0, lv, rv in zip(
            tbl.column("k0").to_pylist(),
            tbl.column("l_value").to_pylist(),
            tbl.column("r_value").to_pylist(),
        )
    }
    # Only pid 1 is present in both maps.
    assert set(rows) == {1}
    assert rows[1] == (2, 30.0)


def test_jit_join_rejects_bad_how():
    with pytest.raises(jit.JitError, match="inner / left / right / full"):
        a = jit.map(key=jit.i64, value=jit.count())
        b = jit.map(key=jit.i64, value=jit.sum())
        jit.join(a, b, how="outer")


def test_jit_join_rejects_non_map_operand():
    with pytest.raises(jit.JitError, match="jit.map declarations"):
        a = jit.map(key=jit.i64, value=jit.count())
        jit.join(a, 123)


def test_jit_rejects_observe_on_join_result():
    with pytest.raises(jit.JitError, match="jit.join result"):

        @jit.plugin
        class Bad:
            counts = jit.map(key=jit.i64, value=jit.count())
            durs = jit.map(key=jit.i64, value=jit.sum())
            joined = jit.join(counts, durs)

            @jit.each_event
            def step(self, e):
                self.joined[e.pid] += 1


@jit.plugin
class _SessionCounts:
    hits = jit.map(key=(jit.i64,), value=jit.count())

    @jit.each_event
    def step(self, e):
        self.hits[(e.pid,)] += 1


@pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler available for the jit backend")
def test_jit_plugin_fused_in_session(tmp_path):
    import dftracer.utils as dftu

    gz = str(tmp_path / "t.pfw.gz")
    _write_trace(gz, 10, [100, 200, 300], ["f0"])
    with dftu.Indexer(files=[gz], index_dir=str(tmp_path)) as ix:
        ix.ensure_indexed()

    tv = dftu.TraceViewer(gz, index_path=str(tmp_path))
    with tv.session() as s:
        by_cat = s.view().group_by("cat").agg("count").collect()
        counts = s.view().plugin(_SessionCounts)

    # collect and the plugin ran over one shared scan.
    cat = pa.table(by_cat.result()).to_pandas()
    assert int(cat["count"].sum()) == 10

    res = counts.result()  # single named result -> our DataFrame
    assert isinstance(res, dftu.DataFrame)
    pdf = pa.table(res).to_pandas()
    assert int(pdf["value"].sum()) == 10  # every event counted on the same scan
    assert len(pdf) == 3  # three distinct pids
