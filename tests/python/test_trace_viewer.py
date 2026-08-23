#!/usr/bin/env python3
"""Tests for the arrow-first TraceViewer."""

import gzip
import json

import pyarrow as pa

import dftracer.utils as dftu_utils
from dftracer.utils import TraceViewer

from .common import Environment


def _indexed(env):
    gz = env.create_test_gzip_file()
    with dftu_utils.Indexer(files=[gz]) as indexer:
        indexer.ensure_indexed()
    return gz


class TestTraceViewer:
    def test_collect_group_by_agg_returns_arrow(self):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            tbl = TraceViewer(gz).group_by("cat").agg("count", "mean:dur", "std:dur").collect()
            df = tbl.to_pandas()
            assert set(["cat", "count", "mean_dur", "std_dur"]).issubset(df.columns)
            # Every event lands in exactly one cat group.
            assert int(df["count"].sum()) == 200
            assert set(df["cat"]) <= {"posix", "stdio"}  # group value lowercases cat

    def test_agg_set_union_distinct_values(self):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            df = TraceViewer(gz).agg("set_union:cat").collect().to_pandas()
            assert df.shape[0] == 1
            # One string column of the distinct cat values, joined by \x1e.
            assert set(df["set_cat"].iloc[0].split("\x1e")) == {"POSIX", "STDIO"}

    def test_statistics_summary(self):
        with Environment(lines=150) as env:
            gz = _indexed(env)
            stats = TraceViewer(gz).statistics()
            assert stats["duration_count"] == 150
            assert stats["duration_stddev_us"] >= 0.0
            assert stats["max_timestamp_us"] >= stats["min_timestamp_us"]

    def test_filter_narrows_events(self):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            full = TraceViewer(gz).statistics()["duration_count"]
            stdio = TraceViewer(gz).filter('cat == "STDIO"').statistics()["duration_count"]
            assert 0 < stdio < full

    def test_materialize_row_view_and_reuse(self, tmp_path):
        import os

        with Environment(lines=200) as env:
            gz = _indexed(env)
            mv_root = str(tmp_path / "mv")

            def stdio_view():
                return TraceViewer(gz).filter('cat == "STDIO"').views_root(mv_root)

            def stdio_count():
                v = stdio_view()
                return sum(pa.record_batch(c).num_rows for c in v.stream(batch_size=128))

            base = stdio_count()
            base_stats = stdio_view().statistics()
            assert base > 0
            assert stdio_view().mv_source() == []  # nothing materialized yet

            # Build the filtered-trace MV with a progress callback.
            ticks = []
            stdio_view().materialize(progress=lambda done, total: ticks.append((done, total)))
            parts = [
                f
                for _root, _dirs, files in os.walk(mv_root)
                for f in files
                if f.endswith(".pfw.gz")
            ]
            assert parts, "materialize should write at least one part file"
            assert ticks and ticks[-1][0] == ticks[-1][1]  # progress reached 100%

            # Observability: a matching query now reports the MV as its source.
            assert stdio_view().mv_source(), "query should be served from the MV"

            # A matching stream and statistics() both stay correct off the MV.
            assert stdio_count() == base
            assert stdio_view().statistics()["duration_count"] == base_stats["duration_count"]

    def test_stream_yields_all_events_as_arrow(self):
        with Environment(lines=500) as env:
            gz = _indexed(env)
            batches = [pa.record_batch(c) for c in TraceViewer(gz).stream(batch_size=128)]
            assert batches, "stream produced no batches"
            tbl = pa.Table.from_batches(batches)
            assert tbl.num_rows == 500
            assert {"name", "cat", "ts", "dur"}.issubset(tbl.column_names)

    def test_stream_respects_filter(self):
        with Environment(lines=400) as env:
            gz = _indexed(env)
            tbl = pa.Table.from_batches(
                [pa.record_batch(c) for c in TraceViewer(gz).filter('cat == "STDIO"').stream()]
            )
            assert 0 < tbl.num_rows < 400
            assert set(tbl.column("cat").to_pylist()) == {"STDIO"}

    def test_export_events_gzip_reindexes(self, tmp_path):
        with Environment(lines=300) as env:
            gz = _indexed(env)
            out = str(tmp_path / "events.pfw.gz")
            TraceViewer(gz).filter('cat == "STDIO"').export_trace(out)
            # The exported trace is a valid, re-indexable dftracer trace.
            with dftu_utils.Indexer(files=[out], index_dir=str(tmp_path)) as ix:
                ix.ensure_indexed()
            reread = TraceViewer(out, index_path=str(tmp_path)).statistics()
            assert 0 < reread["duration_count"] < 300

    def test_export_aggregation_gzip(self, tmp_path):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            out = str(tmp_path / "agg.pfw.gz")
            TraceViewer(gz).group_by("cat").time_bucket(1000).agg("count").export_trace(out)
            with gzip.open(out, "rt") as fh:
                events = [json.loads(line) for line in fh if line.strip()]
            assert events and all(e.get("ph") == 2 for e in events)  # ph=C counters

    def test_export_plain_ndjson(self, tmp_path):
        with Environment(lines=100) as env:
            gz = _indexed(env)
            out = str(tmp_path / "plain.pfw")
            TraceViewer(gz).export_trace(out, compress=False)
            lines = [line for line in open(out) if line.strip()]
            assert len(lines) == 100

    def test_stream_select_pushdown(self):
        with Environment(lines=300) as env:
            gz = _indexed(env)
            tbl = pa.Table.from_batches(
                [pa.record_batch(c) for c in TraceViewer(gz).select("ts", "dur").stream()]
            )
            assert set(tbl.column_names) == {"ts", "dur"}
            assert tbl.num_rows == 300

    def test_stream_dictionary_encoding(self):
        with Environment(lines=600) as env:
            gz = _indexed(env)
            # Small batches force multiple per-batch dictionaries to unify.
            tbl = pa.Table.from_batches(
                [pa.record_batch(c) for c in TraceViewer(gz).stream(batch_size=64, dict=True)]
            )
            assert pa.types.is_dictionary(tbl.schema.field("cat").type)
            assert tbl.num_rows == 600
            plain = pa.Table.from_batches(
                [pa.record_batch(c) for c in TraceViewer(gz).stream(dict=False)]
            )
            assert pa.types.is_string(plain.schema.field("cat").type)

    def test_aggregate_partial_merge_matches_collect(self):
        """Distributed partial+merge equals a single collect (incl. mean/std)."""
        with Environment(lines=300) as env:
            files = [env.create_test_gzip_file(f"f{k}/t{k}.pfw.gz") for k in range(3)]
            with dftu_utils.Indexer(files=files, index_dir=env.temp_dir) as ix:
                ix.ensure_indexed()

            def view(fs):
                return (
                    dftu_utils.TraceViewer(fs, index_path=env.temp_dir)
                    .group_by("cat")
                    .agg("count", "mean:dur", "std:dur")
                )

            whole = pa.table(view(files).collect()).sort_by("cat")
            partials = [view([f]).aggregate_partial() for f in files]
            for b in partials:
                assert isinstance(b, bytes) and b
            merged = pa.table(view([]).merge_partials_to_table(partials)).sort_by("cat")

            assert whole.column("count").to_pylist() == merged.column("count").to_pylist()
            for col in ("mean_dur", "std_dur"):
                for a, b in zip(whole.column(col).to_pylist(), merged.column(col).to_pylist()):
                    assert abs(a - b) < 1e-6

    def test_materialize_rollup_then_collect_reads_it(self):
        with Environment(lines=300) as env:
            gz = _indexed(env)
            order = [("cat", "ascending"), ("time_bucket", "ascending")]

            def view():
                return (
                    TraceViewer(gz)
                    .rollup_root(env.temp_dir + "/vc")
                    .group_by("cat")
                    .time_bucket(1000)
                    .agg("count", "mean:dur")
                )

            fresh = pa.table(view().collect()).sort_by(order)
            assert view().reconstruct_if_cached() is None  # cold
            view().materialize()  # build the rollup
            warm = view().reconstruct_if_cached()
            assert warm is not None
            # reconstruct and a plain collect (auto-reads the rollup) both match.
            assert (
                pa.table(warm)
                .sort_by(order)
                .to_pandas()
                .round(6)
                .equals(fresh.to_pandas().round(6))
            )
            assert (
                pa.table(view().collect())
                .sort_by(order)
                .to_pandas()
                .round(6)
                .equals(fresh.to_pandas().round(6))
            )

    def test_cache_terminals_gated_to_aggregation(self):
        """group_by/agg promote TraceViewer -> AggregatedTraceViewer, which alone
        carries the cache terminals; a raw TraceViewer cannot reach them."""
        import pytest

        from dftracer.utils import AggregatedTraceViewer

        with Environment(lines=100) as env:
            gz = _indexed(env)
            base = TraceViewer(gz)
            assert not hasattr(base, "materialize_partials")
            assert not hasattr(base.filter("cat == POSIX"), "reconstruct_if_cached")
            assert isinstance(base.group_by("cat"), AggregatedTraceViewer)
            assert isinstance(base.agg("count"), AggregatedTraceViewer)
            # Aggregation stays through a further op, and only it has the rollup.
            agg = base.group_by("cat").time_bucket(1000)
            assert isinstance(agg, AggregatedTraceViewer)
            assert hasattr(agg, "materialize_partials")
            # Base collect() takes no cache kwarg.
            with pytest.raises(TypeError):
                base.collect(cache=True)

    def test_io_cat_group_and_sumsq(self, tmp_path):
        """group_by io_cat classifies by function; sumsq:dur == sum of dur^2."""
        gz = str(tmp_path / "io.pfw.gz")
        specs = [("read", 2, 10), ("write", 3, 20), ("open", 1, 5)]  # io_cat 1,2,3
        with gzip.open(gz, "wt") as f:
            for name, n, dur in specs:
                for i in range(n):
                    f.write(
                        '{"ph":"X","name":"%s","cat":"POSIX","pid":1,"tid":1,'
                        '"ts":%d,"dur":%d,"args":{}}\n' % (name, 1000 + i, dur)
                    )
        with dftu_utils.Indexer(files=[gz], index_dir=str(tmp_path)) as ix:
            ix.ensure_indexed()
        df = (
            pa.table(
                dftu_utils.TraceViewer(gz, index_path=str(tmp_path))
                .group_by("io_cat")
                .agg("count", "sumsq:dur")
                .collect()
            )
            .to_pandas()
            .set_index("io_cat")
        )
        # io_cat comes back as the stringified IOCategory int (read=1/write=2/meta=3).
        assert df.loc["1", "count"] == 2 and df.loc["1", "sumsq_dur"] == 2 * 10**2
        assert df.loc["2", "count"] == 3 and df.loc["2", "sumsq_dur"] == 3 * 20**2
        assert df.loc["3", "count"] == 1 and df.loc["3", "sumsq_dur"] == 1 * 5**2

    def test_size_derived_from_ret_for_io(self, tmp_path):
        """For POSIX read/write, ret is the byte count, so size == sum(ret)."""
        gz = str(tmp_path / "io.pfw.gz")
        with gzip.open(gz, "wt") as f:
            for i in range(100):
                f.write(
                    '{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,'
                    '"ts":%d,"dur":%d,"args":{"ret":%d}}\n' % (1000 + i, 10 + i, 10 * (i + 1))
                )
        with dftu_utils.Indexer(files=[gz], index_dir=str(tmp_path)) as ix:
            ix.ensure_indexed()
        tbl = pa.table(
            dftu_utils.TraceViewer(gz, index_path=str(tmp_path))
            .group_by("cat")
            .agg("sum:size")
            .collect()
        )
        assert tbl.column("sum_size").to_pylist()[0] == 10 * (100 * 101 // 2)

    def test_auto_numeric_args_surfaces_size(self, tmp_path):
        gz = str(tmp_path / "sz.pfw.gz")
        with gzip.open(gz, "wt") as f:
            for i in range(100):
                f.write(
                    '{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,'
                    '"ts":%d,"dur":%d,"args":{"size":%d}}\n' % (1000 + i, 10 + i, 100 * (i + 1))
                )
        with dftu_utils.Indexer(files=[gz], index_dir=str(tmp_path)) as ix:
            ix.ensure_indexed()
        tbl = pa.table(
            dftu_utils.TraceViewer(gz, index_path=str(tmp_path))
            .group_by("cat")
            .agg("count")
            .agg_numeric_args()
            .collect()
        )
        assert "size" in tbl.column_names

    def test_enums_compose_with_builder(self):
        from dftracer.utils import AggOp, GroupKey, Phase

        assert Phase.EVENTS == "events"
        assert GroupKey.CAT == "cat"
        assert GroupKey.arg("epoch") == "arg:epoch"
        assert AggOp.STD.of("dur") == "std:dur"
        assert AggOp.ARGMAX.of("name", by="dur") == "argmax:name:dur"
        assert AggOp.COUNT.of("") == "count"
        with Environment(lines=100) as env:
            gz = _indexed(env)
            tbl = pa.table(
                dftu_utils.TraceViewer(gz)
                .phase(Phase.EVENTS)
                .group_by(GroupKey.CAT)
                .agg(AggOp.COUNT.of(""), AggOp.MEAN.of("dur"))
                .collect()
            )
            assert {"cat", "count", "mean_dur"}.issubset(tbl.column_names)

    def test_time_unit_normalizes_from_source(self, tmp_path):
        """A trace declared in SEC normalizes ts/dur to the target unit."""
        from dftracer.utils import TimeUnit

        gz = str(tmp_path / "sec.pfw.gz")
        with gzip.open(gz, "wt") as f:
            f.write(
                '{"ph":"M","name":"CM","pid":0,"tid":0,'
                '"args":{"name":"time_metric","value":"SEC"}}\n'
            )
            for i in range(50):
                f.write(
                    '{"ph":"X","name":"read","cat":"POSIX","pid":1,"tid":1,'
                    '"ts":%d,"dur":2,"args":{}}\n' % (100 + i)
                )
        with dftu_utils.Indexer(files=[gz], index_dir=str(tmp_path)) as ix:
            ix.ensure_indexed()

        tv = dftu_utils.TraceViewer(gz, index_path=str(tmp_path))
        native = pa.table(tv.group_by("cat").agg("mean:dur").collect())
        assert native.column("mean_dur").to_pylist()[0] == 2.0  # seconds

        us = pa.table(tv.time_unit(TimeUnit.US).group_by("cat").agg("mean:dur").collect())
        assert us.column("mean_dur").to_pylist()[0] == 2_000_000.0  # 2 s in us

        strm = pa.Table.from_batches(
            [pa.record_batch(c) for c in tv.time_unit("us").select("ts", "dur").stream()]
        )
        durs = [x for x in strm.column("dur").to_pylist() if x is not None]
        assert set(durs) == {2_000_000} and len(durs) == 50

    def test_builder_ops_are_immutable(self):
        with Environment(lines=50) as env:
            gz = _indexed(env)
            base = TraceViewer(gz)
            narrowed = base.filter('cat == "STDIO"')
            # base is unchanged by the derived view's filter.
            assert base.statistics()["duration_count"] == 50
            assert narrowed.statistics()["duration_count"] < 50
