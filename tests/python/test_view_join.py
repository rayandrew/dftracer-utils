#!/usr/bin/env python3
"""Tests for TraceViewer.join (equi-join of two aggregated views)."""

import pyarrow as pa
import pytest

import dftracer.utils as dftu_utils
from dftracer.utils import TraceViewer

from .common import Environment


def _indexed(env):
    gz = env.create_test_gzip_file()
    with dftu_utils.Indexer(files=[gz]) as indexer:
        indexer.ensure_indexed()
    return gz


def _by_cat(tbl):
    return {row["cat"]: row for row in tbl.to_pylist()}


class TestViewJoin:
    def test_inner_join_carries_both_sides(self):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            left = TraceViewer(gz).group_by("cat").agg("count")
            right = TraceViewer(gz).filter('cat == "POSIX"').group_by("cat").agg("sum:dur")
            tbl = pa.table(left.join(right, how="inner"))
            assert {"cat", "l_count", "r_sum_dur"}.issubset(tbl.column_names)
            rows = _by_cat(tbl)
            # Inner keeps only the shared key; the right side filtered to posix.
            assert set(rows) == {"posix"}
            assert rows["posix"]["l_count"] > 0
            assert rows["posix"]["r_sum_dur"] > 0

    def test_left_join_null_pads_the_right(self):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            left = TraceViewer(gz).group_by("cat").agg("count")
            right = TraceViewer(gz).filter('cat == "POSIX"').group_by("cat").agg("sum:dur")
            tbl = pa.table(left.join(right, how="left"))
            rows = _by_cat(tbl)
            assert {"posix", "stdio"} <= set(rows)
            assert rows["posix"]["r_sum_dur"] is not None  # matched
            assert rows["stdio"]["l_count"] > 0
            assert rows["stdio"]["r_sum_dur"] is None  # left-only row nulls right

    @pytest.mark.parametrize("how", ["semi", "anti"])
    def test_semi_anti_return_left_columns_only(self, how):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            left = TraceViewer(gz).group_by("cat").agg("count")
            right = TraceViewer(gz).filter('cat == "POSIX"').group_by("cat").agg("sum:dur")
            tbl = pa.table(left.join(right, how=how))
            assert {"cat", "l_count"}.issubset(tbl.column_names)
            assert not any(c.startswith("r_") for c in tbl.column_names)
            cats = set(tbl.column("cat").to_pylist())
            assert cats == ({"posix"} if how == "semi" else {"stdio"})

    def test_group_key_mismatch_raises(self):
        with Environment(lines=200) as env:
            gz = _indexed(env)
            left = TraceViewer(gz).group_by("cat").agg("count")
            right = TraceViewer(gz).group_by("pid").agg("count")
            with pytest.raises(ValueError):
                left.join(right, how="inner")

    def test_bad_how_and_bad_other_raise(self):
        with Environment(lines=100) as env:
            gz = _indexed(env)
            left = TraceViewer(gz).group_by("cat").agg("count")
            right = TraceViewer(gz).group_by("cat").agg("count")
            with pytest.raises(ValueError):
                left.join(right, how="outer")
            with pytest.raises(TypeError):
                left.join("not-a-viewer")
