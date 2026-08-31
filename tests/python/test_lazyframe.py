#!/usr/bin/env python3
"""Tests for the LazyFrame deferred-query wrapper.

LazyFrame records ops and runs them on collect(), materializing a DataFrame.
The engine is the same SIMD columnar engine as the eager path, so a lazy
pipeline must match the eager result column-for-column. filter/with_column take
a columnar Expr, resolved by position against the frame's schema at that op.
"""

import pytest

pa = pytest.importorskip("pyarrow")

from dftracer.utils import DataFrame, LazyFrame, col, lazy  # noqa: E402
from dftracer.utils import dftracer_utils_ext as _ext  # noqa: E402

_HAS_ARROW = hasattr(_ext, "_LazyFrame")
pytestmark = pytest.mark.skipif(not _HAS_ARROW, reason="extension built without Arrow support")


def _df(mapping):
    return DataFrame.from_arrow(pa.table(mapping))


def _dict(df):
    return df.to_arrow().to_pydict()


def test_lazy_entry_points():
    df = _df({"a": [1, 2, 3]})
    assert isinstance(df.lazy(), LazyFrame)
    assert isinstance(lazy(df), LazyFrame)


def test_collect_roundtrip_is_identity():
    df = _df({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
    out = df.lazy().collect()
    assert _dict(out) == {"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]}


def test_filter_matches_eager():
    df = _df({"a": [1, 2, 3, 4], "b": [10, 20, 30, 40]})
    out = df.lazy().filter(col("a") > 2).collect()
    assert _dict(out) == {"a": [3, 4], "b": [30, 40]}


def test_with_column_and_select():
    df = _df({"a": [1, 2, 3], "b": [10, 20, 30]})
    out = df.lazy().with_column("c", col("a") + col("b")).select("a", "c").collect()
    assert _dict(out) == {"a": [1, 2, 3], "c": [11, 22, 33]}


def test_chained_pipeline_matches_eager():
    df = _df({"a": [5, 1, 4, 2, 3], "b": [1, 2, 3, 4, 5]})
    out = df.lazy().filter(col("a") >= 2).with_column("d", col("a") * 2).sort_by("a").collect()
    assert _dict(out) == {"a": [2, 3, 4, 5], "b": [4, 5, 3, 1], "d": [4, 6, 8, 10]}


def test_group_by_matches_eager():
    df = _df({"k": [1, 1, 2, 2, 2], "v": [10, 20, 30, 40, 50]})
    out = df.lazy().group_by("k", "count", "sum:v").sort_by("k").collect()
    d = _dict(out)
    assert d["k"] == [1, 2]
    assert d["count"] == [2, 3]
    assert d["sum_v"] == [30, 120]


def test_head_tail_slice():
    df = _df({"a": list(range(10))})
    assert _dict(df.lazy().head(3).collect()) == {"a": [0, 1, 2]}
    assert _dict(df.lazy().tail(2).collect()) == {"a": [8, 9]}
    assert _dict(df.lazy().slice(4, 3).collect()) == {"a": [4, 5, 6]}


def test_unique():
    df = _df({"a": [1, 1, 2, 3, 3, 3]})
    assert _dict(df.lazy().unique().collect()) == {"a": [1, 2, 3]}


def test_schema_and_explain():
    lf = _df({"a": [1], "b": [2]}).lazy().with_column("c", col("a") + col("b"))
    assert lf.schema() == ["a", "b", "c"]
    assert isinstance(lf.explain(), str)
    assert lf.explain()


def test_filter_after_data_dependent_schema_raises():
    lf = _df({"a": [1, 2, 2]}).lazy().to_dummies("a")
    assert lf.schema() == []  # data-dependent until collected
    with pytest.raises(ValueError):
        lf.filter(col("a") > 0)


def test_memory_budget_is_chainable():
    df = _df({"a": [3, 1, 2]})
    out = df.lazy().memory_budget(1 << 20).sort_by("a").collect()
    assert _dict(out) == {"a": [1, 2, 3]}
