#!/usr/bin/env python3
"""Tests for the native analytical / relational DataFrame methods.

These primitives (``window``, ``gap_fill``, ``join``, ``asof``, ``interval``,
``unnest``, ``melt``, ``concat``, ``union``, ``distinct``, ``sample``,
``top_k``, ``sort``) run the SIMD kernels on the native columns and take/return
``DataFrame``s. Arrow is crossed only at the edge (``from_arrow``/``to_arrow``),
never inside a method - one test proves a method runs with pyarrow importing
blocked.
"""

import builtins
import sys

import pytest

pa = pytest.importorskip("pyarrow")

from dftracer.utils import DataFrame  # noqa: E402
from dftracer.utils import dftracer_utils_ext as _ext  # noqa: E402

# The analytical kernels only exist when the extension was built with Arrow.
_HAS_ARROW = hasattr(_ext, "window")
pytestmark = pytest.mark.skipif(not _HAS_ARROW, reason="extension built without Arrow support")


def _df(mapping):
    return DataFrame.from_arrow(pa.table(mapping))


def _dict(df):
    return df.to_arrow().to_pydict()


def _col(df, name):
    return df.to_arrow().column(name).to_pylist()


def test_window_row_number_running_sum_lag():
    df = _df({"pid": [1, 1, 1, 2], "ts": [10, 20, 30, 5], "dur": [1, 2, 3, 4]})
    out = df.window(
        partition_by=["pid"],
        order_by=["ts"],
        specs=[
            ("row_number", "rn"),
            ("running_sum", "dur", "cum"),
            ("lag", "dur", 1, "prev"),
        ],
    )
    assert isinstance(out, DataFrame)
    assert out.to_arrow().column_names == ["pid", "ts", "dur", "rn", "cum", "prev"]
    assert _col(out, "rn") == [1, 2, 3, 1]
    assert _col(out, "cum") == [1, 3, 6, 4]
    assert _col(out, "prev") == [None, 1, 2, None]


def test_window_frame_ntile_first_value():
    df = _df({"g": [1, 1, 1, 1], "ts": [1, 2, 3, 4], "x": [10, 20, 30, 40]})
    out = df.window(
        partition_by=["g"],
        order_by=["ts"],
        specs=[
            ("frame_sum", "x", 1, 1, "fs"),
            ("ntile", 2, "nt"),
            ("first_value", "x", "fv"),
        ],
    )
    assert _col(out, "fs") == [30, 60, 90, 70]
    assert _col(out, "nt") == [1, 1, 2, 2]
    assert _col(out, "fv") == [10, 10, 10, 10]


def test_window_rank_dense_rank_sessionize():
    df = _df({"g": [1, 1, 1, 1], "ts": [10, 10, 20, 100]})
    out = df.window(
        partition_by=["g"],
        order_by=["ts"],
        specs=[
            ("rank", "rk"),
            ("dense_rank", "dr"),
            ("sessionize", "ts", 15, "sess"),
        ],
    )
    assert _col(out, "rk") == [1, 1, 3, 4]
    assert _col(out, "dr") == [1, 1, 2, 3]
    assert _col(out, "sess") == [1, 1, 1, 2]


def test_gap_fill_modes():
    df = _df({"pid": [1, 1, 1], "ts": [0, 10, 30], "v": [100, 110, 130]})

    none = df.gap_fill(["pid"], "ts", 10, "v", "none")
    assert _col(none, "ts") == [0, 10, 20, 30]
    assert _col(none, "v") == [100, 110, None, 130]

    locf = df.gap_fill(["pid"], "ts", 10, "v", "locf")
    assert _col(locf, "v") == [100, 110, 110, 130]

    linear = df.gap_fill(["pid"], "ts", 10, "v", "linear")
    assert _col(linear, "v") == [100.0, 110.0, 120.0, 130.0]


def test_gap_fill_explicit_range():
    df = _df({"pid": [1, 1], "ts": [10, 20], "v": [1, 2]})
    out = df.gap_fill(["pid"], "ts", 10, "v", "locf", start=0, end=30)
    assert _col(out, "ts") == [0, 10, 20, 30]
    assert _col(out, "v") == [None, 1, 2, 2]


def test_join_inner_left_semi_anti():
    left = _df({"k": [1, 2, 3], "lv": [10, 20, 30]})
    right = _df({"k": [2, 3, 4], "rv": [200, 300, 400]})

    inner = left.join(right, "k", "inner")
    assert isinstance(inner, DataFrame)
    assert inner.to_arrow().column_names == ["k", "lv", "rv"]
    assert _dict(inner) == {"k": [2, 3], "lv": [20, 30], "rv": [200, 300]}

    lj = left.join(right, "k", "left")
    assert _dict(lj) == {"k": [1, 2, 3], "lv": [10, 20, 30], "rv": [None, 200, 300]}

    semi = left.join(right, "k", "semi")
    assert semi.to_arrow().column_names == ["k", "lv"]
    assert _dict(semi) == {"k": [2, 3], "lv": [20, 30]}

    anti = left.join(right, "k", "anti")
    assert semi.to_arrow().column_names == ["k", "lv"]
    assert _dict(anti) == {"k": [1], "lv": [10]}


def test_join_multi_key():
    left = _df({"a": [1, 1], "b": [1, 2], "lv": [10, 20]})
    right = _df({"a": [1], "b": [2], "rv": [99]})
    out = left.join(right, ["a", "b"], "inner")
    assert _dict(out) == {"a": [1], "b": [2], "lv": [20], "rv": [99]}


def test_join_int_on_uses_native_leading_keys():
    # An int `on` resolves to the first N shared column names, then joins on them.
    left = _df({"k": [1, 2, 3], "lv": [10, 20, 30]})
    right = _df({"k": [2, 3, 4], "rv": [200, 300, 400]})
    out = left.join(right, on=1, how="inner")
    assert _dict(out) == {"k": [2, 3], "lv": [20, 30], "rv": [200, 300]}


def test_asof_backward():
    left = _df({"ts": [10, 20, 30], "ev": ["a", "b", "c"]})
    right = _df({"ts": [5, 25], "val": [100, 200]})
    out = left.asof(right, "ts", direction="backward")
    assert out.to_arrow().column_names == ["ts", "ev", "val"]
    assert _col(out, "val") == [100, 100, 200]


def test_asof_by_and_tolerance():
    left = _df({"k": [1, 1, 2], "ts": [10, 20, 10], "ev": ["a", "b", "c"]})
    right = _df({"k": [1, 2], "ts": [5, 5], "val": [1, 2]})
    out = left.asof(right, "ts", by="k")
    assert _col(out, "val") == [1, 1, 2]

    tol = left.asof(right, "ts", by="k", tolerance=3)
    assert _col(tol, "val") == [None, None, None]


def test_interval_points_in_spans():
    pts = _df({"p": [5, 15, 25], "name": ["x", "y", "z"]})
    spans = _df({"lo": [0, 10], "hi": [9, 30], "span": ["A", "B"]})
    out = pts.interval(spans, "p", "lo", "hi")
    assert out.to_arrow().column_names == ["p", "name", "span"]
    assert _col(out, "span") == ["A", "B", "B"]


def test_interval_outer_keeps_unmatched():
    pts = _df({"p": [5, 100], "name": ["x", "y"]})
    spans = _df({"lo": [0], "hi": [9], "span": ["A"]})
    inner = pts.interval(spans, "p", "lo", "hi")
    assert _col(inner, "name") == ["x"]
    outer = pts.interval(spans, "p", "lo", "hi", outer=True)
    assert _col(outer, "name") == ["x", "y"]
    assert _col(outer, "span") == ["A", None]


def test_unnest_explodes_list_and_passes_second_list_through():
    df = _df(
        {
            "pid": [1, 2],
            "files": [["a", "b"], ["x"]],
            "tags": [["t1", "t2"], ["t3"]],
        }
    )
    out = df.unnest("files")
    assert out.to_arrow().column_names == ["pid", "files", "tags"]
    assert _col(out, "pid") == [1, 1, 2]
    assert _col(out, "files") == ["a", "b", "x"]
    assert _col(out, "tags") == [["t1", "t2"], ["t1", "t2"], ["t3"]]


def test_unnest_keep_empty():
    df = _df({"pid": [1, 2, 3], "files": [["a"], [], ["b"]]})
    assert _col(df.unnest("files"), "pid") == [1, 3]
    out = df.unnest("files", keep_empty=True)
    assert _col(out, "pid") == [1, 2, 3]
    assert _col(out, "files") == ["a", None, "b"]


def test_melt_concat_union_distinct_sample_topk_sort():
    df = _df({"id": [1, 2], "a": [10, 30], "b": [20, 40]})
    melted = df.melt(["id"], ["a", "b"])
    assert set(melted.to_arrow().column_names) == {"id", "variable", "value"}
    assert melted.num_rows == 4

    a = _df({"k": [1, 2]})
    b = _df({"k": [2, 3]})
    assert sorted(_col(a.concat(b), "k")) == [1, 2, 2, 3]
    assert sorted(_col(a.union(b), "k")) == [1, 2, 3]

    dup = _df({"k": [1, 1, 2]})
    assert sorted(_col(dup.distinct(), "k")) == [1, 2]

    big = _df({"k": list(range(100))})
    assert big.sample(5, seed=1).num_rows == 5

    scores = _df({"name": ["a", "b", "c"], "v": [3, 1, 2]})
    assert _col(scores.top_k("v", 2), "v") == [3, 2]
    assert _col(scores.sort("v"), "v") == [1, 2, 3]
    assert _col(scores.sort("v", descending=True), "v") == [3, 2, 1]


def test_bad_column_name_raises_keyerror():
    df = _df({"pid": [1], "ts": [1], "dur": [1]})
    with pytest.raises(KeyError):
        df.window(["nope"], ["ts"], [("row_number", "rn")])
    with pytest.raises(KeyError):
        df.join(df, "missing")
    with pytest.raises(KeyError):
        df.unnest("nope")


def test_bad_how_mode_direction_raise_valueerror():
    df = _df({"pid": [1], "ts": [1], "dur": [1]})
    with pytest.raises(ValueError):
        df.join(df, "pid", "sideways")
    with pytest.raises(ValueError):
        df.gap_fill(["pid"], "ts", 10, "dur", "bogus")
    with pytest.raises(ValueError):
        df.asof(df, "ts", direction="sideways")


def test_bad_spec_raises_valueerror():
    df = _df({"pid": [1], "ts": [1], "dur": [1]})
    with pytest.raises(ValueError):
        df.window(["pid"], ["ts"], [("lag", "dur", "rn")])  # missing offset
    with pytest.raises(ValueError):
        df.window(["pid"], ["ts"], [("bogus", "rn")])  # unknown function
    with pytest.raises(ValueError):
        df.join(df, [], "inner")  # empty key list
    with pytest.raises(ValueError):
        df.gap_fill(["pid"], "ts", 10, "dur", "none", start=0)  # start w/o end


def test_series_analytical_ops_are_native():
    """Per-column analytical ops are native Series methods: Series in, Series
    out, and they run with pyarrow importing blocked."""
    from dftracer.utils import Series

    df = _df({"v": [30, 10, 20, 10]})
    col = df["v"]
    asc = Series.from_arrow(pa.array([10, 20, 30]))
    needles = Series.from_arrow(pa.array([15, 25]))

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("pyarrow blocked for this test")
        return real_import(name, *args, **kwargs)

    saved = {k: v for k, v in sys.modules.items() if k == "pyarrow" or k.startswith("pyarrow.")}
    for k in saved:
        del sys.modules[k]
    builtins.__import__ = blocked
    try:
        srt = col.sort()
        assert isinstance(srt, Series)
        assert list(srt.to_numpy()) == [10, 10, 20, 30]

        order = col.argsort()
        assert isinstance(order, Series)
        assert list(order.to_numpy()) == [1, 3, 2, 0]

        rk = col.rank(method="dense")
        assert list(rk.to_numpy()) == [3.0, 1.0, 2.0, 1.0]

        roll = col.rolling(window=2, op="sum")
        # rolling leaves row 0 null (window-1), so to_numpy would need pyarrow;
        # verify the pyarrow-free op via reductions instead: values [_, 40, 30, 30].
        assert roll.null_count == 1
        assert roll.sum() == 100.0
        assert roll.max() == 40.0
        assert roll.min() == 30.0

        idx = asc.search_sorted(needles)
        assert list(idx.to_numpy()) == [1, 2]

        assert col.is_sorted() is False
        assert asc.is_sorted() is True
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)


def test_methods_need_no_pyarrow():
    """The method path never imports pyarrow: build frames first, then block
    pyarrow and prove window/join/sort still run on the native columns."""
    left = _df({"pid": [1, 1, 2], "ts": [10, 20, 5], "dur": [1, 2, 4]})
    right = _df({"pid": [1, 2], "rv": [100, 200]})

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "pyarrow" or name.startswith("pyarrow."):
            raise ImportError("pyarrow blocked for this test")
        return real_import(name, *args, **kwargs)

    saved = {k: v for k, v in sys.modules.items() if k == "pyarrow" or k.startswith("pyarrow.")}
    for k in saved:
        del sys.modules[k]
    builtins.__import__ = blocked
    try:
        with pytest.raises(ImportError):
            __import__("pyarrow")

        win = left.window(["pid"], ["ts"], [("row_number", "rn"), ("running_sum", "dur", "cum")])
        assert win.column_names == ["pid", "ts", "dur", "rn", "cum"]
        # to_numpy on a flat non-null numeric column reads the native buffer
        # directly, no pyarrow.
        assert list(win["rn"].to_numpy()) == [1, 2, 1]
        assert list(win["cum"].to_numpy()) == [1, 3, 4]

        joined = left.join(right, "pid", "inner")
        assert joined.num_rows == 3

        srt = left.sort("ts")
        assert list(srt["ts"].to_numpy()) == [5, 10, 20]

        # a per-column primitive is equally pyarrow-free.
        assert left["dur"].is_sorted() is True  # dur == [1, 2, 4]
        assert left["ts"].is_sorted() is False  # ts == [10, 20, 5]
    finally:
        builtins.__import__ = real_import
        sys.modules.update(saved)
