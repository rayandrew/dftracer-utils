#!/usr/bin/env python3
"""Tests for the native ``Series`` ops exposed on the Python wrapper.

The C extension carries the full SIMD kernel surface on ``_Series``; the Python
``Series`` wrapper now exposes each one explicitly (so it is typed and shows up
in the reference) rather than relying on attribute forwarding. These tests prove
the wrappers dispatch correctly and return the right Python type: reductions
return scalars, elementwise/ordering ops return a wrapped ``Series``, and
``value_counts`` returns a ``DataFrame``.
"""

import numpy as np
import pytest

from dftracer.utils import DataFrame, Series


def _s(values):
    return Series.from_numpy(np.asarray(values, dtype=np.float64))


def test_reductions_return_scalars():
    s = _s([3.0, 1.0, 2.0, 5.0, 4.0])
    assert s.sum() == pytest.approx(15.0)
    assert s.mean() == pytest.approx(3.0)
    assert s.min() == pytest.approx(1.0)
    assert s.max() == pytest.approx(5.0)
    assert s.median() == pytest.approx(3.0)
    assert s.quantile(0.5) == pytest.approx(3.0)
    assert s.nunique() == 5
    assert s.count() == 5


def test_moment_statistics():
    s = _s([1.0, 2.0, 3.0, 4.0, 5.0])
    assert s.variance() == pytest.approx(2.5)
    assert s.stddev() == pytest.approx(2.5**0.5)
    # skewness of a symmetric sample is ~0.
    assert abs(s.skewness()) < 1e-9


def test_elementwise_return_series():
    s = _s([-2.0, 3.0, -4.0])
    out = s.abs()
    assert isinstance(out, Series)
    assert out.to_numpy().tolist() == [2.0, 3.0, 4.0]
    assert isinstance(s.round(), Series)
    assert s.clip(-1.0, 2.0).to_numpy().tolist() == [-1.0, 2.0, -1.0]


def test_cumulative_and_windowed():
    s = _s([1.0, 2.0, 3.0, 4.0])
    assert isinstance(s.cumsum(), Series)
    assert s.cumsum().to_numpy().tolist() == [1.0, 3.0, 6.0, 10.0]
    assert isinstance(s.rolling(2, "sum"), Series)


def test_ordering_ops_return_series():
    s = _s([3.0, 1.0, 2.0])
    assert isinstance(s.sort(), Series)
    assert s.sort().to_numpy().tolist() == [1.0, 2.0, 3.0]
    assert isinstance(s.reverse(), Series)
    assert isinstance(s.head(2), Series)
    assert len(s.head(2)) == 2
    assert isinstance(s.top_k(2), Series)


def test_boolean_masks():
    s = _s([1.0, 2.0, 3.0, 4.0])
    assert isinstance(s.is_between(2.0, 3.0), Series)
    assert s.is_between(2.0, 3.0).to_numpy().tolist() == [False, True, True, False]
    assert isinstance(s.is_nan(), Series)  # mask op
    assert s.is_null(0) is False  # element-level null check


def test_properties():
    s = _s([1.0, 2.0, 3.0])
    assert s.length == 3
    assert s.null_count == 0
    assert s.type is not None
    assert s.encoding is not None


def test_value_counts_returns_dataframe():
    s = Series.from_numpy(np.array([1, 1, 2, 3, 3, 3], dtype=np.int64))
    vc = s.value_counts()
    assert isinstance(vc, DataFrame)
    assert vc.num_rows == 3


def test_string_ops():
    pa = pytest.importorskip("pyarrow")
    s = Series.from_arrow(pa.array(["abc", "bcd", "xyz"]))
    mask = s.str_contains("bc")
    assert isinstance(mask, Series)
    assert mask.to_numpy().tolist() == [True, True, False]
    assert isinstance(s.to_uppercase(), Series)
    assert s.to_uppercase().to_pandas().tolist() == ["ABC", "BCD", "XYZ"]


def test_astype_matches_cast_by_name_enum_and_int():
    from dftracer.utils import DType

    s = Series.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    by_name = s.astype("int64")
    by_enum = s.cast(DType.INT64)
    by_int = s.cast(int(DType.INT64))
    assert by_name.type == by_enum.type == by_int.type == int(DType.INT64)
    assert by_name.to_numpy().tolist() == [1, 2, 3]

    # Case-insensitive name lookup, and an unknown name raises.
    assert s.astype("Float64").type == int(DType.FLOAT64)
    with pytest.raises(ValueError):
        s.astype("not_a_dtype")


def test_uint64_float64_arithmetic_promotes():
    from dftracer.utils import DType

    u = Series.from_numpy(np.array([1, 2, 3], dtype=np.uint64))
    f = Series.from_numpy(np.array([0.5, 1.5, 2.5], dtype=np.float64))

    summed = u.add(f)
    assert summed.type == int(DType.FLOAT64)
    assert summed.to_numpy().tolist() == pytest.approx([1.5, 3.5, 5.5])

    i = Series.from_numpy(np.array([1, 2, 3], dtype=np.int64))
    prod = i.mul_scalar(1.0)
    assert prod.type == int(DType.FLOAT64)
    assert prod.to_numpy().tolist() == pytest.approx([1.0, 2.0, 3.0])

    u_prod = u.mul_scalar(1.0)
    assert u_prod.type == int(DType.FLOAT64)
    assert u_prod.to_numpy().tolist() == pytest.approx([1.0, 2.0, 3.0])
