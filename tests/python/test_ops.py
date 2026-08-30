"""The op registry exposed to Python (dftracer.utils.jit.ops): discovery and one
generic runner that marshals Python args per an op's signature."""

import numpy as np
import pytest

from dftracer.utils.jit import ops
from dftracer.utils.series import Series


def _col(vals):
    return Series.from_numpy(np.array(vals, dtype=np.int64))


def _vals(series):
    return np.asarray(series).tolist()


class TestDiscovery:
    def test_list_covers_the_surface(self):
        names = ops.list()
        for n in (
            "add",
            "sub",
            "mul",
            "div",
            "add_scalar",
            "compare",
            "cast",
            "count",
            "reduce",
            "str_contains",
            "mode",
        ):
            assert n in names

    def test_info(self):
        info = ops.info("add")
        assert info["kind"] == "series"
        assert info["arity"] == 2
        assert info["signature"] == "(series, series) -> series"

        red = ops.info("count")
        assert red["kind"] == "aggregate"

    def test_unknown_op(self):
        with pytest.raises(KeyError):
            ops.info("no_such_op")
        with pytest.raises(AttributeError):
            _ = ops.definitely_not_an_op


class TestRun:
    def test_binary_as_attribute(self):
        out = ops.add(_col([1, 2, 3]), _col([10, 20, 30]))
        assert isinstance(out, Series)
        assert _vals(out) == [11, 22, 33]

    def test_run_by_name(self):
        assert _vals(ops.run("mul", _col([1, 2, 3]), _col([4, 5, 6]))) == [4, 10, 18]

    def test_get_bound_callable(self):
        sub = ops.get("sub")
        assert _vals(sub(_col([5, 7]), _col([1, 2]))) == [4, 5]

    def test_scalar_operand(self):
        assert _vals(ops.add_scalar(_col([1, 2, 3]), 100)) == [101, 102, 103]

    def test_reducer_returns_scalar(self):
        col = _col([1, 5, 2])
        assert ops.count(col) == 3
        assert ops.arg_max(col) == 1

    def test_missing_argument_raises(self):
        with pytest.raises(TypeError):
            ops.add(_col([1, 2]))  # add needs two columns
