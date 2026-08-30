"""@jit.series: author a reusable column op as an Expr; it joins the ops
registry and runs via the engine's evaluator (no compile step)."""

import numpy as np
import pytest

from dftracer.utils import jit
from dftracer.utils.jit import ops
from dftracer.utils.series import Series


def _col(vals):
    return Series.from_numpy(np.array(vals, dtype=np.int64))


def _vals(series):
    return np.asarray(series).tolist()


def test_series_op_is_callable_and_registered():
    @jit.series
    def doubled(dur):
        return dur * 2

    s = _col([1, 2, 3])
    assert _vals(doubled(s)) == [2, 4, 6]  # the decorated name is callable
    assert _vals(ops.doubled(s)) == [2, 4, 6]  # and reachable via ops.<name>
    assert _vals(ops.run("doubled", s)) == [2, 4, 6]  # and by name
    assert "doubled" in ops.list()
    info = ops.info("doubled")
    assert info["kind"] == "series" and info["arity"] == 1


def test_series_op_two_args_fuses_expr():
    @jit.series
    def weighted(a, b):
        return a * 2 + b

    assert _vals(ops.run("weighted", _col([1, 2, 3]), _col([10, 20, 30]))) == [12, 24, 36]


def test_series_op_arity_mismatch_raises():
    @jit.series
    def triple(a):
        return a * 3

    with pytest.raises(TypeError):
        ops.run("triple", _col([1]), _col([2]))


def test_series_op_rejects_non_expr_body():
    with pytest.raises(jit.JitError):

        @jit.series
        def bad(a):
            return 5  # not a column expression


def test_series_op_rejects_builtin_name_clash():
    with pytest.raises(ValueError):

        @jit.series
        def add(a, b):  # "add" is a built-in op
            return a + b
