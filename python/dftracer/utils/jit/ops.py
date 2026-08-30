"""The dataframe op registry, as callables.

Every engine column op and reducer lives in one name-keyed registry. Look one up
and call it, discover what is there, or run one dynamically by name::

    from dftracer.utils import ops

    ops.add(a, b)            # a registered op as an attribute
    ops.run("add", a, b)     # by name (name-as-data)
    ops.get("mul")(a, b)     # bound callable
    ops.list()               # every op name
    ops.info("compare")      # {name, kind, arity, signature}

Column arguments are Series; any other operands (a scalar, an op-enum int, a
string, an int, a char) follow the op's signature. The call returns a Series for
a column op, or a Python scalar for a reducer.
"""

from typing import Callable, Dict, List, Union

from . import dftracer_utils_ext as _ext
from .series import Series, _unwrap, _wrap

__all__ = ["run", "get", "list", "names", "info"]

# A column argument is a Series; any other operand follows the op's signature.
Operand = Union[Series, int, float, str]
# A column op returns a Series; a reducer returns a Python scalar.
Result = Union[Series, int, float]


def run(name: str, *args: Operand) -> Result:
    """Run the registered op `name` on `args` (Series unwrapped in, result
    wrapped out)."""
    return _wrap(_ext.op_run(name, *(_unwrap(a) for a in args)))


def get(name: str) -> Callable[..., Result]:
    """The op `name` as a bound callable, so `ops.get("mul")(a, b)` runs it."""

    def call(*args: Operand) -> Result:
        return run(name, *args)

    call.__name__ = name
    call.__qualname__ = f"ops.{name}"
    return call


def names() -> List[str]:
    """Every registered op name."""
    return _ext.op_list()


# ops.list() reads as the natural discovery call; it shadows the builtin only as
# a module attribute, never inside this module.
list = names  # noqa: A001


def info(name: str) -> Dict[str, object]:
    """{name, kind, arity, signature} for a registered op."""
    return _ext.op_info(name)


def __getattr__(name: str) -> Callable[..., Result]:
    # ops.add / ops.str_contains / ... resolve to a bound callable.
    if name in _ext.op_list():
        return get(name)
    raise AttributeError(f"module {__name__!r} has no op or attribute {name!r}")
