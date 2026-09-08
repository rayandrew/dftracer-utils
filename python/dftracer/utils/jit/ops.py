"""The dataframe op registry, as callables.

Every engine column op and reducer lives in one name-keyed registry, alongside
any user ops authored with :func:`dftracer.utils.jit.series`. Look one up and
call it, discover what is there, or run one dynamically by name::

    from dftracer.utils.jit import ops

    ops.add(a, b)            # a registered op as an attribute
    ops.run("add", a, b)     # by name (name-as-data)
    ops.get("mul")(a, b)     # bound callable
    ops.list()               # every op name (built-in + user)
    ops.info("compare")      # {name, kind, arity, signature}

Column arguments are Series; any other operands (a scalar, an op-enum int, a
string, an int, a char) follow the op's signature. The call returns a Series for
a column op, or a Python scalar for a reducer.
"""

from typing import TYPE_CHECKING, Callable, Dict, List, Union

from .. import dftracer_utils_ext as _ext
from ..series import Series, _unwrap, _wrap

if TYPE_CHECKING:
    from ..columnar import Expr

__all__ = ["run", "get", "list", "names", "info"]

# A column argument is a Series; any other operand follows the op's signature.
Operand = Union[Series, int, float, str]
# A column op returns a Series; a reducer returns a Python scalar.
Result = Union[Series, int, float]


class _UserOp:
    """A user op authored with @jit.series: an Expr over N placeholder columns
    (__x0__ .. __xN-1__), applied to the call's Series arguments."""

    __slots__ = ("name", "n_args", "expr")

    def __init__(self, name: str, n_args: int, expr: "Expr") -> None:
        self.name = name
        self.n_args = n_args
        self.expr = expr


_USER: Dict[str, _UserOp] = {}


def _register_user(name: str, n_args: int, expr: "Expr") -> None:
    """Register a user op; rejects a name already used by a built-in or user op
    (no silent shadowing)."""
    if name in _USER or name in _ext.op_list():
        raise ValueError(f"op '{name}' is already registered")
    _USER[name] = _UserOp(name, n_args, expr)


def run(name: str, *args: Operand) -> Result:
    """Run the op `name` on `args` (Series unwrapped in, result wrapped out).
    Resolves a user op (@jit.series) before the built-in registry."""
    uop = _USER.get(name)
    if uop is not None:
        if len(args) != uop.n_args:
            raise TypeError(f"op '{name}' takes {uop.n_args} argument(s), got {len(args)}")
        source = {f"__x{i}__": a for i, a in enumerate(args)}
        return uop.expr.apply(source)
    return _wrap(_ext.op_run(name, *(_unwrap(a) for a in args)))


def get(name: str) -> Callable[..., Result]:
    """The op `name` as a bound callable, so `ops.get("mul")(a, b)` runs it."""

    def call(*args: Operand) -> Result:
        return run(name, *args)

    call.__name__ = name
    call.__qualname__ = f"ops.{name}"
    return call


def names() -> List[str]:
    """Every registered op name (built-in and user)."""
    return [*_ext.op_list(), *_USER]


# ops.list() reads as the natural discovery call; it shadows the builtin only as
# a module attribute, never inside this module.
list = names  # noqa: A001


def info(name: str) -> Dict[str, object]:
    """{name, kind, arity, signature} for a registered op."""
    uop = _USER.get(name)
    if uop is not None:
        return {
            "name": name,
            "kind": "series",
            "arity": uop.n_args,
            "signature": "user (expr)",
        }
    return _ext.op_info(name)


def _is_module(name: str) -> bool:
    """True if `name` is a module prefix, i.e. some user op is `name.<op>`."""
    prefix = name + "."
    return any(k.startswith(prefix) for k in _USER)


class _ModuleNs:
    """A module namespace: ``ops.stats.zscore(a, b)`` runs the user op
    ``stats.zscore``."""

    __slots__ = ("_prefix",)

    def __init__(self, prefix: str) -> None:
        self._prefix = prefix

    def __getattr__(self, name: str) -> Union[Callable[..., Result], "_ModuleNs"]:
        full = f"{self._prefix}.{name}"
        # A dotted module (`pkg.stats.zscore`) nests one namespace per segment.
        if full not in _USER and _is_module(full):
            return _ModuleNs(full)
        return get(full)


def __getattr__(name: str):
    # ops.add / a user op -> a bound callable; ops.<module> -> a namespace.
    if name in _USER or name in _ext.op_list():
        return get(name)
    if _is_module(name):
        return _ModuleNs(name)
    raise AttributeError(f"module {__name__!r} has no op or attribute {name!r}")
