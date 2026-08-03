"""Query DSL for filtering DFTracer trace events.

Build query expressions using Field objects with Python operators:

    from dftracer.utils.query import Field

    cat = Field("cat")
    dur = Field("dur")
    name = Field("name")

    # Simple comparison
    q = cat == "POSIX"

    # Combined with AND/OR
    q = (cat == "POSIX") & (dur > 1000)
    q = (cat == "POSIX") | (cat == "STDIO")

    # NOT
    q = ~(cat == "POSIX")

    # IN / NOT IN
    q = cat.is_in(["POSIX", "STDIO"])
    q = cat.not_in(["MPI"])

    # Nested field paths
    level = Field("args.level")
    q = level == "DEBUG"

    # String matching: SQL LIKE (% wildcard), case-insensitive, regex, substring
    q = Field("name").like("%read%")
    q = Field("name").ilike("READ")
    q = Field("name").regex("^p?read$")
    q = Field("args.file").contains("tmp")  # substring: "tmp" in args.file

    # Resolved virtual fields (resolved.* / r.*): the engine rewrites these to
    # concrete hash lookups against the index.
    q = resolved("hostname") == "node01"
    q = resolved("fpath").like("%/scratch/%")

    # Convert to query string for the C++ parser
    query_string = str(q)
"""

from __future__ import annotations

from typing import Sequence, Union

__all__ = ["Expr", "Field", "resolved", "Value"]

Value = Union[str, int, float, bool]

# Resolved virtual-field leaf names accepted after the resolved./r. prefix.
_RESOLVED_FIELDS = frozenset({"fpath", "cwd", "hostname", "host", "exec", "cmd"})


class Expr:
    """Base class for query expressions."""

    def __and__(self, other: Expr) -> Expr:
        return _BinaryExpr("and", self, other)

    def __or__(self, other: Expr) -> Expr:
        return _BinaryExpr("or", self, other)

    def __invert__(self) -> Expr:
        return _NotExpr(self)

    def __str__(self) -> str:
        raise NotImplementedError


class _CompareExpr(Expr):
    def __init__(self, field: str, op: str, value: Value) -> None:
        self._field = field
        self._op = op
        self._value = value

    def __str__(self) -> str:
        return f"{self._field} {self._op} {_format_value(self._value)}"


class _InExpr(Expr):
    def __init__(self, field: str, values: Sequence[Value]) -> None:
        self._field = field
        self._values = list(values)

    def __str__(self) -> str:
        items = ", ".join(_format_value(v) for v in self._values)
        return f"{self._field} in [{items}]"


class _NotInExpr(Expr):
    def __init__(self, field: str, values: Sequence[Value]) -> None:
        self._field = field
        self._values = list(values)

    def __str__(self) -> str:
        items = ", ".join(_format_value(v) for v in self._values)
        return f"{self._field} not in [{items}]"


class _MatchExpr(Expr):
    """A string-match predicate: `field <op> "pattern"` where op is one of
    like / ilike / ~ (regex) / ~* (case-insensitive regex)."""

    def __init__(self, field: str, op: str, pattern: str) -> None:
        self._field = field
        self._op = op
        self._pattern = pattern

    def __str__(self) -> str:
        return f"{self._field} {self._op} {_format_value(self._pattern)}"


class _ContainsExpr(Expr):
    """A substring predicate, serialized literal-first as `"sub" in field`."""

    def __init__(self, field: str, sub: str) -> None:
        self._field = field
        self._sub = sub

    def __str__(self) -> str:
        return f"{_format_value(self._sub)} in {self._field}"


class _BinaryExpr(Expr):
    def __init__(self, op: str, left: Expr, right: Expr) -> None:
        self._op = op
        self._left = left
        self._right = right

    def __str__(self) -> str:
        return f"({self._left} {self._op} {self._right})"


class _NotExpr(Expr):
    def __init__(self, operand: Expr) -> None:
        self._operand = operand

    def __str__(self) -> str:
        return f"not ({self._operand})"


class Field:
    """A field reference for building query expressions.

    Supports arbitrary field names including dotted paths for nested
    JSON fields (e.g., "args.level", "args.io.size").
    """

    def __init__(self, name: str) -> None:
        self._name = name

    def __eq__(self, other: Value) -> Expr:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return _CompareExpr(self._name, "==", other)

    def __ne__(self, other: Value) -> Expr:  # type: ignore[override]  # ty: ignore[invalid-method-override]
        return _CompareExpr(self._name, "!=", other)

    def __gt__(self, other: Value) -> Expr:
        return _CompareExpr(self._name, ">", other)

    def __lt__(self, other: Value) -> Expr:
        return _CompareExpr(self._name, "<", other)

    def __ge__(self, other: Value) -> Expr:
        return _CompareExpr(self._name, ">=", other)

    def __le__(self, other: Value) -> Expr:
        return _CompareExpr(self._name, "<=", other)

    def is_in(self, values: Sequence[Value]) -> Expr:
        return _InExpr(self._name, values)

    def not_in(self, values: Sequence[Value]) -> Expr:
        return _NotInExpr(self._name, values)

    def like(self, pattern: str) -> Expr:
        """SQL LIKE: `%` matches any run, `_` any single char."""
        return _MatchExpr(self._name, "like", pattern)

    def ilike(self, pattern: str) -> Expr:
        """Case-insensitive LIKE."""
        return _MatchExpr(self._name, "ilike", pattern)

    def regex(self, pattern: str) -> Expr:
        """ECMAScript regex match (anchored as written)."""
        return _MatchExpr(self._name, "~", pattern)

    def iregex(self, pattern: str) -> Expr:
        """Case-insensitive regex match."""
        return _MatchExpr(self._name, "~*", pattern)

    def contains(self, sub: str) -> Expr:
        """Unanchored substring search (`"sub" in field`)."""
        return _ContainsExpr(self._name, sub)


def resolved(name: str) -> Field:
    """A resolved virtual field (`resolved.<name>`), which the engine rewrites
    to a concrete hash lookup against the index. Known names: fpath, cwd,
    hostname (alias host), exec, cmd."""
    if name not in _RESOLVED_FIELDS:
        raise ValueError(
            f"unknown resolved field {name!r}; expected one of {sorted(_RESOLVED_FIELDS)}"
        )
    return Field("resolved." + name)


def _format_value(v: Value) -> str:
    if isinstance(v, str):
        escaped = v.replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)
