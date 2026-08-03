"""Typed vocabularies for the TraceViewer builder API.

These are ``str`` enums, so they pass straight into the string-accepting
builder ops (``phase``/``group_by``/``agg``) - ``phase(Phase.EVENTS)`` is the
same call as ``phase("events")`` - while giving callers a discoverable, typo-safe
set instead of bare strings.
"""

from enum import Enum
from typing import Optional


class Phase(str, Enum):
    """Which records a view scans."""

    EVENTS = "events"  # ph="X"
    COUNTERS = "counters"  # ph="C"
    ANY = "any"


class GroupKey(str, Enum):
    """A group_by dimension."""

    NAME = "name"
    CAT = "cat"
    PID = "pid"
    TID = "tid"
    FHASH = "fhash"

    @staticmethod
    def arg(key: str) -> str:
        """Group by an args-map entry: ``GroupKey.arg("epoch")`` -> ``"arg:epoch"``."""
        return f"arg:{key}"


class AggOp(str, Enum):
    """An aggregation reducer (the ``op`` in an ``"op:field"`` agg spec)."""

    COUNT = "count"
    SUM = "sum"
    SUMSQ = "sumsq"
    MIN = "min"
    MAX = "max"
    MEAN = "mean"
    VAR = "var"
    STD = "std"
    SKEW = "skew"
    KURT = "kurt"
    HIST = "hist"  # raw histogram: an arrow list<struct<lo, hi, count>> column
    ARGMAX = "argmax"

    def of(self, field: str, by: Optional[str] = None) -> str:
        """Build an agg spec: ``AggOp.STD.of("dur")`` -> ``"std:dur"``;
        ``AggOp.ARGMAX.of("name", by="dur")`` -> ``"argmax:name:dur"``."""
        if self is AggOp.COUNT:
            return "count" if not field else f"count:{field}"
        spec = f"{self.value}:{field}"
        return f"{spec}:{by}" if by else spec
