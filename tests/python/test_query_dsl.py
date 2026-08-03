"""Query DSL (Field/Expr) serialization and TraceViewer.filter(Expr) wiring."""

import gzip
import os

import pytest

from dftracer.utils import Field, resolved
from dftracer.utils.query import Expr


def test_comparisons():
    assert str(Field("cat") == "POSIX") == 'cat == "POSIX"'
    assert str(Field("dur") > 1000) == "dur > 1000"
    assert str(Field("dur") <= 5) == "dur <= 5"
    assert str(Field("ok") == True) == "ok == true"  # noqa: E712


def test_boolean_composition():
    q = (Field("cat") == "POSIX") & (Field("dur") > 1000)
    assert str(q) == '(cat == "POSIX" and dur > 1000)'
    q = (Field("cat") == "POSIX") | (Field("cat") == "STDIO")
    assert str(q) == '(cat == "POSIX" or cat == "STDIO")'
    assert str(~(Field("cat") == "POSIX")) == 'not (cat == "POSIX")'


def test_in_and_not_in():
    assert str(Field("cat").is_in(["POSIX", "STDIO"])) == 'cat in ["POSIX", "STDIO"]'
    assert str(Field("cat").not_in(["MPI"])) == 'cat not in ["MPI"]'


def test_string_match_ops():
    assert str(Field("name").like("%read%")) == 'name like "%read%"'
    assert str(Field("name").ilike("READ")) == 'name ilike "READ"'
    assert str(Field("name").regex("^p?read$")) == 'name ~ "^p?read$"'
    assert str(Field("name").iregex("READ")) == 'name ~* "READ"'
    # Substring serializes literal-first as `"sub" in field`.
    assert str(Field("args.file").contains("tmp")) == '"tmp" in args.file'


def test_resolved_fields():
    assert str(resolved("hostname") == "node01") == 'resolved.hostname == "node01"'
    assert str(resolved("fpath").like("%/scratch/%")) == ('resolved.fpath like "%/scratch/%"')
    with pytest.raises(ValueError):
        resolved("not_a_field")


def test_nested_field_path():
    assert str(Field("args.level") == "DEBUG") == 'args.level == "DEBUG"'


def _write_trace(path):
    with gzip.open(path, "wt") as f:
        for i in range(20):
            name = ["read", "write"][i % 2]
            cat = ["POSIX", "STDIO"][i % 2]
            f.write(
                '{"ph":"X","name":"%s","cat":"%s","pid":1,"tid":1,'
                '"ts":%d,"dur":%d,"args":{}}\n' % (name, cat, 1000 + i, 5 + i)
            )


def test_traceviewer_filter_accepts_expr(tmp_path):
    """filter()/query() accept an Expr, producing the same result as its str()."""
    pa = pytest.importorskip("pyarrow")
    from dftracer.utils import AggregationConfig, Indexer, TraceViewer

    trace = os.path.join(tmp_path, "t.pfw.gz")
    _write_trace(trace)
    idx = os.path.join(tmp_path, "idx")
    with Indexer(
        directory=str(tmp_path),
        index_dir=idx,
        require_aggregation=AggregationConfig(time_interval_ms=100000),
    ) as ix:
        ix.ensure_indexed()

    expr = Field("cat") == "POSIX"
    assert isinstance(expr, Expr)

    def count(pred):
        tbl = (
            TraceViewer(str(tmp_path), index_path=idx)
            .phase("events")
            .filter(pred)
            .group_by("cat")
            .agg("count")
            .collect()
        )
        df = pa.table(tbl).to_pandas()
        return int(df["count"].sum())

    # Passing the Expr must match passing its serialized string.
    assert count(expr) == count(str(expr)) == 10
