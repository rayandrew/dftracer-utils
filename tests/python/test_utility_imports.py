"""Tests for utility import paths."""


class TestUtilityImports:
    def test_import_from_utilities_subpackage(self):
        from dftracer.utils.utilities import (
            AggregatorUtility,
            ComparatorUtility,
            MetadataCollectorUtility,
        )

        for cls in [
            AggregatorUtility,
            ComparatorUtility,
            MetadataCollectorUtility,
        ]:
            assert cls is not None

    def test_import_from_ext_directly(self):
        from dftracer.utils.dftracer_utils_ext import (
            AggregatorUtility,
        )

        assert AggregatorUtility is not None

    def test_import_query_field(self):
        from dftracer.utils.query import Expr, Field

        cat = Field("cat")
        q = cat == "POSIX"
        assert isinstance(q, Expr)
        assert 'cat == "POSIX"' in str(q)
