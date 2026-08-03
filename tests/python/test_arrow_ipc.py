"""Tests for Arrow IPC file output and readback via pyarrow."""

import os
import shutil
import subprocess
import tempfile

import pyarrow as pa
import pyarrow.ipc as ipc

import dftracer.utils as dft_utils
from dftracer.utils.dftracer_utils_ext import (
    AggregatorUtility,
)

from .common import Environment


class TestArrowIpcReadback:
    """Verify Arrow output is readable by pyarrow."""

    EXPECTED_BASE_COLUMNS = {
        "batch_type",
        "cat",
        "name",
        "pid",
        "tid",
        "hhash",
        "fhash",
        "time_bucket",
        "count",
        "dur_total",
        "dur_min",
        "dur_max",
        "dur_mean",
        "dur_std",
        "size_total",
        "size_min",
        "size_max",
        "size_mean",
        "size_std",
        "ts",
        "te",
    }

    def test_view_cli_arrow_output(self):
        """dftracer_view --format arrow produces a valid IPC file."""
        binary = shutil.which("dftracer_view")
        if binary is None:
            return  # CLI not installed

        with Environment(lines=20) as env:
            env.create_test_gzip_file()
            directory = env.temp_dir

            with tempfile.NamedTemporaryFile(suffix=".arrows", delete=False) as f:
                output_path = f.name

            try:
                subprocess.run(
                    [
                        binary,
                        "--directory",
                        directory,
                        "--group-by",
                        "name",
                        "--agg",
                        "mean:dur",
                        "--format",
                        "arrow",
                        "--output",
                        output_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )

                assert os.path.exists(output_path)
                assert os.path.getsize(output_path) > 0

                reader = ipc.open_file(output_path)
                table = reader.read_all()

                assert table.num_rows > 0
                # View aggregate table: group columns + aggregated value columns.
                col_names = set(table.column_names)
                assert "name" in col_names
                assert "mean_dur" in col_names

            finally:
                if os.path.exists(output_path):
                    os.unlink(output_path)

    def test_aggregator_python_roundtrip(self):
        """AggregatorUtility Arrow output is readable by pyarrow."""
        with Environment(lines=20) as env:
            env.create_test_gzip_file()
            directory = env.temp_dir

            table = AggregatorUtility().process(directory)
            assert table.num_rows > 0

            for batch in table.batches():
                pa_batch = pa.record_batch(batch)
                assert pa_batch.num_rows > 0
                assert pa_batch.num_columns == len(self.EXPECTED_BASE_COLUMNS)

                schema = pa_batch.schema
                assert set(schema.names) == self.EXPECTED_BASE_COLUMNS
                assert schema.field("cat").type == pa.utf8()
                assert schema.field("count").type == pa.uint64()
                assert schema.field("dur_mean").type == pa.float64()

    def test_trace_viewer_stream_roundtrip(self):
        """TraceViewer.stream Arrow output is readable by pyarrow."""
        with Environment(lines=20) as env:
            gz_file = env.create_test_gzip_file()
            with dft_utils.Indexer(files=[gz_file], index_dir=env.temp_dir) as ix:
                ix.ensure_indexed()

            viewer = dft_utils.TraceViewer(gz_file, index_path=env.temp_dir)
            batches = [pa.record_batch(c) for c in viewer.stream(batch_size=100)]
            assert batches
            for pa_batch in batches:
                assert pa_batch.num_rows > 0
                col_names = set(pa_batch.schema.names)
                assert "name" in col_names or "cat" in col_names
