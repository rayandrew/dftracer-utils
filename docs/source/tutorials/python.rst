Python Tutorial: An I/O Bottleneck Report
=========================================

This tutorial builds a complete, runnable program that turns a directory of
DFTracer traces into an I/O bottleneck report: a per-category summary, the
slowest functions, and a bandwidth time series, printed to the console and saved
as CSV.

It uses :class:`~dftracer.utils.TraceViewer` throughout; see
:doc:`../api/trace_viewer` for the full API and :doc:`../api/query` for the
filter DSL.

The goal
--------

Given ``./traces`` (a directory of ``.pfw.gz`` files), produce:

1. total I/O time and bytes per category,
2. the ten functions with the most total duration,
3. read/write bytes over one-second windows,

and write the tables to ``report/``.

Step 1: point a viewer at the traces
------------------------------------

A :class:`~dftracer.utils.TraceViewer` takes a file, a list of files, or a
directory. Given a directory it scans recursively for ``.pfw.gz`` traces (in
parallel), so point it straight at the folder. Building the index first is
optional - a query builds it on first touch - but doing it explicitly makes the
first report faster and reports progress:

.. code-block:: python

   from dftracer.utils import TraceViewer, Indexer

   trace_dir = "./traces"

   with Indexer(trace_dir) as ix:
       status = ix.ensure_indexed()
       print(f"indexed {status.total_files} files")

   view = TraceViewer(trace_dir)

Step 2: summarize I/O by category
---------------------------------

Group by category, count events, and sum duration and size. ``collect()``
returns a pyarrow ``Table``; convert to pandas for reporting:

.. code-block:: python

   cat_summary = (
       view.phase("events")
           .group_by("cat")
           .agg("count", "sum:dur", "sum:size")
           .collect()
           .to_pandas()
           .rename(columns={"sum_dur": "total_us", "sum_size": "total_bytes"})
           .sort_values("total_us", ascending=False)
   )

Step 3: find the slowest functions
----------------------------------

Group by function name and rank by total time. Keep it to POSIX I/O and take the
top ten:

.. code-block:: python

   slowest = (
       view.filter('cat == "POSIX"')
           .group_by("name")
           .agg("count", "sum:dur", "max:dur", "p99:dur")
           .collect()
           .to_pandas()
           .sort_values("sum_dur", ascending=False)
           .head(10)
   )

``p99:dur`` comes from the index's persisted sketch, so it costs no extra scan.

Step 4: a bandwidth time series
-------------------------------

Bucket time into one-second windows (microseconds) and sum bytes per window and
direction. ``time_bucket`` becomes a group key:

.. code-block:: python

   bandwidth = (
       view.filter('name in ["read", "pread", "write", "pwrite"]')
           .time_bucket(1_000_000)
           .group_by("time_bucket", "name")
           .agg("sum:size")
           .collect()
           .to_pandas()
           .pivot_table(index="time_bucket", columns="name",
                        values="sum_size", fill_value=0)
   )

Step 5: write the report
------------------------

.. code-block:: python

   os.makedirs("report", exist_ok=True)
   cat_summary.to_csv("report/by_category.csv", index=False)
   slowest.to_csv("report/slowest_functions.csv", index=False)
   bandwidth.to_csv("report/bandwidth_over_time.csv")

   print("\n== I/O by category ==")
   print(cat_summary.to_string(index=False))
   print("\n== Slowest functions ==")
   print(slowest.to_string(index=False))

The complete program
--------------------

.. code-block:: python

   #!/usr/bin/env python3
   """io_report.py -- an I/O bottleneck report from DFTracer traces."""
   import os
   import sys

   from dftracer.utils import Indexer, TraceViewer


   def main(trace_dir: str, out_dir: str = "report") -> None:
       with Indexer(trace_dir) as ix:
           status = ix.ensure_indexed()
           print(f"indexed {status.total_files} files")

       view = TraceViewer(trace_dir)   # scans the directory recursively

       cat_summary = (
           view.phase("events")
               .group_by("cat")
               .agg("count", "sum:dur", "sum:size")
               .collect()
               .to_pandas()
               .rename(columns={"sum_dur": "total_us", "sum_size": "total_bytes"})
               .sort_values("total_us", ascending=False)
       )

       slowest = (
           view.filter('cat == "POSIX"')
               .group_by("name")
               .agg("count", "sum:dur", "max:dur", "p99:dur")
               .collect()
               .to_pandas()
               .sort_values("sum_dur", ascending=False)
               .head(10)
       )

       bandwidth = (
           view.filter('name in ["read", "pread", "write", "pwrite"]')
               .time_bucket(1_000_000)
               .group_by("time_bucket", "name")
               .agg("sum:size")
               .collect()
               .to_pandas()
               .pivot_table(index="time_bucket", columns="name",
                            values="sum_size", fill_value=0)
       )

       os.makedirs(out_dir, exist_ok=True)
       cat_summary.to_csv(f"{out_dir}/by_category.csv", index=False)
       slowest.to_csv(f"{out_dir}/slowest_functions.csv", index=False)
       bandwidth.to_csv(f"{out_dir}/bandwidth_over_time.csv")

       print("\n== I/O by category ==")
       print(cat_summary.to_string(index=False))
       print("\n== Slowest functions ==")
       print(slowest.to_string(index=False))
       print(f"\nwrote CSVs to {out_dir}/")


   if __name__ == "__main__":
       main(sys.argv[1] if len(sys.argv) > 1 else "./traces")

Run it:

.. code-block:: bash

   python io_report.py ./traces

High performance: the Runtime
-----------------------------

Every engine call runs on a :class:`~dftracer.utils.Runtime` thread pool. For
best throughput, create one sized to your machine and pass it to the viewers and
indexer so they reuse the same threads instead of spinning a pool per call. This
is the Python counterpart of the C++ :doc:`../pipeline` - ``rt.submit`` fans
embarrassingly-parallel per-file work across the pool, which you then gather.

.. code-block:: python

   #!/usr/bin/env python3
   """runtime_report.py -- fused + per-file summaries on a shared Runtime."""
   import glob
   import sys

   from dftracer.utils import Indexer, Runtime, TraceViewer


   def main(trace_dir: str) -> None:
       # threads: C++ compute/scan threads; python_threads: pool for submit().
       with Runtime(threads=16, python_threads=8) as rt:
           with Indexer(trace_dir, runtime=rt) as ix:
               ix.ensure_indexed()

           # One directory, one fused parallel pass across `threads`.
           overall = (
               TraceViewer(trace_dir, runtime=rt)
               .phase("events")
               .group_by("cat")
               .agg("count", "sum:dur")
               .collect()
           )
           print(overall.to_pandas().to_string(index=False))

           # Independent per-file reports, fanned across the pool and gathered.
           files = sorted(glob.glob(f"{trace_dir}/**/*.pfw.gz", recursive=True))

           def report(f):
               return (
                   TraceViewer([f], runtime=rt)
                   .phase("events")
                   .group_by("name")
                   .agg("sum:dur")
                   .collect()
               )

           handles = [rt.submit(report, f, name=f) for f in files]
           rt.wait_all(raise_on_error=True)
           per_file = {f: h.get() for f, h in zip(files, handles)}
           print(f"gathered {len(per_file)} per-file summaries")


   if __name__ == "__main__":
       main(sys.argv[1] if len(sys.argv) > 1 else "./traces")

Size ``threads`` to physical cores; a single shared Runtime avoids
oversubscription when many queries run back to back. ``rt.wait_all`` blocks until
every submitted task finishes, and ``TaskHandle.get()`` returns each result (or
re-raises its error).

Scaling out with Dask
---------------------

For trace sets too large for one machine, swap ``TraceViewer`` for
:class:`~dftracer.utils.dask.DaskTraceViewer` and keep the same builder calls -
the aggregation fans across the cluster and returns the same table:

.. code-block:: python

   from dask.distributed import Client
   from dftracer.utils.dask import DaskTraceViewer, register_auto_thread_plugin

   client = Client("scheduler:8786")
   register_auto_thread_plugin()

   view = DaskTraceViewer(files, index_path, client=client)
   cat_summary = view.group_by("cat").agg("count", "sum:dur", "sum:size").collect()

Everything else in ``main`` is unchanged. See :doc:`../api/trace_viewer` and
:doc:`../api/dfanalyzer` for the distributed building blocks.
