C++ Tutorial: A Trace Summary Tool
==================================

This tutorial builds a complete C++ command-line tool that prints a per-category
I/O summary of a trace: for each ``(category, function)`` it reports the event
count, total duration, and slowest call. It uses the C++ ``View`` - the
counterpart of the Python :class:`~dftracer.utils.TraceViewer` - and links
against the ``dftracer-utils`` library.

The goal
--------

``trace_summary trace.pfw.gz`` prints:

.. code-block:: text

   cat            name       count      sum_dur       max_dur
   POSIX          read        1240     4821003.0      12904.0
   POSIX          write        512     1900550.0       8800.0
   ...

Query the trace
---------------

The builder is lazy; ``collect()`` is a terminal that returns a
``coro::CoroTask``. Run it by ``co_await``-ing it inside a coroutine - the View
never blocks a thread itself. ``collect()`` yields a ``ResultTable`` by value.

.. code-block:: cpp

   #include <dftracer/utils/utilities/composites/dft/views/view.h>

   using namespace dftracer::utils::utilities::composites::dft::views;

   auto table = co_await View::from_file(trace_path, index_path)
                    .phase(Phase::Events)
                    .group_by({GroupKey::cat(), GroupKey::name()})
                    .agg({{AggOp::Count, "", "count"},
                          {AggOp::Sum, "dur", "sum_dur"},
                          {AggOp::Max, "dur", "max_dur"}});
   // .collect() below runs the whole chain in one pass.

``GroupKey`` has factories (``cat()``, ``name()``, ``pid()``, ``file_path()``,
...); an ``AggSpec`` is ``{AggOp, field, output_name}`` (``AggOp::Count`` takes
an empty field). ``phase()``, ``time_bucket()``, ``limit()`` / ``offset()``
compose as in Python.

Read the result
---------------

``ResultTable`` has ``group_columns`` (the key names), ``value_columns`` (the
numeric aggregate names), and ``rows`` (by value, so use ``.``). Each
``ResultRow`` has ``keys`` (aligned to ``group_columns``) and ``values``
(aligned to ``value_columns``):

.. code-block:: cpp

   for (const auto& col : table.group_columns) std::printf("%-14s", col.c_str());
   for (const auto& col : table.value_columns) std::printf("%14s", col.c_str());
   std::printf("\n");

   for (const auto& row : table.rows) {
       for (const auto& k : row.keys)   std::printf("%-14s", k.c_str());
       for (double v : row.values)      std::printf("%14.1f", v);
       std::printf("\n");
   }

The complete program
--------------------

``collect()`` runs on the coroutine runtime, so wrap the query in a coroutine
and block ``main`` on it once with ``.get()`` - the ``.get()`` is on the
top-level coroutine, never on the View:

.. code-block:: cpp

   // trace_summary.cpp -- per-category I/O summary of a DFTracer trace.
   #include <dftracer/utils/utilities/composites/dft/views/view.h>

   #include <cstdio>
   #include <string>

   using namespace dftracer::utils::utilities::composites::dft::views;
   namespace coro = dftracer::utils::coro;

   coro::CoroTask<int> run(const std::string& trace, const std::string& index) {
       auto table = co_await View::from_file(trace, index)
                        .phase(Phase::Events)
                        .group_by({GroupKey::cat(), GroupKey::name()})
                        .agg({{AggOp::Count, "", "count"},
                              {AggOp::Sum, "dur", "sum_dur"},
                              {AggOp::Max, "dur", "max_dur"}})
                        .collect();

       for (const auto& col : table.group_columns) std::printf("%-14s", col.c_str());
       for (const auto& col : table.value_columns) std::printf("%14s", col.c_str());
       std::printf("\n");
       for (const auto& row : table.rows) {
           for (const auto& k : row.keys) std::printf("%-14s", k.c_str());
           for (double v : row.values)    std::printf("%14.1f", v);
           std::printf("\n");
       }
       co_return 0;
   }

   int main(int argc, char** argv) {
       if (argc < 2) {
           std::fprintf(stderr, "usage: %s <trace.pfw.gz> [index_dir]\n", argv[0]);
           return 1;
       }
       const std::string trace = argv[1];
       const std::string index = argc > 2 ? argv[2] : trace + ".dftindex";
       return run(trace, index).get();   // block main on the coroutine once
   }

Build and link
--------------

``dftracer-utils`` exports CMake targets. A minimal ``CMakeLists.txt``:

.. code-block:: cmake

   cmake_minimum_required(VERSION 3.20)
   project(trace_summary CXX)
   set(CMAKE_CXX_STANDARD 20)

   find_package(dftracer_utils REQUIRED)

   add_executable(trace_summary trace_summary.cpp)
   target_link_libraries(trace_summary PRIVATE dftracer::utils)

Build and run:

.. code-block:: bash

   cmake -S . -B build && cmake --build build
   ./build/trace_summary trace.pfw.gz

High performance: a whole directory in one pass
-----------------------------------------------

The same program scales from one file to a directory tree of traces with a
single change. ``View::from_directory`` scans ``dir`` recursively for
``.pfw.gz`` files (the recursive scan itself runs in parallel), builds a View
over all of them, and the pruned aggregation then fans across the runtime's
compute threads in one pass. You never manage threads, a ``CoroScope``, or a
pipeline yourself - the View owns that. It is a coroutine, so ``co_await`` it.

Here is the complete tool, now over a directory:

.. code-block:: cpp

   // dir_summary.cpp -- per-category I/O summary across a directory of traces.
   #include <dftracer/utils/utilities/composites/dft/views/view.h>

   #include <cstdio>
   #include <string>

   using namespace dftracer::utils::utilities::composites::dft::views;
   namespace coro = dftracer::utils::coro;

   coro::CoroTask<int> run(const std::string& dir, const std::string& index) {
       auto table = co_await View::from_directory(dir, index)
                        .phase(Phase::Events)
                        .group_by({GroupKey::cat(), GroupKey::name()})
                        .agg({{AggOp::Count, "", "count"},
                              {AggOp::Sum, "dur", "sum_dur"},
                              {AggOp::Max, "dur", "max_dur"}})
                        .collect();

       for (const auto& col : table.group_columns) std::printf("%-14s", col.c_str());
       for (const auto& col : table.value_columns) std::printf("%14s", col.c_str());
       std::printf("\n");
       for (const auto& row : table.rows) {
           for (const auto& k : row.keys) std::printf("%-14s", k.c_str());
           for (double v : row.values)    std::printf("%14.1f", v);
           std::printf("\n");
       }
       co_return 0;
   }

   int main(int argc, char** argv) {
       if (argc < 2) {
           std::fprintf(stderr, "usage: %s <trace_dir> [index_dir]\n", argv[0]);
           return 1;
       }
       const std::string dir   = argv[1];
       const std::string index = argc > 2 ? argv[2] : "";  // "" = per-file sidecar
       return run(dir, index).get();   // block main on the coroutine once
   }

Each trace gets its own index, resolved under the second argument; leave it
empty and each file uses the sidecar index beside it (built on first touch).
The whole tree is then one fused parallel pass. To size the compute pool,
construct a ``dftracer::utils::Runtime`` before running (see :doc:`../pipeline`);
the scan uses its threads.

Explicit concurrency: a Pipeline of per-file summaries
------------------------------------------------------

``from_directory`` fuses every file into one aggregation. When you instead want
an *independent* result per file, or to compose the query with other stages
(read, transform, write), drive it with a :doc:`../pipeline`. A producer streams
file paths into a channel; a consumer pulls each path and fans a per-file View
collect across the pool, so the queries run concurrently. You never touch
``run_coro_scope`` - the Pipeline owns the scheduling and hands each task a
``CoroScope`` to ``spawn`` from.

.. code-block:: cpp

   // per_file_summary.cpp -- concurrent per-file I/O summaries via a Pipeline.
   #include <dftracer/utils/core/coro/channel.h>
   #include <dftracer/utils/core/pipeline/pipeline.h>
   #include <dftracer/utils/core/pipeline/pipeline_config.h>
   #include <dftracer/utils/core/tasks/task.h>
   #include <dftracer/utils/utilities/composites/dft/views/view.h>

   #include <cstdio>
   #include <string>
   #include <vector>

   using namespace dftracer::utils;
   using namespace dftracer::utils::coro;
   namespace views = dftracer::utils::utilities::composites::dft::views;

   int main(int argc, char** argv) {
       std::vector<std::string> files(argv + 1, argv + argc);

       auto config =
           PipelineConfig().with_name("summaries").with_compute_threads(16);
       Pipeline pipeline(config);
       auto channel = make_channel<std::string>(256);

       // Producer: stream the file paths into the channel.
       auto producer = make_task(
           [ch = channel->producer(), files](CoroScope&) mutable -> CoroTask<void> {
               auto guard = ch.guard();
               for (const auto& f : files) co_await ch.send(f);
               co_return;
           },
           "producer");

       // Consumer: fan a per-file View aggregation across the pool, then print.
       auto consumer = make_task(
           [ch = channel->consumer()](CoroScope& scope) mutable -> CoroTask<void> {
               std::vector<SpawnFuture<views::ResultTable>> inflight;
               while (auto file = co_await ch.receive()) {
                   std::string f = std::move(*file);
                   inflight.push_back(scope.spawn(
                       [f](CoroScope&) -> CoroTask<views::ResultTable> {
                           co_return co_await views::View::from_file(
                               f, f + ".dftindex")
                               .phase(views::Phase::Events)
                               .group_by({views::GroupKey::cat()})
                               .agg({{views::AggOp::Count, "", "count"},
                                     {views::AggOp::Sum, "dur", "sum_dur"}})
                               .collect();
                       }));
               }
               for (auto& fut : inflight) {
                   views::ResultTable t = co_await fut;
                   for (const auto& row : t.rows) {
                       for (const auto& k : row.keys) std::printf("%-12s", k.c_str());
                       for (double v : row.values)   std::printf("%14.1f", v);
                       std::printf("\n");
                   }
               }
               co_return;
           },
           "consumer");

       pipeline.set_source(producer);
       pipeline.set_destination(consumer);
       pipeline.execute();   // blocks until both tasks finish
       return 0;
   }

The producer's ``ch.guard()`` closes the channel when it exits, which unblocks
the consumer's ``co_await ch.receive()`` loop. ``scope.spawn`` returns a
``SpawnFuture`` per file; collecting them after the loop lets every file's query
run at once rather than one after another. Give the Pipeline more
``with_compute_threads`` to widen the fan-out.

Going further
-------------

- Add ``.query("dur >= 1000")`` before ``group_by`` to filter, or
  ``.time_bucket(1'000'000)`` to aggregate into fixed time windows (see the
  Python :doc:`python` tutorial for the analogous query shapes).
- ``collect_typed()`` returns all three record families
  (``regular``/``aggregated``/``counters``) from one pass.
