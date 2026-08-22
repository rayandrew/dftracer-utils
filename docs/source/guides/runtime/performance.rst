:description: The runtime tuning surface in one place: worker thread count and the knobs to reach for when a scan, aggregation, or plugin run is slow.

Tune runtime performance
=========================

.. admonition:: Goal
   :class: goal

   Know which knob to reach for when a scan, aggregation, or plugin run is
   slower or heavier than it should be. This page collects the tuning surface in
   one place; it does not re-explain the runtime model (see
   :doc:`../../concepts/async-runtime`) or memory budgeting (see
   :doc:`memory-budget`, not duplicated here).

Worker thread count
--------------------

The number of worker threads the runtime schedules coroutines on is the first
knob. More threads help a workload that is CPU-bound across many independent
chunks; too many on a shared or oversubscribed node hurts more than it helps.

.. tab-set::

   .. tab-item:: C++

      ``Runtime`` (``dftracer/utils/core/runtime.h``, namespace
      ``dftracer::utils``) takes a thread count directly, or a full
      ``ExecutorConfig`` for finer control:

      .. code-block:: cpp

         #include <dftracer/utils/core/runtime.h>
         using namespace dftracer::utils;

         Runtime rt(8);   // 8 worker threads; 0 = hardware_concurrency

         ExecutorConfig config;
         config.num_threads = 8;      // running cap; 0 = hardware_concurrency
         config.io_pool_size = 4;     // I/O thread pool; 0 = hardware_concurrency
         config.min_workers = 2;      // elastic floor: start at 2, grow to
                                       // num_threads under backlog, idle-retire
                                       // back down (0 = eager, spawn all up front)
         Runtime elastic_rt(config);

      ``DFTRACER_UTILS_THREADS`` overrides any requested count at process
      start (set it to ``1`` for a single-threaded async loop while
      debugging) - it takes precedence over both the constructor argument and
      ``ExecutorConfig::num_threads``.

   .. tab-item:: Python

      ``Runtime`` (``dftracer.utils.Runtime``) wraps the same C++ executor and
      adds a separate pool for plain Python callables:

      .. code-block:: python

         from dftracer.utils import Runtime, TraceViewer

         rt = Runtime(threads=8, io_threads=8, python_threads=4)
         tv = TraceViewer("./traces", runtime=rt)  # view execution uses rt's pool

      - ``threads``: C++ executor worker threads (0 = hardware_concurrency).
      - ``io_threads``: C++ I/O thread pool (0 = hardware_concurrency).
      - ``python_threads``: a ``ThreadPoolExecutor`` size for
        ``rt.submit(python_callable)`` (0 = ``min(32, threads)``); it does not
        affect C++-side scan/aggregation work.

      Without an explicit ``runtime=``, a viewer runs on a process-wide
      default runtime. Pass a ``Runtime`` explicitly when you need a
      non-default thread count or want several viewers to share one pool.

Data-parallel loops: parallel_for / parallel_reduce
---------------------------------------------------------

``Runtime::parallel_for`` and ``Runtime::parallel_reduce`` (same header) are
C++-only fork-join helpers over an index range, used internally by the
columnar and scan code and available to any C++ caller. There is no Python
binding; from Python, parallelism comes from the view/DataFrame APIs that use
these internally, not from calling them directly.

.. code-block:: cpp

   Runtime& rt = dftracer::utils::default_runtime();

   // Split [0, n) into grain-sized chunks, run body on the pool, block until done.
   rt.parallel_for(n, /*grain=*/4096, [&](std::int64_t begin, std::int64_t end) {
       for (std::int64_t i = begin; i < end; ++i) process(i);
   });

   // Same split, but each chunk maps to a T and partials combine (associative).
   std::int64_t total = rt.parallel_reduce<std::int64_t>(
       n, /*grain=*/4096, /*identity=*/0,
       [&](std::int64_t begin, std::int64_t end) { return partial_sum(begin, end); },
       [](std::int64_t a, std::int64_t b) { return a + b; });

Both pick ``min(chunks, threads())`` workers and drain an atomic chunk counter
(dynamic, load-balanced scheduling), run inline serial for a single chunk, and
detect nesting - a call already inside a ``parallel_for``/``parallel_reduce``
body runs serial rather than forking a nested fork-join. ``grain`` is the
per-chunk unit size: too small and the atomic-counter overhead dominates, too
large and load balancing degrades to one chunk per worker regardless of skew.

Where the memory budget fits in
-------------------------------------

Thread count controls how much CPU-parallel work runs at once; it does not
bound how much memory that work holds. A wide thread count over a large
group-by can build many large in-memory group maps in parallel and exhaust
memory faster than a narrow one would. See :doc:`memory-budget` for
``View::memory_budget`` / ``auto_spill`` (and the Python equivalents), which
cap and spill that side independently of the thread count set here.

See also
--------

- :doc:`../../concepts/async-runtime` for the coroutine/executor model these
  knobs configure.
- :doc:`memory-budget` for bounding memory instead of CPU parallelism.
- :doc:`task-graphs` for building an explicit DAG on top of this runtime.
- :doc:`../../concepts/coroutine-caveats` for rules to follow before sharing
  state across a parallel body.
