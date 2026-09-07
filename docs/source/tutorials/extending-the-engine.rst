:description: Lesson 3: author your own plugin, a fold over the engine's single fused scan, in Python or C++, run it, and read its result.

Extending the engine
====================

The built-in aggregates in :doc:`first-analysis` and :doc:`analysis-in-depth`
cover common needs, but sometimes you need a computation the aggregator does not
offer. A **plugin** is your own fold over the single fused scan the engine
already runs: you see every event once, accumulate into a mergeable state, and
the host handles the parallelism, merging, and materialization for you.

In this lesson you author one plugin - in Python or C++, your choice of tab -
run it, and read its result. This lesson assumes you have finished lessons 1
and 2.

Both languages need a C++ compiler on your PATH: the Python plugin is
AST-compiled to a cached native ``.so`` the first time you load it, and the C++
plugin you compile yourself.

1. Build a trace with two processes
-----------------------------------

Shared setup. The plugin will count events per process, so write a trace that
has more than one ``pid``:

.. code-block:: python

   import gzip

   with gzip.open("plugin_trace.pfw.gz", "wt") as f:
       for i in range(300):
           pid = 2 if i % 3 == 0 else 1
           f.write(
               f'{{"name":"read","cat":"POSIX","pid":{pid},"tid":1,'
               f'"ts":{1000 + i},"dur":10,"ph":"X","args":{{}}}}\n'
           )

That is 100 events for ``pid`` 2 (every third event) and 200 for ``pid`` 1.

2. Author the plugin
--------------------

The state is the same in both languages: a map keyed by ``pid`` whose value is a
count monoid, incremented once per event. The host runs a copy of the plugin per
worker, merges the maps, and materializes the result - you never write reduce or
threading code.

.. tab-set::

   .. tab-item:: Python

      A JIT plugin is a class decorated with ``@jit.plugin``. It declares one or
      more maps - a tuple key and a monoid value - and a single
      ``@jit.each_event`` method that updates them per event. The key is a
      1-tuple ``(jit.i64,)`` because ``pid`` is one integer:

      .. code-block:: python

         from dftracer.utils import jit

         @jit.plugin
         class EventsPerPid:
             hits = jit.map(key=(jit.i64,), value=jit.count())

             @jit.each_event
             def step(self, e):
                 self.hits[(e.pid,)] += 1

      The ``count()`` monoid is what lets the host merge per-worker copies for
      you.

   .. tab-item:: C++

      In C++ you write a *Slice*: a struct constructed from the config, with
      ``step`` (per batch), ``merge`` (combine two slices), and ``finalize``.
      ``make_plugin<Slice>`` fills in the ABI struct, and the ``dftracer_plugin``
      factory exports it (the factory symbol the host looks for). Save this as
      ``events_per_pid.cpp``:

      .. code-block:: cpp

         #include <dftracer/utils/plugins/plugin.h>

         #include <cstdint>

         using namespace dftracer::utils::plugins;

         struct EventsPerPid {
             explicit EventsPerPid(const Config&) {}

             void step(const Batch& b, Host h) {
                 auto hits = h.map("hits", Monoid::Counter, Key<std::int64_t>{});
                 for (const Event& e : b)
                     hits[e.pid()] += 1;
             }

             void merge(EventsPerPid&) {}  // host owns the map; nothing to merge
             void finalize(Host) {}
         };

         extern "C" dftu_plugin* dftracer_plugin(const dftu_value* config) {
             return make_plugin<EventsPerPid>(config);
         }

      ``Batch`` and ``Event`` are zero-copy views over the batch, so
      ``for (const Event& e : b)`` iterates typed events with accessors like
      ``e.pid()`` instead of indexing the raw C struct. ``h.map`` returns a
      typed ``Map``: ``Key<std::int64_t>{}`` fixes the key schema (one integer
      ``pid``) and ``Monoid::Counter`` the value, so you never spell a
      ``DFTU_MONOID_*`` name or hand-encode the key. ``hits[e.pid()] += 1`` reads
      like ``std::map`` but contributes to the merge: the host owns the map, so
      ``merge`` is empty and per-worker copies are combined by the map service,
      not by your code. ``plugin.h`` is header-only, so nothing links the
      library.

3. Run it and read the result
-----------------------------

.. tab-set::

   .. tab-item:: Python

      ``PluginHost`` compiles the class to a cached ``.so``, loads it, and folds
      it over one scan of your traces. ``run()`` wires up the map service and
      returns the emitted results keyed by map name:

      .. code-block:: python

         import pyarrow as pa
         from dftracer.utils.plugins import PluginHost

         host = PluginHost()
         host.load(EventsPerPid)
         results = host.run("plugin_trace.pfw.gz")

         table = pa.table(results["hits"])
         print(table.to_pandas().sort_values("k0").reset_index(drop=True))
         print("scanned:", host.stats["events_scanned"])

      Expected output. The key column is ``k0`` (the ``pid``) and the monoid
      value column is ``value`` (the count):

      .. code-block:: text

              k0  value
           0   1    200
           1   2    100
           scanned: 300

      Pass a directory to ``run()`` and it folds over every ``.pfw.gz`` beneath
      it, exactly like ``TraceViewer``.

   .. tab-item:: C++

      Compile the Slice to a shared library, then fold it over the trace with
      the ``dftracer_run`` binary - no Python involved. Point ``-I`` at the
      installed headers:

      .. code-block:: console

         $ g++ -std=c++20 -shared -fPIC -I/path/to/dftracer-utils/include \
               events_per_pid.cpp -o events_per_pid.so
         $ dftracer_run --plugin ./events_per_pid.so --files plugin_trace.pfw.gz

      ``dftracer_run`` folds the plugin over one shared, index-pruned parallel
      scan and reports the scan on stderr:

      .. code-block:: text

         Run: plugins=1 | Files: 1 | Chunks: scanned=1 skipped=0 | Events: matched=300 scanned=300

      The 300 scanned events are the fold's input; the per-``pid`` map is merged
      in-process by the host. Use ``-d <directory>`` in place of ``--files`` to
      fold over a whole tree. To consume the merged map values programmatically
      (as the Python tab does), drive the same ``.so`` through a ``PluginHost``;
      see :doc:`../plugins`.

What you learned
----------------

- A plugin is a **fold over the one fused scan**: see each event once, keep
  mergeable state.
- Author it in Python with ``@jit.plugin`` + ``jit.map`` + ``@jit.each_event``,
  or in C++ as a ``Slice`` with ``step`` / ``merge`` / ``finalize`` exported by
  ``make_plugin`` and the ``dftracer_plugin`` factory - both compile to the same
  ABI.
- Run it in Python with ``PluginHost.load`` / ``run``, or compile
  the ``.so`` and fold it with ``dftracer_run --plugin``.

To go further, see the Extending the engine section of the
:doc:`../guides/index` and the plugin reference in :doc:`../plugins`.
