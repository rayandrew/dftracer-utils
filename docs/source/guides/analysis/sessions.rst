:description: Run many reads of one trace over a single shared scan with a session: register several branches (aggregates, folds, exports, call trees) and execute them together.

Sessions: many reads, one scan
===============================

.. admonition:: Goal
   :class: goal

   Answer several questions about the same trace without scanning it several
   times. A **session** registers a batch of independent branch views and runs
   them all over one shared pass, so N reads cost one decompression and one
   parse, not N.

Why a session
-------------

Every View terminal (``collect``, ``export``, ``materialize``, ``call_tree``,
...) runs its own scan: it decompresses the gzip members and parses the JSON
once, for that one result. When you want several results from the same base
view - a couple of unrelated aggregations, a custom fold next to a built-in
aggregate, an aggregate next to an export - running each terminal separately
pays that decompression and parse once **per terminal**.

A session pays it **once** and splits the result across every branch. This is
the user-facing capability built on the engine's :doc:`../../concepts/fused-scan`
mechanism: fused-scan is *how* one pass feeds many folds; a session is the API
you use to *ask for* many reads at once.

Reach for a session when two or more reads share the same base files (and the
same scan-wide settings like phase or time range), and you would otherwise
call two terminals back to back.

The shape
---------

1. Open a session off a base view.
2. Register each branch: start a branch view, chain the per-branch builder API
   (its own ``filter`` / ``group_by`` / ``agg`` / ...), and end in a terminal.
   The terminal returns a **handle**, not a result.
3. Execute the session once. Every branch resolves together; then read each
   handle.

A handle read before execution throws - the value is not there yet.

Python
------

Each ``s.view()`` is a full lazy ``TraceViewer`` branch with its own schema and
filters; the terminal registers it and returns a :class:`~dftracer.utils.dataframe.Handle`.
A ``with`` block executes on exit, or call ``s.execute()`` (the first
``handle.result()`` also triggers it):

.. code-block:: python

   from dftracer.utils import TraceViewer

   tv = TraceViewer("trace.pfw.gz")

   with tv.session() as s:
       by_cat  = s.view().group_by("cat").agg("count", "mean:dur").collect()
       by_rank = s.view().group_by("rank").agg("count").sort_by("count").collect()
       posix   = s.view().filter('cat == "POSIX"').export("posix.pfw.gz")
       tree    = s.view().filter('cat == "POSIX"').call_tree(partition=("pid", "tid"))

   cat_df  = by_cat.result()     # a DataFrame, resolved by the one scan
   rank_df = by_rank.result()
   tree_df = tree.result()       # events + level / parent_id

Any branch terminal works: ``collect``, ``export``, ``materialize`` (build-only,
no handle), ``statistics``, ``events``, ``aggregate_partial``, and the
containment terminals ``call_tree`` / ``flamegraph`` / ``containment`` (same
``partition`` / ``ts`` / ``dur`` / ``name`` arguments as on ``TraceViewer``).

C++
---

``View::session()`` returns a ``ViewSession``. Register ops - each returns a
``Deferred<T>`` handle - then ``execute()`` runs the one shared scan:

.. code-block:: cpp

   #include <dftracer/utils/trace/views/view.h>
   using namespace dftracer::utils::trace::views;
   namespace df = dftracer::utils::dataframe;

   View base = View::from_file("trace.pfw.gz");
   ViewSession run = base.session();

   // A built-in aggregate branch.
   auto by_cat = run.collect(
       Query::from_string(R"(cat == "POSIX")").value(), {GroupKey::cat()},
       {{AggOp::Count, "", "n"}, {AggOp::Sum, "dur", "sum_dur"}});

   // A custom fold branch over the same scan (reuses the parsed event).
   struct Small { std::size_t n = 0; double sum = 0; };
   auto small = run.fold<Small>(
       Query::from_string("dur < 25").value(),
       [](Small& s, const json::JsonValue& jv, std::string_view) {
           ++s.n;
           s.sum += jv["dur"].get<double>(0);
       },
       [](Small&& a, Small&& b) { a.n += b.n; a.sum += b.sum; return std::move(a); });

   // A containment branch: a call tree over a filtered sub-view of the base.
   auto tree = run.call_tree(
       base.filter(Query::from_string(R"(cat == "POSIX")").value()),
       {"pid", "tid"});

   run.execute().get();          // the single shared scan

   df::DataFrame cat_df = *by_cat;   // Deferred<T>::get() / operator* / ->
   std::size_t total  = small->n;
   df::DataFrame tree_df = *tree;

Reading a ``Deferred`` before ``execute()`` resolves it throws, exactly like the
Python handle.

Combining two branches: join and compare
----------------------------------------

Two ``collect`` branches that group the same way can be joined or compared
**after** the one scan - the shared key width is inferred - and the combination
is itself a handle:

.. code-block:: python

   with tv.session() as s:
       base = s.view().filter('rank == 0').group_by("name").agg("mean:dur").collect()
       var  = s.view().filter('rank == 1').group_by("name").agg("mean:dur").collect()
       delta = s.compare(base, var)          # or s.join(base, var)
   diff = delta.result()

.. code-block:: cpp

   ViewSession run = View::from_file("trace.pfw.gz").session();
   auto base = run.collect(Query::from_string("rank == 0").value(),
                           {GroupKey::name()}, {{AggOp::Mean, "dur", "mean_dur"}});
   auto var  = run.collect(Query::from_string("rank == 1").value(),
                           {GroupKey::name()}, {{AggOp::Mean, "dur", "mean_dur"}});
   auto delta = run.compare(base, var);      // or run.join(base, var)
   run.execute().get();
   df::DataFrame diff = *delta;

Notes
-----

- Branches are independent: each carries its own filter, group_by, and result
  schema. Only the base files and scan-wide settings (phase, time range, time
  scale) are shared.
- ``materialize`` in a session is build-only - it persists a rollup as a side
  effect and returns no handle.
- A session is the in-process form. Across nodes, the same one-pass-many-results
  idea is the partial/reduce pattern - see :doc:`../scale/distributed-aggregation`.

See also
--------

- :doc:`../../concepts/fused-scan` - the one-scan-many-folds mechanism a session
  drives.
- :doc:`views` - the single-terminal View API each branch is built from.
- :doc:`aggregation` - the ``group_by`` / ``agg`` vocabulary branches use.
- :doc:`../../api/trace_viewer` - the Python ``TraceViewer`` / ``Session`` /
  ``SessionView`` / ``Handle`` reference.
