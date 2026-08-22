:description: Equi-join two frames on their leading key columns: inner, left, right, full, and semi joins over frames sharing key names in order.

Join frames
===========

Combine two frames row-wise on a shared key: line up a per-file summary against a
baseline, attach resolved names to an aggregate, or intersect two result sets.
The join key is a *leading-key-count*: both frames must already carry the join
columns first, in the same order, with the same names. The engine equi-joins on
the first ``n`` columns and matches nothing when the two key schemas disagree.

Join two DataFrames
-------------------

.. tab-set::

   .. tab-item:: C++

      ``join_batches`` is a free function in
      ``dftracer/utils/trace/views/result_join.h``.

      .. code-block:: cpp

         #include <dftracer/utils/trace/views/result_join.h>

         using namespace dftracer::utils::trace::views;
         using dftracer::utils::dataframe::DataFrame;

         // Equi-join on the first n_key columns (shared names, in order).
         DataFrame joined = join_batches(left, right, /*n_key=*/1, JoinType::INNER);

      ``JoinType`` is ``INNER``, ``LEFT``, ``RIGHT``, ``FULL``, ``LEFT_SEMI``, or
      ``LEFT_ANTI``.

   .. tab-item:: Python

      .. code-block:: python

         # on = number of leading key columns, or the shared key name(s).
         joined = df.join(other, on=1, how="inner")
         joined = df.join(other, on="fid", how="left")
         joined = df.join(other, on=["pid", "tid"], how="inner")

      ``on`` is either an int count of leading key columns (both frames must
      carry them first, in order, with the same names) or the shared key column
      name(s). ``how`` is one of ``inner`` / ``left`` / ``right`` / ``full`` /
      ``semi`` / ``anti`` (use ``full``, not ``outer``). ``on`` defaults to
      ``1``.

The result keeps the key columns under their own names, then the non-key
**left** columns prefixed ``l_``, then (unless the join is ``SEMI`` / ``ANTI``)
the non-key **right** columns prefixed ``r_``. An outer join (``LEFT`` /
``RIGHT`` / ``FULL``) nulls the absent side. If the two frames do not share the
leading key schema the result is an empty frame.

There is no ``dftu_dataframe_join`` in the C ABI; from C, drive the join through
the C++ or Python surface above.

Join two aggregated trace queries
---------------------------------

When both sides are trace queries with the same ``group_by``/``agg``, join their
aggregated results directly - each side runs its own pruned scan, then the two
result frames equi-join on the shared group key.

.. tab-set::

   .. tab-item:: C++

      ``AggregatedView::join`` is a terminal on the aggregated view (from
      ``group_by``/``agg``); it returns the joined ``DataFrame``.

      .. code-block:: cpp

         #include <dftracer/utils/trace/views/view.h>

         using namespace dftracer::utils::trace::views;

         auto base = View::from_file("baseline.pfw.gz")
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Count, "", "n"}});
         auto vary = View::from_file("variant.pfw.gz")
                         .group_by({GroupKey::cat()})
                         .agg({{AggOp::Count, "", "n"}});

         auto joined = base.join(vary, JoinType::FULL).get();

   .. tab-item:: Python

      .. code-block:: python

         from dftracer.utils import TraceViewer

         base = TraceViewer("baseline.pfw.gz").group_by("cat").agg("count")
         vary = TraceViewer("variant.pfw.gz").group_by("cat").agg("count")

         joined = base.join(vary, how="inner")

For a ready-made baseline-vs-variant delta on top of this join, see
:doc:`../analysis/comparison`.

Temporal and interval joins (Python)
-------------------------------------

For time-based matching that an equi-join cannot express - "attach the latest
known state as of this event" or "which interval does this point fall in" -
``DataFrame`` carries two more native joins, ``asof`` and ``interval``. Both take
another ``DataFrame`` and return a ``DataFrame`` (no pyarrow), referencing
columns by name:

.. code-block:: python

   # Nearest-by-time match, within an optional equi-key partition.
   matched = left.asof(right, on="ts", by="pid",
                       direction="backward", tolerance=1_000_000)

   # Point-in-range match: left's `point` against right's closed [lo, hi].
   spans = events.interval(phases, point="ts", lo="start", hi="end", by="pid")

``asof`` matches each left row to the nearest right row by the time column
``on`` (``direction`` ``backward`` / ``forward`` / ``nearest``; ``tolerance``
bounds the allowed distance); unmatched left rows get null right values.
``interval`` matches each left row's ``point`` to every right row whose closed
span ``[lo, hi]`` contains it, one output row per match, and ``outer=True`` also
emits an unmatched left row once with null right values. Neither is the same op
as the equi-join above.

See also
--------

- :doc:`dataframe` for the other frame ops (project, filter, sort, reshape) and
  the generated ``asof`` / ``interval`` / ``window`` reference.
- :doc:`time-windows` for window functions and time-grid gap filling.
- :doc:`../analysis/aggregation` for building the aggregated views you join.
- :doc:`../../cpp_api/dataframe` for the full member and C ABI reference.
