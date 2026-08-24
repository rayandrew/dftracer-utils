:description: Roll traces up into a table: group matching events by one or more keys and reduce each group with count and duration aggregates.

Aggregate trace events
======================

Roll a trace up into a table: counts and duration stats per category, per file,
per host, or over time buckets. A trace query groups matching events by one or
more keys and reduces each group with a set of aggregates, pushing the filter
down to the index and scanning candidates in parallel. The result is a native
``DataFrame``.

Group and aggregate
-------------------

Pick group keys, pick aggregates, collect.

.. tab-set::

   .. tab-item:: C++

      Group keys are ``GroupKey`` values; aggregates are ``AggSpec`` values.
      ``group_by``/``agg`` return an ``AggregatedView``; ``collect()`` is a
      coroutine - ``.get()`` drives it to completion for a non-coroutine caller.

      .. code-block:: cpp

         #include <dftracer/utils/trace/views/view.h>

         using namespace dftracer::utils::trace::views;

         auto df = View::from_file("trace.pfw.gz")
                       .group_by({GroupKey::cat()})
                       .agg({{AggOp::Count, "", "count"},
                             {AggOp::Sum, "dur", "sum_dur"},
                             {AggOp::Mean, "dur", "mean_dur"}})
                       .collect()
                       .get();

      Scan a whole directory with ``View::from_directory`` (itself a coroutine):

      .. code-block:: cpp

         View v = View::from_directory("traces/").get();
         auto df = v.group_by({GroupKey::cat()})
                    .agg({{AggOp::Count, "", "count"}})
                    .collect()
                    .get();

   .. tab-item:: Python

      Group keys and aggregates are strings.

      .. code-block:: python

         from dftracer.utils import TraceViewer

         df = (TraceViewer("traces/")
               .group_by("cat")
               .agg("count", "sum:dur", "mean:dur")
               .collect())

Both produce a frame with one row per distinct category and the columns
``count``, ``sum_dur``, ``mean_dur``.

Aggregate with F expressions
----------------------------

The unified ``F`` builds aggregates too, Polars-style, additive to the string /
``AggSpec`` forms above (which stay fully supported and can be mixed in the same
``agg`` call). ``F("dur").sum()`` lowers to the same ``AggSpec`` as ``"sum:dur"``,
so the two forms produce identical columns. Field names resolve by dotted name,
top-level and ``args.*`` alike (``F("args.level").mean()``).

.. tab-set::

   .. tab-item:: C++

      .. code-block:: cpp

         using namespace dftracer::utils::dataframe::field;  // F

         auto df = View::from_file("trace.pfw.gz")
                       .group_by({GroupKey::cat()})
                       .agg(F("dur").sum(), F("dur").mean(), F.any.count())
                       .collect()
                       .get();

   .. tab-item:: Python

      .. code-block:: python

         from dftracer.utils import TraceViewer
         from dftracer.utils.columnar import F

         df = (TraceViewer("trace.pfw.gz")
               .group_by("cat")
               .agg(F.dur.sum(), F("dur").mean(), F.any.count())
               .collect())

The reductions are ``sum`` / ``min`` / ``max`` / ``mean`` / ``var`` / ``std`` /
``skew`` / ``kurt`` / ``count`` (C++ adds ``F("name").argmax_by("dur")``). The
``F.any`` wildcard aggregates every numeric ``args.*`` field: ``F.any.mean()`` is
the scan-time numeric-args path (see below) and ``F.any.count()`` is the group
count. Because that path computes only the mean, any other ``F.any.<op>()`` is
rejected - name a field for those (``F("args.level").sum()``).

Group keys
----------

A ``GroupKey`` names one column of a (possibly composite) key.

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Key
     - C++
     - Python
   * - event name
     - ``GroupKey::name()``
     - ``"name"``
   * - category
     - ``GroupKey::cat()``
     - ``"cat"``
   * - process / thread id
     - ``GroupKey::pid()`` / ``::tid()``
     - ``"pid"`` / ``"tid"``
   * - file hash / resolved path
     - ``GroupKey::fhash()`` / ``::file_path()``
     - ``"fhash"`` / ``"file_path"``
   * - host hash / resolved name
     - ``GroupKey::hhash()`` / ``::host_name()``
     - ``"hhash"`` / ``"host_name"``
   * - an args-map entry
     - ``GroupKey::of_arg("level")``
     - ``"arg:level"``

Group on several keys at once by passing more than one; the group key is the
composite. Also available: ``GroupKey::io_cat()`` / ``::acc_pat()`` /
``::file_name()``.

Aggregates
----------

An ``AggSpec`` is ``{op, field, out_name}`` (plus ``by`` for ``ArgMax`` and
``q`` for ``Pct``). In Python each aggregate is a string.

.. list-table::
   :header-rows: 1
   :widths: 34 33 33

   * - Aggregate
     - C++ ``AggOp``
     - Python spec
   * - group size
     - ``Count`` (field ignored)
     - ``"count"``
   * - sum / min / max
     - ``Sum`` / ``Min`` / ``Max``
     - ``"sum:dur"`` / ``"min:dur"`` / ``"max:dur"``
   * - mean
     - ``Mean``
     - ``"mean:dur"``
   * - variance / stddev
     - ``Var`` / ``Std``
     - ``"var:dur"`` / ``"std:dur"``
   * - skewness / kurtosis
     - ``Skew`` / ``Kurt``
     - ``"skew:dur"`` / ``"kurt:dur"``
   * - quantile (DDSketch)
     - ``Pct`` (``q`` in (0,1))
     - ``"pct:dur:0.99"`` or ``"p99:dur"``
   * - histogram (DDSketch)
     - ``Hist``
     - ``"hist:dur"``
   * - argmax (value at max ``by``)
     - ``ArgMax``
     - ``"argmax:name:dur"``
   * - distinct string values
     - ``SetUnion``
     - ``"set_union:name"`` / ``"uniq:name"``
   * - busy time (occupancy)
     - ``Busy``
     - ``"busy"``
   * - average concurrency
     - ``Concurrency``
     - ``"concurrency"``
   * - utilization
     - ``Utilization``
     - ``"utilization"``
   * - peak concurrency
     - ``Active``
     - ``"active"``

For example, a p99 duration per category is ``AggOp::Pct`` with ``q = 0.99`` (C++)
or ``"p99:dur"`` (Python). Percentile and histogram aggregates are backed by a
DDSketch; see :doc:`statistics`. The last four are the occupancy metrics,
explained below.

Occupancy: concurrency-aware duration
-------------------------------------

``sum(dur)`` adds up every event's duration, so overlapping or nested work
(async I/O in flight at once, threads running in parallel, a span that wholly
contains its children) is counted many times over. It answers "how much event-
time was recorded", not "how much wall-clock time was actually busy". The
occupancy metrics answer the wall-clock question. They are field-less: each is
always measured over the event interval ``[ts, ts + dur)``, so the spec is the
bare op name with no ``:field``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Metric
     - Meaning
   * - ``busy``
     - Wall-clock microseconds during which at least one event was active (the
       union of the intervals, not their sum).
   * - ``concurrency``
     - Average parallelism, ``sum(dur) / busy``: ``1.0`` when events never
       overlap, ``N`` when ``N`` run concurrently on average. This is Little's
       law, where ``sum(dur)`` is the integral of the active-event count over
       time.
   * - ``utilization``
     - ``busy`` divided by the makespan (``max_end - min_ts``): the fraction of
       the elapsed window that was busy.
   * - ``active``
     - Peak concurrent headcount: the largest number of events active at once.

Because the field is fixed, group and bucket as usual to ask sharper questions:
the number of concurrent ``pread`` calls per file is a ``concurrency`` (or
``active``) aggregate grouped by name and file.

.. tab-set::

   .. tab-item:: C++

      .. code-block:: cpp

         auto df = View::from_file("trace.pfw.gz")
                       .group_by({GroupKey::name()})
                       .agg({{AggOp::Concurrency, "dur", "concurrency"},
                             {AggOp::Active, "dur", "active"}})
                       .collect()
                       .get();

   .. tab-item:: Python

      .. code-block:: python

         from dftracer.utils import TraceViewer, AggOp

         df = (TraceViewer("trace.pfw.gz")
               .group_by("name")
               .agg("concurrency", "active")   # or AggOp.CONCURRENCY, AggOp.ACTIVE
               .collect())

Occupancy is computed during the parallel scan from a bounded per-bucket
coverage mask, so it streams in constant memory and merges across files and
ranks the same way the other aggregates do (see
:doc:`../scale/distributed-aggregation`). ``busy`` is an upper bound on the true
interval union that tightens as the time bucket shrinks; add
``time_bucket(interval_us)`` to set that resolution.

Typed aggregate specs
---------------------

The Python spec strings above are convenient but a type checker cannot validate
them: ``"sum:dur"`` is an ordinary ``str``, so a typo in the op (``"sim:dur"``)
is caught only at run time. For a checked spec, build it from the ``AggOp`` enum
instead, which ``agg`` accepts anywhere a string is accepted:

.. code-block:: python

   from dftracer.utils import AggOp

   AggOp.SUM.of("dur")   # -> "sum:dur"; a wrong member name is a type error
   AggOp.BUSY.of()       # -> "busy"; the field-less ops take no field

The ``F`` expression builder (``F("dur").sum()``) is the other checked form; see
the ``F`` examples above.

Time-bucketed counters
----------------------

Add ``time_bucket(interval_us)`` to bucket events by time as well as by group -
one aggregated row per (group, bucket). See :doc:`../data/time-windows`. To emit
the rollup as a re-indexable ``ph="C"`` counter trace instead of collecting it:

.. tab-set::

   .. tab-item:: C++

      ``export_counters`` writes each aggregated row as a counter event to an
      ``ExportSink``.

      .. code-block:: cpp

         View::from_file("trace.pfw.gz")
             .group_by({GroupKey::cat()})
             .time_bucket(1000)
             .agg({{AggOp::Count, "", "n"}})
             .export_counters(sink)
             .get();

   .. tab-item:: Python

      .. code-block:: python

         (TraceViewer("traces/")
          .group_by("cat")
          .time_bucket(1000)
          .agg("count"))
         # collect() for a DataFrame, or export to a counter trace.

Counters without naming the fields
----------------------------------

For counter traces where the numeric fields are not known up front,
``agg_numeric_args`` aggregates every numeric ``args.*`` field as a per-group
mean, one value column per discovered field.

.. tab-set::

   .. tab-item:: C++

      .. code-block:: cpp

         auto df = View::from_file("counters.pfw.gz")
                       .group_by({GroupKey::cat()})
                       .agg_numeric_args()
                       .collect()
                       .get();

   .. tab-item:: Python

      .. code-block:: python

         df = TraceViewer("counters.pfw.gz").group_by("cat").agg_numeric_args().collect()

``F.any.mean()`` in an ``agg`` call is the same thing (``.agg(F.any.mean())`` ==
``.agg_numeric_args()``), so it composes with named aggregates in one call.

Custom metrics (C++)
--------------------

When the built-in aggregate vocabulary cannot express a metric (custom
bucketing, a representative value, a side output), fold over the raw scanned
batches with ``map_batches``: your ``fold`` runs over each batch on one of
``num_slots`` worker slots, and ``combine`` reduces the per-slot partials into
one. This runs over the same index-pruned parallel scan the built-in aggregates
use.

.. code-block:: cpp

   auto result = View::from_file("trace.pfw.gz")
                     .map_batches<MyPartial>(fold, combine, /*num_slots=*/8)
                     .get();

This is a C++ / native surface; there is no Python equivalent. From Python, use
the plugin host (see :doc:`../../plugins`) or aggregate with the built-in specs
above.

See also
--------

- :doc:`statistics` for distributions, percentiles, and sketches.
- :doc:`../data/time-windows` for the time-bucket window model.
- :doc:`../core/query-dsl` for the filter you push down before aggregating.
- :doc:`../../trace-viewer` for the full viewer walkthrough.
