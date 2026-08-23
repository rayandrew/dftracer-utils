:description: Build a call tree over a trace set too large for one process by partitioning PIDs across MPI ranks with dftracer_call_tree_mpi.

Run across ranks with MPI
=========================

.. admonition:: Goal
   :class: goal

   Build a call tree over a trace set too large for one process by spreading
   the work across MPI ranks. ``dftracer_call_tree_mpi`` partitions PIDs across
   ranks, each rank writes a Chrome Tracing JSON shard to a shared staging
   directory, and rank 0 merges them into the final output.

This is CLI-only and requires an MPI build (see below). Call-tree construction is
the one distributed binary; the aggregation and view engines scale within a node
instead (see :doc:`distributed-index` for cross-node indexing via dask).

Build with MPI enabled
----------------------

MPI support is off by default. Configure with the option on to build the ``_mpi``
binary:

.. code-block:: bash

   cmake -S . -B build -DDFTRACER_UTILS_ENABLE_MPI=ON
   cmake --build build

Without ``DFTRACER_UTILS_ENABLE_MPI=ON``, ``dftracer_call_tree_mpi`` is not
built.

Invoke it
---------

Launch it under ``mpirun`` (or your scheduler's launcher) with the rank count:

.. code-block:: bash

   mpirun -n 32 dftracer_call_tree_mpi ./traces -o call_tree.pfw --gzip

.. list-table::
   :header-rows: 1
   :widths: 30 24 46

   * - Flag
     - Default
     - Meaning
   * - ``input``
     - (required)
     - Input directory containing trace files.
   * - ``-o``, ``--output``
     - ``call_tree.pfw``
     - Output JSON path (Chrome Tracing).
   * - ``--staging-dir``
     - ``<output>.shards/``
     - Shared-FS staging root for the per-rank shards.
   * - ``--gzip``
     - off
     - gzip the merged output.
   * - ``--keep-staging``
     - off
     - Keep the per-rank shards after the merge.

Each rank owns a slice of PIDs and emits its own shard; rank 0 performs the
final merge, then all ranks hit a barrier. The staging directory must live on a
filesystem visible to every rank. When you run several ranks per node, the
binary divides each rank's worker and I/O threads by the ranks-per-node so the
ranks do not oversubscribe the cores.

See also
--------

- :doc:`../tools/call-tree` - the single-process ``dftracer_call_tree`` and the
  ``CallTree`` C++ API.
- :doc:`distributed-index` - build an index across a dask cluster.
- :doc:`../runtime/memory-budget` - ``suggested_nodes`` from the budget advice
  sizes how wide to run.
