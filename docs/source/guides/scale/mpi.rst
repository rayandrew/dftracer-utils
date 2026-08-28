:description: Build a folded flamegraph over a trace set too large for one process by folding an arena partial per MPI rank with dftracer_view --flamegraph.

Run across ranks with MPI
=========================

.. admonition:: Goal
   :class: goal

   Build a folded flamegraph over a trace set too large for one process by
   spreading the work across MPI ranks. Under ``mpirun`` each rank folds an
   arena partial over its slice of the input, ranks all-gather the partials,
   and rank 0 merges them into the final output.

This is CLI-only and requires an MPI build (see below). The flamegraph fold is
the one distributed binary; the aggregation and view engines scale within a node
instead (see :doc:`distributed-index` for cross-node indexing via dask).

Build with MPI enabled
----------------------

MPI support is off by default. Configure with the option on:

.. code-block:: bash

   cmake -S . -B build -DDFTRACER_UTILS_ENABLE_MPI=ON
   cmake --build build

The same ``dftracer_view`` binary distributes under MPI; there is no separate
``_mpi`` binary.

Invoke it
---------

Launch ``dftracer_view --flamegraph`` under ``mpirun`` (or your scheduler's
launcher) with the rank count:

.. code-block:: bash

   mpirun -n 32 dftracer_view --flamegraph --files ./traces/*.pfw.gz -o flamegraph.ndjson

Each rank folds an arena partial over its slice of the input; the ranks
all-gather the partials and rank 0 merges them into the final folded flamegraph
(NDJSON). The output path must live on a filesystem visible to rank 0.

See also
--------

- :doc:`../../cli` - ``dftracer_view --call-tree`` / ``--flamegraph`` and the
  full option list.
- :doc:`distributed-index` - build an index across a dask cluster.
- :doc:`../runtime/memory-budget` - ``suggested_nodes`` from the budget advice
  sizes how wide to run.
