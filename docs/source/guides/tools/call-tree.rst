:description: Reconstruct the parent-child call hierarchy from traces, get statistics, and emit Chrome Tracing JSON with dftracer_call_tree or the C++ API.

Build a call tree
=================

.. admonition:: Goal
   :class: goal

   Reconstruct the parent-child call hierarchy from a set of traces, get
   statistics off it, and emit Chrome Tracing JSON. Use the ``dftracer_call_tree``
   binary for a one-shot run, or the C++ ``CallTree`` API when you want to walk the
   tree in your own code.

This is C++ and CLI. There is no Python binding for ``CallTree``.

Command line
------------

``dftracer_call_tree`` takes trace files or directories and emits a merged call
tree as Chrome Tracing JSON.

.. code-block:: bash

   # Build from a directory (recursively) and write JSON
   dftracer_call_tree ./traces -r -o call_tree.pfw

   # gzip the output
   dftracer_call_tree ./traces -r -o call_tree.pfw --gzip

   # Analyze without writing a file
   dftracer_call_tree ./traces --no-save

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Flag
     - Default
     - Meaning
   * - ``inputs``
     - (required)
     - One or more trace files (``.pfw.gz``) or directories.
   * - ``-r``, ``--recursive``
     - off
     - Recurse into input directories.
   * - ``-o``, ``--output``
     - ``<basename>.pfw``
     - Output JSON path (Chrome Tracing).
   * - ``--no-save``
     - off
     - Build and analyze, but write nothing.
   * - ``--gzip``
     - off
     - gzip the output (appends ``.gz`` if needed).

Load the JSON into any Chrome Tracing viewer (``chrome://tracing`` or
Perfetto).

The CallTree C++ API
--------------------

``CallTree`` (header ``dftracer/utils/call_tree/call_tree.h``, namespace
``dftracer::utils::call_tree``) loads traces, generates the tree, then exposes
traversal, lookup, and statistics.

.. code-block:: cpp

   #include <dftracer/utils/call_tree/call_tree.h>
   using namespace dftracer::utils::call_tree;

   CallTree tree;
   tree.load_from_directory("./traces", "*.pfw.gz");   // pattern optional
   tree.generate();

   if (tree.is_generated()) {
       tree.print_statistics();
       tree.print_depth_first(/*max_depth=*/0);   // 0 == unlimited

       CallTreeStats stats = tree.get_statistics();
       // stats.total_nodes, num_levels, num_leaf_nodes, max_depth, ...
   }

Walk the tree by process and thread, or pull every node flat:

.. code-block:: cpp

   for (auto pid : tree.get_process_ids()) {
       for (auto tid : tree.get_thread_ids(pid)) {
           for (const auto& root : tree.get_root_nodes(pid, tid)) {
               visit(root);   // CallTreeNodeInfo
           }
       }
   }

   std::vector<CallTreeNodeInfo> all = tree.get_all_nodes();
   CallTreeNodeInfo n = tree.get_node_by_id(some_id);

Each ``CallTreeNodeInfo`` carries ``id``, ``name``, ``category``,
``start_time_us``, ``duration_us``, ``level``, ``parent_id``, ``num_children``,
``children_ids``, and an ``args`` map. Node and level counts come from
``get_statistics()`` (``CallTreeStats``), not a ``size()`` method. Reuse a
``CallTree`` for another dataset with ``clear()``.

Distributed builds
------------------

For datasets too large for one process, ``dftracer_call_tree_mpi`` partitions
PIDs across ranks, emits per-rank JSON shards, and merges on rank 0. See
:doc:`../scale/mpi`.

See also
--------

- :doc:`../serving/http-server` - the server exposes a merged call tree at
  ``/api/viz/calltree`` without a separate build.
- :doc:`../scale/mpi` - the ``_mpi`` driver for large datasets.
