:description: Reference for the DAG builder: TaskGraph and TaskGroup with fan-out, fan-in, map, and reduce edges, plus the node factory functions.

Task Graph
==========

.. seealso::

   :doc:`../pipeline` for how task graphs run on the pipeline, and
   :doc:`../guides/pipelines/patterns` for fan-out/fan-in patterns.

The DAG builder in ``dftracer::utils::task_graph``: ``TaskGraph`` and
``TaskGroup`` with fan-out, fan-in, map, and reduce edges, plus the factory
functions that construct nodes.

Type relationships
------------------

How the task-graph builder relates to the core Task it composes:

.. mermaid:: /_generated/task_graph.mmd

.. include:: /cpp_api/_generated/task_graph.rst.inc
