:description: Reference for the C++ query DSL: the predicate IR, the field builder, the string codec, and the evaluator that runs predicates over columns.

Query
=====

.. seealso::

   :doc:`../concepts/indexing-and-pushdown` for how predicates push down into
   the index, and :doc:`../c_api/query` for the flat C ABI.

The query DSL in ``dftracer::utils::query``: the predicate IR, the field builder,
the string codec, and the evaluator that runs predicates over columns.

Type relationships
------------------

The predicate IR node types and how they nest:

.. mermaid:: /_generated/query.mmd

.. include:: /cpp_api/_generated/query.rst.inc
