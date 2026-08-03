Python API Reference
====================

This section contains the Python API documentation for dftracer utilities.

.. toctree::
   :maxdepth: 2
   :caption: Python Modules:

   trace_viewer
   query
   utilities
   runtime
   reader
   indexer
   dfanalyzer

Module Overview
---------------

The dftracer utilities Python package provides the following main modules:

- :doc:`trace_viewer` - **TraceViewer**, the primary lazy, Arrow-first query API (filter, group_by, agg, collect)
- :doc:`query` - The filter DSL used by TraceViewer
- :doc:`utilities` - Utility bindings (aggregation, comparison, metadata)
- :doc:`runtime` - Coroutine runtime and Dask integration
- :doc:`reader` - Lazy JSON object type
- :doc:`indexer` - Indexing and searching capabilities
- :doc:`dfanalyzer` - dfanalyzer bridge: View-based HLM, Arrow IPC, index build
