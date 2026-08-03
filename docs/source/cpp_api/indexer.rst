Indexer Components
==================

.. seealso::

   For complete class and member documentation, see the
   :doc:`API Reference <api/utilities/indexer>`.

Indexing and searching functionality for compressed trace files.
All classes are in the ``dftracer::utils::utilities::indexer`` namespace.

.. mermaid:: ../_generated/indexer.mmd

Overview
--------

The indexer module provides a root-local ``.dftindex`` RocksDB store for
efficient random access to compressed trace files. The store keeps index data
in dedicated column families:

- **Checkpoints**: Byte offsets and decompression state for random access
- **Bloom filters**: Per-chunk probabilistic membership tests for event filtering
- **Chunk statistics**: Per-chunk event counts, timestamps, duration distributions

Getting Started
---------------

Build an index for a compressed trace file using the fluent configuration API:

.. code-block:: cpp

    #include <dftracer/utils/utilities/indexer/index_builder_utility.h>

    using namespace dftracer::utils::utilities::indexer;

    std::vector<std::string> dims(DEFAULT_BLOOM_DIMENSIONS.begin(),
                                  DEFAULT_BLOOM_DIMENSIONS.end());

    auto config = std::make_shared<IndexBuildBatchConfig>();
    config->file_paths = {"trace.pfw.gz"};
    config->index_dir = "/tmp/indexes";
    config->checkpoint_size = 32 * 1024 * 1024;
    config->bloom_dimensions = dims;

    IndexBuildBatchResult result =
        co_await IndexBatchBuilderUtility::process(&scope, config);

    // result.indexed / result.skipped / result.failed and result.total_events
    // hold aggregate counts; result.results carries a per-file IndexBuildResult.

Once an index exists, open it directly with ``IndexDatabase`` to query bloom
filters or chunk statistics:

.. code-block:: cpp

    #include <dftracer/utils/utilities/indexer/index_database.h>

    IndexDatabase db(result.results[0].index_path);
    int file_id = db.find_file("trace.pfw.gz");

    // Query time bounds across all chunks
    auto bounds = db.query_time_bounds(file_id);

    // Query bloom filters for a specific dimension
    auto blooms = db.query_chunk_bloom_filters(file_id, "name");


``DEFAULT_BLOOM_DIMENSIONS`` (7 fields) is the default indexed set;
``DEFAULT_EXTRA_DIMENSIONS`` (ret, count, offset, epoch, step) names the
additional argument fields available for dimension statistics.

IndexBuildResult
----------------

Per-file build outcome; ``IndexBuildBatchResult::results`` carries one per
input file. Contains:

- ``index_path`` -- path to the produced ``.dftindex`` store
- ``file_path`` -- the indexed trace file
- ``success`` / ``was_skipped`` / ``index_created`` -- outcome flags
- ``events_processed`` / ``chunks_processed`` / ``total_lines`` -- build statistics
- ``error_message`` -- non-empty on failure

IndexDatabase
-------------

RocksDB-backed handle over the root-local ``.dftindex`` store that holds all
index data across column families. Call ``init_schema()`` once (idempotent) to
create the column families. Writes go through a batched writer context
obtained from ``begin_write()``; the read-only query API is called directly.

Read query API (selected):

- **Bloom data**: ``query_chunk_bloom_filters(file_id, dimension)``, ``query_file_bloom_filter(file_id, dimension)``
- **Chunk statistics**: ``query_chunk_statistics(file_id)``, ``query_time_bounds(file_id)``
- **Dimension stats**: ``query_chunk_dimension_stats(file_id)``
- **Hash tables**: ``resolve_hash(type, hash)``, ``resolve_name_to_hash(type, name)`` where ``type`` is ``IndexDatabase::HashType`` (``FILE``/``HOST``/``STRING``/``PROC``)
- **File lookup**: ``find_file(path)``, ``get_file_info_id(path)``

.. code-block:: cpp

    IndexDatabase db(result.index_path);
    db.init_schema();

    // Batched writes go through a writer context.
    auto writer = db.begin_write();  // IndexDatabaseWriterContext
    // ... index folds emit records into *writer ...

    // Reads use the query API directly.
    int file_id = db.find_file("trace.pfw.gz");
    auto stats = db.query_chunk_statistics(file_id);

IndexVisitor
------------

Abstract visitor interface for index building passes. Implement this to add
custom indexing logic during the checkpoint-by-checkpoint scan. The driver
calls, in order:

1. ``begin(std::size_t num_checkpoints)`` -- called once before the scan starts
2. ``on_checkpoint(std::size_t idx)`` -- ``CoroTask<void>``, called at each checkpoint boundary
3. ``on_chunk(const char* data, std::size_t len, std::size_t checkpoint_idx)`` -- ``CoroTask<void>``; one decompressed chunk of a member's plaintext (a batch of lines), which the consumer splits or parses as it needs
4. ``finalize(IndexDatabaseWriterContext& writer, int file_id)`` -- called once after the scan to persist results

Optional overrides support backpressure and buffering: ``flush()`` and
``drain_pending()`` return ``CoroTask<void>``, and ``wants_drain()`` (default
false, polled after each ``on_chunk``) hints that ``drain_pending()`` should run
to apply backpressure when a downstream channel is full.

Indexer
-------

The only low-level indexer is ``internal::Indexer`` (in the
``dftracer::utils::utilities::indexer::internal`` namespace), an abstract base
for the per-archive indexer implementations (gzip, tar.gz). Application code
should use ``IndexBatchBuilderUtility`` rather than the internal indexer
directly.

IndexBatchBuilderUtility
------------------------

Processes a list of files in parallel through the fold-based index pipeline
against a shared ``IndexDatabaseWriterContext``, yielding an
``IndexBuildBatchResult`` with aggregated metrics. Configured via
``IndexBuildBatchConfig`` (file list, parallelism, checkpoint size, bloom
toggle, shared sink).

IndexBuildBatchConfig
~~~~~~~~~~~~~~~~~~~~~

Configuration struct for ``IndexBatchBuilderUtility``: file slices, output
directory, checkpoint size, bloom flag, and the shared
``IndexBatchSink`` (typically an ``IndexDatabaseWriterContext``) that
receives encoded batches from all workers.

IndexDatabaseWriterContext
--------------------------

Implements ``IndexBatchSink`` and owns a thread-safe writer pipeline into a
RocksDB-backed ``IndexDatabase``. Workers in ``IndexBatchBuilderUtility``
submit encoded index batches to this context, which serializes them into
checkpoint, bloom, and statistics column families.

IndexResolverUtility
--------------------

Resolves a directory or file list into a set of ``FileWorkItem`` entries by
opening or building per-file indexes and emitting line-range work items
suitable for parallel scan / aggregation / replay pipelines. Defined in
``dftracer/utils/utilities/composites/dft/indexing/index_resolver_utility.h``.
