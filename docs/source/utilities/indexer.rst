Indexer
=================

Unified indexing and reading infrastructure for compressed trace files. Builds a sidecar ``.dftindex`` RocksDB store (and optional flat-file SSTs) that enables efficient random access, bloom-filter-accelerated queries, and distributed aggregation, all from a single decompression pass.

.. code-block:: cpp

   #include <dftracer/utils/utilities/indexer/index_builder_utility.h>
   #include <dftracer/utils/utilities/reader/trace_reader.h>

Overview
--------

The indexer writes column families into a shared ``.dftindex`` RocksDB store
(or, for distributed builds, a content-addressed SST staging directory that
is ingested into the store):

- **Checkpoints** - byte offsets and decompression dictionaries for random access
- **Bloom filters** - per-chunk bloom filters for fast event filtering (optional)
- **Chunk statistics** - per-chunk event counts, duration distributions (optional)
- **Aggregation / system metrics** - distributed aggregation CFs populated via
  ``SstFileWriter::Merge`` operands

SST files staged on disk are **content-addressed** (FNV-1a 64-bit fingerprint
over the SST payload) so identical SSTs produced by different ranks collapse
to a single ingest, and re-ingesting is idempotent. String IDs in the
``names`` and ``cats`` CFs are deterministic FNV-1a hashes so the same name
maps to the same id across processes.

IndexBatchBuilderUtility
------------------------

Builds many files in a single pipelined pass. Parses files in parallel
(``parallelism`` workers) and routes their parsed artifacts (bloom rows,
aggregation merge operands, extra-visitor SSTs) to a
write phase. Supports batched flushing (``flush_every_files``) to bound
peak memory, distributed SST sinks via ``sink_factory`` / ``sink_commit``,
preassigned file ids, and per-file gzip-member slicing for cross-rank file
splitting (the MPI driver pre-scans each ``.pfw.gz`` for member boundaries
and assigns disjoint ``[member_begin, member_end)`` ranges to ranks).

.. code-block:: cpp

   #include <dftracer/utils/utilities/indexer/index_builder_utility.h>

   IndexBuildBatchConfig cfg;
   cfg.file_paths = {"a.pfw.gz", "b.pfw.gz", "c.pfw.gz"};
   cfg.index_dir = "/data/.dftindex";
   cfg.parallelism = 16;
   cfg.rebuild_root_summaries = true;
   cfg.flush_every_files = 8;

   auto batch = co_await IndexBatchBuilderUtility::process(scope,
       std::make_shared<IndexBuildBatchConfig>(std::move(cfg)));

IndexDatabaseWriterContext
--------------------------

Implements ``IndexBatchSink`` over a coordinator-owned RocksDB store: each
batch's parsed artifacts are buffered, then committed atomically via
``WriteBatch``. ``IndexDatabaseSstWriterContext`` is the SST-staging
variant used by the distributed indexer; its outputs are content-addressed
SST files later ingested into the coordinator store.

IndexResolverUtility
--------------------

Resolves the index directory for a given trace file, building the index on
demand when ``auto_build_index`` is set. Lives in
``composites/dft/indexing/`` because it depends on the DFT visitor set.

.. code-block:: cpp

   #include <dftracer/utils/utilities/composites/dft/indexing/index_resolver_utility.h>

IndexDatabase
-------------

RocksDB-backed index store (part of the ``.dftindex`` root) with an
additive, idempotent schema across column families.

.. code-block:: cpp

   #include <dftracer/utils/utilities/indexer/index_database.h>

   using namespace dftracer::utils::utilities::indexer;

   IndexDatabase db("trace.pfw.gz.dftindex");
   db.init_schema();  // idempotent; sets up all column families

   // Writes go through a writer context
   auto writer = db.begin_write();

   // Read-only queries
   int fid = db.get_file_info_id("trace.pfw.gz");
   bool has_bloom = db.has_bloom_data(fid);

TraceReader
-----------

Unified reader for all trace file formats. Auto-selects between sequential decompression and indexed random access based on ``.idx`` presence. Supports gzip and plain text files.

Two methods cover all reading modes:

- ``read_lines(ReadConfig)`` - returns parsed ``Line`` objects (``string_view``, zero-copy)
- ``read_raw(ReadConfig)`` - returns raw byte spans (``std::span<const char>``)

``ReadConfig`` controls range (line or byte), alignment, and buffering.

.. code-block:: cpp

   #include <dftracer/utils/utilities/reader/trace_reader.h>

   using namespace dftracer::utils::utilities::reader;

   TraceReader reader({.file_path = "trace.pfw.gz"});

   // Read all lines (default)
   auto gen = reader.read_lines();
   while (auto line = co_await gen.next()) {
       // line->content is string_view, valid until next iteration
   }

   // Line range
   ReadConfig rc;
   rc.start_line = 100;
   rc.end_line = 200;
   auto range = reader.read_lines(rc);

   // Raw bytes - line-aligned, multi-line chunks (fastest for bulk processing)
   auto raw = reader.read_raw();
   while (auto chunk = co_await raw.next()) {
       // chunk is std::span<const char>
   }

   // Raw bytes - single line per yield
   ReadConfig single;
   single.line_aligned = true;
   single.multi_line = false;
   auto line_bytes = reader.read_raw(single);

   // Raw bytes - no line awareness
   ReadConfig raw_cfg;
   raw_cfg.line_aligned = false;
   auto bytes = reader.read_raw(raw_cfg);

   // Byte range
   ReadConfig byte_range;
   byte_range.start_byte = 0;
   byte_range.end_byte = 1024 * 1024;
   auto chunk_gen = reader.read_raw(byte_range);

**ReadConfig to StreamType mapping:**

.. list-table::
   :header-rows: 1

   * - ``read_raw`` flags
     - Internal StreamType
   * - ``line_aligned=true, multi_line=true`` (default)
     - ``MULTI_LINES_BYTES``
   * - ``line_aligned=true, multi_line=false``
     - ``LINE_BYTES``
   * - ``line_aligned=false``
     - ``BYTES``

IndexVisitor
------------

Interface for processing decompressed lines during index building. Implementations receive each line and its checkpoint index.

.. code-block:: cpp

   #include <dftracer/utils/utilities/indexer/index_visitor.h>

   class IndexVisitor {
   public:
       virtual ~IndexVisitor() = default;
       virtual void begin(std::size_t num_checkpoints) = 0;
       virtual void on_checkpoint(std::size_t checkpoint_idx) = 0;
       virtual void on_line(std::string_view line,
                            std::size_t checkpoint_idx) = 0;
       virtual void finalize(IndexDatabase& db, int file_id) = 0;
   };

Built-in event visitors live in ``composites/dft/visitors/`` (they extend
``DftEventVisitor`` and are wrapped by ``DftEventDispatcher``, which
implements ``IndexVisitor``):

- **BloomVisitor** (``composites/dft/visitors/bloom_visitor.h``) - parses
  JSON events, populates bloom filters and chunk statistics
- **AggregationVisitor** (``composites/dft/aggregators/aggregation_visitor.h``)
  - emits per-chunk aggregation and system-metric merge operands

Low-level IndexerFactory
------------------------

Creates checkpoint indexers with automatic format detection (GZIP vs TAR.GZ). Used internally by the index build pipeline.

.. code-block:: cpp

   #include <dftracer/utils/utilities/indexer/internal/indexer_factory.h>

   using namespace dftracer::utils::utilities::indexer::internal;

   auto indexer = IndexerFactory::create(
       "trace.pfw.gz",     // Input file
       "trace.pfw.gz.idx", // Output index path
       32 * 1024 * 1024,   // Checkpoint size (32MB)
       true                // Force rebuild
   );

   co_await indexer->build_async();

   std::size_t num_lines = indexer->get_num_lines();
   auto checkpoints = indexer->get_checkpoints();

Python API
----------

**Indexer:**

.. code-block:: python

   from dftracer.utils import Indexer

   # Checkpoint-only build
   with Indexer("trace.pfw.gz") as indexer:
       indexer.build()
       print(f"Lines: {indexer.get_num_lines()}")

   # Single-pass build with bloom
   with Indexer("trace.pfw.gz", build_bloom=True) as indexer:
       indexer.build()
       assert indexer.has_bloom

   # Incremental: add bloom to existing index
   with Indexer("trace.pfw.gz", build_bloom=True) as indexer:
       indexer.build()  # reuses checkpoints, adds bloom only

   # Wrapper cleanup only; the shared .dftindex store remains available
   indexer = Indexer("trace.pfw.gz")
   indexer.build()
   indexer.close()

   # With explicit Runtime for thread pool control
   from dftracer.utils import Runtime

   with Runtime(threads=8) as rt:
       indexer = Indexer("trace.pfw.gz", build_bloom=True, runtime=rt)
       indexer.build()  # uses rt's thread pool

       # TraceReader can share the same Runtime
       reader = TraceReader("trace.pfw.gz", runtime=rt)
       for line in reader.iter_lines():
           process(line)

**TraceReader:**

.. code-block:: python

   from dftracer.utils import TraceReader

   # Auto-selects sequential vs indexed reading
   reader = TraceReader("trace.pfw.gz")
   lines = reader.read_lines()

   # Line range
   partial = reader.read_lines(start_line=100, end_line=200)

   # Properties
   print(reader.has_index)    # True if .dftindex exists
   print(reader.num_lines)    # precise line count

   # Context manager
   with TraceReader("trace.pfw.gz") as reader:
       for line in reader.read_lines():
           event = json.loads(line)

See Also
--------

- :doc:`/cli` - Command-line tools (``dftracer_index``, ``dftracer_reader``)
- :doc:`composites` - Bloom filter and chunk indexing composites
- :doc:`/cpp_api/dft_indexing` - C++ API reference for indexing classes
