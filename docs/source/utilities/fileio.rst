:description: The file I/O utilities: synchronous readers and writers plus async line and byte generators for gzip-compressed trace files.

File I/O
==================

File reading, writing, and streaming utilities supporting both synchronous and asynchronous operations.

Synchronous I/O:

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/file_reader_utility.h>
   #include <dftracer/utils/utilities/fileio/binary_file_reader_utility.h>
   #include <dftracer/utils/utilities/fileio/types/chunk_iterator.h>
   #include <dftracer/utils/utilities/fileio/streaming_file_writer_utility.h>
   #include <dftracer/utils/utilities/fileio/lines/streaming_line_reader.h>

Asynchronous Generators:

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/lines/sources/async_indexed_file_line_generator.h>
   #include <dftracer/utils/utilities/fileio/lines/sources/async_indexed_file_bytes_generator.h>
   #include <dftracer/utils/utilities/fileio/lines/sources/async_streaming_gz_line_generator.h>

Types
-----

.. code-block:: cpp

   // Zero-copy byte span (see core_infrastructure)
   class ByteView {
       const std::byte* data() const;
       std::size_t size() const;
       template <typename T> const T* as() const;
   };

   // Text content
   struct Text {
       std::string content;
       bool empty() const;
       std::size_t size() const;
   };

   // Line with position
   struct Line {
       std::string_view content;
       std::size_t line_number;  // 1-based
   };

FileReaderUtility
-----------------

Reads entire file into memory as text. A coroutine functor, not a
``.process()``-style object; call it and await the result.

.. code-block:: cpp

   FileReaderUtility reader;
   Text content = co_await reader(FileEntry{"/path/to/file.txt"});

read_binary_file
-----------------

Free function returning a streaming binary generator that yields zero-copy
``ByteView`` chunks (``dftracer/utils/utilities/fileio/binary_file_reader_utility.h``).

.. code-block:: cpp

   auto gen = read_binary_file("/path/to/file.bin");
   while (auto chunk = co_await gen.next()) {
       process(chunk->as<char>(), chunk->size());
   }

ChunkRange
----------

Lazy, input-iterator-based chunk reading over a plain file
(``dftracer/utils/utilities/fileio/types/chunk_iterator.h``). Only one
chunk is buffered in memory at a time; each dereference exposes a
zero-copy ``ByteView`` into that buffer.

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/types/chunk_iterator.h>

   using namespace dftracer::utils::utilities::fileio;

   ChunkRange chunks("/path/to/large_file.dat", 1024 * 1024);

   for (const ByteView& chunk : chunks) {
       process(chunk);
   }

StreamingFileWriterUtility
--------------------------

Writes ``ByteView`` chunks to a file. ``process()`` is a coroutine and must
be awaited; ``close()`` is synchronous.

.. code-block:: cpp

   StreamingFileWriterUtility writer("/output/file.dat");

   for (const auto& chunk : data_chunks) {
       co_await writer.process(chunk);
   }

   writer.close();
   std::cout << "Wrote " << writer.total_bytes() << " bytes\n";

StreamingLineReader
-------------------

Async line reading for gzip trace files, auto-selecting indexed random
access (when a ``.dftindex`` sidecar is given and exists) or single-pass
streaming decompression otherwise. There is no plain-text mode; only
``.gz`` input is recognized.

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/lines/streaming_line_reader.h>

   using namespace dftracer::utils::utilities::fileio::lines;

   auto config = StreamingLineReaderConfig()
       .with_file("trace.pfw.gz")
       .with_index("trace-root/.dftindex")  // omit for streaming decompression
       .with_line_range(1, 1000);           // omit for the whole file

   auto gen = StreamingLineReader::read_async(config);
   while (auto line = co_await gen.next()) {
       std::cout << line->line_number << ": " << line->content << "\n";
   }

Asynchronous File I/O
---------------------

Async generators provide non-blocking line and byte reading using C++20
coroutines. They are ideal for high-concurrency scenarios and integrating
with async task pipelines. Only gzip-compressed input (``.pfw.gz``) is
supported; there is no plain-text generator.

**Indexed (Compressed) Files**

Read lines from a ``.dftindex``-backed archive asynchronously:

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/lines/sources/async_indexed_file_line_generator.h>

   auto config = IndexedFileLineIteratorConfig()
       .with_file("trace.pfw.gz", "trace-dir/.dftindex")
       .with_line_range(1, 1000);

   auto gen = async_indexed_file_lines(config);
   while (auto line = co_await gen.next()) {
       process(*line);
   }

**Indexed Files by Byte Range**

Read lines within a byte range from indexed archives:

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/lines/sources/async_indexed_file_bytes_generator.h>

   auto reader = ReaderFactory::create("trace.pfw.gz", "trace-dir/.dftindex");
   auto gen = async_indexed_file_bytes(reader, 1000, 5000);  // bytes 1000-5000
   while (auto line = co_await gen.next()) {
       process(*line);
   }

**Streaming Gzip Decompression**

Read lines from ``.gz`` files without building an index, using streaming decompression:

.. code-block:: cpp

   #include <dftracer/utils/utilities/fileio/lines/sources/async_streaming_gz_line_generator.h>

   // Decompress and stream lines without building a sidecar index
   auto gen = async_streaming_gz_lines("data.pfw.gz");
   while (auto line = co_await gen.next()) {
       process(*line);
   }

   // With line range filtering
   auto gen = async_streaming_gz_lines("data.pfw.gz", 100, 200);  // lines 100-200
   while (auto line = co_await gen.next()) {
       process(*line);
   }

Parallel Writers
----------------

Layout-aware parallel writers for multi-worker output. The ``ParallelWriter``
interface is implemented by three concrete layouts under
``fileio/parallel/``:

- **StripedWriter** - single output file, atomic-offset ``pwrite`` per
  worker. Used on local FS and PFS without padded stripes.
- **PaddedStripedWriter** - single output file where each worker chunk is
  padded to a full PFS stripe so per-stripe writes never cross workers.
  Recommended for Lustre/GPFS when the stripe size is at least
  ``MIN_PADDED_STRIPE_BYTES`` (1 MiB).
- **ShardedWriter** - N output files, one per worker, glob-named by
  ordinal. Used on NFS where atomic-offset ``pwrite`` is not reliable.

.. code-block:: cpp

    #include <dftracer/utils/utilities/fileio/parallel/parallel_writer.h>
    #include <dftracer/utils/utilities/fileio/parallel/layout.h>

    using namespace dftracer::utils::utilities::fileio::parallel;

    auto info   = detect_layout("/lustre/.../output.pfw.gz");
    auto sizing = compute_writer_sizing(info, /*baseline_workers=*/64,
                                        /*default_flush=*/4 << 20,
                                        /*headroom=*/1 << 20,
                                        /*padded=*/true);

    WriterConfig cfg{
        .layout = info.layout,
        .stripe_size = info.stripe_size,
        .gzip = true,
    };
    auto writer = make_writer(cfg);
    co_await writer->open("output.pfw.gz", sizing.num_workers,
                          /*gzip_extension=*/true, scope);

    co_await writer->write_header(header_bytes);
    co_await writer->write_chunk(worker_id, chunk_bytes);
    auto member = writer->last_member(worker_id);  // offset+length of the gzip member
    co_await writer->write_footer(footer_bytes);
    co_await writer->close();

The writer collects per-chunk ``MemberSpan`` entries (offset + length of
each independently decompressable gzip member) and exposes them via
``member_layout()`` after close. ``shard_base_offsets()`` remaps shard-local
offsets to merged-file offsets for sharded layouts.

Layout detection (``detect_layout``) classifies a path's filesystem as
Lustre, GPFS, BeeGFS, NFS, or LOCAL and picks ``SHARDED`` on NFS,
``STRIPED`` elsewhere; ``compute_writer_sizing`` caps worker count at the
PFS stripe count and sets ``flush_threshold`` to the stripe size for
padded layouts so each compressed flush coalesces into one stripe.

.. note::

    Compressor generators consumed by the parallel writer are wrapped in
    smart pointers (``std::unique_ptr<ManualStreamingCompressorUtility>``)
    so they can be moved across coroutine frames without leaking the
    underlying zlib stream.

Async vs Synchronous
--------------------

Use async generators when:

- Integrating with coroutine-based pipelines (TaskGraph, Channel-based streaming)
- Processing multiple files concurrently without blocking threads
- Operating in high-concurrency environments (many tasks sharing thread pools)

Use synchronous readers when:

- Sequential file processing is acceptable
- Working outside of coroutine contexts
- Simpler error handling is preferred (no need to handle resumable failures)
