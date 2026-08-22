:description: What the .dftindex store holds - checkpoints, zone-map statistics, bloom filters - and how a query prunes chunks before decompressing them.

Indexing and predicate pushdown
================================

What this explains: what the ``.dftindex`` store holds, and why a query can
skip most of a trace file without decompressing it.

The index is built once, queried many times
--------------------------------------------------

Indexing (``trace::indexing/``) reads a ``.pfw.gz`` trace once and writes a
per-directory RocksDB store (a ``.dftindex``) that later queries consult
instead of re-reading the trace. The store holds three kinds of information,
built together by the chunk indexer (``chunk_indexer_utility.h``):

- **Checkpoints** mark where each compressed member of the gzip file starts,
  so a chunk can be located and decompressed on its own instead of requiring
  a sequential decompress of everything before it.
- **Chunk statistics** (``chunk_statistics.h``) are a zone-map per chunk: a
  compact min/max/count summary (timestamp range, duration range, event
  count) and, at a finer grain, per-sub-chunk zone-maps bucketed every
  ``sub_chunk_events`` events. A zone-map answers "could this range contain a
  match" without looking at a single event.
- **Bloom filters** (``scalable_bloom_filter.h``, ``bloom_filter.h``) give
  approximate set-membership over the dimensions configured for indexing -
  name, category, pid, tid, host/file/exec hash, and any extra
  ``args.*`` dimensions requested. A bloom filter can say "definitely not
  present" for a value with no false negatives, which is exactly the
  one-sided guarantee pruning needs.

Predicate pushdown: prune before you decompress
------------------------------------------------------

A query does not scan a trace file end to end and then filter. Before any
chunk is decompressed, the chunk pruner (``chunk_pruner_utility.h``)
evaluates the query's predicates against the index and produces the list of
chunks that *may* match - every other chunk is skipped outright. Two things
make this sound rather than approximate-and-wrong:

- **Bloom filters only prune, never confirm.** A negative bloom test is
  certain (no false negatives), so a chunk can be safely dropped on it; a
  positive test means "maybe," so the chunk is still decompressed and the
  predicate is evaluated for real against the parsed events. The bloom filter
  reduces work; it never changes the answer.
- **Numeric zone-map ranges are extracted conservatively.** A range on a
  field (say ``dur > 1000``) is only usable for pruning when it is pulled
  from a top-level AND of comparisons; an OR or NOT that also references the
  field would let a bucket that fails the range still contribute events that
  the query wants, so pushdown backs off to "unconstrained" for that field
  rather than risk dropping a match (``sub_chunk_prune.h``).

This is the same principle a column-store's zone-maps and a Bloom-filtered
LSM tree both rely on: cheap, sound negatives eliminate the expensive path
(decompression and JSON parsing) for the overwhelming majority of chunks a
query does not care about, and the expensive path only ever runs on chunks
that survived every prune.

.. mermaid::

   graph LR
       Query["Query predicate"] --> Bloom["Bloom filter test<br/>(negative = skip)"]
       Query --> Zone["Zone-map range test<br/>(sound only under AND)"]
       Bloom --> Prune["Chunk pruner"]
       Zone --> Prune
       Prune --> Candidates["Candidate chunks"]
       Candidates --> Scan["Fused scan<br/>(actual decompress + evaluate)"]

Resolved fields: querying by name without an index scan
--------------------------------------------------------------

Trace events reference files, hosts, and executables by an interned hash
(``fhash``, ``hhash``, ``shash``), not by the human-readable string. Writing
a predicate directly against the hash is unusable for a person; writing it
against a live string comparison at scan time would mean decoding every
candidate row before it can even be tested. The index instead keeps a
reverse hash table, and the resolved-field rewriter
(``resolved_field_rewriter.h``) rewrites a predicate on ``resolved.fpath`` /
``r.fpath`` (and the ``hostname``/``exec``/``cmd`` equivalents) into a
concrete ``fhash in [...]`` clause before the query ever reaches the pruner
or the scan. An exact match does a direct reverse lookup; a pattern match
(``like``, ``ilike``, a regex, a substring) enumerates the hash table once
and keeps the hashes whose resolved name matches. Either way, the pruner and
the per-event evaluator downstream see an ordinary hash-valued predicate and
need no special case for "this field is actually a name."

See also
--------

- :doc:`fused-scan` for what happens to the chunks that survive pruning.
- :doc:`architecture` for where indexing sits in the overall query path.
- :doc:`../guides/core/query-dsl` for the predicate syntax itself.
- :doc:`../cpp_api/indexer` for the generated indexer API reference.
