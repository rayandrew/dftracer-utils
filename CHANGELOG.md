# Changelog

All notable changes to dftracer-utils are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project aims to adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

<!-- changelog-body -->

## [Unreleased]

### Added

- Interactive web trace viewer gains a counter timeline track (with malformed-value
  handling), per-counter pid/tid breakdown, bounded-density serving, and active-time
  statistics.
- Rectangle selection in the viewer scopes analysis to a time range, lanes, and rows;
  aggregated (`ph=3`) events are visualized with uniform extrapolation and labeled
  "aggregated" in tooltips.
- DLIO event category and name can be remapped through an event-map file.
- Python: `DaskTraceViewer` is exported from the `dask` module; new `TraceViewer`/`View`
  bindings expose the query DSL and portable C-API glue.
- Query DSL: subsumption-based simplification, string-match operators, and
  `resolved.*` virtual fields.
- Materialized views with rollups and tier-served aggregation; a new `AggregationFold`
  and extended aggregation operators.
- Concurrency-aware occupancy aggregates `busy`, `concurrency`, `utilization` and
  `active` measure wall-clock busy time and parallelism instead of double-counting
  overlapping durations. They are field-less (always over `dur`), work per group and
  time bucket, and merge across files and ranks; exposed on the `View`/`TraceViewer`
  `agg`, the `dftracer_view --agg` CLI, and the typed `AggOp` enum.
- Multi-member split, merge, and reorganize for intra-file parallelism.
- Humanized sizes and durations: every `<bytes>` CLI flag accepts a unit suffix
  (`64MB`, `1.5GiB`, `512KB`, `8kb`; 1024-based, `b` is bits) and every `<s>` flag
  accepts a duration suffix (`30s`, `5m`, `1.5h`); a bare number keeps the flag's
  legacy unit. The Python API's byte/duration arguments accept a number or a string.
- `dftracer_server --timeout` bounds server uptime and then shuts down gracefully
  (accepts a humanized duration; `0`, the default, disables it).
- Full type coverage across the public Python surface (`Series`/`DataFrame`/viewers,
  the jit DSL, and a generic `TaskHandle`).

### Changed

- `Runtime.submit()` no longer accepts a `name=` argument; the task name is derived
  automatically from the callable and its call-site source location.

- Indexer scan core reworked to a fold-fusion, batch-native design; server query,
  viz caching, and frontend paths updated to match.
- `--select` is now applied to raw event queries (SQL-style column projection).
- Substantial internal refactors: async runtime/pipeline/cache, trace reader and
  compression I/O, RocksDB manager lifecycle, shared primitives and umbrella headers,
  domain composites (visitor pattern retired), and CLI binaries/trace generators.

### Removed

- **Breaking:** the legacy per-line visitor scan path is retired in favor of the
  batch-native fold-fusion indexer.

### Fixed

- Indexer recovers a chunk whose batch parse collapses on a single bad line, closes
  cached index handles before removing a stale index, and adds a sharded immutable
  index with a full-path-keyed registry (breaking on-disk change).
- Comparator now defaults to comparing all events rather than only POSIX/STDIO.
- Server: zero-fill simdjson padding in the viz summary fold.
- CI reworked onto a shared Flux allocation with GitLab pipelines mirroring the GitHub
  workflows; wheel version now sees tags via full-history fetch.

## [0.0.12] - 2026-07-13

### Added

- Interactive web trace viewer: a SolidJS canvas frontend backed by a new server
  visualization API, with timeline node grouping (#93), app-span events, and a
  timelapse axis for multi-run traces.
- `dftracer_stats` now shows histogram bounds.
- Indexer detects and rebuilds stale indexes when the source trace changes; this
  rebuild is honored across read consumers and the server.

### Changed

- Leaner wheels; CI compiles RocksDB once and persists ccache for faster builds.

### Fixed

- I/O: scatter-gather writes now complete past the `IOV_MAX` limit.
- Comparator no longer miscounts files.
- Fixed a deadlock in multi-process dfanalyzer and a stale-index / app-span-complement
  run break in the server.
- `dfanalyzer` distributed indexing skips already-indexed files.
- `lcov` coverage compatibility fix.

## [0.0.11] - 2026-07-05

### Added

- New `dftracer_validate` CLI tool.
- `Result<T>` error model with `DFT_TRY`, and Python bindings that map `ErrorCode`
  to typed exceptions.
- Comparator gains a DLIO preset.
- ARM64 (aarch64) build support in CI and Valgrind memory checking for the C++ and
  Python test suites.

### Changed

- **Breaking:** core runtime and I/O consolidated around the new `Result<T>` model;
  single-op utilities migrated to `Result<T>`, with god-file splits and broad dedup.
- Python bindings deduplicated; GIL ordering fixed and an `atexit` shutdown added.

### Fixed

- Executor shutdown lost-wakeup deadlock.
- Object-pool Treiber-stack races (memory ordering plus a DWCAS optimization to fix
  a segfault).
- Noisy aarch64 warnings silenced.

## [0.0.10] - 2026-06-08

### Added

- Distributed HLM filtering and time bucketing.

## [0.0.9] - 2026-06-07

### Fixed

- Wheel build: updated Xcode version for macOS packaging.

## [0.0.8] - 2026-06-05

### Added

- Portable-wheel support via the `DFTRACER_UTILS_LOCAL_PACKAGES` option.

### Fixed

- Portable `to_chars_double` fallback for macOS and improved zstd handling.

## [0.0.7] - 2026-05-22

### Changed

- CI cleanup: removed ccache setup and keyed the ccache on `matrix.os`.

## [0.0.6] - 2026-05-21

Large feature release: the async core, indexing, aggregation, query, and server
capabilities that define the current engine landed here.

### Added

- **Query DSL** for filtering trace events, wired into raw byte reading with
  file-level chunk skipping.
- **HTTP server** (from-scratch async) exposing trace query APIs, with streaming
  iovec-based responses.
- **Arrow data interchange** via nanoarrow (Arrow C Data Interface), zero-copy across
  the Python boundary.
- **Python runtime and bindings**: `Runtime` with async `submit()`/`TaskHandle` and
  Python-callable support, streaming iterators, a `StreamingUtility` with Arrow output,
  the Indexer wired to the Runtime, and a Dask plugin.
- **Aggregator** (`dftracer_aggregator`) with profile and system-event aggregation,
  custom metrics, offset tracking, and time-bucket persistence.
- **Comparator** (`dftracer_comparator`) for comparing trace metrics, injecting trace
  metadata into root SUMMARY rows.
- Bloom-filter multi-index, manifest (`.midx`) sidecars, a DFT view system with bloom
  filtering and predicate support, and an `index_threshold` to skip bloom/manifest for
  small files.
- Parallel statistics with DDSketch and log2 histograms.
- Replay (`dftracer replay`) with provenance-based, semantic trace reorganization.
- Call-tree utility with MPI support.
- DLIO config generation.
- C++20 coroutine core: async I/O backends (io_uring/kqueue), pipelines, task system,
  heterogeneous `when_all`/`when_any`, and channels.

### Changed

- **Breaking:** utilities migrated to the query DSL; core string/memory optimizations
  and a unified index infrastructure.
- Index storage migrated from SQLite to RocksDB.
- Replaced the `nonstd::span` polyfill with C++20 `std::span`; removed the xxhash
  dependency in favor of FNV-1a and standardized hashing.
- Broad performance work across parsing, scanning, serialization, compression,
  zero-copy I/O, and the aggregation pipeline.

### Fixed

- Coroutine foundation rewrite fixing `channel`/`when_all` memory leaks, `when_all`
  await-ready races, and channel lifetime management.
- gzip inflater trailer/window-reset handling and double-counting when an index
  already exists.
- Type-safety, lifetime, concurrency, and portability issues; endianness and query
  fallback handling.

## [0.0.5] - 2025-10-21

### Fixed

- Wheels now link their libraries correctly.
- Promise fulfillment and exception handling in pipeline executors.
- Performance when continuing to read from a gzip stream; default checkpoint size now
  comes from a defined constant.
- Dropped a soon-to-be-deprecated macOS version from the build matrix.

## [0.0.4] - 2025-10-20

### Fixed

- Race condition from mutating shared state.
- Source distribution now excludes unnecessary files and includes `cmake`/`tests`;
  corrected CLI options in the docs.

## [0.0.3] - 2025-10-20

### Fixed

- Corrected the versioning scheme and the link to the DFTracer GitHub repo.
- Excluded unnecessary files from the source distribution and fixed build scripts for
  the non-publishing path.

## [0.0.2] - 2025-10-19

### Fixed

- Added `setup.py` to fix a Python versioning issue and shipped missing scripts needed
  for Python publishing.

## [0.0.1] - 2025-10-19

Initial release.

### Added

- Core trace tooling: `dftracer_index` (gzip and tar indexers), `dftracer_merge`,
  `dftracer_split` (with `--verify`), `dftracer_event_count`, `dftracer_info`, and
  `pgzip`.
- A coroutine-driven task scheduler, executor, and pipeline framework with progress
  callbacks and multiple-output support.
- gzip block-boundary-aware reading with prefix-based hashes and checkpointing.
- Python packaging and Sphinx-based documentation with C++ API reference.
