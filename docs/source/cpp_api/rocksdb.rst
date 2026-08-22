:description: Reference for the RocksDB wrappers behind the on-disk index: database open/close, column families, key encoding, and batched read/write paths.

RocksDB
=======

The RocksDB wrappers in ``dftracer::utils::rocksdb`` that back the on-disk
index: database open/close, column families, key encoding, and the batched read
and write paths used by the indexer.

Type relationships
------------------

Composition among the RocksDB wrappers:

.. mermaid:: /_generated/rocksdb.mmd

.. include:: /cpp_api/_generated/rocksdb.rst.inc
