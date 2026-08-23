:description: Reference for the async I/O backend: an IoBackend over io_uring and kqueue with a portable fallback, and the awaitables that suspend on completion.

Async I/O
=========

.. seealso::

   :doc:`../concepts/async-runtime` for how the I/O backend integrates with the
   scheduler.

The async I/O backend in ``dftracer::utils::io``: an ``IoBackend`` abstraction
over io_uring (Linux) and kqueue (macOS/BSD) with a portable fallback, and the
awaitables that suspend a coroutine on I/O completion.

.. include:: /cpp_api/_generated/io.rst.inc
