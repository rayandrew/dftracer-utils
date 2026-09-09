:description: The stable plugin C ABI: fill a dftu_plugin struct and reach host services through dftu_host and its dftu_svc_* extension groups.

Plugins
=======

.. seealso::

   :doc:`../plugins` for the plugin guide, :doc:`../jit` to author a plugin in
   Python, and :doc:`../cpp_api/plugins` for the C++ SDK that wraps this ABI.

The stable plugin C ABI. A plugin includes only ``abi.h`` and fills a
``dftu_plugin`` struct; host services are reached through ``dftu_host`` and its
optional extension groups (``dftu_svc_coro``, ``dftu_svc_compose``,
``dftu_svc_query``, ``dftu_svc_writer``, ``dftu_svc_sketch``,
``dftu_svc_arrow``, ``dftu_svc_trace``, ``dftu_svc_ports``,
``dftu_svc_result``, ``dftu_svc_agg``, ``dftu_svc_ops``).

.. code-block:: c

   #include <dftracer/utils/plugins/abi.h>

Core ABI
--------

Every type, extension group, and entry point of the ABI. ``dftu_plugin`` is the
Fold a plugin fills; ``dftu_host`` and the ``dftu_svc_*`` structs are the
services the host lends back.

.. doxygenfile:: dftracer/utils/plugins/abi.h
   :project: dftracer-utils

Arrow ABI
---------

The Arrow extension group: build Arrow record batches from a plugin over the
Arrow C Data Interface.

.. doxygenfile:: dftracer/utils/plugins/arrow_abi.h
   :project: dftracer-utils

Numeric primitives
------------------

Header-only ``dftu_*`` helpers for plugin and raw JIT bodies: bit counting, log2
bucketing, power-of-two rounding, integer sqrt/gcd, alignment, and fast integer
hashing. The JIT layer lowers ``jit.ilog2`` and friends to these.

.. doxygenfile:: dftracer/utils/plugins/prims.h
   :project: dftracer-utils
