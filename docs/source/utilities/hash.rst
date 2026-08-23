:description: The incremental FNV-1a 64-bit hasher utility used across the engine for content addressing and deterministic string IDs.

Hash
==============

Incremental FNV-1a 64-bit hash computation.

.. code-block:: cpp

   #include <dftracer/utils/utilities/hash/hasher_utility.h>

Types
-----

.. code-block:: cpp

   struct Hash {
       std::size_t value = 0;
       bool operator==(const Hash& other) const;
       bool operator!=(const Hash& other) const;
   };

``HasherUtility`` is a type alias for ``Fnv1aHasherUtility``
(``dftracer/utils/utilities/hash/fnv1a_hasher_utility.h``). The hasher was
once a runtime-selectable hierarchy with a ``std::hash`` variant; only
FNV-1a was ever used, so it collapsed to the one concrete implementation.
There is no runtime algorithm selection and no thread-safe variant - a
``HasherUtility`` instance holds mutable state and is not safe to share
across threads without external synchronization.

HasherUtility
-------------

Incremental, chunk-order-independent hasher: ``update("Hello");
update("World")`` yields the same hash as ``update("HelloWorld")``.

**Basic hashing:**

.. code-block:: cpp

   #include <dftracer/utils/utilities/hash/hasher_utility.h>

   using namespace dftracer::utils::utilities::hash;

   HasherUtility hasher;
   hasher.reset();
   hasher.update("Hello");
   hasher.update("World");
   Hash h = hasher.get_hash();

**Hashing POD types and binary data:**

.. code-block:: cpp

   HasherUtility hasher;
   hasher.reset();

   hasher.update("name");          // String
   int id = 42;
   hasher.update(id);              // Trivially-copyable type (hashes raw bytes)

   std::vector<unsigned char> bin = {0xFF, 0xFE};
   hasher.update(bin);             // Binary data

   Hash combined = hasher.get_hash();

**Variadic processing:**

.. code-block:: cpp

   HasherUtility hasher;
   hasher.reset();
   Hash h = hasher.process(1, 2, 3);
   Hash h2 = hasher.process("hello", 42, 3.14);

``process(args...)`` updates with each argument in order and returns the
running hash. It is a plain synchronous call, not a coroutine.

**Reusing in hot loops:**

.. code-block:: cpp

   // Per project conventions: reuse a single instance with reset()
   HasherUtility hasher;

   for (const auto& item : items) {
       hasher.reset();
       hasher.update(item.data);
       Hash h = hasher.get_hash();
       process_hash(h);
   }

See Also
--------

- :doc:`/cpp_api/index` - Full C++ API documentation
