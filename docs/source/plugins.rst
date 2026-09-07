:description: Extend trace analysis with a compiled plugin that rides the fused scan, authored with the C++ SDK or the raw C ABI, shown side by side.

Plugins
=======

.. seealso::

   :doc:`cpp_api/plugins` for the generated plugin C ABI reference, :doc:`jit`
   to author a plugin in Python, and :doc:`tutorials/write-a-c-plugin` for a
   compiled-and-run C++ walkthrough.

A plugin extends trace analysis with your own compiled code, loaded at runtime
from a shared library. It rides the same fused scan the built-in analytics use:
N plugins share one traversal that parses each event once, so
``--plugin a --plugin b`` decompresses and parses the trace a single time.

There are two co-equal ways to author one:

- **The C++ SDK** (``<dftracer/utils/plugins/plugin.h>``, namespace
  ``dftracer::utils::plugins``): a plain ``Slice`` struct with ``step`` / ``merge`` /
  ``finalize`` methods, registered with ``make_plugin<Slice>()``. Typed
  ``Batch`` / ``Event`` views, host-owned ``Map`` accumulators, and an RAII
  ``Host`` read like ordinary C++.
- **The raw C ABI** (``<dftracer/utils/plugins/abi.h>``): a small set of C
  structs you fill by hand. This is the stable boundary the SDK compiles down
  to, and the path for a C-only toolchain.

Both compile to the same loadable ``.so`` and link nothing from the internal
library, so a plugin stays compatible across releases. Every section below shows
the C++ and the C form side by side; the SDK is a header-only, zero-cost wrapper
over the ABI, so any SDK call maps to one ABI call underneath.

.. contents:: On this page
   :local:
   :depth: 1

Overview
--------

- **A plugin is a Fold**: it sees each batch of parsed events and folds them
  into its own state. There is no per-event callback overhead and no threading
  code - the host runs one fold slice per worker and merges the results.
- **State is host-owned where it can be**: a mergeable ``Map`` (or ``Handle``)
  is owned and merged by the host, so parallelism and the final Arrow
  materialization are free and ``merge`` stays empty.
- **Strings cross as interned ids**: an ``Event``'s string fields are
  ``dftu_str`` ids, resolved to bytes on demand through the ``Host``, so nothing
  is copied on the hot path and a map key stores the id, not the label.
- **Services are optional extension groups**: maps, async I/O, the query DSL,
  Arrow, writers, sketches, and inter-plugin channels are fetched by id; a
  missing group degrades gracefully to a null/no-op.

1. Entry point and lifecycle
----------------------------

A plugin is a shared library exporting one symbol, ``dftracer_plugin`` (the
``DFTRACER_PLUGIN_FACTORY_SYMBOL`` the loader resolves via ``dlsym``). It
receives the parsed config tree (NULL when none given) and returns a
``dftu_plugin``. The host then drives that plugin as a data-parallel fold:

.. mermaid::

   graph LR
       make["make_slice /<br/>Slice(Config)<br/>(per worker)"] --> step["on_batch / step<br/>(per batch)"]
       step --> step
       step --> merge["merge<br/>(fold slices together)"]
       merge --> fin["on_finalize / finalize<br/>(emit output)"]
       fin --> destroy["destroy_slice /<br/>destroy"]

Each stage maps one-to-one between the SDK and the ABI:

.. list-table::
   :header-rows: 1
   :widths: 34 34 32

   * - ``dftracer::utils::plugins`` Slice
     - ``dftu_plugin`` callback
     - When
   * - ``Slice(const Config&)``
     - ``make_slice``
     - once per worker slice
   * - ``step`` / ``on_batch``
     - ``on_batch``
     - once per event batch
   * - ``merge``
     - ``merge``
     - to fold two slices into one
   * - ``finalize`` / ``on_finalize``
     - ``on_finalize``
     - once on the merged slice, to emit
   * - destructor
     - ``destroy_slice`` / ``destroy``
     - teardown
   * - ``static constexpr needs``
     - ``needs``
     - queried before the scan (section 2)
   * - config ``"query"`` key
     - ``plan_query``
     - queried before the scan (section 3)

``on_batch`` / ``on_finalize`` return ``NULL`` (or nothing, for the sync SDK
``step`` / ``finalize``) when handled synchronously, or a ``dftu_task`` the host
awaits for async work (section 9). The delivered ``dftu_batch`` is valid until
that returned task completes, or only during the call for a synchronous return -
never retain the pointer.

.. tab-set::

   .. tab-item:: C++ (SDK)

      A ``Slice`` provides the constructor and methods; ``make_plugin<Slice>``
      fills the ``dftu_plugin`` struct and adapts exceptions at the boundary
      (every callback catches and logs). ``abi_version`` is set for you.

      .. code-block:: cpp

         #include <dftracer/utils/plugins/plugin.h>

         using namespace dftracer::utils::plugins;

         struct MyPlugin {
             explicit MyPlugin(const Config&) {}   // per-worker state

             void step(const Batch& b, Host h) {   // fold one batch
                 (void)b; (void)h;
             }
             void merge(MyPlugin&) {}               // combine two slices
             void finalize(Host) {}                 // emit output
         };

         extern "C" dftu_plugin* dftracer_plugin(const dftu_value* config) {
             return make_plugin<MyPlugin>(config);
         }

   .. tab-item:: C (raw ABI)

      Fill the ``dftu_plugin`` struct by hand. ``abi_version`` must be set to
      ``DFTRACER_PLUGIN_ABI_VERSION``; ``self`` is the read-only config shared
      across slices.

      .. code-block:: c

         #include <dftracer/utils/plugins/abi.h>
         #include <stdlib.h>

         static uint32_t needs(void* self) { (void)self; return 0; }
         static const char* plan_query(void* self) { (void)self; return NULL; }
         static void* make_slice(void* self) { (void)self; return calloc(1, 1); }
         static dftu_task* on_batch(void* slice, const dftu_batch* b,
                                   const dftu_host* host) {
             (void)slice; (void)b; (void)host; return NULL;  /* synchronous */
         }
         static void merge(void* into, void* other) { (void)into; (void)other; }
         static dftu_task* on_finalize(void* s, const dftu_host* h) {
             (void)s; (void)h; return NULL;
         }
         static void destroy_slice(void* slice) { free(slice); }
         static void destroy(void* self) { (void)self; }

         static dftu_plugin g_plugin;

         dftu_plugin* dftracer_plugin(const dftu_value* config) {
             (void)config;
             g_plugin.abi_version   = DFTRACER_PLUGIN_ABI_VERSION;
             g_plugin.self          = NULL;
             g_plugin.needs         = needs;
             g_plugin.plan_query    = plan_query;
             g_plugin.make_slice    = make_slice;
             g_plugin.on_batch      = on_batch;
             g_plugin.merge         = merge;
             g_plugin.on_finalize   = on_finalize;
             g_plugin.destroy_slice = destroy_slice;
             g_plugin.destroy       = destroy;
             g_plugin.get_extension = NULL;   /* optional (section 8) */
             return &g_plugin;
         }

2. Declaring field needs
------------------------

An event always carries ``cat``, ``name``, ``pid``, ``tid``, ``ts``, ``dur``,
``phase`` and ``has_dur``. Three optional fields cost extra to extract, so the
scan only pays for them when a plugin declares it needs them. ``needs`` returns
the OR of these flags:

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Flag
     - Enables
   * - ``DFTU_NEED_ARGS``
     - ``Event::args()`` / ``arg_count()`` (the flattened, interned arg list);
       ``args`` is NULL otherwise.
   * - ``DFTU_NEED_FHASH``
     - ``Event::fhash_id()`` (the file-name hash).
   * - ``DFTU_NEED_HHASH``
     - ``Event::hhash_id()`` (the host-name hash).

These three (``abi.h``) are the complete set; there is no other ``DFTU_NEED_*``
flag.

.. tab-set::

   .. tab-item:: C++ (SDK)

      Add a ``static constexpr std::uint32_t needs``; ``make_plugin`` detects
      and reports it. Omit it for 0 (no optional fields).

      .. code-block:: cpp

         struct MyPlugin {
             static constexpr std::uint32_t needs =
                 DFTU_NEED_ARGS | DFTU_NEED_FHASH;

             explicit MyPlugin(const Config&) {}
             void step(const Batch& b, Host h) {
                 for (const Event& e : b)
                     for (const Arg& a : e.args()) { /* args now populated */ }
             }
             void merge(MyPlugin&) {}
             void finalize(Host) {}
         };

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         static uint32_t needs(void* self) {
             (void)self;
             return DFTU_NEED_ARGS | DFTU_NEED_FHASH;
         }

3. Coarse predicate pushdown (plan_query)
-----------------------------------------

A plugin can narrow the shared scan to the chunks that can possibly match, using
the same query DSL the analytics use (see :doc:`guides/core/query-dsl`). The
host uses the returned predicate to prune index chunks before the fold ever sees
them; it is a coarse filter, so a plugin still checks per event if it needs
exactness. Returning nothing scans everything.

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``make_plugin`` reads the pushdown predicate from the plugin's ``"query"``
      config value, so set it with ``--parg query=...`` (section 4) or a config
      file. There is no separate ``Slice`` method for it.

      .. code-block:: bash

         dftracer_run --plugin ./p.so --parg query='cat == "POSIX"' \
                      --files trace.pfw.gz

   .. tab-item:: C (raw ABI)

      Implement ``plan_query`` to return any DSL string (scan-lifetime storage),
      or NULL for no filter.

      .. code-block:: c

         static const char* plan_query(void* self) {
             (void)self;
             return "cat == \"POSIX\"";
         }

4. Configuration and arguments
------------------------------

``dftracer_run`` layers a JSON config tree per plugin block and passes it to the
factory. On the command line, arguments attach to the ``--plugin`` they follow:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Flag
     - Effect
   * - ``--parg key=value``
     - Set one config key for the preceding ``--plugin`` block.
   * - ``--pconfig file.json``
     - Merge a JSON config file into the preceding block.
   * - ``--shared-parg key=value``
     - Set a key on every plugin block.
   * - ``--shared-pconfig file.json``
     - Merge a JSON file into every block.

Layering is low to high: shared config file, shared args, block config file,
block args (later wins). A plugin reads the resulting tree from its factory.

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``Config`` (``plugin.h``) is a typed view over the tree; a ``Slice`` may
      keep returned ``string_view``s since the host-owned tree outlives it.
      Accessors: ``get`` (string), ``get_int``, ``get_double``, ``get_bool``,
      ``child`` (nested object), ``array`` / ``get_int_array`` /
      ``get_double_array`` / ``get_string_array``.

      .. code-block:: cpp

         struct MyPlugin {
             std::int64_t threshold_;
             std::string label_;

             explicit MyPlugin(const Config& cfg)
                 : threshold_(cfg.get_int("threshold", 0)),
                   label_(cfg.get("label", "events")) {}

             void step(const Batch&, Host) {}
             void merge(MyPlugin&) {}
             void finalize(Host) {}
         };

   .. tab-item:: C (raw ABI)

      The tree is a ``dftu_value`` passed to the factory; ``self`` should carry
      whatever the slices need. Read it with the inline coercions
      ``dftu_obj_get`` / ``dftu_as_i64`` / ``dftu_as_f64`` / ``dftu_as_bool`` /
      ``dftu_as_str``.

      .. code-block:: c

         dftu_plugin* dftracer_plugin(const dftu_value* config) {
             int64_t threshold = dftu_as_i64(dftu_obj_get(config, "threshold"), 0);
             /* stash `threshold` in a static/self the slices can read */
             (void)threshold;
             /* ... fill g_plugin ... */
             return &g_plugin;
         }

5. Events and batches
---------------------

``on_batch`` / ``step`` receives a batch of parsed events. The SDK's ``Batch``
and ``Event`` are zero-copy typed views valid only for that call. String fields
are interned ``dftu_str`` ids; resolve them to bytes through the ``Host``. Every
event field:

.. list-table::
   :header-rows: 1
   :widths: 26 20 54

   * - ``Event`` (SDK)
     - ``dftu_event`` (C)
     - Meaning
   * - ``pid()`` / ``tid()``
     - ``pid`` / ``tid``
     - process / thread id (``uint64``)
   * - ``ts()`` / ``dur()``
     - ``ts`` / ``dur``
     - start timestamp / duration
   * - ``has_dur()``
     - ``has_dur``
     - whether a duration was present (a complete-phase event)
   * - ``phase()``
     - ``phase``
     - ``dftu_phase``: ``UNKNOWN`` / ``COMPLETE`` / ``COUNTER`` /
       ``AGGREGATED`` / ``METADATA``
   * - ``cat_id()`` / ``name_id()``
     - ``cat`` / ``name``
     - interned category / event-name id
   * - ``fhash_id()`` / ``hhash_id()``
     - ``fhash`` / ``hhash``
     - interned file / host hash (need ``DFTU_NEED_FHASH`` / ``HHASH``)
   * - ``cat(h)`` / ``name(h)`` / ``fhash(h)`` / ``hhash(h)``
     - ``resolve``
     - resolve the id to a ``string_view`` (empty when absent)
   * - ``arg_count()`` / ``args()`` / ``arg(i)``
     - ``arg_count`` / ``args``
     - flattened args (need ``DFTU_NEED_ARGS``); ``args()`` is range-for-able
   * - ``find_arg(key)`` / ``find_arg(h, "a", "b")``
     - iterate ``args``
     - first arg by interned or dotted key

There is no per-event severity/level field; the ABI exposes exactly the fields
above. An ``Arg`` carries an interned ``key_id()`` and a value selected by
``kind()``: ``is_i64()`` / ``i64()``, ``is_f64()`` / ``f64()``, or ``is_str()``
/ ``str_id()`` / ``str(h)`` (``DFTU_ARG_I64`` / ``DFTU_ARG_F64`` /
``DFTU_ARG_STR``).

.. tab-set::

   .. tab-item:: C++ (SDK)

      .. code-block:: cpp

         void step(const Batch& b, Host h) {
             for (const Event& e : b) {
                 if (!e.has_name()) continue;
                 std::string_view cat = e.cat(h);        // resolve id -> bytes
                 std::uint64_t dur = e.has_dur() ? e.dur() : 0;
                 for (const Arg& a : e.args())
                     if (a.is_i64()) { /* a.key(h), a.i64() */ }
             }
         }

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         static dftu_task* on_batch(void* slice, const dftu_batch* b,
                                   const dftu_host* host) {
             (void)slice;
             for (uint32_t i = 0; i < b->count; ++i) {
                 const dftu_event* e = &b->events[i];
                 if (e->name == DFTU_STR_NONE) continue;
                 uint32_t n = 0;
                 const char* cat = host->resolve(host->h, e->cat, &n);
                 (void)cat;
                 for (uint32_t j = 0; j < e->arg_count; ++j) {
                     const dftu_arg* a = &e->args[j];
                     if (a->kind == DFTU_ARG_I64) { /* a->v.i64 */ }
                 }
             }
             return NULL;
         }

6. Mergeable maps (DFTU_EXT_MAP)
--------------------------------

A host-owned map is the usual way a plugin accumulates a result: a fixed tuple
key mapping to a monoid value. The host merges same-named maps across all worker
slices and materializes each at finalize to an Arrow table (key columns, then
one value column per component), returned under the map name. Because the map is
write-only and host-merged, ``merge`` stays empty and there is nothing to free.

Keys are fixed-width integers (``I8``..``I64``, ``U8``..``U64``), ``F32`` / F64
(bit-cast into the key slot), or ``STR`` (an interned id). In the SDK the key
schema is a ``Key<Ts...>`` tag: an arithmetic type maps to its column type, a
string-like type or ``Interned`` maps to a STR column. ``Interned`` carries an
already-interned id (``e.name_id()``, ``e.fhash_id()``) as a STR key without
re-interning its bytes.

**Monoids.** The value is one ``Monoid`` (SDK ``enum class``) or
``dftu_monoid_kind`` (C). The full vocabulary (``abi.h``):

- Scalars fed with an integer add: ``Counter`` (u64 sum), ``Min_U64`` /
  ``Max_U64``, ``Bool_And`` / ``Bool_Or``, ``Bitset_Or``, ``Distinct`` (approx
  distinct count), and the typed ``Min_*`` / ``Max_*`` (``I8``..``I64``,
  ``U8``..``U32``).
- Scalars fed with a floating add: ``Sum_F64``, ``Min_F64`` / ``Max_F64``,
  ``Min_F32`` / ``Max_F32``, and the moment stats ``Mean``, ``Variance``,
  ``Stddev``, ``Skewness``, ``Kurtosis``.
- Two-variable co-moments fed with ``(x, y)``: ``Corr``, ``Covar_Pop`` /
  ``Covar_Samp``, ``Regr_Slope`` / ``Regr_Intercept`` / ``Regr_R2``.
- Collections: ``Set_Str`` / ``List_Str`` / ``Set_I64`` / ``List_I64``.
- Arg-by (min-by/max-by keeping a payload): ``ArgMin_I64`` / ``ArgMax_I64`` /
  ``ArgMin_Str`` / ``ArgMax_Str``, and the whole-row ``ArgMin_Row`` /
  ``ArgMax_Row`` (via ``map_new_argrow``).
- Bounded / approximate: ``TopK_I64`` / ``TopK_Str`` / ``BottomK_I64`` /
  ``BottomK_Str``, ``Approx_TopK_I64`` / ``Approx_TopK_Str`` (heavy hitters),
  ``Sample_I64`` / ``Sample_Str`` (bottom-k-by-hash sample).
- ``Sketch`` (DDSketch quantiles) has no scalar result, so it is created only
  via a sketch map (below), never as a plain ``map_new`` value.

Out-of-core spilling for a very large map is handled transparently by the host;
there is no plugin-facing spill call in the ABI.

Simple keyed counter
~~~~~~~~~~~~~~~~~~~~~~

Count events per ``(pid, event-name)``:

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``counter_map(name, Key<...>)`` is shorthand for
      ``map(name, Monoid::Counter, Key<...>)`` (``sum_map`` maps to
      ``Sum_F64``). ``edges[key] += n`` contributes at a key; a single-column
      key can be subscripted bare, a multi-column key as a ``std::tuple``.

      .. code-block:: cpp

         void step(const Batch& b, Host h) {
             auto edges = h.counter_map("name_edges",
                                        Key<std::uint64_t, Interned>{});
             for (const Event& e : b) {
                 if (!e.has_name()) continue;
                 edges[std::tuple{e.pid(), interned(e.name_id())}] += 1;
             }
         }

   .. tab-item:: C (raw ABI)

      Fetch ``DFTU_EXT_MAP``, ``map_new`` with a key-type array and a monoid,
      then ``map_add_u64`` / ``map_add_f64`` at an ``int64`` key array (STR
      slots carry the interned id).

      .. code-block:: c

         static dftu_task* on_batch(void* slice, const dftu_batch* b,
                                   const dftu_host* host) {
             (void)slice;
             const dftu_ext_map* map =
                 (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);
             if (!map || !map->map_new) return NULL;
             static const dftu_type kt[2] = {DFTU_T_U64, DFTU_T_STR};
             dftu_map* m = map->map_new(host->h, "name_edges", kt, 2,
                                       DFTU_MONOID_COUNTER);
             if (!m) return NULL;
             for (uint32_t i = 0; i < b->count; ++i) {
                 const dftu_event* e = &b->events[i];
                 if (e->name == DFTU_STR_NONE) continue;
                 int64_t key[2] = {(int64_t)e->pid, (int64_t)e->name};
                 map->map_add_u64(host->h, m, key, 1);
             }
             return NULL;
         }

The SDK ``Map`` also exposes ``.add`` / ``.add_at`` (a value component),
``.add_xy`` (co-moments), ``.add_argby`` (arg-by), ``.add_topk`` /
``.add_approx_topk`` / ``.add_sample``, ``.add_ordered`` (list monoids), and
``.set_ordered(true)`` (materialize sorted by key); each mirrors a
``map_add_*_at`` C call. The k for a bounded / approximate monoid is passed on
every add (constant per component).

Product maps
~~~~~~~~~~~~~

A product value is a tuple of monoids, one value column per component. Component
0 takes the bare add / ``+=``; other components take ``_at`` adds.

.. tab-set::

   .. tab-item:: C++ (SDK)

      .. code-block:: cpp

         void step(const Batch& b, Host h) {
             auto stats = h.product_map("per_pid",
                                        {Monoid::Counter, Monoid::Sum_F64},
                                        Key<std::uint64_t>{});
             for (const Event& e : b) {
                 stats[e.pid()] += 1;                        // component 0
                 stats.add_at(e.pid(), 1, double(e.dur()));  // component 1
             }
         }

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         static const dftu_monoid_kind vals[2] =
             {DFTU_MONOID_COUNTER, DFTU_MONOID_SUM_F64};
         dftu_map* m = map->map_new_product(host->h, "per_pid", kt, 1, vals, 2);
         int64_t key[1] = {(int64_t)e->pid};
         map->map_add_u64_at(host->h, m, key, 0, 1);
         map->map_add_f64_at(host->h, m, key, 1, (double)e->dur);

Nested maps
~~~~~~~~~~~

A nested map keeps an inner map per outer key, materialized as one
``list<struct>`` row per outer key.

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``nested_map`` takes an outer ``Key`` and an inner ``Key`` tag; ``add``
      contributes to a value component at ``(outer, inner)``.

      .. code-block:: cpp

         auto nm = h.nested_map("pid_names", {Monoid::Counter},
                                Key<std::uint64_t>{}, Key<Interned>{});
         nm.add(std::tuple{e.pid()}, std::tuple{interned(e.name_id())}, 1u);

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         dftu_map* m = map->map_new_nested(host->h, "pid_names",
                                          okt, 1, ikt, 1, vals, 1);
         int64_t ok[1] = {(int64_t)e->pid};
         int64_t ik[1] = {(int64_t)e->name};
         map->map_add_nested_u64(host->h, m, ok, ik, 0, 1);

Joins
~~~~~

Declare a join to run at finalize on the merged master maps: it joins two named
maps on their shared key tuple and emits the result as an additional named map.
``JoinType`` (SDK) / ``dftu_join_type`` (C) is ``Inner`` / ``Left`` / ``Right``
/ ``Full``.

.. tab-set::

   .. tab-item:: C++ (SDK)

      .. code-block:: cpp

         h.map_declare_join("joined", "left_map", "right_map", JoinType::Inner);

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         map->map_declare_join(host->h, "joined", "left_map", "right_map",
                               DFTU_JOIN_INNER);

The SDK also wraps the specialized constructors on ``Host``:
``map_new_argrow`` / ``map_add_argrow`` (whole-row arg-by),
``map_new_sketch`` (a per-key DDSketch quantile table over requested quantiles),
and ``map_new_fused`` / ``map_add_row`` (several same-key maps sharing one hash
lookup). Each mirrors the matching ``dftu_ext_map`` call.

7. Quantile sketches (DFTU_EXT_SKETCH)
--------------------------------------

A plugin-owned DDSketch accumulates values and reports mergeable quantiles
(``dftu_quantiles``: ``count``, ``min``, ``max``, ``mean``, ``p50``, ``p90``,
``p95``, ``p99``). Unlike a map, the plugin owns the handle and merges it itself
across slices.

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``Host::make_sketch()`` returns an RAII ``Sketch`` that frees itself.
      Keep one per slice, ``add`` during ``step``, ``merge`` in ``merge``, read
      ``result`` at ``finalize``.

      .. code-block:: cpp

         #include <optional>

         struct Durations {
             std::optional<Sketch> sk;      // built lazily from the first Host
             explicit Durations(const Config&) {}
             void step(const Batch& b, Host h) {
                 if (!sk) sk.emplace(h);    // Sketch{h}
                 for (const Event& e : b)
                     if (e.has_dur()) sk->add(double(e.dur()));
             }
             void merge(Durations& o) { if (sk && o.sk) sk->merge(*o.sk); }
             void finalize(Host) {
                 if (sk) { dftu_quantiles q = sk->result(); (void)q; }
             }
         };

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_sketch* sk =
             (const dftu_ext_sketch*)host->get_extension(host->h, DFTU_EXT_SKETCH);
         dftu_sketch* s = sk->sketch_create(host->h);
         sk->sketch_add(host->h, s, (double)e->dur, 1.0);
         /* sk->sketch_merge(host->h, into, other) in merge */
         dftu_quantiles q = sk->sketch_result(host->h, s);
         sk->sketch_free(host->h, s);   /* plugin owns it */

For a per-key quantile table instead of one global sketch, use a sketch map
(``Host::map_new_sketch`` / ``map_new_sketch``) from section 6.

8. Inter-plugin communication
-----------------------------

Within the one shared scan, one plugin can hand values to another. Three
mechanisms, each its own extension group. See
:doc:`guides/plugins/inter-plugin-comms` for the full treatment.

**Ports (DFTU_EXT_PORTS)** are a batch-scoped slot keyed by a capability id: a
producer publishes during a batch, a consumer reads it back during the same
batch (NULL if the producer has not published, or runs after the consumer). The
bus resets between batches. A producer must run before its consumer in fold
order (``--plugin`` / registration order).

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``publish_port<T>`` / ``consume_port<T>`` give a typed ``OutPort`` /
      ``InPort`` (``T`` trivially copyable). ``send`` publishes; ``recv``
      returns ``std::optional<T>`` (a copy, safe to keep).

      .. code-block:: cpp

         constexpr const char* CAP = "com.example.perbatch";
         // producer:
         void step(const Batch& b, Host h) {
             std::uint64_t n = 0;
             for (const Event& e : b) if (e.has_dur()) ++n;
             h.publish_port<std::uint64_t>(CAP).send(n);
         }
         // consumer:
         void step(const Batch&, Host h) {
             if (auto n = h.consume_port<std::uint64_t>(CAP).recv()) { /* *n */ }
         }

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_ports* p =
             (const dftu_ext_ports*)host->get_extension(host->h, DFTU_EXT_PORTS);
         uint64_t key = p->port_key(host->h, "com.example.perbatch");
         uint64_t n = 42;
         p->publish(host->h, key, &n, sizeof(n));         /* producer */
         uint32_t len = 0;
         const void* got = p->consume(host->h, key, &len); /* consumer */

**Cross-worker handles (DFTU_EXT_HANDLES)** are a named monoid accumulator the
host merges across every worker slice: contribute during ``on_batch``, read the
merged value at ``finalize``. ``MonoidValue`` selects ``as_u64()`` /
``as_f64()`` / ``as_quantiles()`` by ``kind()``.

.. tab-set::

   .. tab-item:: C++ (SDK)

      .. code-block:: cpp

         struct Total {
             Handle h_;
             explicit Total(const Config&) {}
             void step(const Batch& b, Host h) {
                 if (!h_) h_ = h.handle("com.example.total", Monoid::Counter);
                 h_.add(std::uint64_t(b.size()));
             }
             void merge(Total&) {}
             void finalize(Host) {
                 if (auto v = h_.result()) { /* v->as_u64() */ }
             }
         };

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_handles* H =
             (const dftu_ext_handles*)host->get_extension(host->h,
                                                         DFTU_EXT_HANDLES);
         dftu_handle* hd = H->shared_get(host->h, "com.example.total",
                                        DFTU_MONOID_COUNTER);
         H->add_u64(host->h, hd, b->count);        /* during on_batch */
         dftu_monoid_value out;                     /* at on_finalize */
         if (H->result(host->h, "com.example.total", &out) == 0) { /* out.as.u64 */ }

**Named results (DFTU_EXT_RESULT)** emit an opaque blob or a user-schema Arrow
array under a name; both surface from ``Plugins::run`` keyed by that name.
Use ``Host::emit_result`` / ``emit_result_arrow`` (C: ``emit`` / ``emit_arrow``
on ``dftu_ext_result``), best called at finalize.

**Capabilities (DFTU_EXT_COMMS)** let a consumer discover whether a compatible
producer was loaded and order the fold accordingly. On the raw ABI this is
reached through the plugin's own ``get_extension`` returning a
``dftu_plugin_comms`` of ``provides`` / ``require_caps`` / ``resolve``; see
:doc:`guides/plugins/inter-plugin-comms` for the full protocol.

A C++ Slice declares the same three hooks as optional ``static`` members and
``make_plugin<Slice>`` wires ``get_extension`` to a per-Slice
``dftu_plugin_comms`` automatically (nothing is emitted when a Slice declares
none, so existing plugins are unaffected). Build the entries with
``dftracer::utils::plugins::capability`` and ``dftracer::utils::plugins::requirement``; the consumer hook
is named ``requires_caps`` because ``requires`` is a C++20 keyword. Inside
``on_resolve`` use ``Host::provider_count`` / ``Host::provider_best`` to inspect
the registry.

.. tab-set::

   .. tab-item:: C++ (SDK)

      .. code-block:: cpp

         struct Slice {
             explicit Slice(const dftracer::utils::plugins::Config&) {}
             void step(const dftracer::utils::plugins::Batch&, dftracer::utils::plugins::Host) {}
             void merge(Slice&) {}
             void finalize(dftracer::utils::plugins::Host) {}

             // Capabilities this plugin publishes.
             static std::array<dftu_capability, 1> provides() {
                 return {dftracer::utils::plugins::capability("com.example.tag", 1, 0, 0)};
             }
             // Capabilities it consumes (optional == graceful fallback).
             static std::array<dftu_requirement, 1> requires_caps() {
                 return {dftracer::utils::plugins::requirement("com.example.peer",
                                                  DFTU_VER_CARET, 1, 0, 0,
                                                  /*required=*/false)};
             }
             // Runs once after every plugin has declared.
             static void on_resolve(dftracer::utils::plugins::Host h) {
                 if (auto v = h.provider_best(requires_caps()[0])) {
                     // a compatible provider is present at version *v
                 }
             }
         };

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         static uint32_t provides(void* self, dftu_capability* out, uint32_t max) {
             (void)self;
             if (max >= 1) {
                 out[0].id = "com.example.tag";
                 out[0].ver.major = 1;
                 out[0].ver.minor = out[0].ver.patch = 0;
             }
             return 1;
         }
         static const dftu_plugin_comms g_comms = {provides, NULL, NULL};
         static const void* get_extension(void* self, const char* ext_id) {
             (void)self;
             if (ext_id && strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_comms;
             return NULL;
         }
         /* ... vt.get_extension = get_extension; in the factory ... */

9. Async work and I/O
---------------------

For overlapped I/O or a custom fan-out, return a task instead of running
synchronously; the host awaits it before considering the batch done. ``step`` /
``on_batch`` must stay CPU-bound - offload real blocking through the async path
or ``run_blocking``.

.. tab-set::

   .. tab-item:: C++ (SDK)

      Name the hook ``on_batch`` / ``on_finalize`` and return a
      ``dftracer::utils::plugins::Task`` coroutine; ``make_plugin`` detects the signature.
      Inside it, ``co_await`` the async accessors. ``Io`` (from ``Host::io()``)
      wraps the full backend surface: ``open`` / ``close`` / ``read`` /
      ``write`` / ``pread`` / ``pwrite`` / ``fsync`` / ``ftruncate`` /
      ``fstat``, the vectored ``readv`` / ``writev`` / ``preadv`` / ``pwritev``,
      ``lseek``, ``sendfile``, and the socket ops ``accept`` / ``recv`` /
      ``send``. ``Host::spawn`` runs a callable on a pool worker, ``Host::then``
      sequences one after a task, ``Host::all`` / ``Host::any`` are ``when_all``
      / ``when_any`` over tasks, and ``Host::run_blocking`` runs a synchronous
      call while releasing the worker's run-permit.

      .. code-block:: cpp

         #include <dftracer/utils/plugins/plugin.h>
         #include <fcntl.h>

         using namespace dftracer::utils::plugins;

         struct Writer {
             explicit Writer(const Config&) {}
             void step(const Batch&, Host) {}
             void merge(Writer&) {}

             Task on_finalize(Host h) {
                 int fd = -1, rc = 0;
                 co_await h.io().open("/tmp/out.bin",
                                      O_WRONLY | O_CREAT | O_TRUNC, 0644, &fd);
                 if (fd >= 0) {
                     std::int64_t n = 0;
                     const char msg[] = "done";
                     co_await h.io().write(fd, msg, sizeof(msg) - 1, &n);
                     co_await h.io().close(fd, &rc);
                 }
             }
         };

         extern "C" dftu_plugin* dftracer_plugin(const dftu_value* config) {
             return make_plugin<Writer>(config);
         }

   .. tab-item:: C (raw ABI)

      Fetch ``DFTU_EXT_IO`` for the ``dftu_io`` calls (each returns a ``dftu_task``
      to compose or await; the out-slot must outlive it) and ``DFTU_EXT_CORO``
      for the combinators ``spawn`` / ``when_all`` / ``when_any`` / ``then`` /
      ``run_blocking``, plus ``drive`` (which the SDK's coroutine adapter uses).
      Return the root task from ``on_batch`` / ``on_finalize``.

      .. code-block:: c

         static dftu_task* on_finalize(void* slice, const dftu_host* host) {
             (void)slice;
             const dftu_io* io =
                 (const dftu_io*)host->get_extension(host->h, DFTU_EXT_IO);
             if (!io) return NULL;
             static int fd = -1;
             /* returns a task the host awaits; compose with dftu_ext_coro */
             return io->open(host->h, "/tmp/out.bin",
                             O_WRONLY | O_CREAT | O_TRUNC, 0644, &fd);
         }

The SDK also offers a pull-model ``Stream<Tag>`` (from ``Host::util_stream``)
and ``util`` / ``util_async`` / ``util_each`` for host utilities. Every
``dftu_ext_coro`` combinator now has an SDK method (``spawn`` / ``then`` /
``all`` / ``any`` / ``run_blocking``), so async composition needs no raw ABI.
See :doc:`guides/plugins/compose-ops` for composing reusable typed ops inside a
plugin.

10. Query DSL against events (DFTU_EXT_QUERY)
---------------------------------------------

Beyond coarse pushdown (section 3), a plugin can compile a query once and test
it against individual events. The compiled ``dftu_query`` is valid for the whole
scan; do not free it.

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``query_compile`` accepts either a raw DSL string or an expression built
      with the ergonomic ``F`` / ``Field`` builder. The SDK re-exports the
      builder into ``dftracer::utils::plugins``, so ``plugin.h`` is the only include and no
      ``query::`` qualifier is needed; the host parses the rendered string, so
      the plugin still links nothing from the query library.

      .. code-block:: cpp

         struct Filtered {
             dftu_query* q_ = nullptr;
             explicit Filtered(const Config&) {}
             void step(const Batch& b, Host h) {
                 if (!q_) q_ = h.query_compile(F("dur") > 1000);
                 // Equivalent: h.query_compile("dur > 1000");
                 for (const Event& e : b)
                     if (h.query_matches(q_, e.raw())) { /* ... */ }
             }
             void merge(Filtered&) {}
             void finalize(Host) {}
         };

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_query* Q =
             (const dftu_ext_query*)host->get_extension(host->h, DFTU_EXT_QUERY);
         dftu_query* q = Q->query_compile(host->h, "dur > 1000", 10);
         if (Q->query_matches(host->h, q, e)) { /* ... */ }

11. Arrow interchange (DFTU_EXT_ARROW)
--------------------------------------

Materialize a batch as an Arrow record batch (``cat``, ``name``, ``pid``,
``tid``, ``ts``, ``dur``, ``phase``), or read and write Arrow IPC files. The
caller owns the exported array/schema and must release them; the SDK's
``OwnedArrow`` does that in its destructor.

.. tab-set::

   .. tab-item:: C++ (SDK)

      .. code-block:: cpp

         void step(const Batch& b, Host h) {
             OwnedArrow a = h.batch_to_arrow(b.raw());
             if (a) h.arrow_write_ipc(a, "/tmp/batch.arrow");
         }

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_arrow* A =
             (const dftu_ext_arrow*)host->get_extension(host->h, DFTU_EXT_ARROW);
         struct ArrowArray arr; struct ArrowSchema sch;
         if (A->batch_to_arrow(host->h, b, &arr, &sch) == 0) {
             A->arrow_write_ipc(host->h, &arr, &sch, "/tmp/batch.arrow");
             arr.release(&arr); sch.release(&sch);   /* caller owns */
         }

12. Output writers
------------------

Two writer groups exist. The **trace writer (DFTU_EXT_TRACE)** appends events to
a gzip ``.pfw.gz`` and has an SDK wrapper on ``Host``; the index is built lazily
on first read, not at close.

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``Host::trace_open_write`` / ``trace_write`` / ``trace_close``, plus
      ``trace_read`` to scan an existing (auto-indexed) trace.

      .. code-block:: cpp

         void finalize(Host h) {
             dftu_trace_writer* w = h.trace_open_write("/tmp/out.pfw.gz");
             if (w) { /* h.trace_write(w, evs, n); */ h.trace_close(w); }
         }

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_trace* T =
             (const dftu_ext_trace*)host->get_extension(host->h, DFTU_EXT_TRACE);
         dftu_trace_writer* w = T->trace_open_write(host->h, "/tmp/out.pfw.gz");
         T->trace_write(host->h, w, evs, n);
         T->trace_close(host->h, w);

The **parallel writer (DFTU_EXT_WRITER)** is a sharded, multi-worker gzip-member
output (``writer_create`` / ``writer_open`` / ``writer_chunk`` / ``writer_close``
/ ``merge_shards``).

.. tab-set::

   .. tab-item:: C++ (SDK)

      ``Host::writer`` returns an ergonomic ``Writer`` (host-owned, nothing to
      free); ``open`` / ``chunk`` / ``close`` are ``co_await``-able AsyncOps, and
      ``Host::merge_shards`` concatenates shard files into a target.

      .. code-block:: cpp

         Task on_finalize(Host h) {
             Writer w = h.writer("/tmp/out.pfw.gz", /*num_workers=*/4,
                                 /*gzip=*/true);
             if (!w) co_return;
             co_await w.open();
             co_await w.chunk(/*worker=*/0, std::string_view{"payload"});
             co_await w.close();
         }

   .. tab-item:: C (raw ABI)

      .. code-block:: c

         const dftu_ext_writer* W =
             (const dftu_ext_writer*)host->get_extension(host->h, DFTU_EXT_WRITER);
         dftu_writer* w = W->writer_create(host->h, "/tmp/out.pfw.gz",
                                          /*num_workers=*/4, /*gzip=*/1);
         /* co-await W->writer_open / writer_chunk / writer_close (tasks) */

13. Getting results back
------------------------

Each host-owned map is materialized at finalize to an Arrow table keyed by the
map's name, with the key columns first and one value column per component;
interned key ids are resolved to their labels once, so a STR key column is a
real ``string``. Named results (section 8) surface the same way, keyed by their
emit name.

**From Python**, load one or more plugins and run them over a single fused scan;
each result comes back as a ``pyarrow.Table``:

.. code-block:: python

   from dftracer.utils.plugins import PluginHost

   host = PluginHost()
   host.load("./name_edges.so")          # a compiled .so, or a @jit.plugin class
   results = host.run("./traces")        # a directory or list of .pfw.gz traces
   table = results["name_edges"]         # pyarrow.Table[pid, name, value]
   df = table.to_pandas()

The output tables cross to NumPy or pandas cheaply (zero-copy where the dtype
allows); the fold itself runs in compiled code, so aggregate in the native scan
and do NumPy / pandas analysis on the (much smaller) result tables:

.. code-block:: python

   values = table.column("value").to_numpy(zero_copy_only=False)   # np.ndarray

**From the command line**, ``dftracer_run`` folds every ``--plugin`` over one
shared, index-pruned scan and reports the scan on stderr (it does not print map
contents - read those through ``PluginHost``):

.. code-block:: bash

   dftracer_run --plugin ./name_edges.so --files trace.pfw.gz
   dftracer_run --plugin ./a.so --plugin ./b.so -d ./traces      # one shared scan

Use ``-d <directory>`` for a whole tree instead of ``--files``.

Host services index
--------------------

Every service is an optional extension group fetched by id
(``host->get_extension``); the SDK ``Host`` memoizes each group and degrades a
missing one to a null/no-op. The map from SDK accessor to group to the section
that covers it:

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - C++ SDK accessor
     - Extension group
     - Covered in
   * - ``Host::map`` / ``counter_map`` / ``sum_map`` / ``product_map`` /
       ``nested_map`` (typed ``Map`` / ``NestedMap``)
     - ``DFTU_EXT_MAP``
     - `6. Mergeable maps (DFTU_EXT_MAP)`_
   * - ``Host::make_sketch`` (RAII ``Sketch``)
     - ``DFTU_EXT_SKETCH``
     - `7. Quantile sketches (DFTU_EXT_SKETCH)`_
   * - ``Host::publish_port`` / ``consume_port`` (``OutPort`` / ``InPort``)
     - ``DFTU_EXT_PORTS``
     - `8. Inter-plugin communication`_
   * - ``Host::handle`` (``Handle``, ``MonoidValue``)
     - ``DFTU_EXT_HANDLES``
     - `8. Inter-plugin communication`_
   * - ``Host::emit_result`` / ``emit_result_arrow``
     - ``DFTU_EXT_RESULT``
     - `8. Inter-plugin communication`_
   * - Slice ``provides`` / ``requires_caps`` / ``on_resolve``
     - ``DFTU_EXT_COMMS``
     - `8. Inter-plugin communication`_
   * - ``Host::all`` / ``any`` / ``run_blocking``
     - ``DFTU_EXT_CORO``
     - `9. Async work and I/O`_
   * - ``Host::io()`` (typed ``Io``), ``util_stream`` (``Stream``)
     - ``DFTU_EXT_IO`` / ``DFTU_EXT_UTIL``
     - `9. Async work and I/O`_
   * - typed compose ops (``dftracer::utils::plugins::make_op`` / ``run``)
     - ``DFTU_EXT_COMPOSE``
     - :doc:`guides/plugins/compose-ops`
   * - ``Host::query_compile`` / ``query_matches``
     - ``DFTU_EXT_QUERY``
     - `10. Query DSL against events (DFTU_EXT_QUERY)`_
   * - ``Host::batch_to_arrow`` / ``arrow_read_ipc`` / ``arrow_write_ipc``
     - ``DFTU_EXT_ARROW``
     - `11. Arrow interchange (DFTU_EXT_ARROW)`_
   * - ``Host::trace_open_write`` / ``trace_write`` / ``trace_read``
     - ``DFTU_EXT_TRACE``
     - `12. Output writers`_
   * - none (raw ``get_extension``)
     - ``DFTU_EXT_WRITER``
     - `12. Output writers`_

Building and scaffolding
------------------------

The ``dftracer_plugin`` console script (installed with the Python package,
``python/dftracer/utils/plugin_cli.py``) scaffolds a template source and builds
it, so you need neither the lifecycle boilerplate nor the compiler flags by
hand. This is distinct from the ``@jit.plugin`` DSL (:doc:`jit`), which writes
and compiles a plugin from Python source at call time.

``dftracer_plugin new`` writes a template plugin (a per-``pid`` counter into a
mergeable map). The default is the raw-C template; ``--cpp`` emits the SDK
template against ``plugin.h``:

.. code-block:: bash

   dftracer_plugin new name_edges --cpp          # -> ./name_edges.cpp (SDK)
   dftracer_plugin new name_edges                # -> ./name_edges.c (raw ABI)
   dftracer_plugin new name_edges --cpp -o ./plugins   # into a chosen directory

``dftracer_plugin build`` compiles a source straight to a loadable ``.so``,
resolving the plugin include directory for you (bundled with an installed wheel,
or walked up from a source checkout; override with the ``DFTRACER_PLUGIN_INCLUDE``
environment variable):

.. code-block:: bash

   dftracer_plugin build name_edges.cpp                  # -> ./name_edges.so
   dftracer_plugin build name_edges.cpp -o out/edges.so   # explicit output path

``dftracer_plugin cflags`` prints the flags ``build`` uses
(``-std=c++20 -fPIC -shared -I<plugin-include-dir>``, plus
``-undefined dynamic_lookup`` on macOS, since a plugin ``.so`` loads into a host
process rather than pre-linking against one), for wiring a plugin into your own
build system. ``build`` and ``cflags`` always target a C++ compiler (``$CXX``,
else ``c++`` / ``clang++``) even for a ``.c`` template, since the flags are
C++20 flags:

.. code-block:: bash

   c++ $(dftracer_plugin cflags) -o name_edges.so name_edges.cpp

For a full compiled-and-run C++ walkthrough (scaffold, edit, build, run), see
:doc:`tutorials/write-a-c-plugin`.
