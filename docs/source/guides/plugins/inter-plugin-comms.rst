:description: Have plugins share data within one scan: publish and consume batch-scoped ports, and negotiate producer capabilities before wiring up.

How to communicate between plugins
====================================

.. admonition:: Goal
   :class: goal

   Have one plugin publish a value during a batch and another plugin
   consume it, within the one shared scan ``--plugin a --plugin b`` already runs,
   and (optionally) have a plugin discover whether a compatible producer is
   even loaded. See :doc:`../../plugins` for the plugin model this builds on.

Two mechanisms cover this, and both are keyed by the same capability id
strings (the ``dftu.`` prefix is reserved for the host, see below):

- **Ports** (``DFTU_EXT_PORTS``) are the data channel: a producer publishes a
  value during a batch, a consumer reads it back in the same batch.
- **Capability negotiation** (``DFTU_EXT_COMMS``) is discovery: does a
  compatible producer exist at all, and at what version, so a consumer can
  adapt its behavior before wiring itself to a port.

Most plugins only need ports; reach for capability negotiation when a
consumer should behave differently depending on whether its producer was
loaded.

Ports: publish and consume
----------------------------

A port is a batch-scoped slot, identified by a capability id string, that
resets between batches. A producer writes to it during the batch fold; a
consumer reads it back during the same batch.

.. tab-set::

   .. tab-item:: C++ (SDK)

      The ``Host`` wrapper (``dftracer/utils/plugins/plugin.h``) exposes ports
      as a typed pair, ``publish_port<T>``/``consume_port<T>``, built once from
      a capability id and used every batch through ``send``/``recv``:

      .. code-block:: cpp

         template <class T> OutPort<T> publish_port(const char* cap_id) const;
         template <class T> InPort<T> consume_port(const char* cap_id) const;

         // OutPort<T>: void send(const T& v) const;
         // InPort<T>:  std::optional<T> recv() const;

      ``T`` must be trivially copyable; the host copies ``sizeof(T)`` bytes per
      publish. A producer built with ``dftracer::utils::plugins::make_plugin<Slice>``
      publishes from its ``step``:

      .. code-block:: cpp

         #include <dftracer/utils/plugins/plugin.h>

         constexpr const char* PORT_CAP = "com.example.perbatch";

         struct ProducerSlice {
             explicit ProducerSlice(const dftracer::utils::plugins::Config&) {}
             void step(const dftracer::utils::plugins::Batch& b, dftracer::utils::plugins::Host h) {
                 std::uint64_t with_dur = 0;
                 for (const dftracer::utils::plugins::Event& e : b)
                     if (e.has_dur()) ++with_dur;
                 h.publish_port<std::uint64_t>(PORT_CAP).send(with_dur);
             }
             void merge(ProducerSlice&) {}
             void finalize(dftracer::utils::plugins::Host) {}
         };

         dftu_plugin* dftracer_plugin(const dftu_value* config) {
             return dftracer::utils::plugins::make_plugin<ProducerSlice>(config);
         }

      A consumer reads it back with the same capability id:

      .. code-block:: cpp

         struct ConsumerSlice {
             explicit ConsumerSlice(const dftracer::utils::plugins::Config&) {}
             void step(const dftracer::utils::plugins::Batch&, dftracer::utils::plugins::Host h) {
                 if (auto with_dur = h.consume_port<std::uint64_t>(PORT_CAP).recv()) {
                     // ... use *with_dur for this batch
                 }
             }
             void merge(ConsumerSlice&) {}
             void finalize(dftracer::utils::plugins::Host) {}
         };

   .. tab-item:: C (raw ABI)

      A plugin fetches the raw port channel from the host through
      ``host->get_extension(host->h, DFTU_EXT_PORTS)``:

      .. code-block:: c

         typedef struct dftu_ext_ports {
             uint64_t (*port_key)(void* h, const char* cap_id);
             void (*publish)(void* h, uint64_t key, const void* data, uint32_t len);
             const void* (*consume)(void* h, uint64_t key, uint32_t* out_len);
         } dftu_ext_ports;

      ``port_key`` interns a capability id string once (cache the returned
      ``uint64_t``); ``publish`` copies ``len`` bytes into the batch-scoped
      slot; ``consume`` returns a borrowed pointer valid only until the
      current ``on_batch`` returns (NULL if nothing was published this batch).
      A producer's ``on_batch`` looks like:

      .. code-block:: c

         static dftu_task* on_batch(void* slice, const dftu_batch* b,
                                    const dftu_host* host) {
             MyState* s = (MyState*)slice;
             const dftu_ext_ports* ports =
                 (const dftu_ext_ports*)host->get_extension(host->h, DFTU_EXT_PORTS);
             if (!s->port_key) s->port_key = ports->port_key(host->h, "com.example.perbatch");

             uint64_t with_dur = 0;
             for (uint32_t i = 0; i < b->count; ++i)
                 if (b->events[i].has_dur) ++with_dur;
             ports->publish(host->h, s->port_key, &with_dur, sizeof(with_dur));
             return NULL;  /* synchronous */
         }

      and a consumer reads it back with the same key:

      .. code-block:: c

         static dftu_task* on_batch(void* slice, const dftu_batch* b,
                                    const dftu_host* host) {
             MyState* s = (MyState*)slice;
             const dftu_ext_ports* ports =
                 (const dftu_ext_ports*)host->get_extension(host->h, DFTU_EXT_PORTS);
             if (!s->port_key) s->port_key = ports->port_key(host->h, "com.example.perbatch");

             uint32_t len = 0;
             const void* v = ports->consume(host->h, s->port_key, &len);
             if (v && len == sizeof(uint64_t)) {
                 uint64_t with_dur; memcpy(&with_dur, v, sizeof(with_dur));
                 /* ... use with_dur for this batch */
             }
             return NULL;
         }

   .. tab-item:: Python (JIT)

      A ``@jit.plugin`` class declares a port with ``jit.publish`` /
      ``jit.consume`` as a class attribute, then writes or reads it inside
      ``@jit.each_event``. A publish port accumulates a per-batch total with
      ``self.<port> += <expr>``; the host sums the per-event contributions and
      publishes the total once the batch finishes. A consume port is read as a
      plain scalar, ``self.<port>`` (``0`` when no producer has published yet):

      .. code-block:: python

         from dftracer.utils import jit

         @jit.plugin
         class Producer:
             sig = jit.publish("com.example.perbatch", of=jit.u64)

             @jit.each_event
             def step(self, e):
                 if e.has_dur:
                     self.sig += 1

         @jit.plugin
         class Consumer:
             sig = jit.consume("com.example.perbatch", of=jit.u64, required=True)

             @jit.each_event
             def step(self, e):
                 if self.sig > 0:               # a producer published this batch
                     pass                        # ... use self.sig for this batch

      ``of`` picks the wire width shared by both ends (``jit.u64`` / ``jit.i64``
      / ``jit.f64``); it must match on both the publisher and the consumer. A
      consume port is read-only and a publish port is write-only; writing to a
      consume port (or reading a publish port) raises ``JitError`` at
      decoration time. ``required=True`` makes a missing producer fail the
      ``PluginHost`` plugin-set build; the default (``required=False``) degrades to
      reading ``0``. A JIT producer and a hand-written C++ or C consumer (or
      vice versa) interoperate freely: all three compile down to the same
      ``DFTU_EXT_PORTS`` machinery.

The ordering rule
-------------------

A consumer sees nothing published (``std::nullopt`` in C++, NULL in C, ``0``
in JIT) whenever the producer has not published for the current batch - most
commonly because it runs after the consumer. A producer must run before its
consumer in the fold order, which by default is the order plugins were
registered (``--plugin a --plugin b`` on the command line, or the injection
order into ``PluginHost``). Order your ``--plugin`` flags accordingly, or use
capability requirements (below) to have the build order them for you.

A value published in one batch does not carry over to the next: the host
clears every port between batches, so a consumer sees nothing published until
that batch's producer publishes again.

In C++, ``recv()`` copies the published bytes into the returned
``std::optional<T>``, so the result is safe to keep past the current
``on_batch``/``step`` call without any manual copying. In C, ``consume``
returns a pointer that is only valid until ``on_batch`` returns; copy it out
if you need it longer.

Discovering a producer: capabilities
---------------------------------------

Ports alone assume both plugins are loaded and agree out of band on a
capability id. When a consumer should adapt to whether a compatible producer
is present - and at what version - a plugin declares what it **provides** and
**requires**.

.. tab-set::

   .. tab-item:: C++ (SDK)

      A ``Slice`` declares the three hooks as optional ``static`` members and
      ``dftracer::utils::plugins::make_plugin<Slice>`` wires ``get_extension`` to a
      per-Slice ``dftu_plugin_comms`` for you (nothing is emitted for a Slice
      that declares none). Build the entries with
      ``dftracer::utils::plugins::capability`` / ``dftracer::utils::plugins::requirement``;
      the consumer hook is ``requires_caps`` because ``requires`` is a C++20
      keyword, and the resolve hook is ``on_resolve(Host)``:

      .. code-block:: cpp

         struct ConsumerSlice {
             explicit ConsumerSlice(const dftracer::utils::plugins::Config&) {}
             void step(const dftracer::utils::plugins::Batch&, dftracer::utils::plugins::Host) {}
             void merge(ConsumerSlice&) {}
             void finalize(dftracer::utils::plugins::Host) {}

             static std::array<dftu_requirement, 1> requires_caps() {
                 return {dftracer::utils::plugins::requirement("com.example.tag", DFTU_VER_CARET,
                                                  1, 0, 0, /*required=*/false)};
             }
             static void on_resolve(dftracer::utils::plugins::Host h) {
                 if (auto ver = h.provider_best(requires_caps()[0])) {
                     // wired: a compatible provider exists at version *ver
                 } else {
                     // standalone fallback
                 }
             }
         };

      A provider mirrors this with a ``static provides()`` returning a range
      of ``dftu_capability``, built with ``dftracer::utils::plugins::capability(...)``.
      ``Host::provider_count`` / ``Host::provider_best`` (used above) query the
      registry from inside ``on_resolve``.

   .. tab-item:: C (raw ABI)

      A plugin declares what it provides and requires through
      ``dftu_plugin_comms``, fetched by the host from the plugin's own
      ``get_extension``:

      .. code-block:: c

         typedef struct dftu_plugin_comms {
             uint32_t (*provides)(void* self, dftu_capability* out, uint32_t max);
             uint32_t (*require_caps)(void* self, dftu_requirement* out, uint32_t max);
             void (*resolve)(void* self, const dftu_host* host);
         } dftu_plugin_comms;

      ``provides`` lists the capability ids (and semantic versions) this
      plugin offers; ``require_caps`` lists what it wants, each with a version
      constraint (``DFTU_VER_GE`` / ``GT`` / ``LE`` / ``LT`` / ``EQ`` /
      ``CARET`` / ``TILDE``) and whether it is required or optional. The host
      calls every plugin's ``resolve`` once, after every plugin has declared,
      passing the ``dftu_host`` so ``resolve`` can query the registry through
      ``DFTU_EXT_COMMS``:

      .. code-block:: c

         typedef struct dftu_ext_comms {
             uint32_t (*provider_count)(void* h, const char* cap_id);
             int (*provider_best)(void* h, const dftu_requirement* req,
                                  dftu_version* out_ver);
         } dftu_ext_comms;

      A consumer's ``resolve`` typically calls ``provider_best`` for each of
      its requirements and switches behavior (wired vs. standalone) based on
      whether a compatible provider was found:

      .. code-block:: c

         static void resolve(void* self, const dftu_host* host) {
             MyState* s = (MyState*)self;
             const dftu_ext_comms* c =
                 (const dftu_ext_comms*)host->get_extension(host->h, DFTU_EXT_COMMS);
             if (!c) return;
             dftu_version best;
             if (c->provider_best(host->h, &s->requirement, &best) == 0) {
                 s->wired = 1;         /* a compatible provider exists: use the port */
             } else {
                 s->wired = 0;         /* fall back to standalone behavior */
             }
         }

         static const dftu_plugin_comms g_comms = { my_provides, my_require_caps, resolve };

         static const void* get_extension(void* self, const char* ext_id) {
             if (strcmp(ext_id, DFTU_EXT_COMMS) == 0) return &g_comms;
             return NULL;
         }

   .. tab-item:: Python (JIT)

      ``jit.publish`` and ``jit.consume`` put a plugin into the same
      provides/requires graph as the C and C++ sides, and both carry a
      semantic version. A producer declares the version it **provides** with
      ``version=``; a consumer declares a version **constraint** on its
      provider with ``min_version=`` (a ``>=`` constraint by default, or pass
      ``version_op=`` for one of ``>= > <= < = ^ ~``, the ABI's ``DFTU_VER_*``
      operators):

      .. code-block:: python

         from dftracer.utils import jit

         @jit.plugin
         class Producer:
             seen = jit.map(key=(jit.i64,), value=jit.count())
             tag = jit.publish("com.example.tag", of=jit.u64, version="2.1.0")

             @jit.each_event
             def step(self, e):
                 self.seen[(e.pid,)] += 1
                 self.tag += 1

      A consumer adapts at resolve time with a ``@jit.on_resolve`` method. It
      runs once, after every plugin has declared, and reads for each consume
      port whether a provider satisfying its constraint was found
      (``self.<port>.resolved``) and that provider's version
      (``self.<port>.version``, comparable to a ``(major, minor, patch)``
      tuple). It assigns the outcome to plugin **flags** that ``each_event``
      then branches on:

      .. code-block:: python

         @jit.plugin
         class Consumer:
             got = jit.map(key=(jit.i64,), value=jit.count())
             tag = jit.consume("com.example.tag", of=jit.u64, min_version="2.0.0")

             @jit.on_resolve
             def on_resolve(self):
                 self.wired = self.tag.resolved
                 self.modern = self.tag.resolved and (self.tag.version >= (2, 1, 0))

             @jit.each_event
             def step(self, e):
                 if self.wired > 0:          # a compatible provider exists
                     self.got[(e.pid,)] += 1
                 if self.modern > 0:         # ... and it is >= 2.1.0
                     self.got[(e.pid,)] += 1

      Under the hood ``on_resolve`` compiles to the same ``comms_resolve`` /
      ``provider_best`` machinery as the C and C++ tabs: the host always
      records each consume port's ``resolved`` / ``version`` before the body
      runs, so even an empty ``on_resolve`` (or none at all) captures provider
      presence. The flags are file-scope state written once at resolve and
      read by every batch, so ``each_event`` reads them but does not write
      them.

      **What the resolve body can and cannot express.** It is a sequence of
      ``self.<flag> = <expr>`` assignments; the expression may read a consume
      port's ``.resolved`` (a 0/1 int) and ``.version`` (compared to a version
      tuple), combined with ``and`` / ``or`` / ``not``, integer comparisons,
      and integer literals. It cannot run arbitrary Python, call host services
      other than the automatic ``provider_best`` query, or set per-worker (as
      opposed to per-plugin) state - the same shape as a C++ ``on_resolve``
      that sets a flag. For logic beyond that, author the plugin in C or C++
      with ``dftu_plugin_comms`` / ``Slice::on_resolve`` above. A guard reads a
      flag as an integer, so write ``if self.wired > 0:`` rather than a bare
      ``if self.wired:``.

Two more rules govern the registry:

- **The dftu. id prefix is reserved for the host.** A plugin declaring a
  ``dftu.*`` capability (or providing one of the host's blessed ids such as
  ``DFTU_CAP_EVENTS``) fails the plugin-set build for the whole run. Requiring a
  blessed id is fine - only providing one is rejected.
- **An unmet required capability fails the build.** A requirement with
  ``required = 1`` and no satisfying provider makes ``Plugins::Builder::build()``
  return an error; an optional one (``required = 0``) just leaves the consumer
  unwired, as in the examples above.
- **A provide/require edge orders the fold**: when plugin B requires
  something plugin A provides, the host's resolved fold order runs A before
  B (matching the ports ordering rule) regardless of ``--plugin`` order. A
  provide/require cycle falls back to declaration order rather than failing.

See also
--------

- :doc:`../../plugins` for the plugin lifecycle (``make_slice`` /
  ``on_batch`` / ``merge`` / ``on_finalize``) these channels plug into.
- :doc:`../../jit` for the full JIT walkthrough, including ports and
  cross-worker shared handles.
- :doc:`../../cpp_api/plugins` for the full C ABI reference.
