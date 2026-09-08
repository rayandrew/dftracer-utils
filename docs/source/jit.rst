:description: Author a DFTracer plugin in Python: an @jit.plugin class is AST-compiled to a native plugin that runs at native speed inside the fused scan.

JIT Plugins
===========

.. seealso::

   :doc:`api/jit` for the JIT module API reference.

The JIT layer lets you author a plugin in Python and get a native one. A class
decorated with ``@jit.plugin`` is AST-compiled to a C plugin against the stable
:doc:`plugins` ABI, built to a cached shared library, and loaded through
``PluginHost`` exactly like a hand-written plugin - so a Python-authored plugin
runs at native speed inside the same fused scan.

Overview
--------

- **Declare maps, write one method**: a plugin is a class with ``jit.map``
  attributes (the aggregations it maintains) and a single ``@jit.each_event``
  method (what to do per event).
- **Typed keys and rich values**: a map's key is a tuple of typed fields
  (``jit.i64``, ``jit.str_``, ``jit.bytes``) and its value is an aggregate. The
  vocabulary is broad: ``count``, ``sum``, ``min`` / ``max``, ``mean``,
  ``variance`` / ``stddev``, ``quantiles`` (DDSketch percentiles), ``argmin`` /
  ``argmax``, ``topk`` / ``bottomk`` / ``approx_topk``, ``sample``,
  ``distinct``, ``set``, ``list``, and ``record`` (several aggregates on one
  key). A ``jit.map`` lowers to a host aggregation accumulator keyed by its key
  columns - the same ``DFTU_EXT_AGG`` surface a C plugin uses.
- **Statically checked**: the structured ``each_event`` body is a deliberately
  small subset (an optional ``if`` guard around one map update); anything outside
  it raises ``JitError`` at decoration time - never a silent miscompile - and
  ``jit.map`` infers the ``Map[tuple[K], V]`` type so a type checker validates
  your keys and values.
- **Compiled once, computed once**: a subexpression used by more than one map in
  the same block - a hash key like ``jit.fastrange(jit.mix64(e.hhash), 256)``
  fed to several maps - is hoisted to a local and evaluated a single time per
  event, never re-run per map. The hoist stays inside
  its ``if`` guard, so a guarded expression keeps its execution condition.
- **One lookup per key**: several plain maps always subscripted at the same key
  (``self.n[(e.pid,)] += 1``; ``self.tot[(e.pid,)] += e.dur``; ...) are coalesced
  into one internal table, so the per-event hash lookup happens once for all of
  them instead of once per map. They still come back as the separate tables you
  declared - the coalescing is invisible, like the CSE above.

Tutorial: a per-file event counter
-----------------------------------

This is the Python equivalent of the C plugin in :doc:`plugins` - it counts,
per ``(pid, event-name)``, how many events referenced a file. Declare the map,
write the one per-event method:

.. code-block:: python

   from dftracer.utils import jit

   @jit.plugin
   class NameEdges:
       edges = jit.map(key=(jit.i64, jit.str_), value=jit.count())

       @jit.each_event
       def step(self, e):
           if e.fhash != jit.NONE:            # only file-referencing events
               self.edges[(e.pid, e.name)] += 1

The event ``e`` exposes the parsed fields (``pid``, ``tid``, ``name``, ``cat``,
``ts``, ``dur``, ``fhash``, ``hhash``, and args when declared); ``jit.NONE`` is
the absent-field sentinel. ``jit.map`` infers ``Map[tuple[int, str], int]`` so a
type checker validates the subscript and the ``+= 1``.

Running it
----------

There is no build step to run yourself - loading the class compiles it to a
cached native ``.so`` and runs it through
:class:`~dftracer.utils.plugins.PluginHost`, exactly like a hand-written plugin.
Each map comes back as a ``pyarrow.Table`` under its attribute name:

.. code-block:: python

   from dftracer.utils.plugins import PluginHost

   host = PluginHost()
   host.load(NameEdges)                  # compiles + caches the .so, then loads
   results = host.run("./traces")
   df = results["edges"].to_pandas()     # columns: pid, name, value

Working with results (NumPy / pandas)
-------------------------------------

The per-event fold runs in compiled C, not Python - there is no NumPy in the hot
loop, which is the point. The interop happens on the *output*: each map is a
small ``pyarrow.Table`` that crosses to NumPy or pandas cheaply (zero-copy where
the dtype allows).

.. code-block:: python

   tbl = results["edges"]                       # pyarrow.Table
   values = tbl.column("value").to_numpy(zero_copy_only=False)   # np.ndarray
   df = tbl.to_pandas()                          # pandas.DataFrame

   # from here it is ordinary NumPy / pandas
   top = df.sort_values("value", ascending=False).head(10)
   total = int(values.sum())

So the pattern is: aggregate in the fused native scan, then do the NumPy / pandas
analysis on the (much smaller) result tables. If you instead want vectorized math
over *raw* events, that is the columnar :doc:`DataFrame / View <columnar-engine>`
path, not a plugin.

Richer aggregations
-------------------

The value is any aggregate, and a plugin can keep several maps. This one, per
category, keeps the event count, the total and mean duration, the slowest event
name, and an approximate top-k of event names - in one pass:

.. code-block:: python

   @jit.plugin
   class CatStats:
       n      = jit.map(key=(jit.str_,), value=jit.count())
       total  = jit.map(key=(jit.str_,), value=jit.sum())
       avg    = jit.map(key=(jit.str_,), value=jit.mean())
       slow   = jit.map(key=(jit.str_,), value=jit.max())
       worst  = jit.map(key=(jit.str_,), value=jit.argmax(of=jit.str_))
       hot    = jit.map(key=(jit.str_,), value=jit.approx_topk(16))

       @jit.each_event
       def step(self, e):
           self.n[(e.cat,)]     += 1               # additive aggregates use +=
           self.total[(e.cat,)] += e.dur           # (an arithmetic expr is fine)
           self.avg[(e.cat,)].observe(e.dur)       # others observe a value
           self.slow[(e.cat,)].observe(e.dur)
           self.worst[(e.cat,)].observe(e.name, by=e.dur)   # name at the max dur
           self.hot[(e.cat,)].observe(e.name)               # frequent event names

Two contribution shapes cover the vocabulary: additive aggregates (``count``,
``sum``) take ``+= <expr>``, and every other aggregate takes ``.observe(x)`` -
with a ``by=`` keyword for the argument-style aggregates (``argmin`` /
``argmax`` / ``topk`` / ``bottomk``) that keep a payload at the extreme of a
score. A single key can carry several aggregates at once with a ``record``:
``value=dict(n=jit.count(), dur=jit.sum())`` then ``self.m[(k,)].n += 1`` /
``self.m[(k,)].dur += e.dur``.

Percentiles
-----------

``jit.quantiles`` keeps a mergeable DDSketch per key, so p50/p90/p99-style
latency percentiles come out of the one pass without collecting the values.
Pass the quantiles you want; each becomes a result column:

.. code-block:: python

   @jit.plugin
   class Lat:
       lat = jit.map(key=(jit.str_,), value=jit.quantiles(qs=(0.5, 0.9, 0.99)))

       @jit.each_event
       def step(self, e):
           self.lat[(e.cat,)].observe(e.dur)

The map materializes to a ``count`` column plus one f64 column per quantile,
named ``p50``, ``p90``, ``p99`` (a non-integer like ``0.999`` becomes
``p99_9``). The estimates are approximate - DDSketch holds ~1% relative error -
which is what makes them mergeable and O(1) in memory.

Numeric primitives
------------------

A key or value expression can call a small set of numeric primitives - the
operations that are non-obvious or slow to spell out in C by hand. Each lowers
to the matching header-only helper in ``prims.h`` (``jit.ilog2`` ->
``dftu_ilog2_u64``), so there is no Python cost and no divergence from the C. A
log2 duration histogram, keyed by bucket:

.. code-block:: python

   @jit.plugin
   class DurHist:
       hist = jit.map(key=(jit.i64,), value=jit.count())

       @jit.each_event
       def step(self, e):
           self.hist[(jit.ilog2(e.dur),)] += 1     # 0..63 log-scale bucket

The vocabulary. The integer ops are 64-bit (a Python ``int`` is unbounded, but
these compile to 64-bit C and wrap there); the float ops are ``double``:

- **bit ops**: ``ilog2`` (floor log2 / top set bit), ``bit_width``, ``clz``,
  ``ctz``, ``popcount``, ``ceil_pow2``, ``floor_pow2``, ``rotl``, ``rotr``.
- **integer math**: ``abs``, ``clamp(x, lo, hi)``, ``div_ceil``, ``div_round``,
  ``align_up(x, a)``, ``align_down(x, a)``, ``isqrt``, ``gcd``.
- **fast hashing**: ``mix64`` (SplitMix64 finalizer), ``mul_hi``, and
  ``fastrange(x, n)`` (map a hash into ``[0, n)`` with no modulo) - e.g.
  ``jit.fastrange(jit.mix64(e.hhash), 256)`` to bucket ids into 256 shards.
- **float math**: ``sqrt``, ``log2`` / ``log`` / ``exp`` (unreachable from a
  structured body otherwise), ``fma(a, b, c)`` (``a*b + c``, one rounding),
  ``lerp(a, b, t)``, ``fmin`` / ``fmax``, ``clampf(x, lo, hi)``,
  ``copysign(x, y)``.

These are authoring markers: they only compile inside a ``@jit.each_event``
body, and calling one directly in Python raises ``JitError``. The same helpers
are available to hand-written and raw-body plugins as ``dftu_*`` from
``<dftracer/utils/plugins/prims.h>``.

Talking to another plugin
--------------------------

Two JIT plugins loaded in the same run can pass a batch-scoped value between
them: ``jit.publish`` declares a port this plugin **provides**, ``jit.consume``
one it **requires** - both keyed by a capability id string, the same
mechanism the hand-written C++ ``Host::publish_port`` / ``consume_port`` API
uses (see :doc:`guides/plugins/inter-plugin-comms`). The host resolves the
producer to run before the consumer regardless of load order.

A publish port is written with ``self.<port> += <expr>`` inside
``each_event``: the host sums the per-event contributions into one per-batch
total and publishes it once the batch finishes. A consume port is read as a
plain scalar, ``self.<port>``, giving the value the producer published for
the current batch (``0`` when no producer has published yet):

.. code-block:: python

   @jit.plugin
   class Producer:
       seen = jit.map(key=(jit.i64,), value=jit.count())
       sig = jit.publish("com.example.batchsig", of=jit.u64)

       @jit.each_event
       def step(self, e):
           self.seen[(e.pid,)] += 1
           self.sig += 1

   @jit.plugin
   class Consumer:
       got = jit.map(key=(jit.i64,), value=jit.count())
       sig = jit.consume("com.example.batchsig", of=jit.u64, required=True)

       @jit.each_event
       def step(self, e):
           if self.sig > 0:               # a producer published this batch
               self.got[(e.pid,)] += 1

``of`` picks the wire width shared by both ends (``jit.u64`` / ``jit.i64`` /
``jit.f64``); it must match on both the publisher and the consumer. A
consume port is read-only (writing to it raises ``JitError`` at decoration
time) and a publish port is write-only (reading it likewise raises). With
``required=True`` a missing producer fails the ``PluginHost`` build; the
default (``required=False``) degrades to reading ``0``.

The ordering rule matters here exactly as it does in C++: a producer's
``step`` must run before the consumer's for the same batch, which is why
``jit.consume`` declares a requirement rather than assuming load order - the
host reorders the fold to satisfy it. A JIT plugin can publish for a
hand-written C++ consumer and vice versa; both sides go through the same
``DFTU_EXT_PORTS`` / ``DFTU_EXT_COMMS`` machinery.

Versioned negotiation and ``@jit.on_resolve``
.............................................

Both ports carry a semantic version. A producer states the version it offers
with ``jit.publish(cap_id, version="2.1.0")``; a consumer states a constraint
on its provider with ``jit.consume(cap_id, min_version="2.0.0")`` (a ``>=``
constraint by default, or pass ``version_op=`` for one of ``>= > <= < = ^ ~``,
the ABI's ``DFTU_VER_*`` operators). Versions are ``MAJOR[.MINOR[.PATCH]]``
strings, each component 0..65535.

An ``@jit.on_resolve`` method adapts behavior to whichever providers are
present. It takes ``(self)``, runs once after every plugin has declared, and
its body is a sequence of ``self.<flag> = <expr>`` assignments. Each
expression may read, for a consume port, ``self.<port>.resolved`` (1 if a
provider satisfying the constraint was found, else 0) and ``self.<port>.version``
(the found provider's version, comparable to a ``(major, minor, patch)`` tuple),
combined with ``and`` / ``or`` / ``not``, integer comparisons, and integer
literals. Each ``<flag>`` becomes plugin state a later ``each_event`` reads as
``self.<flag>`` (as an integer, so branch with ``if self.<flag> > 0:``):

.. code-block:: python

   @jit.plugin
   class Consumer:
       got = jit.map(key=(jit.i64,), value=jit.count())
       tag = jit.consume("com.example.tag", of=jit.u64, min_version="2.0.0")

       @jit.on_resolve
       def on_resolve(self):
           self.wired  = self.tag.resolved
           self.modern = self.tag.resolved and (self.tag.version >= (2, 1, 0))

       @jit.each_event
       def step(self, e):
           if self.wired > 0:              # a compatible provider exists
               self.got[(e.pid,)] += 1

The host always records each consume port's ``resolved`` / ``version`` before
the body runs, so even an empty ``on_resolve`` captures provider presence. The
body compiles to the same ``comms_resolve`` / ``provider_best`` C the C and C++
plugins use; it cannot run arbitrary Python or call other host services, so for
logic beyond flag-setting author the plugin in C or C++ (see
:doc:`guides/plugins/inter-plugin-comms`).

Raw bodies: the full ABI without leaving JIT
--------------------------------------------

The structured subset covers the common counting and aggregation shapes safely.
For anything it cannot express, ``@jit.each_event(raw=True)`` makes the method
return a C/C++ string that is spliced verbatim into the per-event loop - so the
per-event body has the full :doc:`plugins` ABI, while you keep the JIT
framework: the typed ``jit.map`` declarations, the aggregate vocabulary above,
and the compile / cache / load machinery. In scope inside a raw body are
``host`` (the ``dftu_host``) and ``e`` / ``b`` / ``i`` (the current event,
batch, and index).

So JIT covers everything the C ABI can: the structured subset for the common
case, a raw body for the rest - both compiled to one native plugin. A body that
steps outside the structured subset without ``raw=True`` raises ``JitError``
pointing here, never a silent miscompile.
