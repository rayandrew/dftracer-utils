:description: Operate on a native DataFrame in the SIMD engine: project, filter, sort, derive, and reshape columns before converting at the edge.

Compute and reshape columns
===========================

A query returns a native ``DataFrame`` (a set of typed ``Series``). Everything
below stays in the SIMD engine - Arrow, pandas, NumPy, and polars appear only at
the edge, zero-copy, when you ask. See :doc:`../../columnar-engine` for the
engine internals.

Get a DataFrame
---------------

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from dftracer.utils import TraceViewer

         df = TraceViewer("traces/").group_by("cat").agg("count", "sum:dur").collect()
         # or ingest one: DataFrame.from_pandas(pdf) / from_arrow(tbl) / from_numpy(a, columns=...)

   .. tab-item:: C++

      .. code-block:: cpp

         #include <dftracer/utils/trace/views/view.h>
         using namespace dftracer::utils::trace::views;

         auto df = View::from_file("trace.pfw.gz")
                       .group_by({GroupKey::cat()})
                       .agg({{AggOp::Count, "", "count"},
                             {AggOp::Sum, "dur", "sum_dur"}})
                       .collect()
                       .get();

Derived columns
---------------

Build a column expression from ``F.<name>`` and evaluate it with ``.apply`` - no
wrapper needed. Mix columns and scalars; ``+ - * /``, comparisons, and the
numeric prims (``ilog2``, ``popcount``, ...) all lower to SIMD kernels. This
``F`` / ``.apply`` expression layer is **Python-only**; from C++, build the same
derived column directly with ``Series`` arithmetic and ``with_column`` (see
:doc:`../data/series`).

.. code-block:: python

   from dftracer.utils import F, lit

   avg = (F.sum_dur / F.count).apply(df)          # Series
   avg = df.apply(F.sum_dur / F.count)            # frame-first spelling, same result
   ms = (F.dur * lit(0.001)).apply(df)            # int * float -> float
   bucket = F.dur.ilog2().apply(df)               # log2 histogram bucket

Both spellings evaluate the same way; ``df.apply(expr)`` is
:meth:`DataFrame.apply <dftracer.utils.DataFrame.apply>`, useful when chaining
off a frame variable reads better than starting from the expression.

To evaluate several expressions in one CSE'd pass (a shared subexpression is
computed once), use :func:`~dftracer.utils.eval_many`:

.. code-block:: python

   from dftracer.utils import eval_many
   a, b = eval_many([F.sum_dur / F.count, (F.sum_dur / F.count) * lit(2)], df)

DataFrame operations
--------------------

All of these stay in the engine and back both the Python API and the C++ engine;
the C++ ``DataFrame`` (``dftracer/utils/dataframe/dataframe.h``) carries the same
members under the same names. For side-by-side C++ and Python tabs of each op see
:doc:`../data/dataframe`, :doc:`../data/series`, :doc:`../data/joins`,
:doc:`../data/reshape`, and :doc:`../data/time-windows`.

- **project**: ``select("cat", "count")``, ``rename({"count": "n"})``, ``with_column("avg", series)``
- **rows**: ``filter(mask)``, ``head(n)`` / ``tail(n)``, ``slice(off, len)``, ``reverse()``, ``sample(n, seed)``, ``sort_by("count", descending=True)``, ``topk("count", 10)``
- **reshape**: ``unique()`` / ``drop_duplicates()``, ``drop_nulls()``, ``fill_null(v)``, ``unpivot(...)`` / ``melt(...)``, ``explode("col")``, ``to_dummies("cat")``, ``pivot(index, on, values)``
- **summarize**: ``group_by("cat", aggs)``, ``describe()``, ``null_count()``, ``value_counts`` (on a Series)
- **relational**: ``join(other, how="inner", on=...)``
- **windowed**: ``group_by_dynamic(time_col, every, period, aggs)``

.. code-block:: python

   worst = df.sort_by("sum_dur", descending=True).head(10)
   hist = df["cat"].value_counts()                # a two-column DataFrame

Series operations
-----------------

A ``Series`` is NumPy-like: ``a + b``, ``a * 2``, ``s[0]``, ``s[2:5]``,
``np.asarray(s)``, plus reducers and kernels.

- **reduce**: ``sum() min() max() mean() product() count() arg_min() arg_max() mode() all() any()``, ``quantile(q) median() stddev() variance() skewness() kurtosis()``
- **element-wise**: ``abs() clip(lo, hi) round() ceil() floor() trunc() sign() sqrt() exp() log() fillna(v)``
- **scan / window**: ``cumsum() cummax() cummin() diff() pct_change() shift(n) rolling(w, op)``
- **select / test**: ``sort() head(n) reverse() top_k(k) unique() drop_nulls() is_in(values) is_nan() is_finite() is_sorted()``
- **strings**: ``str_contains() str_starts_with() str_ends_with() str_like() str_matches(re) to_lowercase() str_len_bytes() str_split(sep)`` ...

.. code-block:: python

   p99 = df["sum_dur"].quantile(0.99)
   posix = df.filter(df["cat"].str_eq("POSIX"))

Convert out (only at the edge)
------------------------------

.. code-block:: python

   df.to_arrow()     # pyarrow.Table (zero-copy)      df["count"].to_numpy()
   df.to_pandas()    # pandas DataFrame               df["count"].to_pandas()
   df.to_polars()    # polars DataFrame
