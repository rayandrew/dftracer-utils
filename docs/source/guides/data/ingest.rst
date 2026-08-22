:description: Import from pandas, polars, Arrow, Parquet, dicts, or NumPy and hand data back out, crossing the Arrow C Data Interface without copying.

Ingest and interoperate with other data tools
================================================

``DataFrame`` and ``Series`` are not an island: every ``from_*`` classmethod
imports another library's data, and every ``to_*`` method hands it back out,
so you can keep the rest of your analysis in pandas, polars, or plain NumPy
while dftracer-utils does the columnar work. Most of these paths cross the
boundary through the `Arrow C Data Interface
<https://arrow.apache.org/docs/format/CDataInterface.html>`_ - a shared memory
layout both sides read directly - so importing a ``pyarrow.Table`` or a
``polars.DataFrame`` does not copy the underlying buffers.

Bring data in
-------------

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         from dftracer.utils import DataFrame, Series

         df = DataFrame.from_pandas(pandas_df)
         df = DataFrame.from_arrow(pyarrow_table)
         df = DataFrame.from_polars(polars_df)
         df = DataFrame.from_parquet("data.parquet", columns=["dur", "cat"])
         df = DataFrame.from_dict({"cat": ["POSIX", "STDIO"], "dur": [100, 200]})
         df = DataFrame.from_numpy(arr, columns=["a", "b"])  # 2-D array, or {name: 1-D array}

         s = Series.from_pandas(pandas_series)
         s = Series.from_arrow(pyarrow_array)
         s = Series.from_polars(polars_series)
         s = Series.from_numpy(numpy_array)
         s = Series.from_list([1, 2, 3])            # optional dtype= for a pyarrow type

Each path takes a different route to the same zero-copy interface:

- ``Series.from_arrow`` accepts any object implementing the Arrow PyCapsule
  protocol (``__arrow_c_array__``) and imports it with no copy.
  ``DataFrame.from_arrow`` takes a ``pyarrow.Table`` or ``RecordBatch`` (or
  anything exposing the same ``column_names`` / ``column()`` duck-typed API)
  and imports each column the same zero-copy way.
- ``from_pandas`` and ``from_polars`` go through Arrow too: pandas via
  ``pyarrow.Table.from_pandas``/``pyarrow.Array.from_pandas``, polars via its
  own ``to_arrow()`` (polars already stores its columns as Arrow, so this leg
  is zero-copy on the polars side).
- ``from_parquet`` reads through ``pyarrow.parquet``.
- ``from_dict`` builds a ``pyarrow.table`` from the mapping first.
- ``Series.from_numpy`` has its own fast path: a 1-D, C-contiguous,
  fixed-width numeric array is borrowed directly (no pyarrow, no copy); any
  other shape or dtype falls back to the Arrow path, which does need pyarrow
  installed. ``DataFrame.from_numpy`` builds each column through
  ``Series.from_numpy`` and assembles them with pyarrow.
- ``Series.from_list`` is the one path with no zero-copy shortcut - a Python
  list has no buffer to share - but still lands on the same Arrow-backed
  ``Series`` as everything else, so it composes with the rest of the API the
  same way. Pass ``dtype=`` (a pyarrow type) to pin the imported type instead
  of letting pyarrow infer it.

Every ``from_*`` path except ``from_arrow`` requires pyarrow to be installed
(``pip install pyarrow``); ``from_arrow`` itself does not, since it only needs
the capsule protocol, not the pyarrow package.

Get data out
------------

.. tab-set::

   .. tab-item:: Python

      .. code-block:: python

         df.to_arrow()      # pyarrow.Table, zero-copy
         df.to_pandas()      # pandas.DataFrame
         df.to_polars()      # polars.DataFrame (needs polars installed)

         s.to_arrow()        # pyarrow.Array, zero-copy
         s.to_pandas()        # pandas.Series
         s.to_numpy()          # NumPy array
         s.to_polars()          # polars.Series (needs polars installed)

``to_arrow()`` is the zero-copy exit on both classes; ``to_pandas()`` and
``to_polars()`` are built on top of it (one more Arrow-to-library conversion,
which is where pandas or polars may copy on their own side). ``Series.to_numpy()``
has the same fast path as the import direction: a flat, non-null, fixed-width
numeric column is read straight from the native buffer with no pyarrow
involved; a Bool column, a column with nulls, a String column, or a non-flat
encoding (dictionary/selection) falls back through the Arrow path instead,
which needs pyarrow. ``DataFrame`` has no ``to_numpy()`` - convert the column
you want with ``df["dur"].to_numpy()``, or go through ``to_pandas()``/
``to_arrow()`` for the whole frame.

``np.asarray(series)`` also works directly (``Series`` implements the NumPy
array protocol), and Series arithmetic (``+ - * /``) and comparisons against
another ``Series`` or a Python scalar return a ``Series``, so you rarely need
to round-trip through NumPy just to do elementwise math.

A worked round trip
--------------------

.. code-block:: python

   import pandas as pd
   from dftracer.utils import Series, TraceViewer

   result = (
       TraceViewer("traces/")
       .group_by("cat")
       .agg("count", "mean:dur")
       .collect()
   )

   # Do the rest of the analysis in pandas:
   pdf = result.to_pandas()
   pdf["mean_dur"].plot.bar()

   # Or hand a NumPy array straight to something that only wants numbers:
   import numpy as np
   durs = result["mean_dur"].to_numpy()
   np.percentile(durs, 90)

   # Bring an externally-computed column back in:
   adjusted = pd.Series(durs * 1.1, name="adjusted_mean_dur")
   result_with_adjustment = result.with_column(
       "adjusted_mean_dur", Series.from_pandas(adjusted)
   )

The C++ engine and Arrow
--------------------------

The C++ ``dataframe::Series``/``dataframe::DataFrame`` API exposes a zero-copy
Arrow bridge through the installed public header
``dftracer/utils/dataframe/arrow.h``: ``Series::to_arrow`` /
``DataFrame::to_arrow`` return an ``OwnedArrow`` (a move-only owner of the
paired ``ArrowSchema`` / ``ArrowArray``), and ``Series::from_arrow(schema,
array)`` / ``DataFrame::from_arrow(schema, array)`` import back. A
``DataFrame`` is represented as a struct array with one child per column. The
bridge sits on the same C ABI a C caller would use: the flat-column functions
(``dftu_series_new_flat``, ``dftu_series_data``, and the Arrow-layout validity
bitmap they take) already speak Arrow's physical layout, so a C or C++ program
importing from another Arrow-based library builds columns through that
boundary - see :doc:`../core/c-abi`.

See also
--------

- :doc:`dataframe` and :doc:`series` for the rest of the ``DataFrame``/``Series``
  API once your data is in.
- :doc:`../core/c-abi` for the C ABI's own zero-copy column boundary.
