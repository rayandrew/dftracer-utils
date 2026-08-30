// A native vec column exposed to Python: the DSL and pipeline operate on our
// own SIMD columnar format (dataframe::Series, Highway kernels) and only cross
// to Arrow at the edges - construction from a pyarrow array and export via the
// Arrow PyCapsule interface (__arrow_c_array__).

#include <dftracer/utils/core/common/config.h>  // DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/python/series.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/dataframe/arrow_bridge.h>
#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/kernels/kernels.h>
#include <dftracer/utils/dataframe/kernels/prims.h>
#include <dftracer/utils/python/dataframe.h>
#include <dftracer/utils/python/py_method.h>
#include <dftracer/utils/python/py_scalar_helpers.h>
#include <dftracer/utils/python/py_type_helpers.h>
#include <nanoarrow/nanoarrow.h>

#include <cstdint>
#include <new>
#include <utility>

namespace dataframe = dftracer::utils::dataframe;

namespace {

using dataframe::Series;
using dataframe::TypeId;

struct SeriesObject {
    PyObject_HEAD Series col;
    // Buffer-protocol shape[0] and strides[0]; a column is immutable, so these
    // are constant and every export can point at them (no per-export alloc).
    Py_ssize_t buf_shape;
    Py_ssize_t buf_stride;
};

PyTypeObject SeriesType;

PyObject* make_series(Series&& col) {
    if (!col.valid()) {
        PyErr_SetString(PyExc_RuntimeError,
                        "dataframe kernel produced a null column");
        return nullptr;
    }
    auto* self =
        reinterpret_cast<SeriesObject*>(SeriesType.tp_alloc(&SeriesType, 0));
    if (!self) return nullptr;
    new (&self->col) Series(std::move(col));
    return reinterpret_cast<PyObject*>(self);
}

Series* as_series(PyObject* o) {
    if (!PyObject_TypeCheck(o, &SeriesType)) {
        PyErr_SetString(PyExc_TypeError, "expected a Series");
        return nullptr;
    }
    return &reinterpret_cast<SeriesObject*>(o)->col;
}

void Series_dealloc(SeriesObject* self) {
    self->col.~Series();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// ---- kernels -------------------------------------------------------------

PyObject* Series_add(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    Series* b = as_series(other);
    if (!a || !b) return nullptr;
    return make_series(dataframe::add(*a, *b));
}
PyObject* Series_sub(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    Series* b = as_series(other);
    if (!a || !b) return nullptr;
    return make_series(dataframe::sub(*a, *b));
}
PyObject* Series_mul(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    Series* b = as_series(other);
    if (!a || !b) return nullptr;
    return make_series(dataframe::mul(*a, *b));
}
PyObject* Series_div(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    Series* b = as_series(other);
    if (!a || !b) return nullptr;
    return make_series(dataframe::div(*a, *b));
}

// A Python number as either an i64 or an f64 scalar; the kernel converts it to
// the column's element type.
enum class ScalarOp { Add, Sub, Mul, Div };

PyObject* scalar_op(PyObject* self, PyObject* value, ScalarOp op) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    if (PyFloat_Check(value)) {
        double v = PyFloat_AsDouble(value);
        if (v == -1.0 && PyErr_Occurred()) return nullptr;
        if (op == ScalarOp::Mul)
            return make_series(dataframe::mul_scalar(*a, v));
        if (op == ScalarOp::Div)
            return make_series(dataframe::div_scalar(*a, v));
        return make_series(
            dataframe::add_scalar(*a, op == ScalarOp::Sub ? -v : v));
    }
    long long v = PyLong_AsLongLong(value);
    if (v == -1 && PyErr_Occurred()) return nullptr;
    auto iv = static_cast<std::int64_t>(v);
    if (op == ScalarOp::Mul) return make_series(dataframe::mul_scalar(*a, iv));
    if (op == ScalarOp::Div) return make_series(dataframe::div_scalar(*a, iv));
    return make_series(
        dataframe::add_scalar(*a, op == ScalarOp::Sub ? -iv : iv));
}

PyObject* Series_add_scalar(PyObject* self, PyObject* v) {
    return scalar_op(self, v, ScalarOp::Add);
}
PyObject* Series_sub_scalar(PyObject* self, PyObject* v) {
    return scalar_op(self, v, ScalarOp::Sub);
}
PyObject* Series_mul_scalar(PyObject* self, PyObject* v) {
    return scalar_op(self, v, ScalarOp::Mul);
}
PyObject* Series_div_scalar(PyObject* self, PyObject* v) {
    return scalar_op(self, v, ScalarOp::Div);
}

PyObject* Series_cast(PyObject* self, PyObject* target) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long tid = PyLong_AsLong(target);
    if (tid == -1 && PyErr_Occurred()) return nullptr;
    return make_series(dataframe::cast(*a, static_cast<TypeId>(tid)));
}

PyObject* Series_prim(PyObject* self, PyObject* code) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long op = PyLong_AsLong(code);
    if (op == -1 && PyErr_Occurred()) return nullptr;
    return make_series(dataframe::prim(*a, static_cast<dataframe::Prim>(op)));
}

// Compare a column against a scalar (op is a DFTU_CMP_* code) -> Bool
// column.
PyObject* Series_compare(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int op = 0;
    PyObject* value = nullptr;
    if (!PyArg_ParseTuple(args, "iO", &op, &value)) return nullptr;
    dftu_scalar s{};
    if (!py_to_scalar(value, &s)) return nullptr;
    return make_series(Series{
        dftu_series_compare(a->handle(), static_cast<dftu_cmp_op>(op), s)});
}

// Boolean AND/OR of two Bool columns (op is a DFTU_LOGICAL_* code).
PyObject* Series_logical(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int op = 0;
    PyObject* other = nullptr;
    if (!PyArg_ParseTuple(args, "iO", &op, &other)) return nullptr;
    Series* b = as_series(other);
    if (!b) return nullptr;
    return make_series(Series{dftu_series_logical(
        a->handle(), b->handle(), static_cast<dftu_logical_op>(op))});
}

PyObject* Series_logical_not(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_logical_not(a->handle())});
}

// ---- Arrow export (__arrow_c_array__ PyCapsule interface) ----------------

void schema_capsule_destructor(PyObject* cap) {
    auto* s =
        static_cast<ArrowSchema*>(PyCapsule_GetPointer(cap, "arrow_schema"));
    if (s) {
        if (s->release) s->release(s);
        delete s;
    }
}
void array_capsule_destructor(PyObject* cap) {
    auto* a =
        static_cast<ArrowArray*>(PyCapsule_GetPointer(cap, "arrow_array"));
    if (a) {
        if (a->release) a->release(a);
        delete a;
    }
}

PyObject* Series_arrow_c_array(PyObject* self, PyObject* /*args*/) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    auto* schema = new (std::nothrow) ArrowSchema{};
    auto* array = new (std::nothrow) ArrowArray{};
    if (!schema || !array) {
        delete schema;
        delete array;
        return PyErr_NoMemory();
    }
    try {
        dataframe::to_arrow(*a, schema, array);
    } catch (const std::exception& e) {
        delete schema;
        delete array;
        PyErr_SetString(PyExc_RuntimeError, e.what());
        return nullptr;
    }
    PyObject* scap =
        PyCapsule_New(schema, "arrow_schema", schema_capsule_destructor);
    if (!scap) {
        if (schema->release) schema->release(schema);
        delete schema;
        if (array->release) array->release(array);
        delete array;
        return nullptr;
    }
    PyObject* acap =
        PyCapsule_New(array, "arrow_array", array_capsule_destructor);
    if (!acap) {
        Py_DECREF(scap);  // releases + frees the schema
        if (array->release) array->release(array);
        delete array;
        return nullptr;
    }
    PyObject* result = PyTuple_Pack(2, scap, acap);
    Py_DECREF(scap);
    Py_DECREF(acap);
    return result;
}

// ---- statistics + elementwise transforms (dataframe kernels)
// -------------------

PyObject* Series_quantile(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    double q = PyFloat_AsDouble(arg);
    if (q == -1.0 && PyErr_Occurred()) return nullptr;
    return PyFloat_FromDouble(dftu_series_quantile(a->handle(), q));
}
PyObject* Series_median(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyFloat_FromDouble(dftu_series_quantile(a->handle(), 0.5));
}
PyObject* Series_variance(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int sample = 1;
    if (!PyArg_ParseTuple(args, "|p", &sample)) return nullptr;
    return PyFloat_FromDouble(dftu_series_variance(a->handle(), sample));
}
PyObject* Series_stddev(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int sample = 1;
    if (!PyArg_ParseTuple(args, "|p", &sample)) return nullptr;
    return PyFloat_FromDouble(dftu_series_stddev(a->handle(), sample));
}
PyObject* Series_skewness(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyFloat_FromDouble(dftu_series_skewness(a->handle()));
}
PyObject* Series_kurtosis(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyFloat_FromDouble(dftu_series_kurtosis(a->handle()));
}
PyObject* Series_nunique(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(dftu_series_nunique(a->handle()));
}
PyObject* Series_unique(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_unique(a->handle())});
}
PyObject* Series_value_counts(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    try {
        return dftracer::utils::python::wrap_dataframe(
            dataframe::value_counts(*a));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
}
PyObject* Series_abs(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_abs(a->handle())});
}
PyObject* Series_round(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_round(a->handle())});
}
PyObject* Series_cumsum(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_cumsum(a->handle())});
}
PyObject* Series_cummax(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_cummax(a->handle())});
}
PyObject* Series_cummin(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_cummin(a->handle())});
}
PyObject* Series_cum_prod(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_cum_prod(a->handle())});
}
PyObject* Series_cum_count(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_cum_count(a->handle())});
}
PyObject* Series_ceil(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_ceil(a->handle())});
}
PyObject* Series_floor(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_floor(a->handle())});
}
PyObject* Series_trunc(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_trunc(a->handle())});
}
PyObject* Series_sign(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_sign(a->handle())});
}
PyObject* Series_negate(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_negate(a->handle())});
}
PyObject* Series_diff(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_diff(a->handle())});
}
PyObject* Series_pct_change(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_pct_change(a->handle())});
}
PyObject* Series_sqrt(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_sqrt(a->handle())});
}
PyObject* Series_exp(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_exp(a->handle())});
}
PyObject* Series_log(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_log(a->handle())});
}
PyObject* Series_is_nan(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_is_nan(a->handle())});
}
PyObject* Series_is_finite(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_is_finite(a->handle())});
}
PyObject* Series_is_infinite(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_is_infinite(a->handle())});
}
PyObject* Series_is_unique(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_is_unique(a->handle())});
}
PyObject* Series_is_duplicated(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_is_duplicated(a->handle())});
}
PyObject* Series_is_sorted(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int descending = 0;
    static const char* kwlist[] = {"descending", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p",
                                     const_cast<char**>(kwlist), &descending))
        return nullptr;
    return PyBool_FromLong(dftu_series_is_sorted(a->handle(), descending));
}
PyObject* Series_drop_nulls(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_drop_nulls(a->handle())});
}
PyObject* Series_is_in(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    Series* values = as_series(other);
    if (!values) return nullptr;
    return make_series(
        Series{dftu_series_is_in(a->handle(), values->handle())});
}
PyObject* Series_sort(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int descending = 0;
    static const char* kwlist[] = {"descending", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p",
                                     const_cast<char**>(kwlist), &descending))
        return nullptr;
    return make_series(Series{dftu_series_sort(a->handle(), descending)});
}
PyObject* Series_head(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long n = PyLong_AsLongLong(arg);
    if (n == -1 && PyErr_Occurred()) return nullptr;
    return make_series(
        Series{dftu_series_head(a->handle(), static_cast<std::int64_t>(n))});
}
PyObject* Series_tail(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long n = PyLong_AsLongLong(arg);
    if (n == -1 && PyErr_Occurred()) return nullptr;
    return make_series(
        Series{dftu_series_tail(a->handle(), static_cast<std::int64_t>(n))});
}
PyObject* Series_reverse(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_reverse(a->handle())});
}
PyObject* Series_shift(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long n = PyLong_AsLongLong(arg);
    if (n == -1 && PyErr_Occurred()) return nullptr;
    return make_series(
        Series{dftu_series_shift(a->handle(), static_cast<std::int64_t>(n))});
}
PyObject* Series_top_k(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long k = PyLong_AsLongLong(arg);
    if (k == -1 && PyErr_Occurred()) return nullptr;
    return make_series(
        Series{dftu_series_top_k(a->handle(), static_cast<std::int64_t>(k))});
}
PyObject* Series_bottom_k(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long k = PyLong_AsLongLong(arg);
    if (k == -1 && PyErr_Occurred()) return nullptr;
    return make_series(Series{
        dftu_series_bottom_k(a->handle(), static_cast<std::int64_t>(k))});
}
PyObject* Series_sample(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long n = 0;
    unsigned long long seed = 0;
    static const char* kwlist[] = {"n", "seed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|K",
                                     const_cast<char**>(kwlist), &n, &seed))
        return nullptr;
    return make_series(
        Series{dftu_series_sample(a->handle(), static_cast<std::int64_t>(n),
                                  static_cast<std::uint64_t>(seed))});
}
PyObject* Series_rank(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    const char* method = "average";
    int descending = 0;
    static const char* kwlist[] = {"method", "descending", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|sp",
                                     const_cast<char**>(kwlist), &method,
                                     &descending))
        return nullptr;
    const std::string m(method);
    int code = 0;
    if (m == "average")
        code = 0;
    else if (m == "min")
        code = 1;
    else if (m == "dense")
        code = 2;
    else if (m == "ordinal")
        code = 3;
    else {
        PyErr_SetString(PyExc_ValueError,
                        "rank method must be average|min|dense|ordinal");
        return nullptr;
    }
    return make_series(Series{dftu_series_rank(
        a->handle(), static_cast<dftu_rank_method>(code), descending)});
}
PyObject* Series_rolling(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    Py_ssize_t window = 0;
    const char* op = "sum";
    static const char* kwlist[] = {"window", "op", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "n|s",
                                     const_cast<char**>(kwlist), &window, &op))
        return nullptr;
    const std::string o(op);
    int code = 0;
    if (o == "sum")
        code = 0;
    else if (o == "mean")
        code = 1;
    else if (o == "min")
        code = 2;
    else if (o == "max")
        code = 3;
    else {
        PyErr_SetString(PyExc_ValueError,
                        "rolling op must be sum|mean|min|max");
        return nullptr;
    }
    return make_series(Series{
        dftu_series_rolling(a->handle(), static_cast<std::int64_t>(window),
                            static_cast<dftu_rolling_op>(code))});
}
PyObject* Series_rolling_var(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long window = PyLong_AsLongLong(arg);
    if (window == -1 && PyErr_Occurred()) return nullptr;
    return make_series(Series{dftu_series_rolling_var(
        a->handle(), static_cast<std::int64_t>(window))});
}
PyObject* Series_rolling_std(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long window = PyLong_AsLongLong(arg);
    if (window == -1 && PyErr_Occurred()) return nullptr;
    return make_series(Series{dftu_series_rolling_std(
        a->handle(), static_cast<std::int64_t>(window))});
}
PyObject* Series_rolling_median(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long window = PyLong_AsLongLong(arg);
    if (window == -1 && PyErr_Occurred()) return nullptr;
    return make_series(Series{dftu_series_rolling_median(
        a->handle(), static_cast<std::int64_t>(window))});
}
PyObject* Series_rolling_quantile(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long window = 0;
    double q = 0.0;
    if (!PyArg_ParseTuple(args, "Ld", &window, &q)) return nullptr;
    return make_series(Series{dftu_series_rolling_quantile(
        a->handle(), static_cast<std::int64_t>(window), q)});
}
PyObject* Series_ewm_mean(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    double alpha = PyFloat_AsDouble(arg);
    if (alpha == -1.0 && PyErr_Occurred()) return nullptr;
    return make_series(Series{dftu_series_ewm_mean(a->handle(), alpha)});
}
PyObject* Series_ewm_std(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    double alpha = PyFloat_AsDouble(arg);
    if (alpha == -1.0 && PyErr_Occurred()) return nullptr;
    return make_series(Series{dftu_series_ewm_std(a->handle(), alpha)});
}
PyObject* Series_cut(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    Series* breaks = as_series(other);
    if (!breaks) return nullptr;
    return make_series(Series{dftu_series_cut(a->handle(), breaks->handle())});
}
PyObject* Series_qcut(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long q = PyLong_AsLong(arg);
    if (q == -1 && PyErr_Occurred()) return nullptr;
    return make_series(
        Series{dftu_series_qcut(a->handle(), static_cast<std::int32_t>(q))});
}
PyObject* Series_search_sorted(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    Series* values = as_series(other);
    if (!values) return nullptr;
    return make_series(
        Series{dftu_series_search_sorted(a->handle(), values->handle())});
}
PyObject* Series_interpolate(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(Series{dftu_series_interpolate(a->handle())});
}
PyObject* Series_is_between(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* lo = nullptr;
    PyObject* hi = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &lo, &hi)) return nullptr;
    dftu_scalar slo{}, shi{};
    if (!py_to_scalar(lo, &slo) || !py_to_scalar(hi, &shi)) return nullptr;
    return make_series(Series{dftu_series_is_between(a->handle(), slo, shi)});
}
PyObject* Series_fillna(PyObject* self, PyObject* value) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    dftu_scalar s{};
    if (!py_to_scalar(value, &s)) return nullptr;
    return make_series(Series{dftu_series_fillna(a->handle(), s)});
}
PyObject* Series_clip(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* lo = nullptr;
    PyObject* hi = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &lo, &hi)) return nullptr;
    dftu_scalar slo{}, shi{};
    if (!py_to_scalar(lo, &slo) || !py_to_scalar(hi, &shi)) return nullptr;
    return make_series(Series{dftu_series_clip(a->handle(), slo, shi)});
}

// ---- reducers (return a Python scalar) -----------------------------------

// A tagged scalar as the matching Python number.
PyObject* scalar_to_py(dftu_scalar v) {
    using dataframe::ScalarTag;
    if (v.kind < static_cast<std::int32_t>(ScalarTag::I64) ||
        v.kind > static_cast<std::int32_t>(ScalarTag::F64)) {
        PyErr_SetString(PyExc_RuntimeError, "unknown scalar tag");
        return nullptr;
    }
    switch (static_cast<ScalarTag>(v.kind)) {
        case ScalarTag::I64:
            return PyLong_FromLongLong(v.value.i);
        case ScalarTag::U64:
            return PyLong_FromUnsignedLongLong(v.value.u);
        case ScalarTag::F64:
            return PyFloat_FromDouble(v.value.d);
    }
    PyErr_SetString(PyExc_RuntimeError, "unknown scalar tag");
    return nullptr;
}

PyObject* Series_sum(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return scalar_to_py(a->sum());
}
PyObject* Series_min(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return scalar_to_py(a->min());
}
PyObject* Series_max(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return scalar_to_py(a->max());
}
PyObject* Series_mean(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyFloat_FromDouble(a->mean());
}
PyObject* Series_count(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(a->count());
}
PyObject* Series_product(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return scalar_to_py(a->product());
}
PyObject* Series_mode(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return scalar_to_py(a->mode());
}
PyObject* Series_all(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyBool_FromLong(a->all());
}
PyObject* Series_any(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyBool_FromLong(a->any());
}
PyObject* Series_arg_min(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(a->arg_min());
}
PyObject* Series_arg_max(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(a->arg_max());
}
PyObject* Series_dot(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    Series* b = as_series(other);
    if (!b) return nullptr;
    return scalar_to_py(dftu_series_dot(a->handle(), b->handle()));
}

// ---- row / reshape ops (return a Series) ---------------------------------

PyObject* Series_take(PyObject* self, PyObject* seq) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* fast = PySequence_Fast(seq, "take() expects a sequence of ints");
    if (!fast) return nullptr;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    std::vector<std::int64_t> idx;
    idx.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        long long v = PyLong_AsLongLong(PySequence_Fast_GET_ITEM(fast, i));
        if (v == -1 && PyErr_Occurred()) {
            Py_DECREF(fast);
            return nullptr;
        }
        idx.push_back(static_cast<std::int64_t>(v));
    }
    Py_DECREF(fast);
    return make_series(a->take(idx));
}

PyObject* Series_filter(PyObject* self, PyObject* other) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    Series* mask = as_series(other);
    if (!mask) return nullptr;
    return make_series(a->filter(*mask));
}

PyObject* Series_argsort(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    int descending = 0;
    static const char* kwlist[] = {"descending", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|p",
                                     const_cast<char**>(kwlist), &descending))
        return nullptr;
    return make_series(a->argsort(descending != 0));
}

PyObject* Series_dictionary_encode(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->dictionary_encode());
}
PyObject* Series_materialize(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->materialize());
}
PyObject* Series_share(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->share());
}
PyObject* Series_slice(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long offset = 0, length = 0;
    if (!PyArg_ParseTuple(args, "LL", &offset, &length)) return nullptr;
    return make_series(a->slice(static_cast<std::int64_t>(offset),
                                static_cast<std::int64_t>(length)));
}

// ---- string predicates (return a Bool Series) ----------------------------

// Read a Python str as a string_view; the buffer stays owned by `arg`.
bool as_str_view(PyObject* arg, std::string_view* out) {
    Py_ssize_t len = 0;
    const char* data = PyUnicode_AsUTF8AndSize(arg, &len);
    if (!data) return false;
    *out = std::string_view(data, static_cast<std::size_t>(len));
    return true;
}

PyObject* Series_str_eq(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view rhs;
    if (!as_str_view(arg, &rhs)) return nullptr;
    return make_series(a->str_eq(rhs));
}
PyObject* Series_str_contains(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view needle;
    if (!as_str_view(arg, &needle)) return nullptr;
    return make_series(a->str_contains(needle));
}
PyObject* Series_str_starts_with(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view prefix;
    if (!as_str_view(arg, &prefix)) return nullptr;
    return make_series(a->str_starts_with(prefix));
}
PyObject* Series_str_ends_with(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view suffix;
    if (!as_str_view(arg, &suffix)) return nullptr;
    return make_series(a->str_ends_with(suffix));
}
PyObject* Series_str_matches(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view pattern;
    if (!as_str_view(arg, &pattern)) return nullptr;
    return make_series(a->str_matches(pattern));
}
PyObject* Series_str_like(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view pattern;
    if (!as_str_view(arg, &pattern)) return nullptr;
    return make_series(a->str_like(pattern));
}
PyObject* Series_str_len_bytes(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_len_bytes());
}
PyObject* Series_str_len_chars(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_len_chars());
}
PyObject* Series_str_find(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view needle;
    if (!as_str_view(arg, &needle)) return nullptr;
    return make_series(a->str_find(needle));
}
PyObject* Series_to_lowercase(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->to_lowercase());
}
PyObject* Series_to_uppercase(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->to_uppercase());
}
PyObject* Series_str_strip(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_strip());
}
PyObject* Series_str_lstrip(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_lstrip());
}
PyObject* Series_str_rstrip(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_rstrip());
}
PyObject* Series_str_replace(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* pat = nullptr;
    PyObject* repl = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &pat, &repl)) return nullptr;
    std::string_view p, r;
    if (!as_str_view(pat, &p) || !as_str_view(repl, &r)) return nullptr;
    return make_series(a->str_replace(p, r));
}
PyObject* Series_str_replace_all(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* pat = nullptr;
    PyObject* repl = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &pat, &repl)) return nullptr;
    std::string_view p, r;
    if (!as_str_view(pat, &p) || !as_str_view(repl, &r)) return nullptr;
    return make_series(a->str_replace_all(p, r));
}
PyObject* Series_str_slice(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long start = 0;
    long long length = -1;
    if (!PyArg_ParseTuple(args, "L|L", &start, &length)) return nullptr;
    return make_series(a->str_slice(static_cast<std::int64_t>(start),
                                    static_cast<std::int64_t>(length)));
}
// A single-character pad string -> its first byte; empty is rejected.
bool as_fill_char(PyObject* arg, char* out) {
    std::string_view s;
    if (!as_str_view(arg, &s)) return false;
    if (s.empty()) {
        PyErr_SetString(PyExc_ValueError, "fill must be a non-empty string");
        return false;
    }
    *out = s[0];
    return true;
}
PyObject* Series_str_pad_start(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long width = 0;
    PyObject* fill = nullptr;
    static const char* kwlist[] = {"width", "fill", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|O",
                                     const_cast<char**>(kwlist), &width, &fill))
        return nullptr;
    char f = ' ';
    if (fill && !as_fill_char(fill, &f)) return nullptr;
    return make_series(a->str_pad_start(static_cast<std::int64_t>(width), f));
}
PyObject* Series_str_pad_end(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long width = 0;
    PyObject* fill = nullptr;
    static const char* kwlist[] = {"width", "fill", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|O",
                                     const_cast<char**>(kwlist), &width, &fill))
        return nullptr;
    char f = ' ';
    if (fill && !as_fill_char(fill, &f)) return nullptr;
    return make_series(a->str_pad_end(static_cast<std::int64_t>(width), f));
}
PyObject* Series_str_zfill(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long width = PyLong_AsLongLong(arg);
    if (width == -1 && PyErr_Occurred()) return nullptr;
    return make_series(a->str_zfill(static_cast<std::int64_t>(width)));
}
PyObject* Series_str_split(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view sep;
    if (!as_str_view(arg, &sep)) return nullptr;
    return make_series(a->str_split(sep));
}

// ---- null / metadata accessors -------------------------------------------

PyObject* Series_is_null(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long i = PyLong_AsLongLong(arg);
    if (i == -1 && PyErr_Occurred()) return nullptr;
    return PyBool_FromLong(a->is_null(static_cast<std::int64_t>(i)));
}
PyObject* Series_num_children(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(a->num_children());
}
PyObject* Series_child(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long i = PyLong_AsLongLong(arg);
    if (i == -1 && PyErr_Occurred()) return nullptr;
    return make_series(a->child(static_cast<std::int64_t>(i)));
}

// Arrow/pandas/polars conversion and pickling live in the Python wrapper
// (dftracer.utils.frame); the native handle only exposes the zero-copy
// __arrow_c_array__ capsule below.

Py_ssize_t Series_length_sq(PyObject* self) {
    Series* a = as_series(self);
    if (!a) return -1;
    return static_cast<Py_ssize_t>(a->length());
}

// Expose a FLAT, non-null, fixed-width numeric column as a 1-D read-only buffer
// so numpy reads it (np.asarray) with no pyarrow. Bit-packed Bool, strings,
// nulls, and non-FLAT encodings raise BufferError; the Python wrapper then
// falls back to the Arrow path.
int Series_getbuffer(PyObject* self, Py_buffer* view, int flags) {
    view->obj = nullptr;
    Series* a = as_series(self);
    if (!a) return -1;
    if ((flags & PyBUF_WRITABLE) == PyBUF_WRITABLE) {
        PyErr_SetString(PyExc_BufferError, "Series buffer is read-only");
        return -1;
    }
    dftu_series* h = a->handle();
    if (dftu_series_encoding(h) != 0 || dftu_series_null_count(h) != 0) {
        PyErr_SetString(PyExc_BufferError,
                        "Series is not a flat, non-null column");
        return -1;
    }
    const char* fmt = nullptr;
    Py_ssize_t itemsize = 0;
    switch (static_cast<TypeId>(dftu_series_type(h))) {
        case TypeId::Int8:
            fmt = "b";
            itemsize = 1;
            break;
        case TypeId::Int16:
            fmt = "h";
            itemsize = 2;
            break;
        case TypeId::Int32:
            fmt = "i";
            itemsize = 4;
            break;
        case TypeId::Int64:
            fmt = "q";
            itemsize = 8;
            break;
        case TypeId::Uint8:
            fmt = "B";
            itemsize = 1;
            break;
        case TypeId::Uint16:
            fmt = "H";
            itemsize = 2;
            break;
        case TypeId::Uint32:
            fmt = "I";
            itemsize = 4;
            break;
        case TypeId::Uint64:
            fmt = "Q";
            itemsize = 8;
            break;
        case TypeId::Float32:
            fmt = "f";
            itemsize = 4;
            break;
        case TypeId::Float64:
            fmt = "d";
            itemsize = 8;
            break;
        default:
            PyErr_SetString(PyExc_BufferError,
                            "Series type has no plain numeric buffer");
            return -1;
    }
    const void* data = dftu_series_data(h);
    std::int64_t n = dftu_series_length(h);
    if (!data && n > 0) {
        PyErr_SetString(PyExc_BufferError, "Series has no contiguous buffer");
        return -1;
    }
    // The values buffer is exposed as-is (zero-copy); shape/stride live in the
    // object (constant for an immutable column), so there is nothing to free.
    auto* obj = reinterpret_cast<SeriesObject*>(self);
    obj->buf_shape = static_cast<Py_ssize_t>(n);
    obj->buf_stride = itemsize;
    Py_INCREF(self);
    view->obj = self;
    view->buf = const_cast<void*>(data);
    view->len = static_cast<Py_ssize_t>(n) * itemsize;
    view->readonly = 1;
    view->itemsize = itemsize;
    view->format = (flags & PyBUF_FORMAT) ? const_cast<char*>(fmt) : nullptr;
    view->ndim = 1;
    view->shape = &obj->buf_shape;
    view->strides = &obj->buf_stride;
    view->suboffsets = nullptr;
    view->internal = nullptr;
    return 0;
}

// ---- getters -------------------------------------------------------------

PyObject* Series_get_type(PyObject* self, void*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLong(static_cast<long>(a->type()));
}
PyObject* Series_get_length(PyObject* self, void*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(a->length());
}
PyObject* Series_get_null_count(PyObject* self, void*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLongLong(a->null_count());
}
PyObject* Series_get_encoding(PyObject* self, void*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return PyLong_FromLong(static_cast<long>(a->encoding()));
}

PyMethodDef Series_methods[] = {
    {"add", Series_add, METH_O, "Elementwise a + b."},
    {"sub", Series_sub, METH_O, "Elementwise a - b."},
    {"mul", Series_mul, METH_O, "Elementwise a * b."},
    {"div", Series_div, METH_O, "Elementwise a / b."},
    {"add_scalar", Series_add_scalar, METH_O, "Broadcast a + scalar."},
    {"sub_scalar", Series_sub_scalar, METH_O, "Broadcast a - scalar."},
    {"mul_scalar", Series_mul_scalar, METH_O, "Broadcast a * scalar."},
    {"div_scalar", Series_div_scalar, METH_O, "Broadcast a / scalar."},
    {"cast", Series_cast, METH_O, "Cast to a numeric TypeId code."},
    {"prim", Series_prim, METH_O,
     "Apply a unary numeric primitive by Prim code (i64 in, i64 out)."},
    {"compare", Series_compare, METH_VARARGS,
     "compare(op, scalar) -> Bool column (op is a DFTU_CMP_* code)."},
    {"logical", Series_logical, METH_VARARGS,
     "logical(op, other) -> Bool column (op is a DFTU_LOGICAL_* code)."},
    {"logical_not", Series_logical_not, METH_NOARGS,
     "Boolean NOT of a Bool column."},
    {"__arrow_c_array__", Series_arrow_c_array, METH_VARARGS,
     "Arrow PyCapsule export: (schema_capsule, array_capsule)."},
    {"quantile", Series_quantile, METH_O,
     "quantile(q) -> float (linear interpolation, SIMD sort)."},
    {"median", Series_median, METH_NOARGS, "Median (quantile 0.5)."},
    {"variance", Series_variance, METH_VARARGS,
     "variance(sample=True) -> float."},
    {"stddev", Series_stddev, METH_VARARGS, "stddev(sample=True) -> float."},
    {"skewness", Series_skewness, METH_NOARGS, "Population skewness."},
    {"kurtosis", Series_kurtosis, METH_NOARGS, "Excess kurtosis."},
    {"nunique", Series_nunique, METH_NOARGS, "Number of distinct values."},
    {"unique", Series_unique, METH_NOARGS,
     "Distinct values (ascending) -> Series."},
    {"value_counts", Series_value_counts, METH_NOARGS,
     "value_counts() -> DataFrame of distinct values and counts, "
     "most-frequent first."},
    {"abs", Series_abs, METH_NOARGS, "Elementwise absolute value (SIMD)."},
    {"clip", Series_clip, METH_VARARGS,
     "clip(lo, hi) -> Series clamped to [lo, hi] (SIMD)."},
    {"round", Series_round, METH_NOARGS, "Round floats to nearest (SIMD)."},
    {"fillna", Series_fillna, METH_O,
     "fillna(value) -> Series with nulls replaced."},
    {"cumsum", Series_cumsum, METH_NOARGS, "Cumulative sum -> Series."},
    {"cummax", Series_cummax, METH_NOARGS, "Running maximum -> Series."},
    {"cummin", Series_cummin, METH_NOARGS, "Running minimum -> Series."},
    {"cum_prod", Series_cum_prod, METH_NOARGS, "Running product -> Series."},
    {"cum_count", Series_cum_count, METH_NOARGS,
     "Running count of non-null rows -> Int64 Series."},
    {"ceil", Series_ceil, METH_NOARGS, "Round floats toward +inf (SIMD)."},
    {"floor", Series_floor, METH_NOARGS, "Round floats toward -inf (SIMD)."},
    {"trunc", Series_trunc, METH_NOARGS, "Round floats toward zero (SIMD)."},
    {"sign", Series_sign, METH_NOARGS, "Sign as -1/0/1 (SIMD)."},
    {"negate", Series_negate, METH_NOARGS, "Unary minus (SIMD)."},
    {"diff", Series_diff, METH_NOARGS,
     "First difference x[i]-x[i-1]; row 0 null."},
    {"pct_change", Series_pct_change, METH_NOARGS,
     "Percent change -> Float64 Series; row 0 null."},
    {"sqrt", Series_sqrt, METH_NOARGS, "Square root -> Float64 Series (SIMD)."},
    {"exp", Series_exp, METH_NOARGS, "Exponential -> Float64 Series."},
    {"log", Series_log, METH_NOARGS, "Natural log -> Float64 Series."},
    {"is_nan", Series_is_nan, METH_NOARGS, "Bool mask: value is NaN (SIMD)."},
    {"is_finite", Series_is_finite, METH_NOARGS,
     "Bool mask: value is finite (SIMD)."},
    {"is_infinite", Series_is_infinite, METH_NOARGS,
     "Bool mask: value is +/-Inf (SIMD)."},
    {"is_unique", Series_is_unique, METH_NOARGS,
     "Bool mask: value occurs exactly once."},
    {"is_duplicated", Series_is_duplicated, METH_NOARGS,
     "Bool mask: value is repeated."},
    {"is_sorted", DFTU_PYCFUNCTION(Series_is_sorted),
     METH_VARARGS | METH_KEYWORDS, "is_sorted(descending=False) -> bool."},
    {"drop_nulls", Series_drop_nulls, METH_NOARGS, "Drop null rows -> Series."},
    {"is_in", Series_is_in, METH_O,
     "is_in(values) -> Bool mask where the value is in the values Series."},
    {"sort", DFTU_PYCFUNCTION(Series_sort), METH_VARARGS | METH_KEYWORDS,
     "sort(descending=False) -> sorted Series."},
    {"head", Series_head, METH_O, "head(n) -> first n rows Series."},
    {"tail", Series_tail, METH_O, "tail(n) -> last n rows Series."},
    {"reverse", Series_reverse, METH_NOARGS,
     "Rows in reverse order -> Series."},
    {"shift", Series_shift, METH_O,
     "shift(n) -> Series shifted by n (vacated rows null)."},
    {"top_k", Series_top_k, METH_O, "top_k(k) -> the k largest rows Series."},
    {"bottom_k", Series_bottom_k, METH_O,
     "bottom_k(k) -> the k smallest rows Series."},
    {"sample", DFTU_PYCFUNCTION(Series_sample), METH_VARARGS | METH_KEYWORDS,
     "sample(n, seed=0) -> deterministic sample of n rows Series."},
    {"rank", DFTU_PYCFUNCTION(Series_rank), METH_VARARGS | METH_KEYWORDS,
     "rank(method='average', descending=False) -> Float64 Series "
     "(method: average|min|dense|ordinal)."},
    {"rolling", DFTU_PYCFUNCTION(Series_rolling), METH_VARARGS | METH_KEYWORDS,
     "rolling(window, op='sum') -> Float64 Series (op: sum|mean|min|max)."},
    {"rolling_var", Series_rolling_var, METH_O,
     "rolling_var(window) -> Float64 Series of per-window sample variance."},
    {"rolling_std", Series_rolling_std, METH_O,
     "rolling_std(window) -> Float64 Series of per-window sample std."},
    {"rolling_median", Series_rolling_median, METH_O,
     "rolling_median(window) -> Float64 Series of per-window median."},
    {"rolling_quantile", Series_rolling_quantile, METH_VARARGS,
     "rolling_quantile(window, q) -> Float64 Series of per-window quantile q."},
    {"ewm_mean", Series_ewm_mean, METH_O,
     "ewm_mean(alpha) -> Float64 Series, exponentially weighted mean."},
    {"ewm_std", Series_ewm_std, METH_O,
     "ewm_std(alpha) -> Float64 Series, exponentially weighted sample std."},
    {"cut", Series_cut, METH_O,
     "cut(breaks) -> Int32 Series binning by the ascending breaks Series."},
    {"qcut", Series_qcut, METH_O,
     "qcut(q) -> Int32 Series binning by the column's q-quantile edges."},
    {"search_sorted", Series_search_sorted, METH_O,
     "search_sorted(values) -> Int64 Series of lower-bound insertion indices."},
    {"interpolate", Series_interpolate, METH_NOARGS,
     "interpolate() -> Float64 Series with null interiors linearly filled."},
    {"is_between", Series_is_between, METH_VARARGS,
     "is_between(lo, hi) -> Bool mask where lo <= value <= hi (SIMD)."},
    {"dot", Series_dot, METH_O,
     "dot(other) -> scalar sum of x[i]*y[i] over non-null pairs (SIMD)."},
    {"sum", Series_sum, METH_NOARGS, "Sum of non-null values -> scalar."},
    {"min", Series_min, METH_NOARGS, "Minimum non-null value -> scalar."},
    {"max", Series_max, METH_NOARGS, "Maximum non-null value -> scalar."},
    {"mean", Series_mean, METH_NOARGS, "Mean of non-null values -> float."},
    {"count", Series_count, METH_NOARGS, "Non-null row count -> int."},
    {"product", Series_product, METH_NOARGS,
     "Product of non-null values -> scalar."},
    {"mode", Series_mode, METH_NOARGS,
     "Most frequent non-null value -> scalar."},
    {"all", Series_all, METH_NOARGS,
     "Whether every non-null Bool value is true."},
    {"any", Series_any, METH_NOARGS,
     "Whether any non-null Bool value is true."},
    {"arg_min", Series_arg_min, METH_NOARGS,
     "Index of the minimum non-null value (-1 if none)."},
    {"arg_max", Series_arg_max, METH_NOARGS,
     "Index of the maximum non-null value (-1 if none)."},
    {"take", Series_take, METH_O,
     "take(indices) -> Series gathered at the given row indices."},
    {"filter", Series_filter, METH_O,
     "filter(mask) -> Series keeping rows where the Bool mask is true."},
    {"argsort", DFTU_PYCFUNCTION(Series_argsort), METH_VARARGS | METH_KEYWORDS,
     "argsort(descending=False) -> Int64 Series of the sorted row order."},
    {"dictionary_encode", Series_dictionary_encode, METH_NOARGS,
     "Dictionary-encode a String/Binary column -> Series."},
    {"materialize", Series_materialize, METH_NOARGS,
     "Materialize any encoding to a new FLAT Series (gather)."},
    {"share", Series_share, METH_NOARGS,
     "A new owned Series sharing this column's buffers zero-copy."},
    {"slice", Series_slice, METH_VARARGS,
     "slice(offset, length) -> zero-copy view Series of a FLAT row range."},
    {"str_eq", Series_str_eq, METH_O,
     "str_eq(rhs) -> Bool Series where the string equals rhs."},
    {"str_contains", Series_str_contains, METH_O,
     "str_contains(needle) -> Bool Series where the string contains needle."},
    {"str_starts_with", Series_str_starts_with, METH_O,
     "str_starts_with(prefix) -> Bool Series where the string starts with "
     "prefix."},
    {"str_ends_with", Series_str_ends_with, METH_O,
     "str_ends_with(suffix) -> Bool Series where the string ends with suffix."},
    {"str_like", Series_str_like, METH_O,
     "str_like(pattern) -> Bool Series matching the SQL LIKE / glob pattern "
     "(% any run, _ one char, backslash escapes a literal %/_/backslash)."},
    {"str_matches", Series_str_matches, METH_O,
     "str_matches(pattern) -> Bool Series where the whole string matches the "
     "ECMAScript regex."},
    {"str_len_bytes", Series_str_len_bytes, METH_NOARGS,
     "str_len_bytes() -> Int64 Series of per-row byte length."},
    {"str_len_chars", Series_str_len_chars, METH_NOARGS,
     "str_len_chars() -> Int64 Series of per-row UTF-8 codepoint count."},
    {"str_find", Series_str_find, METH_O,
     "str_find(needle) -> Int64 Series of the first byte index, or -1."},
    {"to_lowercase", Series_to_lowercase, METH_NOARGS,
     "to_lowercase() -> String Series, ASCII case fold (non-ASCII unchanged)."},
    {"to_uppercase", Series_to_uppercase, METH_NOARGS,
     "to_uppercase() -> String Series, ASCII case fold (non-ASCII unchanged)."},
    {"str_strip", Series_str_strip, METH_NOARGS,
     "str_strip() -> String Series trimmed of ASCII whitespace both ends."},
    {"str_lstrip", Series_str_lstrip, METH_NOARGS,
     "str_lstrip() -> String Series trimmed of leading ASCII whitespace."},
    {"str_rstrip", Series_str_rstrip, METH_NOARGS,
     "str_rstrip() -> String Series trimmed of trailing ASCII whitespace."},
    {"str_replace", Series_str_replace, METH_VARARGS,
     "str_replace(pat, repl) -> String Series, first literal occurrence."},
    {"str_replace_all", Series_str_replace_all, METH_VARARGS,
     "str_replace_all(pat, repl) -> String Series, all literal occurrences."},
    {"str_slice", Series_str_slice, METH_VARARGS,
     "str_slice(start, length=-1) -> String Series substring by byte offset."},
    {"str_pad_start", DFTU_PYCFUNCTION(Series_str_pad_start),
     METH_VARARGS | METH_KEYWORDS,
     "str_pad_start(width, fill=' ') -> left-padded String Series."},
    {"str_pad_end", DFTU_PYCFUNCTION(Series_str_pad_end),
     METH_VARARGS | METH_KEYWORDS,
     "str_pad_end(width, fill=' ') -> right-padded String Series."},
    {"str_zfill", Series_str_zfill, METH_O,
     "str_zfill(width) -> String Series left-padded with '0' after any sign."},
    {"str_split", Series_str_split, METH_O,
     "str_split(sep) -> List<String> Series split on the literal sep."},
    {"is_null", Series_is_null, METH_O, "is_null(i) -> whether row i is null."},
    {"num_children", Series_num_children, METH_NOARGS,
     "Child count: 1 for a List, the field count for a Struct, else 0."},
    {"child", Series_child, METH_O,
     "child(i) -> child Series (List: 0=values; Struct: field i)."},
    {nullptr, nullptr, 0, nullptr}};

PySequenceMethods Series_as_sequence = {};
PyBufferProcs Series_as_buffer = {};

PyGetSetDef Series_getset[] = {
    {"type", Series_get_type, nullptr, "TypeId code.", nullptr},
    {"length", Series_get_length, nullptr, "Row count.", nullptr},
    {"null_count", Series_get_null_count, nullptr, "Null row count.", nullptr},
    {"encoding", Series_get_encoding, nullptr, "Encoding code.", nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

// ---- Arrow import factory ------------------------------------------------

// Import one Arrow array (any object with __arrow_c_array__) into a
// dataframe::Series. Returns an invalid Series with a Python error set on
// failure.
Series import_arrow_column(PyObject* obj) {
    PyObject* capsules = PyObject_CallMethod(obj, "__arrow_c_array__", nullptr);
    if (!capsules) return Series{};
    if (!PyTuple_Check(capsules) || PyTuple_GET_SIZE(capsules) != 2) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_TypeError,
                        "__arrow_c_array__ must return (schema, array)");
        return Series{};
    }
    auto* schema = static_cast<ArrowSchema*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 0), "arrow_schema"));
    auto* array = static_cast<ArrowArray*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 1), "arrow_array"));
    if (!schema || !array) {
        Py_DECREF(capsules);
        return Series{};
    }
    Series col;
    try {
        // from_arrow adopts `array` (nulls its release), so the array capsule's
        // destructor becomes a no-op; the schema capsule still owns the schema.
        col = dataframe::from_arrow(schema, array);
        if (!col.valid())
            PyErr_SetString(PyExc_ValueError, "unsupported Arrow column");
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        col = Series{};
    }
    Py_DECREF(capsules);
    return col;
}

PyObject* vec_from_arrow(PyObject* /*self*/, PyObject* obj) {
    Series col = import_arrow_column(obj);
    if (!col.valid()) return nullptr;
    return make_series(std::move(col));
}

// ---- native numpy import (buffer protocol, no pyarrow) -------------------

// Map a native-byte-order buffer-protocol format code to a fixed-width TypeId.
// Rejects explicit non-native byte order (we borrow the raw bytes) and any
// non-numeric or composite format. `itemsize` disambiguates 'l'/'L' (int vs
// long) across platforms.
bool numpy_format_type(const char* fmt, Py_ssize_t itemsize, TypeId* out) {
    if (!fmt || !*fmt) return false;
    char order = *fmt;
    if (order == '@' || order == '=' || order == '<' || order == '>' ||
        order == '!') {
        if (order != '@' && order != '=') return false;
        ++fmt;
    }
    if (fmt[0] == '\0' || fmt[1] != '\0') return false;  // one type code only
    switch (fmt[0]) {
        case 'b':
            *out = TypeId::Int8;
            return itemsize == 1;
        case 'B':
            *out = TypeId::Uint8;
            return itemsize == 1;
        case 'h':
            *out = TypeId::Int16;
            return itemsize == 2;
        case 'H':
            *out = TypeId::Uint16;
            return itemsize == 2;
        case 'i':
            *out = TypeId::Int32;
            return itemsize == 4;
        case 'I':
            *out = TypeId::Uint32;
            return itemsize == 4;
        case 'l':
            *out = itemsize == 8 ? TypeId::Int64 : TypeId::Int32;
            return itemsize == 8 || itemsize == 4;
        case 'L':
            *out = itemsize == 8 ? TypeId::Uint64 : TypeId::Uint32;
            return itemsize == 8 || itemsize == 4;
        case 'q':
            *out = TypeId::Int64;
            return itemsize == 8;
        case 'Q':
            *out = TypeId::Uint64;
            return itemsize == 8;
        case 'f':
            *out = TypeId::Float32;
            return itemsize == 4;
        case 'd':
            *out = TypeId::Float64;
            return itemsize == 8;
        default:
            return false;
    }
}

// Drop a borrowed numpy buffer once the column's last reference is gone. Runs
// from Series destruction, which may be off the interpreter thread, so it
// re-acquires the GIL before touching CPython state.
void numpy_buffer_release(void* ctx) {
    auto* view = static_cast<Py_buffer*>(ctx);
    PyGILState_STATE gil = PyGILState_Ensure();
    PyBuffer_Release(view);
    PyGILState_Release(gil);
    delete view;
}

// Import a 1-D C-contiguous numeric numpy array (anything exposing the buffer
// protocol) into a native Series with NO pyarrow. Zero-copy borrow: the
// column's data buffer points straight at the numpy memory and holds the
// Py_buffer alive until the column is freed. On any unsupported case a Python
// error is set and NULL returned, so the Python wrapper falls back to Arrow.
PyObject* vec_from_numpy(PyObject* /*self*/, PyObject* obj) {
    auto* view = new (std::nothrow) Py_buffer{};
    if (!view) return PyErr_NoMemory();
    if (PyObject_GetBuffer(obj, view,
                           PyBUF_FORMAT | PyBUF_ND | PyBUF_STRIDES) != 0) {
        delete view;
        return nullptr;  // no buffer protocol; error already set
    }
    TypeId type = TypeId::Int64;
    const bool contiguous =
        view->strides == nullptr || view->strides[0] == view->itemsize;
    if (view->ndim != 1 ||
        !numpy_format_type(view->format, view->itemsize, &type) ||
        !contiguous) {
        PyBuffer_Release(view);
        delete view;
        PyErr_SetString(
            PyExc_TypeError,
            "_series_from_numpy needs a 1-D C-contiguous fixed-width "
            "numeric array");
        return nullptr;
    }
    std::int64_t n = view->shape ? view->shape[0] : 0;
    // An empty array has no memory to borrow (the release would not run on a
    // null/zero-length buffer), so copy the (empty) column and free the view.
    if (n == 0 || view->buf == nullptr) {
        Series col = Series::flat(type, view->buf, n, nullptr);
        PyBuffer_Release(view);
        delete view;
        if (!col.valid()) {
            PyErr_SetString(PyExc_RuntimeError,
                            "failed to build an empty column");
            return nullptr;
        }
        return make_series(std::move(col));
    }
    dftu_series* h =
        dftu_series_new_flat_borrowed(static_cast<dftu_dtype>(type), view->buf,
                                      n, nullptr, numpy_buffer_release, view);
    if (!h) {
        numpy_buffer_release(view);  // releases the buffer and deletes the view
        PyErr_SetString(PyExc_RuntimeError,
                        "failed to borrow the numpy buffer");
        return nullptr;
    }
    return make_series(Series{h});
}

}  // namespace

namespace dftracer::utils::python {

PyObject* wrap_vec_column(dftracer::utils::dataframe::Series&& col) {
    return make_series(std::move(col));
}

const dftracer::utils::dataframe::Series* unwrap_vec_column(PyObject* o) {
    if (!PyObject_TypeCheck(o, &SeriesType)) return nullptr;
    return &reinterpret_cast<SeriesObject*>(o)->col;
}

dftracer::utils::dataframe::Series column_from_arrow(PyObject* arrow_array) {
    return import_arrow_column(arrow_array);
}

int init_series(PyObject* m) {
    SeriesType = {};
    SeriesType.ob_base = {PyObject_HEAD_INIT(nullptr) 0};
    SeriesType.tp_name = "dftracer_utils_ext._Series";
    SeriesType.tp_basicsize = sizeof(SeriesObject);
    SeriesType.tp_flags = Py_TPFLAGS_DEFAULT;
    SeriesType.tp_doc =
        "Native columnar Series handle (SIMD ops + Arrow capsule).";
    SeriesType.tp_dealloc = reinterpret_cast<destructor>(Series_dealloc);
    Series_as_sequence.sq_length = Series_length_sq;
    SeriesType.tp_as_sequence = &Series_as_sequence;
    Series_as_buffer.bf_getbuffer = Series_getbuffer;
    SeriesType.tp_as_buffer = &Series_as_buffer;
    SeriesType.tp_methods = Series_methods;
    SeriesType.tp_getset = Series_getset;
    SeriesType.tp_new =
        nullptr;  // created only via _series_from_arrow / kernels
    if (register_type(m, &SeriesType, "_Series") < 0) return -1;

    static PyMethodDef from_arrow_def = {
        "_series_from_arrow", vec_from_arrow, METH_O,
        "_series_from_arrow(arrow_array) -> _Series: import a pyarrow array "
        "into a native columnar Series."};
    PyObject* fn = PyCFunction_NewEx(&from_arrow_def, nullptr, nullptr);
    if (!fn) return -1;
    if (PyModule_AddObject(m, "_series_from_arrow", fn) < 0) {
        Py_DECREF(fn);
        return -1;
    }

    static PyMethodDef from_numpy_def = {
        "_series_from_numpy", vec_from_numpy, METH_O,
        "_series_from_numpy(array) -> _Series: import a 1-D C-contiguous "
        "numeric numpy array (buffer protocol, zero-copy borrow, no pyarrow)."};
    PyObject* np_fn = PyCFunction_NewEx(&from_numpy_def, nullptr, nullptr);
    if (!np_fn) return -1;
    if (PyModule_AddObject(m, "_series_from_numpy", np_fn) < 0) {
        Py_DECREF(np_fn);
        return -1;
    }
    return 0;
}

}  // namespace dftracer::utils::python

#else   // !DFTRACER_UTILS_ENABLE_ARROW

namespace dftracer::utils::python {
int init_series(PyObject*) { return 0; }
const dftracer::utils::dataframe::Series* unwrap_vec_column(PyObject*) {
    return nullptr;
}
}  // namespace dftracer::utils::python

#endif  // DFTRACER_UTILS_ENABLE_ARROW
