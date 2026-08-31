// A native LazyFrame exposed to Python: a deferred query over an in-memory
// dataframe::DataFrame. Builder methods record ops and return a new LazyFrame;
// nothing runs until collect(), which materializes a native _DataFrame. A
// filter/with_column expr's col(i) refers to the i-th column of the frame at
// that point, so the Python wrapper resolves names to indices against schema().

#include <dftracer/utils/core/common/config.h>  // DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/python/lazyframe.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/core/common/memory_budget.h>  // NO_SPILL_BUDGET
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/expr.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/python/columnar_eval.h>
#include <dftracer/utils/python/dataframe.h>
#include <dftracer/utils/python/py_agg_helpers.h>
#include <dftracer/utils/python/py_list_helpers.h>
#include <dftracer/utils/python/py_method.h>
#include <dftracer/utils/python/py_scalar_helpers.h>
#include <dftracer/utils/python/py_type_helpers.h>

#include <cstdint>
#include <new>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dataframe = dftracer::utils::dataframe;

namespace {

using dataframe::GroupAgg;
using dataframe::LazyFrame;
using dftracer::utils::python::aggs_from_seq;
using dftracer::utils::python::group_agg_from_spec;

struct LazyFrameObject {
    PyObject_HEAD LazyFrame lf;
};

PyTypeObject LazyFrameType;

LazyFrameObject* as_lazyframe(PyObject* o) {
    if (!PyObject_TypeCheck(o, &LazyFrameType)) {
        PyErr_SetString(PyExc_TypeError, "expected a LazyFrame");
        return nullptr;
    }
    return reinterpret_cast<LazyFrameObject*>(o);
}

PyObject* make_lazyframe(LazyFrame&& lf) {
    auto* self = reinterpret_cast<LazyFrameObject*>(
        LazyFrameType.tp_alloc(&LazyFrameType, 0));
    if (!self) return nullptr;
    new (&self->lf) LazyFrame(std::move(lf));
    return reinterpret_cast<PyObject*>(self);
}

void LazyFrame_dealloc(LazyFrameObject* self) {
    self->lf.~LazyFrame();
    Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self));
}

// Wrap a builder call that returns a LazyFrame, turning C++ exceptions into
// Python errors.
template <class Fn>
PyObject* run_lazy_op(Fn&& fn) {
    try {
        return make_lazyframe(fn());
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
}

// filter(ast) -> LazyFrame. `ast` is the columnar.py post-order Expr AST; the
// engine rebuilds a dataframe::Expr referencing columns by position.
PyObject* LazyFrame_filter(PyObject* self, PyObject* ast) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    dataframe::Expr pred;
    if (!dftracer::utils::python::build_expr_from_ast(ast, &pred))
        return nullptr;
    return run_lazy_op([&] { return b->lf.filter(pred); });
}

PyObject* LazyFrame_with_column(PyObject* self, PyObject* args) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* name = nullptr;
    PyObject* ast = nullptr;
    if (!PyArg_ParseTuple(args, "sO", &name, &ast)) return nullptr;
    dataframe::Expr e;
    if (!dftracer::utils::python::build_expr_from_ast(ast, &e)) return nullptr;
    return run_lazy_op([&] { return b->lf.with_column(name, e); });
}

PyObject* LazyFrame_select(PyObject* self, PyObject* names) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    std::vector<std::string> ns;
    if (!parse_str_list(names, "names", ns)) return nullptr;
    return run_lazy_op([&] { return b->lf.select(std::move(ns)); });
}

PyObject* LazyFrame_rename(PyObject* self, PyObject* names) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    std::vector<std::string> ns;
    if (!parse_str_list(names, "names", ns)) return nullptr;
    return run_lazy_op([&] { return b->lf.rename(std::move(ns)); });
}

PyObject* LazyFrame_slice(PyObject* self, PyObject* args) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    long long offset = 0, len = 0;
    if (!PyArg_ParseTuple(args, "LL", &offset, &len)) return nullptr;
    return run_lazy_op([&] { return b->lf.slice(offset, len); });
}

PyObject* LazyFrame_head(PyObject* self, PyObject* n) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    long long v = PyLong_AsLongLong(n);
    if (v == -1 && PyErr_Occurred()) return nullptr;
    return run_lazy_op([&] { return b->lf.head(v); });
}

PyObject* LazyFrame_tail(PyObject* self, PyObject* n) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    long long v = PyLong_AsLongLong(n);
    if (v == -1 && PyErr_Occurred()) return nullptr;
    return run_lazy_op([&] { return b->lf.tail(v); });
}

PyObject* LazyFrame_drop_nulls(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.drop_nulls(); });
}

PyObject* LazyFrame_fill_null(PyObject* self, PyObject* value) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    dftu_scalar s{};
    if (!py_to_scalar(value, &s)) return nullptr;
    return run_lazy_op([&] { return b->lf.fill_null(s); });
}

PyObject* LazyFrame_with_row_index(PyObject* self, PyObject* name) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* s = PyUnicode_AsUTF8(name);
    if (!s) return nullptr;
    return run_lazy_op([&] { return b->lf.with_row_index(s); });
}

PyObject* LazyFrame_null_count(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.null_count(); });
}

PyObject* LazyFrame_explode(PyObject* self, PyObject* column) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* s = PyUnicode_AsUTF8(column);
    if (!s) return nullptr;
    return run_lazy_op([&] { return b->lf.explode(s); });
}

// unpivot(id_vars, value_vars); melt is the same call.
PyObject* LazyFrame_unpivot(PyObject* self, PyObject* args) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    PyObject* id_obj = nullptr;
    PyObject* val_obj = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &id_obj, &val_obj)) return nullptr;
    std::vector<std::string> id_vars, value_vars;
    if (!parse_str_list(id_obj, "id_vars", id_vars) ||
        !parse_str_list(val_obj, "value_vars", value_vars))
        return nullptr;
    return run_lazy_op([&] {
        return b->lf.unpivot(std::move(id_vars), std::move(value_vars));
    });
}

PyObject* LazyFrame_topk(PyObject* self, PyObject* args, PyObject* kwds) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* name = nullptr;
    long long k = 0;
    int largest = 1;
    static const char* kw[] = {"name", "k", "largest", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sL|p", const_cast<char**>(kw),
                                     &name, &k, &largest))
        return nullptr;
    return run_lazy_op([&] { return b->lf.topk(name, k, largest != 0); });
}

// group_by(key, aggs): aggs is a sequence of "op[:column]" string specs.
PyObject* LazyFrame_group_by(PyObject* self, PyObject* args) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* key = nullptr;
    PyObject* aggs_obj = nullptr;
    if (!PyArg_ParseTuple(args, "sO", &key, &aggs_obj)) return nullptr;
    std::vector<GroupAgg> aggs;
    if (!aggs_from_seq(aggs_obj, aggs)) return nullptr;
    return run_lazy_op([&] { return b->lf.group_by(key, std::move(aggs)); });
}

PyObject* LazyFrame_sort_by(PyObject* self, PyObject* args, PyObject* kwds) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* name = nullptr;
    int descending = 0;
    static const char* kw[] = {"name", "descending", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "s|p", const_cast<char**>(kw),
                                     &name, &descending))
        return nullptr;
    return run_lazy_op([&] { return b->lf.sort_by(name, descending != 0); });
}

PyObject* LazyFrame_unique(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.unique(); });
}

PyObject* LazyFrame_sample(PyObject* self, PyObject* args, PyObject* kwds) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    long long n = 0;
    unsigned long long seed = 0;
    static const char* kw[] = {"n", "seed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|K", const_cast<char**>(kw),
                                     &n, &seed))
        return nullptr;
    return run_lazy_op([&] { return b->lf.sample(n, seed); });
}

PyObject* LazyFrame_is_duplicated(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.is_duplicated(); });
}

PyObject* LazyFrame_is_unique(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.is_unique(); });
}

// group_by_dynamic(time_col, every, period=None, aggs=[], origin=0,
// origin_min=False). period defaults to every (tumbling window).
PyObject* LazyFrame_group_by_dynamic(PyObject* self, PyObject* args,
                                     PyObject* kwds) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* time_col = nullptr;
    long long every = 0;
    PyObject* period_obj = Py_None;
    PyObject* aggs_obj = nullptr;
    long long origin = 0;
    int origin_min = 0;
    static const char* kw[] = {"time_col", "every",      "period", "aggs",
                               "origin",   "origin_min", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "sL|OOLp", const_cast<char**>(kw), &time_col, &every,
            &period_obj, &aggs_obj, &origin, &origin_min))
        return nullptr;
    long long period = every;
    if (period_obj && period_obj != Py_None) {
        period = PyLong_AsLongLong(period_obj);
        if (period == -1 && PyErr_Occurred()) return nullptr;
    }
    std::vector<GroupAgg> aggs;
    if (aggs_obj && aggs_obj != Py_None && !aggs_from_seq(aggs_obj, aggs))
        return nullptr;
    return run_lazy_op([&] {
        return b->lf.group_by_dynamic(time_col, every, period, std::move(aggs),
                                      origin, origin_min != 0);
    });
}

PyObject* LazyFrame_pivot(PyObject* self, PyObject* args, PyObject* kwds) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* index = nullptr;
    const char* on = nullptr;
    const char* values = nullptr;
    const char* agg = "first";
    static const char* kw[] = {"index", "on", "values", "agg", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sss|s",
                                     const_cast<char**>(kw), &index, &on,
                                     &values, &agg))
        return nullptr;
    return run_lazy_op([&] { return b->lf.pivot(index, on, values, agg); });
}

PyObject* LazyFrame_to_dummies(PyObject* self, PyObject* column) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    const char* s = PyUnicode_AsUTF8(column);
    if (!s) return nullptr;
    return run_lazy_op([&] { return b->lf.to_dummies(s); });
}

PyObject* LazyFrame_describe(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.describe(); });
}

PyObject* LazyFrame_memory_budget(PyObject* self, PyObject* bytes) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    unsigned long long v = PyLong_AsUnsignedLongLong(bytes);
    if (v == static_cast<unsigned long long>(-1) && PyErr_Occurred())
        return nullptr;
    return run_lazy_op([&] { return b->lf.memory_budget(v); });
}

PyObject* LazyFrame_auto_spill(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    return run_lazy_op([&] { return b->lf.auto_spill(); });
}

PyObject* LazyFrame_schema(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    std::vector<std::string> names;
    try {
        names = b->lf.schema();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    return str_list_from(names);
}

PyObject* LazyFrame_explain(PyObject* self, PyObject*) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    std::string text;
    try {
        text = b->lf.explain();
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
    return PyUnicode_FromStringAndSize(text.data(),
                                       static_cast<Py_ssize_t>(text.size()));
}

PyObject* LazyFrame_collect(PyObject* self, PyObject* args, PyObject* kwds) {
    LazyFrameObject* b = as_lazyframe(self);
    if (!b) return nullptr;
    long long morsel_rows = 65536;
    static const char* kw[] = {"morsel_rows", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|L", const_cast<char**>(kw),
                                     &morsel_rows))
        return nullptr;
    try {
        return dftracer::utils::python::wrap_dataframe(
            b->lf.collect(morsel_rows));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
}

PyMethodDef LazyFrame_methods[] = {
    {"filter", LazyFrame_filter, METH_O,
     "filter(ast) -> LazyFrame keeping rows where the predicate holds."},
    {"with_column", LazyFrame_with_column, METH_VARARGS,
     "with_column(name, ast) -> LazyFrame with a column added or replaced."},
    {"select", LazyFrame_select, METH_O,
     "select(names) -> LazyFrame projected to the given columns."},
    {"rename", LazyFrame_rename, METH_O,
     "rename(names) -> LazyFrame with columns renamed positionally."},
    {"slice", LazyFrame_slice, METH_VARARGS,
     "slice(offset, len) -> LazyFrame of the row window."},
    {"head", LazyFrame_head, METH_O,
     "head(n) -> LazyFrame of the first n rows."},
    {"tail", LazyFrame_tail, METH_O,
     "tail(n) -> LazyFrame of the last n rows."},
    {"drop_nulls", LazyFrame_drop_nulls, METH_NOARGS,
     "drop_nulls() -> LazyFrame with rows holding any null removed."},
    {"fill_null", LazyFrame_fill_null, METH_O,
     "fill_null(value) -> LazyFrame with nulls filled in every column."},
    {"with_row_index", LazyFrame_with_row_index, METH_O,
     "with_row_index(name) -> LazyFrame with a prepended Int64 index column."},
    {"null_count", LazyFrame_null_count, METH_NOARGS,
     "null_count() -> LazyFrame of each column's null count (one row)."},
    {"explode", LazyFrame_explode, METH_O,
     "explode(column) -> LazyFrame with the List column expanded per element."},
    {"unpivot", LazyFrame_unpivot, METH_VARARGS,
     "unpivot(id_vars, value_vars) -> LazyFrame reshaped wide to long."},
    {"melt", LazyFrame_unpivot, METH_VARARGS, "melt(...) -> alias of unpivot."},
    {"topk", DFTU_PYCFUNCTION(LazyFrame_topk), METH_VARARGS | METH_KEYWORDS,
     "topk(name, k, largest=True) -> LazyFrame of the k extreme rows."},
    {"group_by", LazyFrame_group_by, METH_VARARGS,
     "group_by(key, aggs) -> LazyFrame grouped by key with string-spec aggs."},
    {"sort_by", DFTU_PYCFUNCTION(LazyFrame_sort_by),
     METH_VARARGS | METH_KEYWORDS,
     "sort_by(name, descending=False) -> LazyFrame sorted by one column."},
    {"unique", LazyFrame_unique, METH_NOARGS,
     "unique() -> LazyFrame with duplicate rows removed (keep first)."},
    {"drop_duplicates", LazyFrame_unique, METH_NOARGS,
     "drop_duplicates() -> alias of unique."},
    {"sample", DFTU_PYCFUNCTION(LazyFrame_sample), METH_VARARGS | METH_KEYWORDS,
     "sample(n, seed=0) -> LazyFrame deterministic n-row sample."},
    {"is_duplicated", LazyFrame_is_duplicated, METH_NOARGS,
     "is_duplicated() -> LazyFrame Bool column, true where the row repeats."},
    {"is_unique", LazyFrame_is_unique, METH_NOARGS,
     "is_unique() -> LazyFrame Bool column, true where the row is unique."},
    {"group_by_dynamic", DFTU_PYCFUNCTION(LazyFrame_group_by_dynamic),
     METH_VARARGS | METH_KEYWORDS,
     "group_by_dynamic(time_col, every, period=None, aggs=[], origin=0, "
     "origin_min=False) -> LazyFrame of time-window aggregates."},
    {"pivot", DFTU_PYCFUNCTION(LazyFrame_pivot), METH_VARARGS | METH_KEYWORDS,
     "pivot(index, on, values, agg='first') -> LazyFrame reshaped long to "
     "wide."},
    {"to_dummies", LazyFrame_to_dummies, METH_O,
     "to_dummies(column) -> LazyFrame one-hot encoding of the column."},
    {"describe", LazyFrame_describe, METH_NOARGS,
     "describe() -> LazyFrame of per-column summary statistics."},
    {"memory_budget", LazyFrame_memory_budget, METH_O,
     "memory_budget(bytes) -> LazyFrame with the out-of-core spill budget "
     "set."},
    {"auto_spill", LazyFrame_auto_spill, METH_NOARGS,
     "auto_spill() -> LazyFrame with the spill budget at ~1/3 of memory."},
    {"schema", LazyFrame_schema, METH_NOARGS,
     "schema() -> list[str] of output column names, or [] if data-dependent."},
    {"explain", LazyFrame_explain, METH_NOARGS,
     "explain() -> str of the optimized plan."},
    {"collect", DFTU_PYCFUNCTION(LazyFrame_collect),
     METH_VARARGS | METH_KEYWORDS,
     "collect(morsel_rows=65536) -> _DataFrame, running the pipeline."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace

namespace dftracer::utils::python {

int init_lazyframe(PyObject* m) {
    LazyFrameType = {};
    LazyFrameType.ob_base = {PyObject_HEAD_INIT(nullptr) 0};
    LazyFrameType.tp_name = "dftracer_utils_ext._LazyFrame";
    LazyFrameType.tp_basicsize = sizeof(LazyFrameObject);
    LazyFrameType.tp_flags = Py_TPFLAGS_DEFAULT;
    LazyFrameType.tp_doc = "A deferred query over a native vec batch.";
    LazyFrameType.tp_dealloc = reinterpret_cast<destructor>(LazyFrame_dealloc);
    LazyFrameType.tp_methods = LazyFrame_methods;
    LazyFrameType.tp_new = nullptr;  // created only by DataFrame.lazy()
    return register_type(m, &LazyFrameType, "_LazyFrame");
}

PyObject* wrap_lazyframe(dataframe::LazyFrame&& lf) {
    return make_lazyframe(std::move(lf));
}

}  // namespace dftracer::utils::python

#else   // !DFTRACER_UTILS_ENABLE_ARROW

namespace dftracer::utils::python {
int init_lazyframe(PyObject*) { return 0; }
PyObject* wrap_lazyframe(dftracer::utils::dataframe::LazyFrame&&) {
    PyErr_SetString(PyExc_RuntimeError, "LazyFrame requires the Arrow build");
    return nullptr;
}
}  // namespace dftracer::utils::python

#endif  // DFTRACER_UTILS_ENABLE_ARROW
