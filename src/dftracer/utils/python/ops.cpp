// The Python view of the dataframe op registry: op_list / op_info for
// discovery, and one generic op_run that decodes an op's packed signature to
// marshal the Python arguments into dftu_op_arg and hand columns to/from the
// engine. All work is the C ABI's (dftu_op_*); this file only marshals.

#include <dftracer/utils/core/common/config.h>  // DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/dataframe/abi.h>
#include <dftracer/utils/python/ops.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/dataframe/series.h>
#include <dftracer/utils/python/series.h>
#endif

namespace dftracer::utils::python {
namespace {

PyObject* op_list(PyObject*, PyObject*) {
    uint32_t n = dftu_op_count();
    PyObject* out = PyList_New(n);
    if (!out) return nullptr;
    for (uint32_t i = 0; i < n; ++i) {
        const dftu_op_desc* op = dftu_op_at(i);
        PyObject* name = PyUnicode_FromString(op ? op->name : "");
        if (!name) {
            Py_DECREF(out);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, name);  // steals name
    }
    return out;
}

PyObject* op_info(PyObject*, PyObject* args) {
    const char* name = nullptr;
    if (!PyArg_ParseTuple(args, "s", &name)) return nullptr;
    const dftu_op_desc* op = dftu_op_find(name);
    if (!op) {
        PyErr_Format(PyExc_KeyError, "no op named '%s'", name);
        return nullptr;
    }
    const char* kind = dftu_op_kind_of(op->sig) == DFTU_OP_KIND_SERIES
                           ? "series"
                           : "aggregate";
    return Py_BuildValue("{s:s,s:s,s:I,s:s}", "name", op->name, "kind", kind,
                         "arity", dftu_op_arity(op->sig), "signature",
                         dftu_op_signature(op->sig));
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW

// Fill dftu_scalar from a Python number (float -> F64, int -> I64).
bool to_scalar(PyObject* o, dftu_scalar* s) {
    if (PyFloat_Check(o)) {
        s->kind = DFTU_SCALAR_TAG_F64;
        s->value.d = PyFloat_AsDouble(o);
        return !PyErr_Occurred();
    }
    long long v = PyLong_AsLongLong(o);
    if (v == -1 && PyErr_Occurred()) return false;
    s->kind = DFTU_SCALAR_TAG_I64;
    s->value.i = v;
    return true;
}

PyObject* scalar_to_py(dftu_scalar s) {
    switch (s.kind) {
        case DFTU_SCALAR_TAG_U64:
            return PyLong_FromUnsignedLongLong(s.value.u);
        case DFTU_SCALAR_TAG_F64:
            return PyFloat_FromDouble(s.value.d);
        default:
            return PyLong_FromLongLong(s.value.i);
    }
}

PyObject* op_run(PyObject*, PyObject* args) {
    Py_ssize_t nargs = PyTuple_GET_SIZE(args);
    if (nargs < 1) {
        PyErr_SetString(PyExc_TypeError,
                        "op_run(name, *args): name is required");
        return nullptr;
    }
    const char* name = PyUnicode_AsUTF8(PyTuple_GET_ITEM(args, 0));
    if (!name) return nullptr;
    const dftu_op_desc* op = dftu_op_find(name);
    if (!op) {
        PyErr_Format(PyExc_KeyError, "no op named '%s'", name);
        return nullptr;
    }

    // Walk the signature's operand tokens in order, pulling each from the next
    // positional argument: a column into `in`, everything else into `arg`.
    const dftu_series* in[2] = {nullptr, nullptr};
    uint32_t n_series = 0;
    dftu_op_arg arg{};
    int n_str = 0, n_i64 = 0;
    Py_ssize_t next = 1;
    for (int i = 0; i < 3; ++i) {
        dftu_op_tok t = DFTU_OP_SIG_ARG(op->sig, i);
        if (t == DFTU_TOK_NONE) break;
        if (next >= nargs) {
            PyErr_Format(PyExc_TypeError, "op '%s' expects more arguments",
                         name);
            return nullptr;
        }
        PyObject* a = PyTuple_GET_ITEM(args, next++);
        switch (t) {
            case DFTU_TOK_SERIES: {
                const dataframe::Series* s = unwrap_vec_column(a);
                if (!s) {
                    PyErr_Format(PyExc_TypeError,
                                 "op '%s' argument %zd must be a Series", name,
                                 next - 1);
                    return nullptr;
                }
                in[n_series++] = s->handle();
                break;
            }
            case DFTU_TOK_SCALAR:
                if (!to_scalar(a, &arg.scalar)) return nullptr;
                break;
            case DFTU_TOK_CMP:
            case DFTU_TOK_PRIM:
            case DFTU_TOK_LOGICAL:
            case DFTU_TOK_DTYPE:
            case DFTU_TOK_REDUCE:
                arg.op_code = static_cast<int32_t>(PyLong_AsLong(a));
                if (arg.op_code == -1 && PyErr_Occurred()) return nullptr;
                break;
            case DFTU_TOK_STR: {
                Py_ssize_t len = 0;
                const char* p = PyUnicode_AsUTF8AndSize(a, &len);
                if (!p) return nullptr;
                if (n_str++ == 0) {
                    arg.s0 = p;
                    arg.s0_len = static_cast<int32_t>(len);
                } else {
                    arg.s1 = p;
                    arg.s1_len = static_cast<int32_t>(len);
                }
                break;
            }
            case DFTU_TOK_I64: {
                long long v = PyLong_AsLongLong(a);
                if (v == -1 && PyErr_Occurred()) return nullptr;
                if (n_i64++ == 0)
                    arg.i0 = v;
                else
                    arg.i1 = v;
                break;
            }
            case DFTU_TOK_CHAR: {
                Py_ssize_t len = 0;
                const char* p = PyUnicode_AsUTF8AndSize(a, &len);
                if (!p) return nullptr;
                if (len != 1) {
                    PyErr_Format(PyExc_ValueError,
                                 "op '%s' expects a single character", name);
                    return nullptr;
                }
                arg.ch = p[0];
                break;
            }
            case DFTU_TOK_NONE:
            case DFTU_TOK_BOOL:
                break;
        }
    }
    if (next != nargs) {
        PyErr_Format(PyExc_TypeError, "op '%s' got too many arguments", name);
        return nullptr;
    }

    if (dftu_op_kind_of(op->sig) == DFTU_OP_KIND_SERIES) {
        dftu_series* out = dftu_op_run(op, in, n_series, &arg);
        if (!out) {
            PyErr_Format(PyExc_RuntimeError, "op '%s' failed", name);
            return nullptr;
        }
        return wrap_vec_column(dataframe::Series{out});
    }
    int ok = 0;
    dftu_scalar r = dftu_op_run_aggregate(op, in[0], &arg, &ok);
    if (!ok) {
        PyErr_Format(PyExc_RuntimeError, "op '%s' failed", name);
        return nullptr;
    }
    return scalar_to_py(r);
}

#else  // no Arrow: columns are unavailable, so op_run cannot marshal a Series.

PyObject* op_run(PyObject*, PyObject*) {
    PyErr_SetString(PyExc_RuntimeError,
                    "op_run requires the Arrow-enabled build");
    return nullptr;
}

#endif

PyMethodDef ops_methods[] = {
    {"op_list", op_list, METH_NOARGS,
     "op_list(): the names of every registered op."},
    {"op_info", op_info, METH_VARARGS,
     "op_info(name): {name, kind, arity, signature} for a registered op."},
    {"op_run", op_run, METH_VARARGS,
     "op_run(name, *args): run a registered op; columns are Series, other "
     "operands follow the op's signature. Returns a Series or a scalar."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace

int init_ops(PyObject* m) { return PyModule_AddFunctions(m, ops_methods); }

}  // namespace dftracer::utils::python
