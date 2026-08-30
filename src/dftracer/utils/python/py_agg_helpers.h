#ifndef DFTRACER_UTILS_PYTHON_PY_AGG_HELPERS_H
#define DFTRACER_UTILS_PYTHON_PY_AGG_HELPERS_H

#include <Python.h>
#include <dftracer/utils/dataframe/dataframe.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace dftracer::utils::python {

// Parse an "op[:column]" aggregate spec; throws std::out_of_range on a bad op.
inline dftracer::utils::dataframe::GroupAgg group_agg_from_spec(
    const std::string& spec) {
    namespace dataframe = dftracer::utils::dataframe;
    std::size_t colon = spec.find(':');
    std::string op = spec.substr(0, colon);
    dataframe::GroupAgg a;
    a.op = dataframe::agg_from_string(op);
    if (colon == std::string::npos) {
        a.out = op;
    } else {
        a.column = spec.substr(colon + 1);
        a.out = op + "_" + a.column;
    }
    return a;
}

// Parse a sequence of agg spec strings ("count" or "<op>:<column>") into
// GroupAgg records, mirroring the group_by string-spec form.
inline bool aggs_from_seq(
    PyObject* obj, std::vector<dftracer::utils::dataframe::GroupAgg>& out) {
    PyObject* seq = PySequence_Fast(obj, "aggs must be a sequence of str");
    if (!seq) return false;
    Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(seq, i));
        if (!s) {
            Py_DECREF(seq);
            return false;
        }
        try {
            out.push_back(group_agg_from_spec(s));
        } catch (const std::exception& e) {
            PyErr_SetString(PyExc_ValueError, e.what());
            Py_DECREF(seq);
            return false;
        }
    }
    Py_DECREF(seq);
    return true;
}

}  // namespace dftracer::utils::python

#endif  // DFTRACER_UTILS_PYTHON_PY_AGG_HELPERS_H
