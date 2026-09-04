#ifndef DFTRACER_UTILS_PYTHON_PY_AGG_HELPERS_H
#define DFTRACER_UTILS_PYTHON_PY_AGG_HELPERS_H

#include <Python.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/python/py_seq_helpers.h>

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
    return parse_seq<dftracer::utils::dataframe::GroupAgg>(
        obj, "aggs must be a sequence of str", out,
        [](PyObject* item,
           std::vector<dftracer::utils::dataframe::GroupAgg>& o) {
            const char* s = PyUnicode_AsUTF8(item);
            if (!s) return false;
            try {
                o.push_back(group_agg_from_spec(s));
            } catch (const std::exception& e) {
                PyErr_SetString(PyExc_ValueError, e.what());
                return false;
            }
            return true;
        });
}

// Parse a group_by key argument that is either one column name or a sequence
// of names (native multi-key entry point). Appends to `out`.
inline bool strings_from_str_or_seq(PyObject* obj,
                                    std::vector<std::string>& out) {
    if (PyUnicode_Check(obj)) {
        const char* s = PyUnicode_AsUTF8(obj);
        if (!s) return false;
        out.emplace_back(s);
        return true;
    }
    return parse_string_seq(obj, "key must be a str or a sequence of str", out);
}

}  // namespace dftracer::utils::python

#endif  // DFTRACER_UTILS_PYTHON_PY_AGG_HELPERS_H
