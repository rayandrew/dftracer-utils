#ifndef DFTRACER_UTILS_PYTHON_PLUGIN_HOST_H
#define DFTRACER_UTILS_PYTHON_PLUGIN_HOST_H

#include <Python.h>

// PluginHost: the Python surface over the C++ plugins::PluginHost. load()
// dlopens a compiled .so plugin, resolve() wires plugin capabilities, and run()
// drives every loaded plugin as a fold over one fused parallel scan of a trace
// set, returning each plugin's named results and exposing the scan counters on
// the `stats` attribute.

typedef struct {
    PyObject_HEAD PyObject
        *runtime_obj;  // RuntimeObject* or NULL (uses default)
    void *host_ptr;    // plugins::PluginHost*
    PyObject *stats;   // dict of the last run's scan counters, or NULL
} PluginHostObject;

extern PyTypeObject PluginHostType;

namespace dftracer::utils::python {

int init_plugin_host(PyObject *m);

// Read a PluginHostObject's named results as a {name: pyarrow|bytes} dict, once
// its session has executed. NULL with a Python error set on failure.
PyObject *plugin_host_results_dict(PyObject *host);

}  // namespace dftracer::utils::python

#endif  // DFTRACER_UTILS_PYTHON_PLUGIN_HOST_H
