#ifndef DFTRACER_UTILS_PYTHON_PLUGIN_HOST_H
#define DFTRACER_UTILS_PYTHON_PLUGIN_HOST_H

#include <Python.h>
#include <dftracer/utils/plugins/plugins.h>

// PluginHost: the Python surface over the C++ plugins::Plugins set. load()
// queues a compiled .so plugin and run() builds the set (dlopen, ABI gate,
// capability resolution) and drives every plugin as a fold over one fused
// parallel scan of a trace set, returning each plugin's named results and
// exposing the scan counters on the `stats` attribute.

typedef struct {
    PyObject_HEAD PyObject
        *runtime_obj;  // RuntimeObject* or NULL (uses default)
    void *host_ptr;    // python::PluginHostState*
    PyObject *stats;   // dict of the last run's scan counters, or NULL
} PluginHostObject;

extern PyTypeObject PluginHostType;

namespace dftracer::utils::python {

int init_plugin_host(PyObject *m);

// Read a PluginHostObject's named results as a {name: pyarrow|bytes} dict, once
// its session has executed. NULL with a Python error set on failure.
PyObject *plugin_host_results_dict(PyObject *host);

// The built plugin set behind a PluginHostObject, built on first use. NULL with
// a Python error set on a load/ABI/capability failure.
const plugins::Plugins *plugin_host_plugins(PyObject *host);

// Where a co-scan should deposit that host's named results; owned by the host
// object, so it outlives the session. NULL with a Python error set.
plugins::NamedResultRegistry *plugin_host_results(PyObject *host);

}  // namespace dftracer::utils::python

#endif  // DFTRACER_UTILS_PYTHON_PLUGIN_HOST_H
