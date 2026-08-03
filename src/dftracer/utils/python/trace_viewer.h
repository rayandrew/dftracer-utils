#ifndef DFTRACER_UTILS_PYTHON_TRACE_VIEWER_H
#define DFTRACER_UTILS_PYTHON_TRACE_VIEWER_H

#include <Python.h>

// TraceViewer: the arrow-first Python surface over the C++ View. Builder
// methods
// (filter/query/phase/group_by/agg/time_bucket/time_range/select/memory_budget/
// auto_spill/limit/offset) return a new TraceViewer holding an updated plan;
// terminals execute once (collect -> pyarrow.Table, stream -> arrow batch
// iterator, export_*). The plan is carried as a Python-owned config and lowered
// onto a fresh View in each terminal, so a TraceViewer is cheap to copy.

typedef struct {
    PyObject_HEAD PyObject *files;  // list[str] of trace paths
    PyObject *index_path;           // str or None
    void *plan_ptr;                 // ViewerPlan*: the accumulated builder ops
    PyObject *runtime_obj;          // RuntimeObject* or NULL (uses default)
} TraceViewerObject;

extern PyTypeObject TraceViewerType;
// group_by/agg/agg_numeric_args promote to this; it adds the cache terminals.
extern PyTypeObject AggregatedTraceViewerType;
int init_trace_viewer(PyObject *m);

#endif  // DFTRACER_UTILS_PYTHON_TRACE_VIEWER_H
