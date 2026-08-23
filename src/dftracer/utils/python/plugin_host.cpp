#define PY_SSIZE_T_CLEAN
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/plugins/config.h>
#include <dftracer/utils/plugins/host.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/python/plugin_host.h>
#include <dftracer/utils/python/py_dict_helpers.h>
#include <dftracer/utils/python/py_errors.h>
#include <dftracer/utils/python/py_method.h>
#include <dftracer/utils/python/py_runtime_mixin.h>
#include <dftracer/utils/python/py_str_helpers.h>
#include <dftracer/utils/python/py_type_helpers.h>
#include <dftracer/utils/python/runtime.h>
#include <dftracer/utils/trace/indexing/resolve_and_build.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/utilities/filesystem/pattern_directory_scanner_utility.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/python/arrow_helpers.h>
#include <dftracer/utils/python/dataframe.h>
#include <dftracer/utils/utilities/common/arrow/explode.h>
#include <dftracer/utils/utilities/common/arrow/gapfill.h>
#include <dftracer/utils/utilities/common/arrow/join.h>
#include <dftracer/utils/utilities/common/arrow/window.h>
#include <nanoarrow/nanoarrow.h>
#endif

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <new>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace {

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::coro::CoroTask;
using dftracer::utils::plugins::ConfigTree;
using dftracer::utils::plugins::NamedResult;
using dftracer::utils::plugins::OwnedArrow;
using dftracer::utils::plugins::PluginHost;
namespace indexing = dftracer::utils::trace::indexing;
namespace internal = dftracer::utils::trace::internal;
namespace views = dftracer::utils::trace::views;
namespace filesystem = dftracer::utils::utilities::filesystem;

PluginHost* host_of(PluginHostObject* self) {
    return static_cast<PluginHost*>(self->host_ptr);
}

// Expand any directory in `inputs` to its .pfw/.pfw.gz files, index (unless
// disabled), then drive every loaded plugin as a fold over one fused scan.
CoroTask<void> run_host_scan(CoroScope& scope, std::vector<std::string> inputs,
                             std::string index_dir, const PluginHost* host,
                             bool auto_index, views::ExportStats* out) {
    std::vector<std::string> files;
    for (const auto& p : inputs) {
        std::error_code ec;
        if (fs::is_directory(p, ec)) {
            filesystem::PatternDirectoryScannerUtilityInput in(
                p, {".pfw", ".pfw.gz"}, /*recursive=*/true,
                /*populate_size=*/false);
            filesystem::PatternDirectoryScannerUtility scanner;
            auto entries = co_await scanner(scope, in);
            for (auto& e : entries) files.push_back(e.path.string());
        } else {
            files.push_back(p);
        }
    }
    std::sort(files.begin(), files.end());
    if (files.empty())
        throw dftracer::utils::DFTUtilsException(
            dftracer::utils::ErrorCode::NOT_FOUND,
            "no .pfw or .pfw.gz trace files found");

    if (auto_index) {
        auto norm = co_await indexing::normalize_members_for_ingest(files, 0);
        files = std::move(norm.files);
        co_await indexing::ensure_indexes_fresh(&scope, "", files, index_dir);
    }

    std::vector<views::ViewFile> view_files;
    view_files.reserve(files.size());
    for (const auto& f : files) {
        views::ViewFile vf;
        vf.file_path = f;
        vf.index_path = internal::determine_index_path(f, index_dir);
        view_files.push_back(std::move(vf));
    }
    views::View view = views::View::from_files(std::move(view_files));
    *out = co_await host->run(view);
}

bool collect_inputs(PyObject* traces, std::vector<std::string>& out) {
    if (PyUnicode_Check(traces)) {
        const char* s = PyUnicode_AsUTF8(traces);
        if (!s) return false;
        out.emplace_back(s);
        return true;
    }
    PyObject* seq =
        PySequence_Fast(traces, "traces must be a str or list[str]");
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = as_utf8(PySequence_Fast_GET_ITEM(seq, i));
        if (!s) {
            Py_DECREF(seq);
            return false;
        }
        out.emplace_back(s);
    }
    Py_DECREF(seq);
    return true;
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
using dftracer::utils::plugins::OwnedArrowBatches;

// Backs a pull-based ArrowArrayStream over in-memory same-schema batches. Each
// batch is self-contained (independent of the intern table and producing fold),
// so nothing external must stay alive while Python pulls.
struct MapStreamState {
    std::vector<OwnedArrow> batches;
    std::size_t idx = 0;
    ArrowSchema schema_template{};
};

int map_stream_get_schema(ArrowArrayStream* s, ArrowSchema* out) {
    auto* p = static_cast<MapStreamState*>(s->private_data);
    if (!p || !p->schema_template.release) return EINVAL;
    return ArrowSchemaDeepCopy(&p->schema_template, out);
}

int map_stream_get_next(ArrowArrayStream* s, ArrowArray* out) {
    auto* p = static_cast<MapStreamState*>(s->private_data);
    if (!p || p->idx >= p->batches.size()) {
        out->release = nullptr;  // end of stream per the Arrow C spec
        return 0;
    }
    ArrowArrayMove(&p->batches[p->idx].array, out);
    ++p->idx;
    return 0;
}

const char* map_stream_get_last_error(ArrowArrayStream*) { return nullptr; }

void map_stream_release(ArrowArrayStream* s) {
    auto* p = static_cast<MapStreamState*>(s->private_data);
    if (p) {
        if (p->schema_template.release)
            p->schema_template.release(&p->schema_template);
        delete p;  // OwnedArrow destructors release any unconsumed batch
    }
    s->private_data = nullptr;
    s->release = nullptr;
}

void map_stream_capsule_release(PyObject* capsule) {
    auto* stream = static_cast<ArrowArrayStream*>(
        PyCapsule_GetPointer(capsule, "arrow_array_stream"));
    if (stream && stream->release) stream->release(stream);
    delete stream;
}

struct MapResultStreamObject {
    PyObject_HEAD std::vector<OwnedArrow> batches;
    ArrowSchema schema_template;
    bool consumed;
};

PyObject* map_result_stream_arrow_c_stream(MapResultStreamObject* self,
                                           PyObject* args) {
    PyObject* requested_schema = Py_None;
    if (!PyArg_ParseTuple(args, "|O", &requested_schema)) return nullptr;
    if (requested_schema != Py_None) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "map stream does not support requested_schema casting; "
                        "pass None to use the native schema.");
        return nullptr;
    }
    if (self->consumed) {
        PyErr_SetString(PyExc_RuntimeError,
                        "map result stream already exported; each stream can "
                        "be exported only once.");
        return nullptr;
    }
    auto* state = new MapStreamState;
    state->batches = std::move(self->batches);
    state->schema_template = self->schema_template;
    self->schema_template = ArrowSchema{};
    self->consumed = true;

    auto* stream = new ArrowArrayStream;
    std::memset(stream, 0, sizeof(*stream));
    stream->get_schema = map_stream_get_schema;
    stream->get_next = map_stream_get_next;
    stream->get_last_error = map_stream_get_last_error;
    stream->release = map_stream_release;
    stream->private_data = state;

    PyObject* capsule =
        PyCapsule_New(stream, "arrow_array_stream", map_stream_capsule_release);
    if (!capsule) {
        stream->release(stream);
        delete stream;
        return nullptr;
    }
    return capsule;
}

void map_result_stream_dealloc(MapResultStreamObject* self) {
    if (self->schema_template.release)
        self->schema_template.release(&self->schema_template);
    self->batches.~vector<OwnedArrow>();
    Py_TYPE(self)->tp_free((PyObject*)self);
}

PyMethodDef map_result_stream_methods[] = {
    {"__arrow_c_stream__", DFTU_PYCFUNCTION(map_result_stream_arrow_c_stream),
     METH_VARARGS, "Export as an Arrow C Data Interface stream PyCapsule."},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject MapResultStreamType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "dftracer_utils_ext._MapResultStream",
    sizeof(MapResultStreamObject),
    0,
    (destructor)map_result_stream_dealloc,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    "Streamed plugin-map result backed by in-memory Arrow batches.",
    0,
    0,
    0,
    0,
    0,
    0,
    map_result_stream_methods,
};

// Move `batches` into a fresh RecordBatchReader. Returns a new reference or
// nullptr with a Python error set.
PyObject* batches_to_reader(std::vector<OwnedArrow>& batches) {
    if (PyType_Ready(&MapResultStreamType) < 0) return nullptr;
    auto* obj = (MapResultStreamObject*)MapResultStreamType.tp_alloc(
        &MapResultStreamType, 0);
    if (!obj) return nullptr;
    new (&obj->batches) std::vector<OwnedArrow>(std::move(batches));
    obj->schema_template = ArrowSchema{};
    obj->consumed = false;
    if (!obj->batches.empty() && obj->batches.front().schema.release &&
        ArrowSchemaDeepCopy(&obj->batches.front().schema,
                            &obj->schema_template) != 0) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_RuntimeError, "failed to copy map stream schema");
        return nullptr;
    }
    PyObject* pa = PyImport_ImportModule("pyarrow");
    if (!pa) {
        Py_DECREF(obj);
        return nullptr;
    }
    PyObject* rbr = PyObject_GetAttrString(pa, "RecordBatchReader");
    Py_DECREF(pa);
    if (!rbr) {
        Py_DECREF(obj);
        return nullptr;
    }
    PyObject* reader =
        PyObject_CallMethod(rbr, "from_stream", "O", (PyObject*)obj);
    Py_DECREF(rbr);
    Py_DECREF(obj);
    return reader;
}

// Wrap a finished Arrow record batch as a native _DataFrame, zero-copy through
// our own bridge (no pyarrow). Consumes `result`: from_arrow adopts the array.
PyObject* arrow_result_to_dataframe(
    dftracer::utils::utilities::common::arrow::ArrowExportResult result) {
    if (!result.valid()) {
        PyErr_SetString(PyExc_RuntimeError, "invalid Arrow result");
        return nullptr;
    }
    try {
        dftracer::utils::dataframe::DataFrame df =
            dftracer::utils::dataframe::DataFrame::from_arrow(
                result.get_schema(), result.get_array());
        if (df.num_columns() == 0 && result.num_columns() != 0) {
            PyErr_SetString(PyExc_ValueError,
                            "failed to import analytical-op result columns");
            return nullptr;
        }
        return dftracer::utils::python::wrap_dataframe(std::move(df));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
        return nullptr;
    }
}

// `batch` is any object exposing __arrow_c_array__; the Python wrapper
// collapses a Table/reader to one batch first.
PyObject* py_unnest(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"batch", "column", "keep_empty", nullptr};
    PyObject* batch = nullptr;
    const char* column = nullptr;
    int keep_empty = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "Os|p",
                                     const_cast<char**>(kwlist), &batch,
                                     &column, &keep_empty))
        return nullptr;

    PyObject* capsules =
        PyObject_CallMethod(batch, "__arrow_c_array__", nullptr);
    if (!capsules) return nullptr;
    if (!PyTuple_Check(capsules) || PyTuple_GET_SIZE(capsules) != 2) {
        Py_DECREF(capsules);
        PyErr_SetString(
            PyExc_TypeError,
            "unnest: __arrow_c_array__ must return (schema, array)");
        return nullptr;
    }
    auto* schema = static_cast<ArrowSchema*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 0), "arrow_schema"));
    auto* array = static_cast<ArrowArray*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 1), "arrow_array"));
    if (!schema || !array) {
        Py_DECREF(capsules);
        return nullptr;
    }

    std::size_t col_idx = 0;
    bool found = false;
    for (std::int64_t i = 0; i < schema->n_children; ++i) {
        const char* nm = schema->children[i]->name;
        if (nm && std::strcmp(nm, column) == 0) {
            col_idx = static_cast<std::size_t>(i);
            found = true;
            break;
        }
    }
    if (!found) {
        Py_DECREF(capsules);
        PyErr_Format(PyExc_KeyError, "unnest: no column named '%s'", column);
        return nullptr;
    }

    namespace arr = dftracer::utils::utilities::common::arrow;
    PyObject* out = nullptr;
    try {
        arr::ArrowExportResult result =
            arr::explode(schema, array, col_idx, keep_empty != 0);
        out = arrow_result_to_dataframe(std::move(result));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    Py_DECREF(capsules);
    return out;
}

PyMethodDef unnest_methods[] = {
    {"unnest", DFTU_PYCFUNCTION(py_unnest), METH_VARARGS | METH_KEYWORDS,
     "unnest(batch, column, keep_empty=False) -> _DataFrame: explode a "
     "list-typed column into one row per element, repeating other columns. A "
     "list<struct> flattens its fields into columns. Empty/null lists drop the "
     "row unless keep_empty=True (then one null-exploded row)."},
    {nullptr, nullptr, 0, nullptr}};

namespace arr = dftracer::utils::utilities::common::arrow;
using dftracer::utils::python::arrow_result_to_table;

// Holds the source (schema, array) capsules alive while the C++ op reads them.
struct ArrowInput {
    PyObject* capsules = nullptr;
    ArrowSchema* schema = nullptr;
    ArrowArray* array = nullptr;
    ArrowInput() = default;
    ArrowInput(const ArrowInput&) = delete;
    ArrowInput& operator=(const ArrowInput&) = delete;
    ~ArrowInput() { Py_XDECREF(capsules); }
};

bool import_arrow(PyObject* obj, ArrowInput& in) {
    PyObject* capsules = PyObject_CallMethod(obj, "__arrow_c_array__", nullptr);
    if (!capsules) return false;
    if (!PyTuple_Check(capsules) || PyTuple_GET_SIZE(capsules) != 2) {
        Py_DECREF(capsules);
        PyErr_SetString(PyExc_TypeError,
                        "__arrow_c_array__ must return (schema, array)");
        return false;
    }
    auto* schema = static_cast<ArrowSchema*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 0), "arrow_schema"));
    auto* array = static_cast<ArrowArray*>(
        PyCapsule_GetPointer(PyTuple_GET_ITEM(capsules, 1), "arrow_array"));
    if (!schema || !array) {
        Py_DECREF(capsules);
        return false;
    }
    in.capsules = capsules;
    in.schema = schema;
    in.array = array;
    return true;
}

bool resolve_col(const ArrowSchema* s, const char* name, std::uint32_t* out) {
    for (std::int64_t i = 0; i < s->n_children; ++i) {
        const char* nm = s->children[i]->name;
        if (nm && std::strcmp(nm, name) == 0) {
            *out = static_cast<std::uint32_t>(i);
            return true;
        }
    }
    PyErr_Format(PyExc_KeyError, "no column named '%s'", name);
    return false;
}

bool resolve_cols(const ArrowSchema* s, PyObject* names,
                  std::vector<std::uint32_t>& out) {
    PyObject* fast = PySequence_Fast(names, "expected a sequence of names");
    if (!fast) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    out.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* nm = as_utf8(PySequence_Fast_GET_ITEM(fast, i));
        std::uint32_t idx = 0;
        if (!nm || !resolve_col(s, nm, &idx)) {
            Py_DECREF(fast);
            return false;
        }
        out.push_back(idx);
    }
    Py_DECREF(fast);
    return true;
}

bool window_func_from_str(const char* name, arr::WindowFunc* out) {
    struct Entry {
        const char* name;
        arr::WindowFunc func;
    };
    static const Entry table[] = {
        {"row_number", arr::WindowFunc::ROW_NUMBER},
        {"rank", arr::WindowFunc::RANK},
        {"dense_rank", arr::WindowFunc::DENSE_RANK},
        {"lag", arr::WindowFunc::LAG},
        {"lead", arr::WindowFunc::LEAD},
        {"running_sum", arr::WindowFunc::RUNNING_SUM},
        {"running_min", arr::WindowFunc::RUNNING_MIN},
        {"running_max", arr::WindowFunc::RUNNING_MAX},
        {"running_count", arr::WindowFunc::RUNNING_COUNT},
        {"delta", arr::WindowFunc::DELTA},
        {"rate", arr::WindowFunc::RATE},
        {"sessionize", arr::WindowFunc::SESSIONIZE},
        {"frame_sum", arr::WindowFunc::FRAME_SUM},
        {"frame_min", arr::WindowFunc::FRAME_MIN},
        {"frame_max", arr::WindowFunc::FRAME_MAX},
        {"frame_count", arr::WindowFunc::FRAME_COUNT},
        {"frame_mean", arr::WindowFunc::FRAME_MEAN},
        {"ntile", arr::WindowFunc::NTILE},
        {"first_value", arr::WindowFunc::FIRST_VALUE},
        {"last_value", arr::WindowFunc::LAST_VALUE},
        {"nth_value", arr::WindowFunc::NTH_VALUE}};
    for (const auto& e : table)
        if (std::strcmp(e.name, name) == 0) {
            *out = e.func;
            return true;
        }
    PyErr_Format(PyExc_ValueError, "window: unknown function '%s'", name);
    return false;
}

// One spec is a fixed 9-tuple normalized by the Python wrapper:
// (func, value_col|None, offset, name, time_col|None, threshold, counter,
//  frame_preceding, frame_following).
bool parse_window_spec(const ArrowSchema* s, PyObject* t,
                       arr::WindowSpec& out) {
    const char* func = nullptr;
    PyObject* value_obj = nullptr;
    Py_ssize_t offset = 0;
    const char* name = nullptr;
    PyObject* time_obj = nullptr;
    double threshold = 0.0;
    int counter = 0;
    Py_ssize_t frame_pre = 0;
    Py_ssize_t frame_post = 0;
    if (!PyArg_ParseTuple(t, "sOnsOdpnn", &func, &value_obj, &offset, &name,
                          &time_obj, &threshold, &counter, &frame_pre,
                          &frame_post))
        return false;
    if (!window_func_from_str(func, &out.func)) return false;
    out.value_col = 0;
    if (value_obj != Py_None) {
        const char* vc = as_utf8(value_obj);
        if (!vc || !resolve_col(s, vc, &out.value_col)) return false;
    }
    out.time_col = 0;
    if (time_obj != Py_None) {
        const char* tc = as_utf8(time_obj);
        if (!tc || !resolve_col(s, tc, &out.time_col)) return false;
    }
    out.offset = static_cast<std::int64_t>(offset);
    out.name = name;
    out.threshold = threshold;
    out.counter = counter != 0;
    out.frame_preceding = static_cast<std::int64_t>(frame_pre);
    out.frame_following = static_cast<std::int64_t>(frame_post);
    return true;
}

bool gapfill_mode_from_str(const char* s, arr::GapFillMode* out) {
    if (std::strcmp(s, "none") == 0)
        *out = arr::GapFillMode::NONE;
    else if (std::strcmp(s, "locf") == 0)
        *out = arr::GapFillMode::LOCF;
    else if (std::strcmp(s, "linear") == 0)
        *out = arr::GapFillMode::LINEAR;
    else {
        PyErr_Format(PyExc_ValueError, "gap_fill: unknown mode '%s'", s);
        return false;
    }
    return true;
}

bool join_type_from_str(const char* s, arr::JoinType* out) {
    if (std::strcmp(s, "inner") == 0)
        *out = arr::JoinType::INNER;
    else if (std::strcmp(s, "left") == 0)
        *out = arr::JoinType::LEFT;
    else if (std::strcmp(s, "right") == 0)
        *out = arr::JoinType::RIGHT;
    else if (std::strcmp(s, "full") == 0)
        *out = arr::JoinType::FULL;
    else if (std::strcmp(s, "semi") == 0)
        *out = arr::JoinType::LEFT_SEMI;
    else if (std::strcmp(s, "anti") == 0)
        *out = arr::JoinType::LEFT_ANTI;
    else {
        PyErr_Format(PyExc_ValueError, "join: unknown how '%s'", s);
        return false;
    }
    return true;
}

bool asof_dir_from_str(const char* s, arr::AsofDirection* out) {
    if (std::strcmp(s, "backward") == 0)
        *out = arr::AsofDirection::BACKWARD;
    else if (std::strcmp(s, "forward") == 0)
        *out = arr::AsofDirection::FORWARD;
    else if (std::strcmp(s, "nearest") == 0)
        *out = arr::AsofDirection::NEAREST;
    else {
        PyErr_Format(PyExc_ValueError, "asof: unknown direction '%s'", s);
        return false;
    }
    return true;
}

PyObject* py_window(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"batch", "partition_by", "order_by", "specs",
                                   nullptr};
    PyObject* batch = nullptr;
    PyObject* part = nullptr;
    PyObject* order = nullptr;
    PyObject* specs = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOO",
                                     const_cast<char**>(kwlist), &batch, &part,
                                     &order, &specs))
        return nullptr;
    ArrowInput in;
    if (!import_arrow(batch, in)) return nullptr;
    std::vector<std::uint32_t> pcols;
    std::vector<std::uint32_t> ocols;
    if (!resolve_cols(in.schema, part, pcols) ||
        !resolve_cols(in.schema, order, ocols))
        return nullptr;
    PyObject* fast = PySequence_Fast(specs, "window: specs must be a sequence");
    if (!fast) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    std::vector<arr::WindowSpec> specv(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        if (!parse_window_spec(in.schema, PySequence_Fast_GET_ITEM(fast, i),
                               specv[static_cast<std::size_t>(i)])) {
            Py_DECREF(fast);
            return nullptr;
        }
    }
    Py_DECREF(fast);
    PyObject* out = nullptr;
    try {
        out = arrow_result_to_dataframe(
            arr::window(in.schema, in.array, pcols.data(),
                        static_cast<std::uint32_t>(pcols.size()), ocols.data(),
                        static_cast<std::uint32_t>(ocols.size()), specv.data(),
                        static_cast<std::uint32_t>(specv.size())));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    return out;
}

PyObject* py_gap_fill(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"batch",  "partition_by", "time",
                                   "bucket", "values",       "mode",
                                   "start",  "end",          nullptr};
    PyObject* batch = nullptr;
    PyObject* part = nullptr;
    const char* time = nullptr;
    long long bucket = 0;
    PyObject* values = nullptr;
    const char* mode = nullptr;
    PyObject* start_obj = Py_None;
    PyObject* end_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "OOsLOs|OO", const_cast<char**>(kwlist), &batch, &part,
            &time, &bucket, &values, &mode, &start_obj, &end_obj))
        return nullptr;
    arr::GapFillMode m;
    if (!gapfill_mode_from_str(mode, &m)) return nullptr;
    ArrowInput in;
    if (!import_arrow(batch, in)) return nullptr;
    std::vector<std::uint32_t> pcols;
    std::vector<std::uint32_t> vcols;
    std::uint32_t time_col = 0;
    if (!resolve_cols(in.schema, part, pcols) ||
        !resolve_col(in.schema, time, &time_col) ||
        !resolve_cols(in.schema, values, vcols))
        return nullptr;
    const bool has_range = start_obj != Py_None;
    std::int64_t range_start = 0;
    std::int64_t range_end = 0;
    if (has_range) {
        range_start = PyLong_AsLongLong(start_obj);
        range_end = PyLong_AsLongLong(end_obj);
        if (PyErr_Occurred()) return nullptr;
    }
    PyObject* out = nullptr;
    try {
        out = arrow_result_to_dataframe(
            arr::gap_fill(in.schema, in.array, pcols.data(),
                          static_cast<std::uint32_t>(pcols.size()), time_col,
                          static_cast<std::int64_t>(bucket), vcols.data(),
                          static_cast<std::uint32_t>(vcols.size()), m,
                          has_range, range_start, range_end));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    return out;
}

PyObject* py_join(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"left", "right", "on", "how", nullptr};
    PyObject* left = nullptr;
    PyObject* right = nullptr;
    PyObject* on = nullptr;
    const char* how = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOOs",
                                     const_cast<char**>(kwlist), &left, &right,
                                     &on, &how))
        return nullptr;
    arr::JoinType jt;
    if (!join_type_from_str(how, &jt)) return nullptr;
    ArrowInput li;
    ArrowInput ri;
    if (!import_arrow(left, li) || !import_arrow(right, ri)) return nullptr;
    std::vector<std::uint32_t> lkeys;
    std::vector<std::uint32_t> rkeys;
    if (!resolve_cols(li.schema, on, lkeys) ||
        !resolve_cols(ri.schema, on, rkeys))
        return nullptr;
    if (lkeys.empty()) {
        PyErr_SetString(PyExc_ValueError, "join: on must name >=1 key column");
        return nullptr;
    }
    PyObject* out = nullptr;
    try {
        out = arrow_result_to_dataframe(arr::join(
            li.schema, li.array, lkeys.data(), ri.schema, ri.array,
            rkeys.data(), static_cast<std::uint32_t>(lkeys.size()), jt));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    return out;
}

PyObject* py_asof(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"left",      "right",     "on",   "by",
                                   "direction", "tolerance", nullptr};
    PyObject* left = nullptr;
    PyObject* right = nullptr;
    const char* on = nullptr;
    PyObject* by = nullptr;
    const char* direction = nullptr;
    PyObject* tol_obj = Py_None;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOsOsO",
                                     const_cast<char**>(kwlist), &left, &right,
                                     &on, &by, &direction, &tol_obj))
        return nullptr;
    arr::AsofDirection dir;
    if (!asof_dir_from_str(direction, &dir)) return nullptr;
    ArrowInput li;
    ArrowInput ri;
    if (!import_arrow(left, li) || !import_arrow(right, ri)) return nullptr;
    std::uint32_t lts = 0;
    std::uint32_t rts = 0;
    std::vector<std::uint32_t> lequi;
    std::vector<std::uint32_t> requi;
    if (!resolve_col(li.schema, on, &lts) ||
        !resolve_col(ri.schema, on, &rts) ||
        !resolve_cols(li.schema, by, lequi) ||
        !resolve_cols(ri.schema, by, requi))
        return nullptr;
    const bool has_tol = tol_obj != Py_None;
    std::int64_t tol = 0;
    if (has_tol) {
        tol = PyLong_AsLongLong(tol_obj);
        if (PyErr_Occurred()) return nullptr;
    }
    PyObject* out = nullptr;
    try {
        out = arrow_result_to_dataframe(arr::asof_join(
            li.schema, li.array, lts, lequi.data(), ri.schema, ri.array, rts,
            requi.data(), static_cast<std::uint32_t>(lequi.size()), dir,
            has_tol, tol));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    return out;
}

PyObject* py_interval(PyObject*, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"left", "right", "point", "lo",
                                   "hi",   "by",    "outer", nullptr};
    PyObject* left = nullptr;
    PyObject* right = nullptr;
    const char* point = nullptr;
    const char* lo = nullptr;
    const char* hi = nullptr;
    PyObject* by = nullptr;
    int outer = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "OOsssOp",
                                     const_cast<char**>(kwlist), &left, &right,
                                     &point, &lo, &hi, &by, &outer))
        return nullptr;
    ArrowInput li;
    ArrowInput ri;
    if (!import_arrow(left, li) || !import_arrow(right, ri)) return nullptr;
    std::uint32_t point_col = 0;
    std::uint32_t lo_col = 0;
    std::uint32_t hi_col = 0;
    std::vector<std::uint32_t> lequi;
    std::vector<std::uint32_t> requi;
    if (!resolve_col(li.schema, point, &point_col) ||
        !resolve_col(ri.schema, lo, &lo_col) ||
        !resolve_col(ri.schema, hi, &hi_col) ||
        !resolve_cols(li.schema, by, lequi) ||
        !resolve_cols(ri.schema, by, requi))
        return nullptr;
    PyObject* out = nullptr;
    try {
        out = arrow_result_to_dataframe(arr::interval_join(
            li.schema, li.array, point_col, lequi.data(), ri.schema, ri.array,
            lo_col, hi_col, requi.data(),
            static_cast<std::uint32_t>(lequi.size()), outer != 0));
    } catch (const std::exception& e) {
        PyErr_SetString(PyExc_ValueError, e.what());
    }
    return out;
}

PyMethodDef arrow_ops_methods[] = {
    {"window", DFTU_PYCFUNCTION(py_window), METH_VARARGS | METH_KEYWORDS,
     "window(batch, partition_by, order_by, specs) -> _DataFrame: SQL "
     "window functions over one batch."},
    {"gap_fill", DFTU_PYCFUNCTION(py_gap_fill), METH_VARARGS | METH_KEYWORDS,
     "gap_fill(batch, partition_by, time, bucket, values, mode, start, end) -> "
     "_DataFrame: materialize a regular time grid with fills."},
    {"join", DFTU_PYCFUNCTION(py_join), METH_VARARGS | METH_KEYWORDS,
     "join(left, right, on, how) -> _DataFrame: same-key equi join."},
    {"asof", DFTU_PYCFUNCTION(py_asof), METH_VARARGS | METH_KEYWORDS,
     "asof(left, right, on, by, direction, tolerance) -> _DataFrame: "
     "temporal nearest-match join."},
    {"interval", DFTU_PYCFUNCTION(py_interval), METH_VARARGS | METH_KEYWORDS,
     "interval(left, right, point, lo, hi, by, outer) -> _DataFrame: "
     "point-in-range join."},
    {nullptr, nullptr, 0, nullptr}};
#endif

// Convert one emitted result to Python: an opaque blob to bytes, a single
// user-schema Arrow array to a pyarrow-compatible table, and a streamed
// multi-batch map result to a pull-based pyarrow.RecordBatchReader. Returns a
// new reference, or nullptr with a Python error set.
PyObject* result_to_py(NamedResult& result) {
    if (auto* blob = std::get_if<std::vector<std::byte>>(&result))
        return PyBytes_FromStringAndSize(
            reinterpret_cast<const char*>(blob->data()),
            static_cast<Py_ssize_t>(blob->size()));
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    namespace arr = dftracer::utils::utilities::common::arrow;
    if (auto* batches = std::get_if<OwnedArrowBatches>(&result))
        return batches_to_reader(batches->batches);
    auto* owned = std::get_if<OwnedArrow>(&result);
    nanoarrow::UniqueSchema schema;
    nanoarrow::UniqueArray array;
    ArrowSchemaMove(&owned->schema, schema.get());
    ArrowArrayMove(&owned->array, array.get());
    return dftracer::utils::python::arrow_result_to_table(
        arr::ArrowExportResult(std::move(schema), std::move(array)));
#else
    PyErr_SetString(PyExc_RuntimeError,
                    "plugin emitted an Arrow result but the extension was "
                    "built without Arrow support");
    return nullptr;
#endif
}

PyObject* results_to_dict(PluginHost* host) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    for (auto& [name, result] : host->results().results()) {
        PyObject* val = result_to_py(result);
        if (!val) {
            Py_DECREF(d);
            return nullptr;
        }
        if (PyDict_SetItemString(d, name.c_str(), val) < 0) {
            Py_DECREF(val);
            Py_DECREF(d);
            return nullptr;
        }
        Py_DECREF(val);
    }
    return d;
}

PyObject* ph_new(PyTypeObject* type, PyObject*, PyObject*) {
    PluginHostObject* self = (PluginHostObject*)type->tp_alloc(type, 0);
    if (!self) return nullptr;
    self->runtime_obj = nullptr;
    self->stats = nullptr;
    self->host_ptr = new PluginHost();
    return (PyObject*)self;
}

void ph_dealloc(PluginHostObject* self) {
    delete host_of(self);
    Py_XDECREF(self->runtime_obj);
    Py_XDECREF(self->stats);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

int ph_init(PluginHostObject* self, PyObject* args, PyObject* kwds) {
    return runtime_backed_init(self, args, kwds);
}

// load(path, config=None): config is a JSON object string (the Python wrapper
// serializes a dict). Raises on config-parse, load, symbol, or ABI failure.
PyObject* ph_load(PluginHostObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"path", "config", nullptr};
    const char* path = nullptr;
    const char* config_json = nullptr;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "s|z", const_cast<char**>(kwlist), &path, &config_json))
        return nullptr;

    PluginHost* host = host_of(self);
    const dftu_value* root = nullptr;
    if (config_json) {
        try {
            root = host->add_config(ConfigTree::from_json_string(config_json));
        } catch (const std::exception& e) {
            PyErr_SetString(dftracer::utils::python::g_dft_value_error
                                ? dftracer::utils::python::g_dft_value_error
                                : PyExc_ValueError,
                            e.what());
            return nullptr;
        }
    }
    if (!host->load(path, root)) {
        PyErr_Format(PyExc_ImportError,
                     "failed to load plugin '%s' (see log for the reason)",
                     path);
        return nullptr;
    }
    Py_RETURN_NONE;
}

PyObject* ph_resolve(PluginHostObject* self, PyObject*) {
    const bool ok = host_of(self)->resolve();
    return PyBool_FromLong(ok ? 1 : 0);
}

// run(traces, index_dir=None, auto_index=True) -> {name: result}. Drives the
// fused scan with the GIL released; scan counters land on the `stats` attr.
PyObject* ph_run(PluginHostObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"traces", "index_dir", "auto_index",
                                   nullptr};
    PyObject* traces = nullptr;
    const char* index_dir = nullptr;
    int auto_index = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|zp",
                                     const_cast<char**>(kwlist), &traces,
                                     &index_dir, &auto_index))
        return nullptr;

    std::vector<std::string> inputs;
    if (!collect_inputs(traces, inputs)) return nullptr;

    Runtime* rt = resolve_runtime(self);
    PluginHost* host = host_of(self);
    std::string index_dir_s = index_dir ? index_dir : "";
    views::ExportStats stats;
    if (!run_blocking([&] {
            rt->submit(
                  dftracer::utils::run_coro_scope(
                      rt->executor(), run_host_scan, std::move(inputs),
                      std::move(index_dir_s), host, auto_index != 0, &stats),
                  "plugin-host-run")
                .get();
        }))
        return nullptr;

    PyObject* stats_dict = PyDict_New();
    if (!stats_dict) return nullptr;
    dict_set_i64(stats_dict, "events_scanned", (long long)stats.events_scanned);
    dict_set_i64(stats_dict, "events_matched", (long long)stats.events_matched);
    Py_XSETREF(self->stats, stats_dict);

    return results_to_dict(host);
}

PyObject* ph_get_stats(PluginHostObject* self, void*) {
    PyObject* s = self->stats ? self->stats : Py_None;
    Py_INCREF(s);
    return s;
}

PyGetSetDef ph_getset[] = {
    {"stats", (getter)ph_get_stats, nullptr,
     "dict of the last run's scan counters {events_scanned, events_matched}, "
     "or None before the first run().",
     nullptr},
    {nullptr, nullptr, nullptr, nullptr, nullptr}};

PyMethodDef ph_methods[] = {
    {"load", DFTU_PYCFUNCTION(ph_load), METH_VARARGS | METH_KEYWORDS,
     "load(path, config=None): dlopen a compiled plugin .so; config is a JSON "
     "object string. Raises on load/symbol/ABI/config failure."},
    {"resolve", DFTU_PYCFUNCTION(ph_resolve), METH_NOARGS,
     "resolve() -> bool: wire plugin capabilities; False on an unmet/reserved "
     "capability (already logged)."},
    {"run", DFTU_PYCFUNCTION(ph_run), METH_VARARGS | METH_KEYWORDS,
     "run(traces, index_dir=None, auto_index=True) -> dict: fold every loaded "
     "plugin over one fused scan. Returns {name: bytes | pyarrow table} of the "
     "emitted results; scan counters land on the .stats attribute."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace

PyTypeObject PluginHostType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "dftracer_utils_ext.PluginHost",
    sizeof(PluginHostObject),
    0,
    (destructor)ph_dealloc,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    Py_TPFLAGS_DEFAULT,
    "Load and run compiled DFTracer plugins over trace files.",
    0,
    0,
    0,
    0,
    0,
    0,
    ph_methods,
    0,
    ph_getset,
    0,
    0,
    0,
    0,
    0,
    (initproc)ph_init,
    0,
    ph_new,
};

int dftracer::utils::python::init_plugin_host(PyObject* m) {
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    if (PyModule_AddFunctions(m, unnest_methods) < 0) return -1;
    if (PyModule_AddFunctions(m, arrow_ops_methods) < 0) return -1;
#endif
    return register_type(m, &PluginHostType, "PluginHost");
}
