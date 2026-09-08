#define PY_SSIZE_T_CLEAN
#include <dftracer/utils/core/common/error.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/plugins/config.h>
#include <dftracer/utils/plugins/plugins.h>
#include <dftracer/utils/plugins/result_registry.h>
#include <dftracer/utils/python/plugin_host.h>
#include <dftracer/utils/python/py_dict_helpers.h>
#include <dftracer/utils/python/py_errors.h>
#include <dftracer/utils/python/py_method.h>
#include <dftracer/utils/python/py_runtime_mixin.h>
#include <dftracer/utils/python/py_seq_helpers.h>
#include <dftracer/utils/python/py_str_helpers.h>
#include <dftracer/utils/python/py_type_helpers.h>
#include <dftracer/utils/python/runtime.h>
#include <dftracer/utils/trace/indexing/resolve_and_build.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/utilities/filesystem/pattern_directory_scanner_utility.h>
// Unconditional: the native-frame result path (OwnedDataFrame) needs the
// DataFrame wrapper regardless of the Arrow build option.
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/internal/lazyframe_handle.h>
#include <dftracer/utils/dataframe/lazyframe.h>
#include <dftracer/utils/python/dataframe.h>
#include <dftracer/utils/python/lazyframe.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/python/arrow_helpers.h>
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
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace {

using dftracer::utils::CoroScope;
using dftracer::utils::Runtime;
using dftracer::utils::coro::CoroTask;
using dftracer::utils::plugins::ConfigTree;
using dftracer::utils::plugins::NamedResult;
using dftracer::utils::plugins::NamedResultRegistry;
using dftracer::utils::plugins::OwnedArrow;
using dftracer::utils::plugins::OwnedDataFrame;
using dftracer::utils::plugins::OwnedLazyFrame;
using dftracer::utils::plugins::PluginRun;
using dftracer::utils::plugins::Plugins;
using dftracer::utils::python::parse_seq;
using dftracer::utils::python::parse_string_seq;
namespace indexing = dftracer::utils::trace::indexing;
namespace internal = dftracer::utils::trace::internal;
namespace views = dftracer::utils::trace::views;
namespace filesystem = dftracer::utils::utilities::filesystem;

}  // namespace

namespace dftracer::utils::python {

// A host object's mutable side: the built set (immutable once constructed) and
// the named results of the most recent run or session.
struct PluginHostState {
    std::optional<Plugins> set;
    NamedResultRegistry results;
    // wire id -> attribute name, so a caller indexes results[...] with the
    // plugin's declared name rather than its package-qualified wire identity.
    std::unordered_map<std::string, std::string> result_names;
};

}  // namespace dftracer::utils::python

namespace {

using dftracer::utils::python::PluginHostState;

PluginHostState* state_of(PluginHostObject* self) {
    return static_cast<PluginHostState*>(self->host_ptr);
}

// The set built at construction. NULL with a Python error set if construction
// did not leave a built set (should not happen: __init__ raises on failure).
const Plugins* built_set(PluginHostObject* self) {
    PluginHostState* st = state_of(self);
    if (!st->set) {
        PyErr_SetString(PyExc_RuntimeError, "Plugins set was not built");
        return nullptr;
    }
    return &*st->set;
}

// Expand any directory in `inputs` to its .pfw/.pfw.gz files, index (unless
// disabled), then drive every loaded plugin as a fold over one fused scan.
CoroTask<void> run_host_scan(CoroScope& scope, std::vector<std::string> inputs,
                             std::string index_dir, const Plugins* plugins,
                             bool auto_index, PluginRun* out) {
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
    auto run = co_await plugins->run(view);
    if (!run)
        throw dftracer::utils::DFTUtilsException(run.error().code,
                                                 run.error().message);
    *out = std::move(*run);
}

bool collect_inputs(PyObject* traces, std::vector<std::string>& out) {
    if (PyUnicode_Check(traces)) {
        const char* s = PyUnicode_AsUTF8(traces);
        if (!s) return false;
        out.emplace_back(s);
        return true;
    }
    return parse_string_seq(traces, "traces must be a str or list[str]", out);
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW

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
    return parse_seq<std::uint32_t>(
        names, "expected a sequence of names", out,
        [&](PyObject* item, std::vector<std::uint32_t>& o) {
            const char* nm = as_utf8(item);
            std::uint32_t idx = 0;
            if (!nm || !resolve_col(s, nm, &idx)) return false;
            o.push_back(idx);
            return true;
        });
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
    if (auto* frame = std::get_if<OwnedDataFrame>(&result)) {
        // Adopt the handle's columns into a native DataFrame (shared buffers,
        // zero-copy) and hand it back as a _DataFrame; the OwnedDataFrame frees
        // the emptied handle.
        namespace df_ns = dftracer::utils::dataframe;
        dftu_dataframe* h = frame->handle;
        df_ns::DataFrame df;
        std::int32_t n = dftu_dataframe_num_columns(h);
        df.names.reserve(static_cast<std::size_t>(n));
        df.columns.reserve(static_cast<std::size_t>(n));
        for (std::int32_t i = 0; i < n; ++i) {
            const char* nm = dftu_dataframe_column_name(h, i);
            df.names.emplace_back(nm ? nm : "");
            df.columns.emplace_back(
                df_ns::Series{dftu_dataframe_column(h, nm)});
        }
        return dftracer::utils::python::wrap_dataframe(std::move(df));
    }
    if (auto* lazy = std::get_if<OwnedLazyFrame>(&result)) {
        // Move the plan out into the native Python LazyFrame wrapper; the
        // OwnedLazyFrame frees the emptied handle. The plan must be
        // self-contained (see the emit_lazyframe contract).
        namespace df_ns = dftracer::utils::dataframe;
        df_ns::LazyFrame lf =
            std::move(df_ns::lazyframe_handle_unwrap(lazy->handle));
        return dftracer::utils::python::wrap_lazyframe(std::move(lf));
    }
#ifdef DFTRACER_UTILS_ENABLE_ARROW
    namespace arr = dftracer::utils::utilities::common::arrow;
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

PyObject* results_to_dict(PluginHostState* st) {
    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    for (auto& [wire_name, result] : st->results.results()) {
        PyObject* val = result_to_py(result);
        if (!val) {
            Py_DECREF(d);
            return nullptr;
        }
        auto it = st->result_names.find(wire_name);
        const std::string& key =
            it != st->result_names.end() ? it->second : wire_name;
        if (PyDict_SetItemString(d, key.c_str(), val) < 0) {
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
    self->host_ptr = new PluginHostState();
    return (PyObject*)self;
}

void ph_dealloc(PluginHostObject* self) {
    delete state_of(self);
    Py_XDECREF(self->runtime_obj);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

bool parse_result_names(PyObject* obj,
                        std::unordered_map<std::string, std::string>& out) {
    if (!obj || obj == Py_None) return true;
    if (!PyDict_Check(obj)) {
        PyErr_SetString(PyExc_TypeError, "result_names must be a dict or None");
        return false;
    }
    PyObject *key = nullptr, *value = nullptr;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
        const char* k = as_utf8(key);
        const char* v = as_utf8(value);
        if (!k || !v) {
            PyErr_SetString(PyExc_TypeError,
                            "result_names keys and values must be str");
            return false;
        }
        out.emplace(k, v);
    }
    return true;
}

// __init__(specs, result_names=None, runtime=None): specs is a sequence of
// (path, config_json_or_None) pairs; config_json is a JSON object string (the
// Python wrapper serializes a dict). Builds the set now - dlopen, the ABI
// gate, and capability resolution all run here, so a load/symbol/ABI/
// capability failure raises ImportError from the constructor.
int ph_init(PluginHostObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"specs", "result_names", "runtime", nullptr};
    PyObject* specs = nullptr;
    PyObject* result_names_obj = nullptr;
    PyObject* runtime_arg = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO",
                                     const_cast<char**>(kwlist), &specs,
                                     &result_names_obj, &runtime_arg))
        return -1;
    if (bind_runtime_arg(self, runtime_arg) != 0) return -1;

    PluginHostState* st = state_of(self);
    if (!parse_result_names(result_names_obj, st->result_names)) return -1;

    Plugins::Builder builder;
    PyObject* iter = PyObject_GetIter(specs);
    if (!iter) return -1;
    PyObject* item;
    while ((item = PyIter_Next(iter)) != nullptr) {
        const char* path = nullptr;
        const char* config_json = nullptr;
        bool ok = PyArg_ParseTuple(item, "s|z", &path, &config_json) != 0;
        if (ok) {
            try {
                if (config_json)
                    builder.add(path,
                                ConfigTree::from_json_string(config_json));
                else
                    builder.add(path);
            } catch (const std::exception& e) {
                PyErr_SetString(dftracer::utils::python::g_dft_value_error
                                    ? dftracer::utils::python::g_dft_value_error
                                    : PyExc_ValueError,
                                e.what());
                ok = false;
            }
        }
        Py_DECREF(item);
        if (!ok) {
            Py_DECREF(iter);
            return -1;
        }
    }
    Py_DECREF(iter);
    if (PyErr_Occurred()) return -1;

    auto built = builder.build();
    if (!built) {
        PyErr_SetString(PyExc_ImportError, built.error().message.c_str());
        return -1;
    }
    st->set.emplace(std::move(*built));
    return 0;
}

// run(traces, index_dir=None, auto_index=True) -> (results, stats). Drives the
// fused scan with the GIL released.
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
    const Plugins* plugins = built_set(self);
    if (!plugins) return nullptr;
    std::string index_dir_s = index_dir ? index_dir : "";
    PluginRun run;
    if (!run_blocking([&] {
            rt->submit(
                  dftracer::utils::run_coro_scope(
                      rt->executor(), run_host_scan, std::move(inputs),
                      std::move(index_dir_s), plugins, auto_index != 0, &run),
                  "plugin-host-run")
                .get();
        }))
        return nullptr;

    PyObject* stats_dict = PyDict_New();
    if (!stats_dict) return nullptr;
    dict_set_i64(stats_dict, "events_scanned",
                 (long long)run.stats.events_scanned);
    dict_set_i64(stats_dict, "events_matched",
                 (long long)run.stats.events_matched);

    PluginHostState* st = state_of(self);
    st->results = std::move(run.results);
    PyObject* results_dict = results_to_dict(st);
    if (!results_dict) {
        Py_DECREF(stats_dict);
        return nullptr;
    }
    PyObject* out = PyTuple_Pack(2, results_dict, stats_dict);
    Py_DECREF(results_dict);
    Py_DECREF(stats_dict);
    return out;
}

PyMethodDef ph_methods[] = {
    {"run", DFTU_PYCFUNCTION(ph_run), METH_VARARGS | METH_KEYWORDS,
     "run(traces, index_dir=None, auto_index=True) -> (dict, dict): fold every "
     "plugin over one fused scan. Returns ({name: bytes | pyarrow table}, "
     "{events_scanned, events_matched})."},
    {nullptr, nullptr, 0, nullptr}};

}  // namespace

PyTypeObject PluginHostType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "dftracer_utils_ext.Plugins",
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
    0,
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
    return register_type(m, &PluginHostType, "Plugins");
}

PyObject* dftracer::utils::python::plugin_host_results_dict(PyObject* host) {
    if (!PyObject_TypeCheck(host, &PluginHostType)) {
        PyErr_SetString(PyExc_TypeError, "expected a Plugins instance");
        return nullptr;
    }
    return results_to_dict(state_of((PluginHostObject*)host));
}

const dftracer::utils::plugins::Plugins*
dftracer::utils::python::plugin_host_plugins(PyObject* host) {
    if (!PyObject_TypeCheck(host, &PluginHostType)) {
        PyErr_SetString(PyExc_TypeError, "expected a Plugins instance");
        return nullptr;
    }
    return built_set((PluginHostObject*)host);
}

dftracer::utils::plugins::NamedResultRegistry*
dftracer::utils::python::plugin_host_results(PyObject* host) {
    if (!PyObject_TypeCheck(host, &PluginHostType)) {
        PyErr_SetString(PyExc_TypeError, "expected a Plugins instance");
        return nullptr;
    }
    return &state_of((PluginHostObject*)host)->results;
}
