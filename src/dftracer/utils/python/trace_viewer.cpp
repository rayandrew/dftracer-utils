#define PY_SSIZE_T_CLEAN
#include <dftracer/utils/core/common/config.h>
#include <dftracer/utils/core/common/filesystem.h>
#include <dftracer/utils/core/runtime.h>
#include <dftracer/utils/dataframe/arrow_bridge.h>
#include <dftracer/utils/dataframe/batch_ops.h>
#include <dftracer/utils/dataframe/dataframe.h>
#include <dftracer/utils/dataframe/internal/column_read.h>
#include <dftracer/utils/plugins/host.h>
#include <dftracer/utils/python/dataframe.h>
#include <dftracer/utils/python/lazyframe.h>
#include <dftracer/utils/python/plugin_host.h>
#include <dftracer/utils/python/py_dict_helpers.h>
#include <dftracer/utils/python/py_errors.h>
#include <dftracer/utils/python/py_list_helpers.h>
#include <dftracer/utils/python/py_method.h>
#include <dftracer/utils/python/py_runtime_mixin.h>
#include <dftracer/utils/python/py_str_helpers.h>
#include <dftracer/utils/python/py_type_helpers.h>
#include <dftracer/utils/python/runtime.h>
#include <dftracer/utils/python/series.h>
#include <dftracer/utils/python/trace_viewer.h>
#include <dftracer/utils/query/query.h>
#include <dftracer/utils/trace/comparator/compare_view.h>
#include <dftracer/utils/trace/internal/utils.h>
#include <dftracer/utils/trace/time_metric.h>
#include <dftracer/utils/trace/trace_config.h>
#include <dftracer/utils/trace/views/result_batch.h>
#include <dftracer/utils/trace/views/result_join.h>
#include <dftracer/utils/trace/views/view.h>
#include <dftracer/utils/utilities/common/arrow/arrow_export.h>
#include <dftracer/utils/utilities/filesystem/pattern_directory_scanner_utility.h>

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#include <dftracer/utils/core/common/memory_budget.h>
#include <dftracer/utils/core/common/string_arena.h>
#include <dftracer/utils/core/tasks/coro_scope.h>
#include <dftracer/utils/python/arrow_helpers.h>
#include <dftracer/utils/python/batch_byte_size.h>
#include <dftracer/utils/python/streaming_iterator.h>
#endif

#include <dftracer/utils/utilities/fileio/compress/libdeflate_gzip.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dataframe = dftracer::utils::dataframe;

namespace {

using dftracer::utils::Runtime;
using dftracer::utils::dataframe::DataFrame;
using dftracer::utils::query::Query;
using dftracer::utils::trace::views::AggOp;
using dftracer::utils::trace::views::AggregatedView;
using dftracer::utils::trace::views::AggSpec;
using dftracer::utils::trace::views::Deferred;
using dftracer::utils::trace::views::ExportStats;
using dftracer::utils::trace::views::GroupKey;
using dftracer::utils::trace::views::Phase;
using dftracer::utils::trace::views::TypedResult;
using dftracer::utils::trace::views::View;
using dftracer::utils::trace::views::ViewFile;
using dftracer::utils::trace::views::ViewSession;
using dftracer::utils::utilities::filesystem::FileEntry;
using dftracer::utils::utilities::filesystem::PatternDirectoryScannerUtility;
using dftracer::utils::utilities::filesystem::
    PatternDirectoryScannerUtilityInput;
namespace dftint = dftracer::utils::trace::internal;

// Parallel scan of a directory for trace files (recursive, .pfw.gz only).
// Returns false with a Python error set on failure. Sorted for determinism.
bool scan_dir_trace_files(const std::string& dir,
                          std::vector<std::string>& out) {
    auto* rt = dftracer::utils::python::get_default_runtime();
    PatternDirectoryScannerUtilityInput input(dir, {".pfw.gz"},
                                              /*recursive=*/true,
                                              /*populate_size=*/false);
    std::vector<FileEntry> entries;
    if (!run_blocking([&] {
            rt->submit(dftracer::utils::run_coro_scope(
                           rt->executor(),
                           [](dftracer::utils::CoroScope& scope,
                              PatternDirectoryScannerUtilityInput in,
                              std::vector<FileEntry>* o)
                               -> dftracer::utils::coro::CoroTask<void> {
                               PatternDirectoryScannerUtility scanner;
                               *o = co_await scanner(scope, in);
                           },
                           std::move(input), &entries),
                       "tv-scan-dir")
                .get();
        })) {
        return false;
    }
    out.clear();
    out.reserve(entries.size());
    for (auto& e : entries) out.push_back(e.path.string());
    std::sort(out.begin(), out.end());
    return true;
}

// The accumulated builder ops, lowered onto a fresh View by each terminal.
struct ViewerPlan {
    std::vector<std::string> filters;  // DSL predicates, AND-combined
    int phase = -1;                    // -1 = View default; else Phase value
    std::vector<GroupKey> group_by;
    std::vector<AggSpec> agg;
    std::uint64_t time_bucket_us = 0;
    std::uint64_t bucket_origin_us = 0;
    bool bucket_origin_min = false;
    std::uint64_t occ_cell_us = 0;
    std::optional<std::pair<double, double>> time_range;
    std::vector<std::string> select;
    std::uint64_t memory_budget = 0;
    bool auto_spill = false;
    bool auto_numeric = false;
    std::vector<AggSpec> numeric_arg_aggs;  // reductions per discovered arg
    double time_scale = 1.0;  // ts/dur normalization factor (1.0 = none)
    std::uint64_t limit = 0;
    std::uint64_t offset = 0;
    std::string rollup_root;  // "" = derive the aggregation-cache dir
    std::string views_root;   // "" = derive <parent-of-index>/.dftindex-views
    // Post-aggregation ordering applied to the collect() result DataFrame (same
    // dataframe kernels as DataFrame.sort_by/topk), so the View and dataframe
    // surfaces match.
    std::string sort_col;
    bool sort_desc = false;
    std::string topk_col;
    std::int64_t topk_k = -1;  // -1 = no topk
    bool topk_largest = true;
};

ViewerPlan* plan_of(TraceViewerObject* self) {
    return static_cast<ViewerPlan*>(self->plan_ptr);
}

// name | cat | pid | tid | fhash | hhash | io_cat | acc_pat | rank | arg:<key>
// "fn(key)" or "fn(key, 'a', 'b')" -> transform + inner key text; returns
// false when `t` is not a call, leaving `t` to parse as a plain key.
bool split_transform(const std::string& t, GroupKey::Transform& tf,
                     std::vector<std::string>& targs, std::string& inner) {
    const auto lp = t.find('(');
    if (lp == std::string::npos || t.back() != ')') return false;
    const std::string fn = t.substr(0, lp);
    if (fn == "dirname") {
        tf = GroupKey::Transform::Dirname;
    } else if (fn == "basename") {
        tf = GroupKey::Transform::Basename;
    } else if (fn == "lower") {
        tf = GroupKey::Transform::Lower;
    } else if (fn == "bucket") {
        tf = GroupKey::Transform::Bucket;
    } else {
        return false;
    }
    std::string body = t.substr(lp + 1, t.size() - lp - 2);
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : body) {
        if (ch == ',') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur += ch;
        }
    }
    parts.push_back(cur);
    auto trim = [](std::string x) {
        const auto b = x.find_first_not_of(" \t'\"");
        const auto e = x.find_last_not_of(" \t'\"");
        return b == std::string::npos ? std::string() : x.substr(b, e - b + 1);
    };
    if (parts.empty()) return false;
    inner = trim(parts[0]);
    for (std::size_t i = 1; i < parts.size(); ++i)
        targs.push_back(trim(parts[i]));
    return !inner.empty();
}

bool parse_group_key(const char* s, GroupKey& out) {
    std::string t(s);
    GroupKey::Transform tf = GroupKey::Transform::None;
    std::vector<std::string> targs;
    std::string inner;
    if (split_transform(t, tf, targs, inner)) t = inner;
    if (t == "name") {
        out = GroupKey::name();
    } else if (t == "cat") {
        out = GroupKey::cat();
    } else if (t == "pid") {
        out = GroupKey::pid();
    } else if (t == "tid") {
        out = GroupKey::tid();
    } else if (t == "fhash") {
        out = GroupKey::fhash();
    } else if (t == "hhash") {
        out = GroupKey::hhash();
    } else if (t == "io_cat") {
        out = GroupKey::io_cat();
    } else if (t == "acc_pat") {
        out = GroupKey::acc_pat();
    } else if (t == "file_path" || t == "resolved.fpath" || t == "r.fpath") {
        // resolved.fpath/hostname resolve fhash/hhash, so group_by resolves the
        // same aliases as select().
        out = GroupKey::file_path();
    } else if (t == "file_name") {
        out = GroupKey::file_name();
    } else if (t == "host_name" || t == "resolved.hostname" ||
               t == "r.hostname" || t == "resolved.host" || t == "r.host") {
        out = GroupKey::host_name();
    } else if (t == "rank") {
        out = GroupKey::rank();
    } else if (t.rfind("arg:", 0) == 0) {
        out = GroupKey::of_arg(t.substr(4));
    } else {
        // Schemaless fallback: any other name is a field, resolved like a
        // filter field (bare name = top-level then args; dotted/bracketed path
        // = that path from the event root).
        out = GroupKey::field(t);
    }
    out.transform = tf;
    out.transform_args = std::move(targs);
    return true;
}

// "count" | "op:field" | "argmax:field:by"  (op in sum/min/max/mean/var/std)
bool parse_agg_spec(const char* s, AggSpec& out) {
    std::string t(s);
    auto c1 = t.find(':');
    std::string op = t.substr(0, c1);
    std::string field, by;
    if (c1 != std::string::npos) {
        std::string rest = t.substr(c1 + 1);
        auto c2 = rest.find(':');
        field = rest.substr(0, c2);
        if (c2 != std::string::npos) by = rest.substr(c2 + 1);
    }
    if (op == "count")
        out = AggSpec(AggOp::Count, "", "", "");
    else if (op == "sum")
        out = AggSpec(AggOp::Sum, field);
    else if (op == "sumsq")
        out = AggSpec(AggOp::SumSq, field);
    else if (op == "min")
        out = AggSpec(AggOp::Min, field);
    else if (op == "max")
        out = AggSpec(AggOp::Max, field);
    else if (op == "mean")
        out = AggSpec(AggOp::Mean, field);
    else if (op == "var")
        out = AggSpec(AggOp::Var, field);
    else if (op == "std")
        out = AggSpec(AggOp::Std, field);
    else if (op == "skew")
        out = AggSpec(AggOp::Skew, field);
    else if (op == "kurt")
        out = AggSpec(AggOp::Kurt, field);
    else if (op == "hist")
        out = AggSpec(AggOp::Hist, field);
    else if (op == "argmax")
        out = AggSpec(AggOp::ArgMax, field, "", by);
    else if (op == "set_union" || op == "uniq")
        out = AggSpec(AggOp::SetUnion, field);
    else if (op == "busy")
        out = AggSpec(AggOp::Busy, field.empty() ? "dur" : field);
    else if (op == "concurrency")
        out = AggSpec(AggOp::Concurrency, field.empty() ? "dur" : field);
    else if (op == "utilization")
        out = AggSpec(AggOp::Utilization, field.empty() ? "dur" : field);
    else if (op == "active")
        out = AggSpec(AggOp::Active, field.empty() ? "dur" : field);
    else if (op == "pct")
        // pct:field:q  (explicit quantile)
        out = AggSpec(AggOp::Pct, field, "", "",
                      by.empty() ? 0.0 : std::stod(by));
    else if (op.size() >= 2 && op[0] == 'p') {
        // pNN shorthand: p50/p90/p99/p999 -> 0.NN
        double denom = 1.0;
        for (std::size_t i = 1; i < op.size(); ++i) {
            if (op[i] < '0' || op[i] > '9') return false;
            denom *= 10.0;
        }
        out = AggSpec(AggOp::Pct, field, op + "_" + field, "",
                      std::stod(op.substr(1)) / denom);
    } else
        return false;
    return true;
}

// Throws DFTUtilsException on a bad filter DSL.
View build_view_from_data(const std::vector<std::string>& file_paths,
                          const std::string& index_dir, const ViewerPlan& p,
                          bool aggregate = true) {
    std::vector<ViewFile> files;
    files.reserve(file_paths.size());
    for (const auto& fp : file_paths) {
        ViewFile vf;
        vf.file_path = fp;
        vf.index_path = dftint::determine_index_path(vf.file_path, index_dir);
        files.push_back(std::move(vf));
    }

    View v = View::from_files(std::move(files));
    for (const auto& q : p.filters) {
        auto parsed = Query::from_string(q);
        if (!parsed)
            throw dftracer::utils::DFTUtilsException(
                dftracer::utils::ErrorCode::INVALID_ARGUMENT,
                "invalid filter query: " + q);
        v = v.filter(parsed.value());
    }
    if (p.phase >= 0) v = v.phase(static_cast<Phase>(p.phase));
    if (p.time_scale != 1.0) v = v.time_scale(p.time_scale);
    if (p.time_bucket_us) {
        if (p.bucket_origin_min)
            v = v.time_bucket_min(p.time_bucket_us);
        else if (p.bucket_origin_us)
            v = v.time_bucket(p.time_bucket_us, p.bucket_origin_us);
        else
            v = v.time_bucket(p.time_bucket_us);
    }
    if (p.occ_cell_us) v = v.occ_cell(p.occ_cell_us);
    if (p.time_range)
        v = v.time_range(p.time_range->first, p.time_range->second);
    // The aggregation and everything that operates on the aggregated result;
    // CompareView owns this step, so it asks for a base view (aggregate=false).
    if (aggregate) {
        if (!p.group_by.empty()) v = v.group_by(p.group_by);
        if (!p.agg.empty()) v = v.agg(p.agg);
        if (p.auto_numeric)
            v = p.numeric_arg_aggs.empty()
                    ? v.agg_numeric_args()
                    : v.agg_numeric_args(p.numeric_arg_aggs);
        if (!p.select.empty()) v = v.select(p.select);
        if (!p.sort_col.empty()) v = v.sort_by(p.sort_col, p.sort_desc);
        if (!p.topk_col.empty())
            v = v.topk(p.topk_col, p.topk_k, p.topk_largest);
        if (p.limit) v = v.limit(p.limit);
        if (p.offset) v = v.offset(p.offset);
    }
    if (p.memory_budget)
        v = v.memory_budget(p.memory_budget);
    else if (p.auto_spill)
        v = v.auto_spill();
    if (!p.rollup_root.empty()) v = v.rollup_root(p.rollup_root);
    if (!p.views_root.empty()) v = v.views_root(p.views_root);
    return v;
}

std::vector<std::string> extract_files(TraceViewerObject* self) {
    std::vector<std::string> out;
    parse_str_list(self->files, "files", out);  // already a validated list[str]
    return out;
}

std::string extract_index_dir(TraceViewerObject* self) {
    return (self->index_path && self->index_path != Py_None)
               ? as_utf8(self->index_path)
               : "";
}

// Same plan, typed as an AggregatedView so the cache terminals are reachable.
// group_by is idempotent (re-applies the plan's own keys), so this only shifts
// the type, not the plan. The runtime guard rejects a plan with no aggregation.
AggregatedView build_agg_view(const std::vector<std::string>& files,
                              const std::string& index_dir,
                              const ViewerPlan& p) {
    return build_view_from_data(files, index_dir, p).group_by(p.group_by);
}

PyObject* tv_new(PyTypeObject* type, PyObject*, PyObject*) {
    TraceViewerObject* self = (TraceViewerObject*)type->tp_alloc(type, 0);
    if (!self) return nullptr;
    self->files = nullptr;
    self->index_path = nullptr;
    self->runtime_obj = nullptr;
    self->plan_ptr = new ViewerPlan();
    return (PyObject*)self;
}

void tv_dealloc(TraceViewerObject* self) {
    Py_XDECREF(self->files);
    Py_XDECREF(self->index_path);
    Py_XDECREF(self->runtime_obj);
    delete plan_of(self);
    Py_TYPE(self)->tp_free((PyObject*)self);
}

int tv_init(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"files", "index_path", "runtime", nullptr};
    PyObject* files = nullptr;
    PyObject* index_path = nullptr;
    PyObject* runtime_arg = nullptr;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|OO",
                                     const_cast<char**>(kwlist), &files,
                                     &index_path, &runtime_arg))
        return -1;

    // Accept a directory, a single path, or an iterable of paths. A directory
    // is expanded to its .pfw.gz files via the parallel scanner.
    PyObject* list = nullptr;
    if (PyUnicode_Check(files)) {
        const char* path = PyUnicode_AsUTF8(files);
        std::error_code ec;
        if (path && fs::is_directory(path, ec)) {
            std::vector<std::string> scanned;
            if (!scan_dir_trace_files(path, scanned)) return -1;
            list = PyList_New(static_cast<Py_ssize_t>(scanned.size()));
            if (!list) return -1;
            for (std::size_t i = 0; i < scanned.size(); ++i) {
                PyObject* item = PyUnicode_FromString(scanned[i].c_str());
                if (!item) {
                    Py_DECREF(list);
                    return -1;
                }
                PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), item);
            }
        } else {
            list = PyList_New(1);
            if (!list) return -1;
            Py_INCREF(files);
            PyList_SET_ITEM(list, 0, files);
        }
    } else {
        list = PySequence_List(files);
        if (!list) return -1;
    }
    self->files = list;

    if (index_path && index_path != Py_None) {
        Py_INCREF(index_path);
        self->index_path = index_path;
    }
    if (runtime_arg && runtime_arg != Py_None) {
        if (PyObject_TypeCheck(runtime_arg, &RuntimeType)) {
            Py_INCREF(runtime_arg);
            self->runtime_obj = runtime_arg;
        } else {
            PyObject* native = PyObject_GetAttrString(runtime_arg, "_native");
            if (native && PyObject_TypeCheck(native, &RuntimeType)) {
                self->runtime_obj = native;
            } else {
                Py_XDECREF(native);
                PyErr_SetString(PyExc_TypeError,
                                "runtime must be a Runtime instance or None");
                return -1;
            }
        }
    }
    return 0;
}

// Shallow-clone the object with a deep-copied plan; the caller mutates the
// returned plan to implement one builder op (immutable-value semantics).
TraceViewerObject* clone_as(TraceViewerObject* self, PyTypeObject* type) {
    TraceViewerObject* c = (TraceViewerObject*)type->tp_alloc(type, 0);
    if (!c) return nullptr;
    Py_XINCREF(self->files);
    c->files = self->files;
    Py_XINCREF(self->index_path);
    c->index_path = self->index_path;
    Py_XINCREF(self->runtime_obj);
    c->runtime_obj = self->runtime_obj;
    c->plan_ptr = new ViewerPlan(*plan_of(self));
    return c;
}

// Builder ops preserve the object's type (a chained AggregatedTraceViewer stays
// aggregated); group_by/agg/agg_numeric_args promote to AggregatedTraceViewer,
// which alone carries the cache terminals.
TraceViewerObject* clone(TraceViewerObject* self) {
    return clone_as(self, Py_TYPE(self));
}
TraceViewerObject* clone_agg(TraceViewerObject* self) {
    return clone_as(self, &AggregatedTraceViewerType);
}

PyObject* tv_filter(TraceViewerObject* self, PyObject* arg) {
    // Accept a DSL string or a query.Expr (or any object whose str() is a DSL
    // predicate); str(str) is the string itself, so this stays zero-cost for
    // the common string case.
    PyObject* s = PyObject_Str(arg);
    if (!s) return nullptr;
    const char* dsl = PyUnicode_AsUTF8(s);
    if (!dsl) {
        Py_DECREF(s);
        return nullptr;
    }
    TraceViewerObject* c = clone(self);
    if (!c) {
        Py_DECREF(s);
        return nullptr;
    }
    plan_of(c)->filters.emplace_back(dsl);
    Py_DECREF(s);
    return (PyObject*)c;
}

PyObject* tv_phase(TraceViewerObject* self, PyObject* arg) {
    const char* s = as_utf8(arg);
    if (!s) return nullptr;
    int ph;
    std::string t(s);
    if (t == "events")
        ph = (int)Phase::Events;
    else if (t == "counters")
        ph = (int)Phase::Counters;
    else if (t == "aggregated")
        ph = (int)Phase::Aggregated;
    else if (t == "metadata")
        ph = (int)Phase::Metadata;
    else if (t == "any")
        ph = (int)Phase::Any;
    else {
        PyErr_SetString(PyExc_ValueError,
                        "phase must be 'events', 'counters', 'aggregated', "
                        "'metadata', or 'any'");
        return nullptr;
    }
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->phase = ph;
    return (PyObject*)c;
}

PyObject* tv_group_by(TraceViewerObject* self, PyObject* args) {
    std::vector<GroupKey> keys;
    const Py_ssize_t n = PyTuple_Size(args);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = as_utf8(PyTuple_GetItem(args, i));
        if (!s) return nullptr;
        GroupKey k;
        if (!parse_group_key(s, k)) {
            PyErr_Format(PyExc_ValueError, "unknown group key: %s", s);
            return nullptr;
        }
        keys.push_back(k);
    }
    TraceViewerObject* c = clone_agg(self);
    if (!c) return nullptr;
    plan_of(c)->group_by = std::move(keys);
    return (PyObject*)c;
}

PyObject* tv_agg(TraceViewerObject* self, PyObject* args) {
    std::vector<AggSpec> specs;
    const Py_ssize_t n = PyTuple_Size(args);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = as_utf8(PyTuple_GetItem(args, i));
        if (!s) return nullptr;
        AggSpec spec;
        if (!parse_agg_spec(s, spec)) {
            PyErr_Format(PyExc_ValueError, "unknown agg spec: %s", s);
            return nullptr;
        }
        specs.push_back(spec);
    }
    TraceViewerObject* c = clone_agg(self);
    if (!c) return nullptr;
    plan_of(c)->agg = std::move(specs);
    return (PyObject*)c;
}

PyObject* tv_time_bucket(TraceViewerObject* self, PyObject* args,
                         PyObject* kwds) {
    long long us = 0;
    PyObject* normalize_to = nullptr;  // None, an int origin, or "min"
    static const char* kwlist[] = {"interval_us", "normalize_to", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "L|O", const_cast<char**>(kwlist), &us, &normalize_to))
        return nullptr;
    if (us < 0) {
        PyErr_SetString(PyExc_ValueError, "interval_us must be >= 0");
        return nullptr;
    }
    std::uint64_t origin = 0;
    bool origin_min = false;
    if (normalize_to && normalize_to != Py_None) {
        if (PyUnicode_Check(normalize_to)) {
            const char* s = PyUnicode_AsUTF8(normalize_to);
            if (!s) return nullptr;
            if (std::strcmp(s, "min") != 0) {
                PyErr_SetString(PyExc_ValueError,
                                "normalize_to must be an int origin or 'min'");
                return nullptr;
            }
            origin_min = true;
        } else {
            const long long o = PyLong_AsLongLong(normalize_to);
            if (o < 0 && PyErr_Occurred()) return nullptr;
            if (o < 0) {
                PyErr_SetString(PyExc_ValueError, "normalize_to must be >= 0");
                return nullptr;
            }
            origin = static_cast<std::uint64_t>(o);
        }
    }
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->time_bucket_us = static_cast<std::uint64_t>(us);
    plan_of(c)->bucket_origin_us = origin;
    plan_of(c)->bucket_origin_min = origin_min;
    return (PyObject*)c;
}

PyObject* tv_occ_cell(TraceViewerObject* self, PyObject* arg) {
    long long us = PyLong_AsLongLong(arg);
    if (us < 0 && PyErr_Occurred()) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->occ_cell_us = (std::uint64_t)us;
    return (PyObject*)c;
}

PyObject* tv_time_unit(TraceViewerObject* self, PyObject* arg) {
    const char* s = as_utf8(arg);
    if (!s) return nullptr;
    std::string t(s);
    dftracer::utils::trace::TimeMetric target;
    if (t == "ns")
        target = dftracer::utils::trace::TimeMetric::NS;
    else if (t == "us")
        target = dftracer::utils::trace::TimeMetric::US;
    else if (t == "ms")
        target = dftracer::utils::trace::TimeMetric::MS;
    else if (t == "sec" || t == "s")
        target = dftracer::utils::trace::TimeMetric::SEC;
    else {
        PyErr_SetString(PyExc_ValueError, "time_unit must be ns/us/ms/sec/s");
        return nullptr;
    }
    // Source unit read once from the first file (assumes the run is uniform).
    auto files = extract_files(self);
    dftracer::utils::trace::TimeMetric source =
        files.empty() ? dftracer::utils::trace::TimeMetric::US
                      : dftracer::utils::trace::read_time_metric(files[0]);
    double scale =
        static_cast<double>(
            dftracer::utils::trace::time_metric_ns_per_unit(source)) /
        static_cast<double>(
            dftracer::utils::trace::time_metric_ns_per_unit(target));
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->time_scale = scale;
    return (PyObject*)c;
}

// Raw scale primitive matching C++ View::time_scale: multiply ts/dur/te by
// `ns_ratio` (source_ns / target_ns; 1.0 = none). time_unit() is the higher-
// level helper that derives this ratio from unit names.
PyObject* tv_time_scale(TraceViewerObject* self, PyObject* arg) {
    const double ratio = PyFloat_AsDouble(arg);
    if (ratio == -1.0 && PyErr_Occurred()) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->time_scale = ratio;
    return (PyObject*)c;
}

PyObject* tv_time_range(TraceViewerObject* self, PyObject* args) {
    double begin, end;
    if (!PyArg_ParseTuple(args, "dd", &begin, &end)) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->time_range = std::make_pair(begin, end);
    return (PyObject*)c;
}

PyObject* tv_select(TraceViewerObject* self, PyObject* args) {
    std::vector<std::string> cols;
    const Py_ssize_t n = PyTuple_Size(args);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = as_utf8(PyTuple_GetItem(args, i));
        if (!s) return nullptr;
        cols.emplace_back(s);
    }
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->select = std::move(cols);
    return (PyObject*)c;
}

PyObject* tv_memory_budget(TraceViewerObject* self, PyObject* arg) {
    long long b = PyLong_AsLongLong(arg);
    if (b < 0 && PyErr_Occurred()) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->memory_budget = (std::uint64_t)b;
    return (PyObject*)c;
}

PyObject* tv_rollup_root(TraceViewerObject* self, PyObject* arg) {
    const char* dir = as_utf8(arg);
    if (!dir) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->rollup_root = dir;
    return (PyObject*)c;
}

PyObject* tv_views_root(TraceViewerObject* self, PyObject* arg) {
    const char* dir = as_utf8(arg);
    if (!dir) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->views_root = dir;
    return (PyObject*)c;
}

PyObject* tv_auto_spill(TraceViewerObject* self, PyObject*) {
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->auto_spill = true;
    return (PyObject*)c;
}

PyObject* tv_auto_numeric_args(TraceViewerObject* self, PyObject* args) {
    // Optional op names (e.g. "sum", "max", "mean") apply that reduction to
    // every discovered numeric arg; no args keeps the legacy bare-mean column.
    std::vector<AggSpec> reductions;
    const Py_ssize_t n = PyTuple_Size(args);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = as_utf8(PyTuple_GetItem(args, i));
        if (!s) return nullptr;
        AggSpec spec;
        if (!parse_agg_spec(s, spec)) {
            PyErr_Format(PyExc_ValueError, "unknown agg spec: %s", s);
            return nullptr;
        }
        reductions.push_back(spec);
    }
    TraceViewerObject* c = clone_agg(self);
    if (!c) return nullptr;
    plan_of(c)->auto_numeric = true;
    plan_of(c)->numeric_arg_aggs = std::move(reductions);
    return (PyObject*)c;
}

PyObject* tv_limit(TraceViewerObject* self, PyObject* arg) {
    long long v = PyLong_AsLongLong(arg);
    if (v < 0 && PyErr_Occurred()) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->limit = (std::uint64_t)v;
    return (PyObject*)c;
}

// sort_by(name, descending=False): order the collect() result by a column.
PyObject* tv_sort_by(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
    const char* name = nullptr;
    int descending = 0;
    static const char* kwlist[] = {"name", "descending", nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "s|p", const_cast<char**>(kwlist), &name, &descending))
        return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->sort_col = name;
    plan_of(c)->sort_desc = descending != 0;
    return (PyObject*)c;
}

// topk(name, k, largest=True): keep the k best rows of the collect() result.
PyObject* tv_topk(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
    const char* name = nullptr;
    Py_ssize_t k = 0;
    int largest = 1;
    static const char* kwlist[] = {"name", "k", "largest", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "sn|p",
                                     const_cast<char**>(kwlist), &name, &k,
                                     &largest))
        return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->topk_col = name;
    plan_of(c)->topk_k = static_cast<std::int64_t>(k);
    plan_of(c)->topk_largest = largest != 0;
    return (PyObject*)c;
}

PyObject* tv_offset(TraceViewerObject* self, PyObject* arg) {
    long long v = PyLong_AsLongLong(arg);
    if (v < 0 && PyErr_Occurred()) return nullptr;
    TraceViewerObject* c = clone(self);
    if (!c) return nullptr;
    plan_of(c)->offset = (std::uint64_t)v;
    return (PyObject*)c;
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
#endif

// collect(cache=False) -> a LazyFrame over this viewer's plan; nothing runs
// until the caller materializes it (LazyFrame.collect() -> DataFrame, or
// .to_arrow()/.to_pandas()/.to_polars()). cache uses the materialized-view
// cache (reconstruct on hit, else scan + persist + return); it requires an
// aggregation (group_by/agg), enforced by the runtime guard.
PyObject* tv_collect(TraceViewerObject* self, PyObject*) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "collect() requires the arrow-enabled build");
    return nullptr;
#else
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    View v = build_view_from_data(files, index_dir, plan);
    return dftracer::utils::python::wrap_lazyframe(v.collect());
#endif
}

static bool parse_partition(PyObject* seq_obj, std::vector<std::string>& out) {
    PyObject* seq = PySequence_Fast(seq_obj, "partition must be a sequence");
    if (!seq) return false;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    for (Py_ssize_t i = 0; i < n; ++i) {
        const char* s = PyUnicode_AsUTF8(PySequence_Fast_GET_ITEM(seq, i));
        if (!s) {
            Py_DECREF(seq);
            return false;
        }
        out.emplace_back(s);
    }
    Py_DECREF(seq);
    return true;
}

// Decode a session containment branch's sink "partition_csv\x1f ts\x1f dur\x1f
// name" back into its fields (defaults to the dftracer schema on a short cfg).
static void parse_containment_cfg(const std::string& sink,
                                  std::vector<std::string>& partition,
                                  std::string& ts, std::string& dur,
                                  std::string& name) {
    std::vector<std::string> parts;
    std::string cur;
    for (char ch : sink) {
        if (ch == '\x1f') {
            parts.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(ch);
        }
    }
    parts.push_back(cur);
    if (parts.size() == 4 && !parts[0].empty()) {
        std::string p;
        for (char ch : parts[0]) {
            if (ch == ',') {
                partition.push_back(p);
                p.clear();
            } else {
                p.push_back(ch);
            }
        }
        partition.push_back(p);
    }
    ts = parts.size() == 4 ? parts[1] : "ts";
    dur = parts.size() == 4 ? parts[2] : "dur";
    name = parts.size() == 4 ? parts[3] : "name";
}

// call_tree(partition, ts_col, dur_col) -> scan the view, then the columnar
// containment fold; returns the events DataFrame plus level/parent_id.
PyObject* tv_call_tree(TraceViewerObject* self, PyObject* args) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "call_tree() requires the arrow-enabled build");
    return nullptr;
#else
    PyObject* part = nullptr;
    const char* ts = "ts";
    const char* dur = "dur";
    const char* name = "name";
    if (!PyArg_ParseTuple(args, "O|sss", &part, &ts, &dur, &name))
        return nullptr;
    std::vector<std::string> partition;
    if (!parse_partition(part, partition)) return nullptr;
    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    DataFrame table;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            table = rt->submit(v.call_tree(partition, ts, dur, name)).get();
        }))
        return nullptr;
    return dftracer::utils::python::wrap_dataframe(std::move(table));
#endif
}

// flamegraph(partition) -> scan the view, then fold by name path; returns the
// node DataFrame.
PyObject* tv_flamegraph(TraceViewerObject* self, PyObject* args) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "flamegraph() requires the arrow-enabled build");
    return nullptr;
#else
    PyObject* part = nullptr;
    const char* ts = "ts";
    const char* dur = "dur";
    const char* name = "name";
    PyObject* group_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O|sssO", &part, &ts, &dur, &name, &group_obj))
        return nullptr;
    std::vector<std::string> partition, group;
    if (!parse_partition(part, partition)) return nullptr;
    if (group_obj && group_obj != Py_None && !parse_partition(group_obj, group))
        return nullptr;
    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    DataFrame table;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            table =
                rt->submit(v.flamegraph(partition, ts, dur, name, group)).get();
        }))
        return nullptr;
    return dftracer::utils::python::wrap_dataframe(std::move(table));
#endif
}

// columns() -> list[str]: the distinct columns discoverable from the index
// (base axis + harvested scalar leaves + resolved.* aliases). No trace scan.
PyObject* tv_columns(TraceViewerObject* self, PyObject*) {
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::vector<std::string> cols;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            cols = v.columns();
        }))
        return nullptr;
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(cols.size()));
    if (!list) return nullptr;
    for (std::size_t i = 0; i < cols.size(); ++i) {
        PyObject* s = PyUnicode_FromString(cols[i].c_str());
        if (!s) {
            Py_DECREF(list);
            return nullptr;
        }
        PyList_SET_ITEM(list, static_cast<Py_ssize_t>(i), s);
    }
    return list;
}

// schema() -> dict[str, str]: each column mapped to its type ("int64" /
// "float64" / "string"). Same discovery as columns(); no trace scan.
PyObject* tv_schema(TraceViewerObject* self, PyObject*) {
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::vector<View::ColumnInfo> sc;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            sc = v.schema();
        }))
        return nullptr;
    PyObject* dict = PyDict_New();
    if (!dict) return nullptr;
    for (const auto& c : sc) {
        PyObject* val = PyUnicode_FromString(c.type.c_str());
        if (!val || PyDict_SetItemString(dict, c.name.c_str(), val) != 0) {
            Py_XDECREF(val);
            Py_DECREF(dict);
            return nullptr;
        }
        Py_DECREF(val);
    }
    return dict;
}

// time_metric() -> str: the trace's native time unit ("us"/"ns"/"ms"/"sec")
// from the first file's CM record. Head-read only, no scan; "us" with no files.
PyObject* tv_time_metric(TraceViewerObject* self, PyObject*) {
    auto files = extract_files(self);
    const dftracer::utils::trace::TimeMetric m =
        files.empty() ? dftracer::utils::trace::TimeMetric::US
                      : dftracer::utils::trace::read_time_metric(files[0]);
    std::string s(dftracer::utils::trace::time_metric_to_string(m));
    for (char& c : s) c = static_cast<char>(std::tolower(c));
    return PyUnicode_FromString(s.c_str());
}

// containment(partition, ts, dur, name) -> (call_tree_df, flamegraph_df) from
// ONE scan and one buffered fold.
PyObject* tv_containment(TraceViewerObject* self, PyObject* args) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "containment() requires the arrow-enabled build");
    return nullptr;
#else
    PyObject* part = nullptr;
    const char* ts = "ts";
    const char* dur = "dur";
    const char* name = "name";
    PyObject* group_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O|sssO", &part, &ts, &dur, &name, &group_obj))
        return nullptr;
    std::vector<std::string> partition, group;
    if (!parse_partition(part, partition)) return nullptr;
    if (group_obj && group_obj != Py_None && !parse_partition(group_obj, group))
        return nullptr;
    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::pair<DataFrame, DataFrame> pr;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            pr = rt->submit(v.containment(partition, ts, dur, name, group))
                     .get();
        }))
        return nullptr;
    PyObject* ct = dftracer::utils::python::wrap_dataframe(std::move(pr.first));
    if (!ct) return nullptr;
    PyObject* fg =
        dftracer::utils::python::wrap_dataframe(std::move(pr.second));
    if (!fg) {
        Py_DECREF(ct);
        return nullptr;
    }
    PyObject* t = PyTuple_Pack(2, ct, fg);  // Pack INCREFs both
    Py_DECREF(ct);
    Py_DECREF(fg);
    return t;
#endif
}

// A file-backed NDJSON sink for a session export branch; closes its file on
// destruction.
class SessionFileSink : public dftracer::utils::trace::views::ExportSink {
   public:
    explicit SessionFileSink(FILE* f) : f_(f) {}
    ~SessionFileSink() override {
        if (f_) std::fclose(f_);
    }
    void write(std::string_view data) override {
        if (f_) std::fwrite(data.data(), 1, data.size(), f_);
    }

   private:
    FILE* f_;
};

// One shared scan driving several ViewSession branches. `branches` is a list of
// (kind:str, viewer:_TraceViewer, sink:str|None) tuples; each viewer carries
// the branch's full plan. Results come back in branch order. The base viewer's
// files/phase/time settings scope the shared scan.
// Map a join-type name to the enum. Returns false on an unknown name.
bool parse_join_type(const char* how,
                     dftracer::utils::trace::views::JoinType* out) {
    namespace views = dftracer::utils::trace::views;
    const std::string h(how);
    if (h == "inner")
        *out = views::JoinType::INNER;
    else if (h == "left")
        *out = views::JoinType::LEFT;
    else if (h == "right")
        *out = views::JoinType::RIGHT;
    else if (h == "full")
        *out = views::JoinType::FULL;
    else if (h == "semi")
        *out = views::JoinType::LEFT_SEMI;
    else if (h == "anti")
        *out = views::JoinType::LEFT_ANTI;
    else
        return false;
    return true;
}

PyObject* tv_session_run(TraceViewerObject* self, PyObject* branches) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "session() requires the arrow-enabled build");
    return nullptr;
#else
    if (!PyList_Check(branches)) {
        PyErr_SetString(PyExc_TypeError, "session branches must be a list");
        return nullptr;
    }
    const Py_ssize_t nb = PyList_Size(branches);

    enum class Kind {
        Collect,
        Materialize,
        Export,
        Events,
        Plugin,
        Partial,
        Join,
        Compare,
        CallTree,
        Flamegraph,
        Containment
    };
    struct BranchData {
        Kind kind = Kind::Collect;
        std::vector<std::string> files;
        std::string index_dir;
        ViewerPlan plan;
        std::string sink;
        // Plugin branch only: the C++ host attached to the session, and its
        // Python object whose named results are read after execute.
        dftracer::utils::plugins::PluginHost* host_cpp = nullptr;
        PyObject* host_obj = nullptr;  // borrowed; the branches list holds it
        // Combine branch (Join/Compare) only: the two source branch indices and
        // the join type; n_key is inferred natively from the source branch.
        Py_ssize_t left_idx = -1;
        Py_ssize_t right_idx = -1;
        dftracer::utils::trace::views::JoinType how =
            dftracer::utils::trace::views::JoinType::INNER;
    };
    std::vector<BranchData> bdata(nb);

    // Snapshot each branch viewer's plan while the GIL is held.
    for (Py_ssize_t i = 0; i < nb; ++i) {
        PyObject* t = PyList_GetItem(branches, i);
        if (!PyTuple_Check(t) || PyTuple_Size(t) != 3) {
            PyErr_SetString(
                PyExc_TypeError,
                "each branch must be a 3-tuple (kind, viewer, sink)");
            return nullptr;
        }
        const char* kind = as_utf8(PyTuple_GetItem(t, 0));
        if (!kind) return nullptr;
        PyObject* vobj = PyTuple_GetItem(t, 1);
        PyObject* sk = PyTuple_GetItem(t, 2);

        const std::string k(kind);
        if (k == "collect")
            bdata[i].kind = Kind::Collect;
        else if (k == "materialize")
            bdata[i].kind = Kind::Materialize;
        else if (k == "export")
            bdata[i].kind = Kind::Export;
        else if (k == "events")
            bdata[i].kind = Kind::Events;
        else if (k == "partial")
            bdata[i].kind = Kind::Partial;
        else if (k == "plugin")
            bdata[i].kind = Kind::Plugin;
        else if (k == "join")
            bdata[i].kind = Kind::Join;
        else if (k == "compare")
            bdata[i].kind = Kind::Compare;
        else if (k == "call_tree")
            bdata[i].kind = Kind::CallTree;
        else if (k == "flamegraph")
            bdata[i].kind = Kind::Flamegraph;
        else if (k == "containment")
            bdata[i].kind = Kind::Containment;
        else {
            PyErr_Format(PyExc_ValueError, "unknown session branch kind: %s",
                         kind);
            return nullptr;
        }

        if (bdata[i].kind == Kind::Join || bdata[i].kind == Kind::Compare) {
            // vobj is (left_idx, right_idx[, how]); both indices must refer to
            // earlier collect branches. n_key is inferred natively.
            if (!PyTuple_Check(vobj) || PyTuple_Size(vobj) < 2) {
                PyErr_SetString(PyExc_TypeError,
                                "a join/compare branch needs (left, right[, "
                                "how]) branch indices");
                return nullptr;
            }
            bdata[i].left_idx = PyLong_AsSsize_t(PyTuple_GetItem(vobj, 0));
            bdata[i].right_idx = PyLong_AsSsize_t(PyTuple_GetItem(vobj, 1));
            if (PyErr_Occurred()) return nullptr;
            if (bdata[i].left_idx < 0 || bdata[i].left_idx >= i ||
                bdata[i].right_idx < 0 || bdata[i].right_idx >= i) {
                PyErr_SetString(PyExc_ValueError,
                                "join/compare indices must refer to earlier "
                                "branches");
                return nullptr;
            }
            if (bdata[i].kind == Kind::Join && PyTuple_Size(vobj) >= 3) {
                const char* h = as_utf8(PyTuple_GetItem(vobj, 2));
                if (!h) return nullptr;
                if (!parse_join_type(h, &bdata[i].how)) {
                    PyErr_Format(PyExc_ValueError, "unknown join type: %s", h);
                    return nullptr;
                }
            }
            continue;
        }

        if (bdata[i].kind == Kind::Plugin) {
            if (!PyObject_TypeCheck(vobj, &PluginHostType)) {
                PyErr_SetString(PyExc_TypeError,
                                "a plugin branch needs a PluginHost");
                return nullptr;
            }
            bdata[i].host_obj = vobj;  // borrowed; branches list holds it
            bdata[i].host_cpp =
                static_cast<dftracer::utils::plugins::PluginHost*>(
                    ((PluginHostObject*)vobj)->host_ptr);
            continue;
        }

        if (!PyObject_TypeCheck(vobj, &TraceViewerType)) {
            PyErr_SetString(PyExc_TypeError,
                            "session branch viewer must be a TraceViewer");
            return nullptr;
        }
        auto* v = (TraceViewerObject*)vobj;
        bdata[i].files = extract_files(v);
        bdata[i].index_dir = extract_index_dir(v);
        bdata[i].plan = *plan_of(v);
        if (sk != Py_None) {
            const char* s = as_utf8(sk);
            if (!s) return nullptr;
            bdata[i].sink = s;
        }
        if (bdata[i].kind == Kind::Export && bdata[i].sink.empty()) {
            PyErr_SetString(PyExc_ValueError,
                            "an export branch needs a sink path");
            return nullptr;
        }
    }

    Runtime* rt = resolve_runtime(self);
    auto base_files = extract_files(self);
    auto base_index = extract_index_dir(self);
    ViewerPlan base_plan = *plan_of(self);

    // AND-combine a branch's DSL filters into one export predicate ("" = all).
    auto combine_filters = [](const std::vector<std::string>& fs) {
        std::string out;
        for (std::size_t i = 0; i < fs.size(); ++i) {
            if (i) out += " and ";
            out += "(" + fs[i] + ")";
        }
        return out;
    };

    std::vector<DataFrame> results(nb);
    std::vector<DataFrame> results2(nb);  // containment's second (flamegraph)
    std::vector<ExportStats> export_stats(nb);
    std::vector<std::string> partial_results(nb);
    std::string err;
    if (!run_blocking([&] {
            View base = build_view_from_data(base_files, base_index, base_plan,
                                             /*aggregate=*/false);
            ViewSession sess = base.session();
            std::vector<Deferred<DataFrame>> agg_handles(nb);
            std::vector<Deferred<DataFrame>> agg_handles2(nb);
            std::vector<Deferred<ExportStats>> exp_handles(nb);
            std::vector<Deferred<std::string>> partial_handles(nb);
            std::vector<std::unique_ptr<SessionFileSink>> sinks;
            for (Py_ssize_t i = 0; i < nb; ++i) {
                switch (bdata[i].kind) {
                    case Kind::Collect: {
                        View bview = build_view_from_data(
                            bdata[i].files, bdata[i].index_dir, bdata[i].plan,
                            /*aggregate=*/true);
                        agg_handles[i] = sess.collect(bview);
                        break;
                    }
                    case Kind::Events: {
                        View bview = build_view_from_data(
                            bdata[i].files, bdata[i].index_dir, bdata[i].plan,
                            /*aggregate=*/true);
                        agg_handles[i] = sess.collect_events(bview);
                        break;
                    }
                    case Kind::CallTree:
                    case Kind::Flamegraph:
                    case Kind::Containment: {
                        View bview = build_view_from_data(
                            bdata[i].files, bdata[i].index_dir, bdata[i].plan,
                            /*aggregate=*/true);
                        std::vector<std::string> partition;
                        std::string ts, dur, nm;
                        parse_containment_cfg(bdata[i].sink, partition, ts, dur,
                                              nm);
                        if (bdata[i].kind == Kind::Containment) {
                            dftracer::utils::trace::views::ContainmentHandles
                                ch = sess.containment(bview, partition, ts, dur,
                                                      nm);
                            agg_handles[i] = ch.call_tree;
                            agg_handles2[i] = ch.flamegraph;
                        } else if (bdata[i].kind == Kind::Flamegraph) {
                            agg_handles[i] =
                                sess.flamegraph(bview, partition, ts, dur, nm);
                        } else {
                            agg_handles[i] =
                                sess.call_tree(bview, partition, ts, dur, nm);
                        }
                        break;
                    }
                    case Kind::Partial: {
                        View bview = build_view_from_data(
                            bdata[i].files, bdata[i].index_dir, bdata[i].plan,
                            /*aggregate=*/true);
                        partial_handles[i] = sess.aggregate_partial(bview);
                        break;
                    }
                    case Kind::Materialize:
                        sess.materialize(bdata[i].plan.group_by,
                                         bdata[i].plan.agg);
                        break;
                    case Kind::Export: {
                        std::optional<Query> q;
                        const std::string f =
                            combine_filters(bdata[i].plan.filters);
                        if (!f.empty()) {
                            auto parsed = Query::from_string(f);
                            if (!parsed) {
                                err = "invalid filter query: " + f;
                                return;
                            }
                            q = std::move(parsed.value());
                        }
                        FILE* fp = std::fopen(bdata[i].sink.c_str(), "wb");
                        if (!fp) {
                            err = "cannot open export sink: " + bdata[i].sink;
                            return;
                        }
                        sinks.push_back(std::make_unique<SessionFileSink>(fp));
                        exp_handles[i] =
                            q ? sess.export_json(std::move(*q), *sinks.back())
                              : sess.export_json(*sinks.back());
                        break;
                    }
                    case Kind::Plugin:
                        // C++-only, safe with the GIL released; named results
                        // are read back after execute.
                        bdata[i].host_cpp->attach_to_session(sess);
                        break;
                    case Kind::Join:
                        agg_handles[i] = sess.join(
                            agg_handles[bdata[i].left_idx],
                            agg_handles[bdata[i].right_idx], bdata[i].how);
                        break;
                    case Kind::Compare:
                        agg_handles[i] =
                            sess.compare(agg_handles[bdata[i].left_idx],
                                         agg_handles[bdata[i].right_idx]);
                        break;
                }
            }
            rt->submit(sess.execute()).get();
            for (Py_ssize_t i = 0; i < nb; ++i) {
                if (bdata[i].kind == Kind::Collect ||
                    bdata[i].kind == Kind::Events ||
                    bdata[i].kind == Kind::Join ||
                    bdata[i].kind == Kind::Compare ||
                    bdata[i].kind == Kind::CallTree ||
                    bdata[i].kind == Kind::Flamegraph)
                    results[i] = std::move(agg_handles[i].get());
                else if (bdata[i].kind == Kind::Containment) {
                    results[i] = std::move(agg_handles[i].get());
                    results2[i] = std::move(agg_handles2[i].get());
                } else if (bdata[i].kind == Kind::Export)
                    export_stats[i] = exp_handles[i].get();
                else if (bdata[i].kind == Kind::Partial)
                    partial_results[i] = std::move(partial_handles[i].get());
            }
        }))
        return nullptr;
    if (!err.empty()) {
        PyErr_SetString(PyExc_ValueError, err.c_str());
        return nullptr;
    }

    PyObject* out = PyList_New(nb);
    if (!out) return nullptr;
    for (Py_ssize_t i = 0; i < nb; ++i) {
        PyObject* item = nullptr;
        if (bdata[i].kind == Kind::Collect || bdata[i].kind == Kind::Events ||
            bdata[i].kind == Kind::Join || bdata[i].kind == Kind::Compare ||
            bdata[i].kind == Kind::CallTree ||
            bdata[i].kind == Kind::Flamegraph) {
            item =
                dftracer::utils::python::wrap_dataframe(std::move(results[i]));
        } else if (bdata[i].kind == Kind::Containment) {
            PyObject* ct =
                dftracer::utils::python::wrap_dataframe(std::move(results[i]));
            if (!ct) {
                Py_DECREF(out);
                return nullptr;
            }
            PyObject* fg =
                dftracer::utils::python::wrap_dataframe(std::move(results2[i]));
            if (!fg) {
                Py_DECREF(ct);
                Py_DECREF(out);
                return nullptr;
            }
            item = PyTuple_Pack(2, ct, fg);
            Py_DECREF(ct);
            Py_DECREF(fg);
        } else if (bdata[i].kind == Kind::Export) {
            const ExportStats& st = export_stats[i];
            item = Py_BuildValue(
                "{s:K,s:K,s:K}", "events_matched",
                (unsigned long long)st.events_matched, "events_scanned",
                (unsigned long long)st.events_scanned, "chunks_scanned",
                (unsigned long long)st.chunks_scanned);
        } else if (bdata[i].kind == Kind::Partial) {
            item = PyBytes_FromStringAndSize(
                partial_results[i].data(),
                (Py_ssize_t)partial_results[i].size());
        } else if (bdata[i].kind == Kind::Plugin) {
            // {name: pyarrow|bytes}; the Python Session shapes it to DataFrame.
            item = dftracer::utils::python::plugin_host_results_dict(
                bdata[i].host_obj);
        } else {
            Py_INCREF(Py_None);
            item = Py_None;
        }
        if (!item) {
            Py_DECREF(out);
            return nullptr;
        }
        PyList_SET_ITEM(out, i, item);
    }
    return out;
#endif
}

// Aggregate this viewer and `other`, then equi-join their result tables on the
// shared group key. Raises ValueError on a bad `how` or when the two group-key
// schemas differ; TypeError when `other` is not a TraceViewer.
PyObject* tv_join(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "join() requires the arrow-enabled build");
    return nullptr;
#else
    namespace views = dftracer::utils::trace::views;
    PyObject* other = nullptr;
    const char* how = "inner";
    static const char* kwlist[] = {"other", "how", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O|s",
                                     const_cast<char**>(kwlist), &other, &how))
        return nullptr;
    if (!PyObject_TypeCheck(other, &TraceViewerType)) {
        PyErr_SetString(PyExc_TypeError, "join() other must be a TraceViewer");
        return nullptr;
    }
    views::JoinType type = views::JoinType::INNER;
    if (!parse_join_type(how, &type)) {
        PyErr_Format(PyExc_ValueError,
                     "join() how must be inner|left|right|full|semi|anti, "
                     "got '%s'",
                     how);
        return nullptr;
    }

    TraceViewerObject* o = (TraceViewerObject*)other;
    Runtime* rt = resolve_runtime(self);
    auto lf = extract_files(self);
    auto li = extract_index_dir(self);
    ViewerPlan lp = *plan_of(self);
    auto rf = extract_files(o);
    auto ri = extract_index_dir(o);
    ViewerPlan rp = *plan_of(o);
    DataFrame joined;
    if (!run_blocking([&] {
            AggregatedView lv = build_agg_view(lf, li, lp);
            AggregatedView rv = build_agg_view(rf, ri, rp);
            joined = rt->submit(lv.join(rv, type)).get();
        }))
        return nullptr;
    if (joined.num_columns() == 0) {  // mismatched group-key schemas
        PyErr_SetString(
            PyExc_ValueError,
            "join() requires both views to share a group-key schema");
        return nullptr;
    }
    return dftracer::utils::python::wrap_dataframe(std::move(joined));
#endif
}

// compare(other): aggregate this viewer and `other` with THIS viewer's
// group_by + agg plan, in parallel, and return the comparison DataFrame (group
// key, l_/r_ per metric, plus delta_/pct_). Wraps
// trace::comparator::CompareView.
PyObject* tv_compare(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
    namespace comparator = dftracer::utils::trace::comparator;
    PyObject* other = nullptr;
    static const char* kwlist[] = {"other", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "O",
                                     const_cast<char**>(kwlist), &other))
        return nullptr;
    if (!PyObject_TypeCheck(other, &TraceViewerType)) {
        PyErr_SetString(PyExc_TypeError,
                        "compare() other must be a TraceViewer");
        return nullptr;
    }
    TraceViewerObject* o = (TraceViewerObject*)other;
    ViewerPlan lp = *plan_of(self);
    if (lp.agg.empty()) {
        PyErr_SetString(PyExc_ValueError,
                        "compare() needs a group_by + agg plan on the baseline "
                        "viewer (both sides aggregate the same way)");
        return nullptr;
    }
    Runtime* rt = resolve_runtime(self);
    auto lf = extract_files(self);
    auto li = extract_index_dir(self);
    auto rf = extract_files(o);
    auto ri = extract_index_dir(o);
    DataFrame result;
    if (!run_blocking([&] {
            View base = build_view_from_data(lf, li, lp, /*aggregate=*/false);
            View variant =
                build_view_from_data(rf, ri, lp, /*aggregate=*/false);
            result = rt->submit(comparator::CompareView::of(std::move(base),
                                                            std::move(variant))
                                    .group_by(lp.group_by)
                                    .agg(lp.agg)
                                    .collect())
                         .get();
        }))
        return nullptr;
    return dftracer::utils::python::wrap_dataframe(std::move(result));
}

// One-pass read of the aggregation index's three record families, returned as a
// dict of native DataFrames: {"regular", "aggregated", "counters"}.
PyObject* tv_collect_typed(TraceViewerObject* self, PyObject* args,
                           PyObject* kwds) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "collect_typed() requires the arrow-enabled build");
    return nullptr;
#else
    int shard_begin = 0, shard_end = 0;  // shard_end <= 0 = all shards
    PyObject* progress_obj = nullptr;
    static const char* kwlist[] = {"shard_begin", "shard_end", "progress",
                                   nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|iiO",
                                     const_cast<char**>(kwlist), &shard_begin,
                                     &shard_end, &progress_obj))
        return nullptr;

    // Bridge a Python (done, total) callback to the C++ scan; it fires from a
    // runtime worker thread while run_blocking holds the GIL released, so each
    // call re-acquires the GIL.
    namespace views = dftracer::utils::trace::views;
    views::ProgressFn progress_fn;
    const views::ProgressFn* progress_ptr = nullptr;
    if (progress_obj && progress_obj != Py_None) {
        Py_INCREF(progress_obj);
        std::shared_ptr<PyObject> cb(progress_obj, [](PyObject* p) {
            PyGILState_STATE g = PyGILState_Ensure();
            Py_DECREF(p);
            PyGILState_Release(g);
        });
        progress_fn = [cb](std::size_t done, std::size_t total) {
            PyGILState_STATE g = PyGILState_Ensure();
            PyObject* r = PyObject_CallFunction(cb.get(), "nn",
                                                static_cast<Py_ssize_t>(done),
                                                static_cast<Py_ssize_t>(total));
            if (r)
                Py_DECREF(r);
            else
                PyErr_Clear();
            PyGILState_Release(g);
        };
        progress_ptr = &progress_fn;
    }

    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    TypedResult typed;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            typed = rt->submit(
                          v.collect_typed(shard_begin, shard_end, progress_ptr))
                        .get();
        }))
        return nullptr;
    namespace py = dftracer::utils::python;
    PyObject* regular = py::wrap_dataframe(std::move(typed.regular));
    if (!regular) return nullptr;
    PyObject* aggregated = py::wrap_dataframe(std::move(typed.aggregated));
    if (!aggregated) {
        Py_DECREF(regular);
        return nullptr;
    }
    PyObject* counters = py::wrap_dataframe(std::move(typed.counters));
    if (!counters) {
        Py_DECREF(regular);
        Py_DECREF(aggregated);
        return nullptr;
    }
    PyObject* d = PyDict_New();
    if (!d) {
        Py_DECREF(regular);
        Py_DECREF(aggregated);
        Py_DECREF(counters);
        return nullptr;
    }
    PyDict_SetItemString(d, "regular", regular);
    PyDict_SetItemString(d, "aggregated", aggregated);
    PyDict_SetItemString(d, "counters", counters);
    Py_DECREF(regular);
    Py_DECREF(aggregated);
    Py_DECREF(counters);
    return d;
#endif
}

// Aggregate this (shard's) files into an opaque, combinable partial. The bytes
// carry the running accumulators (count/sum/sumsq/sketch), so distributed
// callers merge them with merge_partials() - correct for mean/std/percentiles,
// unlike re-aggregating finalized per-shard values.
PyObject* tv_aggregate_partial(TraceViewerObject* self, PyObject*) {
    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::string out;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            out = rt->submit(v.aggregate_partial()).get();
        }))
        return nullptr;
    return PyBytes_FromStringAndSize(out.data(),
                                     static_cast<Py_ssize_t>(out.size()));
}

// Combine partials from aggregate_partial() into the final aggregation Table.
// Uses this viewer's group_by/agg as the shape; files are not scanned.
PyObject* tv_merge_partials(TraceViewerObject* self, PyObject* arg) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "merge_partials() requires the arrow-enabled build");
    return nullptr;
#else
    PyObject* seq = PySequence_Fast(arg, "merge_partials expects a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    std::vector<std::string> owned;
    owned.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        char* buf = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(PySequence_Fast_GET_ITEM(seq, i), &buf,
                                    &len) < 0) {
            Py_DECREF(seq);
            return nullptr;
        }
        owned.emplace_back(buf, static_cast<std::size_t>(len));
    }
    Py_DECREF(seq);

    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    DataFrame table;
    if (!run_blocking([&] {
            std::vector<std::string_view> parts(owned.begin(), owned.end());
            View v = build_view_from_data(files, index_dir, plan);
            table = v.merge_partials_to_table(parts);
        }))
        return nullptr;
    return dftracer::utils::python::wrap_dataframe(std::move(table));
#endif
}

// Scan this rank's files into a serialized flamegraph arena partial (bytes).
PyObject* tv_flamegraph_partial(TraceViewerObject* self, PyObject* args) {
    PyObject* part = nullptr;
    const char* ts = "ts";
    const char* dur = "dur";
    const char* name = "name";
    PyObject* group_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O|sssO", &part, &ts, &dur, &name, &group_obj))
        return nullptr;
    std::vector<std::string> partition, group;
    if (!parse_partition(part, partition)) return nullptr;
    if (group_obj && group_obj != Py_None && !parse_partition(group_obj, group))
        return nullptr;
    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::string out;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            out = rt->submit(
                        v.flamegraph_partial(partition, ts, dur, name, group))
                      .get();
        }))
        return nullptr;
    return PyBytes_FromStringAndSize(out.data(),
                                     static_cast<Py_ssize_t>(out.size()));
}

// Merge flamegraph arena partials (from flamegraph_partial across ranks) into
// the final node DataFrame. No scan.
PyObject* tv_merge_flamegraph_partials(TraceViewerObject*, PyObject* arg) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "merge_flamegraph_partials() requires the arrow build");
    return nullptr;
#else
    PyObject* seq =
        PySequence_Fast(arg, "merge_flamegraph_partials expects a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    std::vector<std::string> owned;
    owned.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        char* buf = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(PySequence_Fast_GET_ITEM(seq, i), &buf,
                                    &len) < 0) {
            Py_DECREF(seq);
            return nullptr;
        }
        owned.emplace_back(buf, static_cast<std::size_t>(len));
    }
    Py_DECREF(seq);
    std::vector<std::string_view> parts(owned.begin(), owned.end());
    DataFrame table = View::merge_flamegraph_partials(parts);
    return dftracer::utils::python::wrap_dataframe(std::move(table));
#endif
}

// Distributed materialize: reduce rank-local partials and write the rollup.
PyObject* tv_materialize_partials(TraceViewerObject* self, PyObject* arg) {
    PyObject* seq =
        PySequence_Fast(arg, "materialize_partials expects a sequence");
    if (!seq) return nullptr;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(seq);
    std::vector<std::string> owned;
    owned.reserve(static_cast<std::size_t>(n));
    for (Py_ssize_t i = 0; i < n; ++i) {
        char* buf = nullptr;
        Py_ssize_t len = 0;
        if (PyBytes_AsStringAndSize(PySequence_Fast_GET_ITEM(seq, i), &buf,
                                    &len) < 0) {
            Py_DECREF(seq);
            return nullptr;
        }
        owned.emplace_back(buf, static_cast<std::size_t>(len));
    }
    Py_DECREF(seq);

    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    if (!run_blocking([&] {
            std::vector<std::string_view> parts(owned.begin(), owned.end());
            AggregatedView v = build_agg_view(files, index_dir, plan);
            rt->submit(v.materialize_partials(parts)).get();
        }))
        return nullptr;
    Py_RETURN_NONE;
}

// Build-only materialize (fire and forget): persist this query as a
// materialized view so a later matching read reuses it. A row query writes a
// filtered trace split into `part_size`-byte files at `checkpoint_size`
// granularity; an aggregation persists a rollup. Idempotent. Returns None.
PyObject* tv_materialize(TraceViewerObject* self, PyObject* args,
                         PyObject* kwds) {
    long long checkpoint_size = 0, part_size = 0;
    PyObject* progress_obj = nullptr;
    static const char* kwlist[] = {"checkpoint_size", "part_size", "progress",
                                   nullptr};
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "|LLO", const_cast<char**>(kwlist), &checkpoint_size,
            &part_size, &progress_obj))
        return nullptr;

    namespace views = dftracer::utils::trace::views;
    views::ProgressFn progress_fn;
    const views::ProgressFn* progress_ptr = nullptr;
    if (progress_obj && progress_obj != Py_None) {
        Py_INCREF(progress_obj);
        std::shared_ptr<PyObject> cb(progress_obj, [](PyObject* p) {
            PyGILState_STATE g = PyGILState_Ensure();
            Py_DECREF(p);
            PyGILState_Release(g);
        });
        progress_fn = [cb](std::size_t done, std::size_t total) {
            PyGILState_STATE g = PyGILState_Ensure();
            PyObject* r = PyObject_CallFunction(cb.get(), "nn",
                                                static_cast<Py_ssize_t>(done),
                                                static_cast<Py_ssize_t>(total));
            if (r)
                Py_DECREF(r);
            else
                PyErr_Clear();
            PyGILState_Release(g);
        };
        progress_ptr = &progress_fn;
    }

    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            rt->submit(
                  v.materialize(static_cast<std::uint64_t>(checkpoint_size),
                                static_cast<std::uint64_t>(part_size))
                      .run(progress_ptr))
                .get();
        }))
        return nullptr;
    Py_RETURN_NONE;
}

// Observability: the MV trace file(s) that would serve this query, or an empty
// list if a read would scan the base. Lets callers report/assert MV reuse.
PyObject* tv_mv_source(TraceViewerObject* self, PyObject*) {
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::vector<std::string> paths;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            paths = v.mv_source();
        }))
        return nullptr;
    PyObject* list = PyList_New(static_cast<Py_ssize_t>(paths.size()));
    if (!list) return nullptr;
    for (std::size_t i = 0; i < paths.size(); ++i)
        PyList_SET_ITEM(
            list, static_cast<Py_ssize_t>(i),
            PyUnicode_FromStringAndSize(
                paths[i].data(), static_cast<Py_ssize_t>(paths[i].size())));
    return list;
}

// Distributed row-MV coordinator: create and return the shared MV directory
// derived from this (full-file-set) view's plan. Ranks export their filtered
// files into subdirs of it; the coordinator then calls register_materialized.
PyObject* tv_materialize_dir(TraceViewerObject* self, PyObject*) {
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::string dir;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            dir = v.materialize_dir();
        }))
        return nullptr;
    return PyUnicode_FromStringAndSize(dir.data(),
                                       static_cast<Py_ssize_t>(dir.size()));
}

// Write the MV manifest at `dir` over this (full) view's base set, after every
// rank has materialized its shard subdir. Makes the MV discoverable.
PyObject* tv_register_materialized(TraceViewerObject* self, PyObject* arg) {
    const char* dir = as_utf8(arg);
    if (!dir) return nullptr;
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::string d(dir);
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            v.register_materialized(d);
        }))
        return nullptr;
    Py_RETURN_NONE;
}

// Reconstruct the cached aggregation into a pyarrow.Table, or None on a cache
// miss (no scan/decode). Lets a distributed caller pick read vs recompute.
PyObject* tv_reconstruct_if_cached(TraceViewerObject* self, PyObject*) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "reconstruct requires the arrow-enabled build");
    return nullptr;
#else
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    std::optional<DataFrame> table;
    if (!run_blocking([&] {
            AggregatedView v = build_agg_view(files, index_dir, plan);
            table = v.reconstruct_if_cached();
        }))
        return nullptr;
    if (!table) Py_RETURN_NONE;
    return dftracer::utils::python::wrap_dataframe(std::move(*table));
#endif
}

// One-row summary aggregation: count, mean/std of dur, min/max ts.
PyObject* tv_statistics(TraceViewerObject* self, PyObject*) {
    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);
    DataFrame table;
    if (!run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan)
                         .group_by({})
                         .agg({AggSpec(AggOp::Count, "", "duration_count"),
                               AggSpec(AggOp::Mean, "dur", "duration_mean_us"),
                               AggSpec(AggOp::Std, "dur", "duration_stddev_us"),
                               AggSpec(AggOp::Min, "ts", "min_timestamp_us"),
                               AggSpec(AggOp::Max, "ts", "max_timestamp_us")});
            table = rt->submit(v.collect().collect()).get();
        }))
        return nullptr;

    PyObject* d = PyDict_New();
    if (!d) return nullptr;
    double count = 0, mean = 0, stddev = 0, mn = 0, mx = 0;
    if (table.num_rows() >= 1 && table.columns.size() >= 5) {
        count = dftracer::utils::dataframe::read_f64(table.columns[0], 0);
        mean = dftracer::utils::dataframe::read_f64(table.columns[1], 0);
        stddev = dftracer::utils::dataframe::read_f64(table.columns[2], 0);
        mn = dftracer::utils::dataframe::read_f64(table.columns[3], 0);
        mx = dftracer::utils::dataframe::read_f64(table.columns[4], 0);
    }
    dict_set_i64(d, "duration_count", (long long)count);
    dict_set_f64(d, "duration_mean_us", mean);
    dict_set_f64(d, "duration_stddev_us", stddev);
    dict_set_i64(d, "min_timestamp_us", (long long)mn);
    dict_set_i64(d, "max_timestamp_us", (long long)mx);
    return d;
}

class FileSink : public dftracer::utils::trace::views::ExportSink {
   public:
    explicit FileSink(FILE* f) : f_(f) {}
    void write(std::string_view data) override {
        std::fwrite(data.data(), 1, data.size(), f_);
    }

   private:
    FILE* f_;
};

// Streaming gzip sink: buffers writes and flushes whole-line gzip members once
// past MEMBER_TARGET, so the aggregation output is a multi-member re-indexable
// trace. finish() flushes the remainder.
class GzipSink : public dftracer::utils::trace::views::ExportSink {
   public:
    GzipSink(FILE* f, int level) : f_(f), comp_(level) {}
    void write(std::string_view data) override {
        buf_.append(data);
        while (buf_.size() >= MEMBER_TARGET) {
            std::size_t cut = buf_.rfind('\n', buf_.size());
            if (cut == std::string::npos || cut + 1 < MEMBER_TARGET) break;
            flush(cut + 1);
        }
    }
    void finish() {
        if (!buf_.empty()) flush(buf_.size());
    }

   private:
    static constexpr std::size_t MEMBER_TARGET = 4 * 1024 * 1024;
    void flush(std::size_t n) {
        if (comp_.compress_member_into(scratch_, buf_.data(), n))
            std::fwrite(scratch_.data(), 1, scratch_.size(), f_);
        buf_.erase(0, n);
    }
    FILE* f_;
    dftracer::utils::utilities::fileio::compress::GzipMemberCompressor comp_;
    std::string buf_;
    std::vector<std::uint8_t> scratch_;
};

// The single export terminal: writes a dftracer trace whose content follows the
// plan (aggregation when group_by/agg is set, else the matching events).
// Gzip-and-re-indexable by default; compress=False writes plain NDJSON; index
// builds the index inline (events only).
PyObject* tv_export(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
    static const char* kwlist[] = {"path",  "compress",  "index", "member_size",
                                   "level", "part_size", nullptr};
    const char* path = nullptr;
    int compress = 1;
    int index = 0;
    long long member_size = 0;
    int level = 6;
    long long part_size = 0;
    if (!PyArg_ParseTupleAndKeywords(
            args, kwds, "s|ppLiL", const_cast<char**>(kwlist), &path, &compress,
            &index, &member_size, &level, &part_size))
        return nullptr;

    const ViewerPlan& p = *plan_of(self);
    const bool aggregated = !p.group_by.empty() || !p.agg.empty();
    std::string path_s(path);

    Runtime* rt = resolve_runtime(self);
    auto files = extract_files(self);
    auto index_dir = extract_index_dir(self);
    ViewerPlan plan = p;

    if (aggregated) {
        FILE* f = std::fopen(path, "wb");
        if (!f) {
            PyErr_SetFromErrnoWithFilename(PyExc_OSError, path);
            return nullptr;
        }
        bool ok = run_blocking([&] {
            View v = build_view_from_data(files, index_dir, plan);
            if (compress) {
                GzipSink sink(f, level);
                rt->submit(v.export_counters(sink)).get();
                sink.finish();
            } else {
                FileSink sink(f);
                rt->submit(v.export_counters(sink)).get();
            }
        });
        std::fclose(f);
        if (!ok) return nullptr;
        Py_RETURN_NONE;
    }

    if (!run_blocking([&] {
            using dftracer::utils::trace::views::TraceWriteOptions;
            View v = build_view_from_data(files, index_dir, plan);
            TraceWriteOptions opts;
            opts.output_path = path_s;
            opts.member_size = (std::size_t)member_size;
            opts.compress = compress != 0;
            opts.level = level;
            opts.build_index = index != 0;
            opts.part_size = (std::size_t)part_size;
            rt->submit(v.export_trace(std::move(opts))).get();
        }))
        return nullptr;
    Py_RETURN_NONE;
}

#ifdef DFTRACER_UTILS_ENABLE_ARROW
// Drives View::stream() (raw-event morsels via LazyFrame's streaming cursor)
// into the StreamingState queue. push() blocks when the bounded queue is
// full; the Python consumer drains it with the GIL released, converting to
// Arrow only if it calls .to_arrow() on a chunk.
dftracer::utils::coro::CoroTask<void> run_viewer_stream(
    dftracer::utils::CoroScope& scope,
    std::shared_ptr<
        dftracer::utils::python::StreamingState<dataframe::DataFrame>>
        state,
    std::vector<std::string> files, std::string index_dir, ViewerPlan plan,
    std::int64_t batch_size) {
    (void)scope;
    try {
        View v = build_view_from_data(files, index_dir, plan, /*aggregate=*/
                                      false);
        if (!plan.select.empty()) v = v.select(plan.select);
        if (!plan.sort_col.empty())
            v = v.sort_by(plan.sort_col, plan.sort_desc);
        if (!plan.topk_col.empty())
            v = v.topk(plan.topk_col, plan.topk_k, plan.topk_largest);
        if (plan.limit) v = v.limit(plan.limit);
        if (plan.offset) v = v.offset(plan.offset);

        auto gen = v.stream(batch_size);
        while (auto df = co_await gen.next()) {
            if (state->cancelled()) break;
            const std::size_t bytes =
                static_cast<std::size_t>(df->num_rows()) *
                static_cast<std::size_t>(df->num_columns() + 1) * 16;
            if (!state->push(std::move(*df), bytes)) break;
        }
        state->complete();
    } catch (...) {
        state->fail(std::current_exception());
    }
}
#endif

PyObject* tv_stream(TraceViewerObject* self, PyObject* args, PyObject* kwds) {
#ifndef DFTRACER_UTILS_ENABLE_ARROW
    PyErr_SetString(PyExc_RuntimeError,
                    "stream() requires the arrow-enabled build");
    return nullptr;
#else
    static const char* kwlist[] = {"batch_size", "workers", "normalize", "dict",
                                   nullptr};
    long long batch_size = 65536;
    long long workers = 0;  // 0 -> runtime worker count
    int normalize = 0;
    int dict_strings = 1;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|LLpp",
                                     const_cast<char**>(kwlist), &batch_size,
                                     &workers, &normalize, &dict_strings))
        return nullptr;
    if (batch_size <= 0) batch_size = 65536;
    Runtime* rt = resolve_runtime(self);
    (void)workers;
    (void)normalize;
    (void)dict_strings;

    std::vector<std::string> files = extract_files(self);
    std::string index_dir = extract_index_dir(self);
    ViewerPlan plan = *plan_of(self);

    // 0 (unset) falls back to the RAM-fraction default.
    auto state = std::make_shared<
        dftracer::utils::python::StreamingState<dataframe::DataFrame>>(
        dftracer::utils::compute_memory_budget(plan.memory_budget));

    auto* iter_obj =
        (dftracer::utils::python::ArrowStreamingIteratorObject*)
            dftracer::utils::python::ArrowStreamingIteratorType.tp_new(
                &dftracer::utils::python::ArrowStreamingIteratorType, nullptr,
                nullptr);
    if (!iter_obj) return nullptr;
    iter_obj->cpp_state->state = state;
    iter_obj->cpp_state->pull_df =
        [state]() -> std::optional<dataframe::DataFrame> {
        return state->pull();
    };
    iter_obj->cpp_state->get_error = [state]() -> std::exception_ptr {
        return state->error();
    };
    iter_obj->cpp_state->cancel = [state]() { state->cancel(); };

    Py_BEGIN_ALLOW_THREADS rt->submit(
        dftracer::utils::run_coro_scope(
            rt->executor(), run_viewer_stream, state, std::move(files),
            std::move(index_dir), std::move(plan), (std::int64_t)batch_size),
        "trace_viewer_stream");
    Py_END_ALLOW_THREADS return (PyObject*)iter_obj;
#endif
}

}  // namespace

static PyMethodDef tv_methods[] = {
    {"filter", DFTU_PYCFUNCTION(tv_filter), METH_O,
     "Keep events matching a query-DSL predicate (AND-combined)."},
    {"query", DFTU_PYCFUNCTION(tv_filter), METH_O, "Alias of filter()."},
    {"phase", DFTU_PYCFUNCTION(tv_phase), METH_O,
     "Select 'events' (ph=X), 'counters' (ph=C), 'aggregated' (ph=A), "
     "'metadata' (ph=M), or 'any'."},
    {"group_by", DFTU_PYCFUNCTION(tv_group_by), METH_VARARGS,
     "Group by keys: name/cat/pid/tid/fhash/arg:<key>."},
    {"agg", DFTU_PYCFUNCTION(tv_agg), METH_VARARGS,
     "Aggregations: count, sum:/min:/max:/mean:/var:/std:<field>, "
     "argmax:<field>:<by>, set_union:<field> (distinct values, one string "
     "column joined by \\x1e)."},
    {"time_bucket", DFTU_PYCFUNCTION(tv_time_bucket),
     METH_VARARGS | METH_KEYWORDS,
     "time_bucket(interval_us, normalize_to=None): bucket events into fixed "
     "intervals. normalize_to aligns bucket boundaries: an int origin, or "
     "'min' "
     "for the trace's minimum timestamp (from the index, no scan); None = "
     "aligned to 0."},
    {"occ_cell", DFTU_PYCFUNCTION(tv_occ_cell), METH_O,
     "Occupancy cell size (busy quantum) in microseconds; 0 = default. Finer "
     "resolves overlap on short events (honored with time_range)."},
    {"time_unit", DFTU_PYCFUNCTION(tv_time_unit), METH_O,
     "Normalize ts/dur to a target unit (ns/us/ms/sec, s=sec); source read "
     "from "
     "the "
     "trace's CM time_metric. Higher-level helper over time_scale()."},
    {"time_scale", DFTU_PYCFUNCTION(tv_time_scale), METH_O,
     "Multiply ts/dur/te by this ratio (source_ns/target_ns; 1.0 = none). The "
     "raw primitive behind time_unit()."},
    {"time_range", DFTU_PYCFUNCTION(tv_time_range), METH_VARARGS,
     "Restrict to a [begin, end) timestamp window."},
    {"select", DFTU_PYCFUNCTION(tv_select), METH_VARARGS, "Project columns."},
    {"memory_budget", DFTU_PYCFUNCTION(tv_memory_budget), METH_O,
     "Spill aggregation above this many bytes (0 = in-memory)."},
    {"auto_spill", DFTU_PYCFUNCTION(tv_auto_spill), METH_NOARGS,
     "Spill at ~1/3 of available memory."},
    {"agg_numeric_args", DFTU_PYCFUNCTION(tv_auto_numeric_args), METH_VARARGS,
     "Aggregate every auto-discovered numeric arg (size, ret, ...). No args = "
     "one bare-named mean column per arg; pass op names (sum/min/max/mean/var/"
     "std/skew/kurt) for one <op>_<arg> column per (arg, op)."},
    {"limit", DFTU_PYCFUNCTION(tv_limit), METH_O, "Cap produced rows/events."},
    {"offset", DFTU_PYCFUNCTION(tv_offset), METH_O,
     "Skip the first N rows/events."},
    {"sort_by", DFTU_PYCFUNCTION(tv_sort_by), METH_VARARGS | METH_KEYWORDS,
     "sort_by(name, descending=False): order the collect() result by a "
     "column (same dataframe kernel as DataFrame.sort_by)."},
    {"topk", DFTU_PYCFUNCTION(tv_topk), METH_VARARGS | METH_KEYWORDS,
     "topk(name, k, largest=True): keep the k best rows of the collect() "
     "result (same dataframe kernel as DataFrame.topk)."},
    {"columns", DFTU_PYCFUNCTION(tv_columns), METH_NOARGS,
     "List the columns discoverable from the index (no trace scan)."},
    {"schema", DFTU_PYCFUNCTION(tv_schema), METH_NOARGS,
     "Map each column to its type (no trace scan)."},
    {"time_metric", DFTU_PYCFUNCTION(tv_time_metric), METH_NOARGS,
     "The trace's native time unit ('us'/'ns'/'ms'/'sec') from the first "
     "file's CM record (head-read only, no scan)."},
    {"collect", DFTU_PYCFUNCTION(tv_collect), METH_NOARGS,
     "Build the group_by+agg plan and return a LazyFrame; nothing scans "
     "until you call .collect() (-> DataFrame) or .to_arrow()/.to_pandas() "
     "on the result."},
    {"call_tree", DFTU_PYCFUNCTION(tv_call_tree), METH_VARARGS,
     "call_tree(partition) -> scan, then the events DataFrame plus "
     "level/parent_id (containment nesting per lane)."},
    {"flamegraph", DFTU_PYCFUNCTION(tv_flamegraph), METH_VARARGS,
     "flamegraph(partition) -> scan, then a folded node DataFrame (node_id, "
     "parent, name, level, total, self, count)."},
    {"containment", DFTU_PYCFUNCTION(tv_containment), METH_VARARGS,
     "containment(partition) -> (call_tree_df, flamegraph_df) from one scan "
     "and "
     "one buffered fold."},
    {"join", DFTU_PYCFUNCTION(tv_join), METH_VARARGS | METH_KEYWORDS,
     "Aggregate and equi-join another viewer on the shared group key; "
     "how=inner|left|right|full|semi|anti. Returns a DataFrame with l_/r_ "
     "prefixed value columns."},
    {"compare", DFTU_PYCFUNCTION(tv_compare), METH_VARARGS | METH_KEYWORDS,
     "Compare this viewer (baseline) against another (variant) using this "
     "viewer's group_by + agg plan. Both sides aggregate in parallel; returns "
     "a DataFrame with the group key, l_/r_ per metric, and delta_/pct_."},
    {"collect_typed", DFTU_PYCFUNCTION(tv_collect_typed),
     METH_VARARGS | METH_KEYWORDS,
     "One-pass read of the aggregation index's three record families over "
     "shard range [shard_begin, shard_end) (shard_end<=0 = all); returns a "
     "dict {'regular','aggregated','counters'} of pyarrow.Table."},
    {"stream", DFTU_PYCFUNCTION(tv_stream), METH_VARARGS | METH_KEYWORDS,
     "Iterate matching events as native DataFrame chunks (parallel, bounded "
     "memory). kwargs: batch_size, workers, normalize."},
    {"_session_execute", DFTU_PYCFUNCTION(tv_session_run), METH_O,
     "Internal: run a Session's branches over one shared scan. Arg: a list of "
     "(kind, viewer, sink) tuples; returns a list of _DataFrame (collect) / "
     "stats dict (export) / None (materialize) in the same order."},
    {"statistics", DFTU_PYCFUNCTION(tv_statistics), METH_NOARGS,
     "One-row summary: count, mean/stddev dur, min/max ts (dict)."},
    {"aggregate_partial", DFTU_PYCFUNCTION(tv_aggregate_partial), METH_NOARGS,
     "Combinable aggregation partial (bytes) for distributed merge."},
    {"merge_partials_to_table", DFTU_PYCFUNCTION(tv_merge_partials), METH_O,
     "Merge aggregate_partial() bytes into the final pyarrow.Table."},
    {"flamegraph_partial", DFTU_PYCFUNCTION(tv_flamegraph_partial),
     METH_VARARGS,
     "flamegraph_partial(partition) -> serialized flamegraph arena (bytes) for "
     "a distributed merge (combine with _ext.merge_flamegraph_partials)."},
    {"rollup_root", DFTU_PYCFUNCTION(tv_rollup_root), METH_O,
     "Override the aggregation-cache root dir (default: derive from index)."},
    {"views_root", DFTU_PYCFUNCTION(tv_views_root), METH_O,
     "Override the materialized-view root dir (default: derive "
     "<parent-of-index>/.dftindex-views)."},
    {"materialize", DFTU_PYCFUNCTION(tv_materialize),
     METH_VARARGS | METH_KEYWORDS,
     "Build-only: persist this query as a materialized view so a later "
     "matching read reuses it (row query -> filtered trace, aggregation -> "
     "rollup). kwargs: checkpoint_size, part_size, progress (called with "
     "(done, total) scan units). Returns None."},
    {"mv_source", DFTU_PYCFUNCTION(tv_mv_source), METH_NOARGS,
     "The materialized-view trace file(s) that would serve this query, or an "
     "empty list if a read would scan the base."},
    {"materialize_dir", DFTU_PYCFUNCTION(tv_materialize_dir), METH_NOARGS,
     "Distributed row-MV coordinator: create and return the shared MV dir for "
     "this full-file-set view. Ranks export into subdirs of it."},
    {"register_materialized", DFTU_PYCFUNCTION(tv_register_materialized),
     METH_O,
     "Write the MV manifest at the given dir over this view's base set, after "
     "ranks have materialized their shard subdirs."},
    {"export_trace", DFTU_PYCFUNCTION(tv_export), METH_VARARGS | METH_KEYWORDS,
     "Write a dftracer trace: the aggregation when group_by/agg is set, else "
     "the matching events. gzip+re-indexable by default. kwargs: compress, "
     "index, member_size, level, part_size."},
    {nullptr, nullptr, 0, nullptr}};

// AggregatedTraceViewer adds the distributed rollup terminals; collect() and
// every builder op / other terminal are inherited from TraceViewer via tp_base.
// A plain collect() already reads a subsuming rollup, so there is no cache
// flag.
static PyMethodDef atv_methods[] = {
    {"materialize_partials", DFTU_PYCFUNCTION(tv_materialize_partials), METH_O,
     "Materialize the rollup from aggregate_partial() bytes; no rescan."},
    {"reconstruct_if_cached", DFTU_PYCFUNCTION(tv_reconstruct_if_cached),
     METH_NOARGS,
     "Materialized aggregation as a pyarrow.Table, or None on a miss."},
    {nullptr, nullptr, 0, nullptr}};

PyTypeObject TraceViewerType = {
    PyVarObject_HEAD_INIT(nullptr, 0) "dftracer_utils_ext.TraceViewer",
    sizeof(TraceViewerObject),
    0,
    (destructor)tv_dealloc,
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
    Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    "Arrow-first composable view over a trace.",
    0,
    0,
    0,
    0,
    0,
    0,
    tv_methods,
    0,
    0,
    0,
    0,
    0,
    0,
    0,
    (initproc)tv_init,
    0,
    tv_new,
};

// Same object layout and builders as TraceViewer (tp_base), plus the cache
// terminals. group_by/agg/agg_numeric_args return this type.
PyTypeObject AggregatedTraceViewerType = {
    PyVarObject_HEAD_INIT(nullptr,
                          0) "dftracer_utils_ext.AggregatedTraceViewer",
    sizeof(TraceViewerObject),
    0,
    0,  // tp_dealloc inherited
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
    "A TraceViewer with a group_by/agg; adds the materialized-view cache.",
    0,
    0,
    0,
    0,
    0,
    0,
    atv_methods,
    0,
    0,
    &TraceViewerType,  // tp_base
    0,
    0,
    0,
    0,
    0,  // tp_init inherited
    0,
    0,  // tp_new inherited
};

int dftracer::utils::python::init_trace_viewer(PyObject* m) {
    if (register_type(m, &TraceViewerType, "_TraceViewer") < 0) return -1;
    if (register_type(m, &AggregatedTraceViewerType, "_AggregatedTraceViewer") <
        0)
        return -1;

    // A pure reduce (no scan / viewer state), so it is a module function, not a
    // viewer method. tv_merge_flamegraph_partials ignores its first argument.
    static PyMethodDef merge_fg_def = {
        "merge_flamegraph_partials",
        DFTU_PYCFUNCTION(tv_merge_flamegraph_partials), METH_O,
        "merge_flamegraph_partials(partials) -> node DataFrame (no scan)."};
    PyObject* fn = PyCFunction_NewEx(&merge_fg_def, nullptr, nullptr);
    if (!fn) return -1;
    if (PyModule_AddObject(m, "merge_flamegraph_partials", fn) < 0) {
        Py_DECREF(fn);
        return -1;
    }
    return 0;
}
