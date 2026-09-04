#include <dftracer/utils/core/common/config.h>  // DFTRACER_UTILS_ENABLE_ARROW

#ifdef DFTRACER_UTILS_ENABLE_ARROW

#include <dftracer/utils/python/series_detail.h>

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace dftracer::utils::python::series_detail {

namespace {

// Read a Python str as a string_view; the buffer stays owned by `arg`.
bool as_str_view(PyObject* arg, std::string_view* out) {
    Py_ssize_t len = 0;
    const char* data = PyUnicode_AsUTF8AndSize(arg, &len);
    if (!data) return false;
    *out = std::string_view(data, static_cast<std::size_t>(len));
    return true;
}

// A single-character pad string -> its first byte; empty is rejected.
bool as_fill_char(PyObject* arg, char* out) {
    std::string_view s;
    if (!as_str_view(arg, &s)) return false;
    if (s.empty()) {
        PyErr_SetString(PyExc_ValueError, "fill must be a non-empty string");
        return false;
    }
    *out = s[0];
    return true;
}

}  // namespace

PyObject* Series_str_eq(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view rhs;
    if (!as_str_view(arg, &rhs)) return nullptr;
    return make_series(a->str_eq(rhs));
}
PyObject* Series_str_contains(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view needle;
    if (!as_str_view(arg, &needle)) return nullptr;
    return make_series(a->str_contains(needle));
}
PyObject* Series_str_starts_with(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view prefix;
    if (!as_str_view(arg, &prefix)) return nullptr;
    return make_series(a->str_starts_with(prefix));
}
PyObject* Series_str_ends_with(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view suffix;
    if (!as_str_view(arg, &suffix)) return nullptr;
    return make_series(a->str_ends_with(suffix));
}
PyObject* Series_str_matches(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view pattern;
    if (!as_str_view(arg, &pattern)) return nullptr;
    return make_series(a->str_matches(pattern));
}
PyObject* Series_str_like(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view pattern;
    if (!as_str_view(arg, &pattern)) return nullptr;
    return make_series(a->str_like(pattern));
}
PyObject* Series_str_len_bytes(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_len_bytes());
}
PyObject* Series_str_len_chars(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_len_chars());
}
PyObject* Series_str_find(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view needle;
    if (!as_str_view(arg, &needle)) return nullptr;
    return make_series(a->str_find(needle));
}
PyObject* Series_to_lowercase(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->to_lowercase());
}
PyObject* Series_to_uppercase(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->to_uppercase());
}
PyObject* Series_str_strip(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_strip());
}
PyObject* Series_str_lstrip(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_lstrip());
}
PyObject* Series_str_rstrip(PyObject* self, PyObject*) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    return make_series(a->str_rstrip());
}
PyObject* Series_str_replace(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* pat = nullptr;
    PyObject* repl = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &pat, &repl)) return nullptr;
    std::string_view p, r;
    if (!as_str_view(pat, &p) || !as_str_view(repl, &r)) return nullptr;
    return make_series(a->str_replace(p, r));
}
PyObject* Series_str_replace_all(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    PyObject* pat = nullptr;
    PyObject* repl = nullptr;
    if (!PyArg_ParseTuple(args, "OO", &pat, &repl)) return nullptr;
    std::string_view p, r;
    if (!as_str_view(pat, &p) || !as_str_view(repl, &r)) return nullptr;
    return make_series(a->str_replace_all(p, r));
}
PyObject* Series_str_slice(PyObject* self, PyObject* args) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long start = 0;
    long long length = -1;
    if (!PyArg_ParseTuple(args, "L|L", &start, &length)) return nullptr;
    return make_series(a->str_slice(static_cast<std::int64_t>(start),
                                    static_cast<std::int64_t>(length)));
}
PyObject* Series_str_pad_start(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long width = 0;
    PyObject* fill = nullptr;
    static const char* kwlist[] = {"width", "fill", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|O",
                                     const_cast<char**>(kwlist), &width, &fill))
        return nullptr;
    char f = ' ';
    if (fill && !as_fill_char(fill, &f)) return nullptr;
    return make_series(a->str_pad_start(static_cast<std::int64_t>(width), f));
}
PyObject* Series_str_pad_end(PyObject* self, PyObject* args, PyObject* kwds) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long width = 0;
    PyObject* fill = nullptr;
    static const char* kwlist[] = {"width", "fill", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "L|O",
                                     const_cast<char**>(kwlist), &width, &fill))
        return nullptr;
    char f = ' ';
    if (fill && !as_fill_char(fill, &f)) return nullptr;
    return make_series(a->str_pad_end(static_cast<std::int64_t>(width), f));
}
PyObject* Series_str_zfill(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    long long width = PyLong_AsLongLong(arg);
    if (width == -1 && PyErr_Occurred()) return nullptr;
    return make_series(a->str_zfill(static_cast<std::int64_t>(width)));
}
PyObject* Series_str_split(PyObject* self, PyObject* arg) {
    Series* a = as_series(self);
    if (!a) return nullptr;
    std::string_view sep;
    if (!as_str_view(arg, &sep)) return nullptr;
    return make_series(a->str_split(sep));
}

}  // namespace dftracer::utils::python::series_detail

#endif  // DFTRACER_UTILS_ENABLE_ARROW
