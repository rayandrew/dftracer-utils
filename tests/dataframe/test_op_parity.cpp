// dftu_op_count()/dftu_op_at() must contain exactly the rows in
// exported_series_ops.def and exported_frame_ops.def, plus the fixed set
// host_ops.cpp registers at runtime - no more, no less.
#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <dftracer/utils/dataframe/op.h>
#include <doctest/doctest.h>

#include <cstdint>
#include <initializer_list>
#include <set>
#include <string>

using dftracer::utils::dataframe::op_at;
using dftracer::utils::dataframe::op_count;
using dftracer::utils::dataframe::OpKind;

namespace {

std::set<std::string> series_def_names() {
    std::set<std::string> names;
#define DFTU_SERIES_OP(name, fn, ret, o0, o1, o2) names.insert(#name);
#include <dftracer/utils/dataframe/exported_series_ops.def>
    return names;
}

std::set<std::string> frame_def_names() {
    std::set<std::string> names;
#define DFTU_FRAME_OP(name, fn, ret, o0, o1, o2, o3, o4) \
    names.insert("dftu.frame." #name);
#include <dftracer/utils/dataframe/exported_frame_ops.def>
    return names;
}

// Registered by utilities/host_ops.cpp outside the .def tables (I/O ops the
// columnar engine itself has no business depending on, per
// exported_series_ops.def's own header comment). Shrinks only if one of
// these moves into the .def; grows only with a matching host_ops.cpp change.
const std::set<std::string>& runtime_registered_series_ops() {
    static const std::set<std::string> names = {
        "dftu.fs.scan_dir",      "dftu.fs.scan_dir_pattern",
        "dftu.file.compress",    "dftu.file.decompress",
        "dftu.text.line_filter",
    };
    return names;
}

std::string joined(const std::set<std::string>& names) {
    std::string out;
    for (const auto& n : names) {
        if (!out.empty()) out += ", ";
        out += n;
    }
    return out;
}

bool kind_in(OpKind k, std::initializer_list<OpKind> kinds) {
    for (OpKind want : kinds) {
        if (k == want) return true;
    }
    return false;
}

void check_bijection(const std::set<std::string>& def_names,
                     const std::set<std::string>& known_extra,
                     std::initializer_list<OpKind> kinds) {
    std::set<std::string> registered;
    std::uint32_t n = op_count();
    for (std::uint32_t i = 0; i < n; ++i) {
        auto op = op_at(i);
        REQUIRE(static_cast<bool>(op));
        if (kind_in(op.sig().kind(), kinds)) {
            registered.insert(std::string(op.name()));
        }
    }

    std::set<std::string> missing_from_registry;
    for (const auto& name : def_names) {
        if (registered.count(name) == 0) missing_from_registry.insert(name);
    }
    std::set<std::string> unexpected_in_registry;
    for (const auto& name : registered) {
        if (def_names.count(name) != 0) continue;
        if (known_extra.count(name) != 0) continue;
        unexpected_in_registry.insert(name);
    }

    INFO(def_names.size() << " .def rows, " << registered.size()
                          << " registered, " << known_extra.size()
                          << " known runtime-registered extras");
    if (!missing_from_registry.empty()) {
        FAIL(".def op(s) not in the registry: "
             << joined(missing_from_registry));
    }
    if (!unexpected_in_registry.empty()) {
        FAIL(
            "registry op(s) not backed by a .def row or the runtime allowlist: "
            << joined(unexpected_in_registry));
    }
}

}  // namespace

TEST_SUITE("op_parity") {
    TEST_CASE(
        "every exported_series_ops.def row is registered, and vice versa") {
        check_bijection(series_def_names(), runtime_registered_series_ops(),
                        {OpKind::Series, OpKind::Aggregate});
    }

    TEST_CASE(
        "every exported_frame_ops.def row is registered, and vice versa") {
        check_bijection(frame_def_names(), {}, {OpKind::Frame});
    }
}
