#!/usr/bin/env python3
"""Checks Series/DataFrame/LazyFrame public methods against the op registry.

Reads C++ method names from the Doxygen XML the docs build emits
(docs/Doxyfile: GENERATE_XML = YES), never a header regex - Doxygen already
tells a declaration from a comment or a disabled #if branch. Not wired into
the default build or ctest: this needs docs/doxygen/xml to already exist,
which `cmake --build --preset tests` does not generate. Run it after
`doxygen docs/Doxyfile` (or from a docs CI job):

    scripts/check_op_parity_doxygen.py docs/doxygen/xml \\
        build/build-tests/lib/libdftracer_utils_dataframe.dylib

Reads the registered ops straight out of that shared library with ctypes
(dftu_op_count/dftu_op_at are a plain C ABI), so no helper binary is needed.

Exits non-zero and lists the drifted names when a public method has no op of
the same leaf name in the registry and is not in the allowlist below.
"""

import argparse
import ctypes
import re
import sys
from pathlib import Path

CLASSES = {
    "Series": "classdftracer_1_1utils_1_1dataframe_1_1_series.xml",
    "DataFrame": "structdftracer_1_1utils_1_1dataframe_1_1_data_frame.xml",
    "LazyFrame": "classdftracer_1_1utils_1_1dataframe_1_1_lazy_frame.xml",
}

# Which registry bucket (see dftu_op_kind in abi.h) backs each class.
REGISTRY_BUCKET = {"Series": "series", "DataFrame": "frame", "LazyFrame": "lazy"}

MEMBER_RE = re.compile(r'<memberdef kind="function"[^>]*>.*?</memberdef>', re.S)
PROT_RE = re.compile(r'prot="([a-z]+)"')
NAME_RE = re.compile(r"<name>([^<]*)</name>")

# Debt list: a public method name with no registry op of the same leaf name.
# Should only shrink - either the op gets registered under this leaf name, or
# a future version of this script understands the fold/rename that maps it to
# an existing op.
ALLOWLIST = {
    "Series": {
        "eq", "ne", "lt", "le", "gt", "ge",
        "sum", "mean", "median", "max", "min",
        "logical_and", "logical_or",
        "hex64_parse", "hex64_format",
        "type", "encoding", "length", "null_count", "is_null", "data",
        "values", "is_flat", "offsets", "offsets_span", "valid", "handle",
        "release", "child", "num_children", "list", "strings", "structs",
        "nulls", "flat", "flat_i64", "flat_f64", "string_at", "share",
        "materialize",
        "from_arrow", "from_borrowed", "to_arrow",
        "slice", "take", "sample", "value_counts",
    },
    "DataFrame": {
        "column", "column_index", "num_columns", "num_rows",
        "is_unique", "is_duplicated",
        "from_arrow", "to_arrow", "to_ipc",
        "stream", "lazy",
        "group_by", "group_by_dynamic", "mask", "melt", "sample", "take",
    },
    "LazyFrame": {
        "collect", "collect_group_state", "explain", "group_by_dynamic",
        "scan", "schema", "stream", "take",
    },
}


class OpDesc(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("sig", ctypes.c_int32),
        ("fn", ctypes.c_void_p),
    ]


def public_methods(xml_path: Path, class_name: str) -> set[str]:
    xml = xml_path.read_text()
    names = set()
    for block in MEMBER_RE.findall(xml):
        prot = PROT_RE.search(block)
        if not prot or prot.group(1) != "public":
            continue
        name_m = NAME_RE.search(block)
        if not name_m:
            continue
        name = name_m.group(1)
        if name in (class_name, "~" + class_name) or name.startswith("operator"):
            continue
        names.add(name)
    return names


def registry_leaf_names(lib_path: Path) -> dict[str, set[str]]:
    lib = ctypes.CDLL(str(lib_path))
    lib.dftu_op_count.restype = ctypes.c_uint32
    lib.dftu_op_at.restype = ctypes.POINTER(OpDesc)
    lib.dftu_op_at.argtypes = [ctypes.c_uint32]
    lib.dftu_op_kind_of.restype = ctypes.c_int
    lib.dftu_op_kind_of.argtypes = [ctypes.c_int32]

    DFTU_OP_KIND_SERIES, DFTU_OP_KIND_AGGREGATE = 0, 1
    DFTU_OP_KIND_FRAME, DFTU_OP_KIND_LAZY = 2, 3

    by_bucket: dict[str, set[str]] = {"series": set(), "frame": set(), "lazy": set()}
    for i in range(lib.dftu_op_count()):
        desc = lib.dftu_op_at(i).contents
        leaf = desc.name.decode().rsplit(".", 1)[-1]
        kind = lib.dftu_op_kind_of(desc.sig)
        if kind in (DFTU_OP_KIND_SERIES, DFTU_OP_KIND_AGGREGATE):
            by_bucket["series"].add(leaf)
        elif kind == DFTU_OP_KIND_FRAME:
            by_bucket["frame"].add(leaf)
        elif kind == DFTU_OP_KIND_LAZY:
            by_bucket["lazy"].add(leaf)
    return by_bucket


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doxygen_xml_dir", type=Path)
    ap.add_argument("dataframe_library", type=Path,
                    help="the built libdftracer_utils_dataframe shared library")
    args = ap.parse_args()

    registered = registry_leaf_names(args.dataframe_library)

    failed = False
    for class_name, xml_name in CLASSES.items():
        xml_path = args.doxygen_xml_dir / xml_name
        if not xml_path.is_file():
            print(f"missing {xml_path}", file=sys.stderr)
            return 1
        names = public_methods(xml_path, class_name)
        bucket = REGISTRY_BUCKET[class_name]
        reg = registered[bucket] if bucket else set()
        drifted = sorted(
            n for n in names if n not in reg and n not in ALLOWLIST[class_name]
        )
        print(f"{class_name}: {len(names)} methods, {len(reg)} registered, "
             f"{len(ALLOWLIST[class_name])} allowlisted")
        if drifted:
            failed = True
            print(f"  DRIFT: {', '.join(drifted)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
