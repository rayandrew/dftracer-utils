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
the same leaf name in the registry, no entry in FOLDS naming the op that backs
it under another name, and no entry in the ALLOWLIST of non-ops below.
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

# Methods a registered op backs under a different leaf name: an op code folds
# a family of methods onto one op (compare, logical, reduce), or the registry
# uses the canonical name for a method spelled as its alias. The value is the
# registry leaf, looked up across every bucket - Series::value_counts returns a
# frame, so its op is keyed dftu.frame.value_counts.
FOLDS = {
    "Series": {
        "eq": "compare", "ne": "compare", "lt": "compare",
        "le": "compare", "gt": "compare", "ge": "compare",
        "logical_and": "logical", "logical_or": "logical",
        "hex64_parse": "parse64", "hex64_format": "format64",
        "sum": "reduce", "mean": "reduce", "median": "reduce",
        "max": "reduce", "min": "reduce",
        "value_counts": "value_counts",
    },
    "DataFrame": {"melt": "unpivot"},
    "LazyFrame": {},
}

# Methods that are not ops and never will be: accessors into a handle, the
# Arrow/IPC boundary, and the lazy terminals that end a plan instead of
# extending it.
ALLOWLIST = {
    "Series": {
        "type", "encoding", "length", "null_count", "is_null", "data",
        "values", "is_flat", "offsets", "offsets_span", "valid", "handle",
        "release", "child", "num_children", "list", "strings", "structs",
        "nulls", "flat", "flat_i64", "flat_f64", "string_at", "share",
        "from_arrow", "from_borrowed", "to_arrow",
    },
    "DataFrame": {
        "column", "column_index", "num_columns", "num_rows",
        "from_arrow", "to_arrow", "to_ipc",
        "stream", "lazy",
    },
    "LazyFrame": {
        "collect", "collect_group_state", "explain",
        "scan", "schema", "stream",
    },
}


class OpDesc(ctypes.Structure):
    _fields_ = [
        ("name", ctypes.c_char_p),
        ("sig", ctypes.c_int64),
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
    # Bucketed by registry name prefix, not by dftu_op_kind: kind comes from
    # the return token, so a frame -> series op (mask, is_unique) is kind
    # SERIES while still being a DataFrame method. The prefix is what says
    # which class owns the op.
    lib = ctypes.CDLL(str(lib_path))
    lib.dftu_op_count.restype = ctypes.c_uint32
    lib.dftu_op_at.restype = ctypes.POINTER(OpDesc)
    lib.dftu_op_at.argtypes = [ctypes.c_uint32]

    by_bucket: dict[str, set[str]] = {"series": set(), "frame": set(), "lazy": set()}
    for i in range(lib.dftu_op_count()):
        name = lib.dftu_op_at(i).contents.name.decode()
        leaf = name.rsplit(".", 1)[-1]
        if name.startswith("dftu.frame."):
            by_bucket["frame"].add(leaf)
        elif name.startswith("dftu.lazy."):
            by_bucket["lazy"].add(leaf)
        else:
            # dftu.series.* plus the host utility families (dftu.hash.*,
            # dftu.hex.*), which are reached as Series methods.
            by_bucket["series"].add(leaf)
    return by_bucket


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("doxygen_xml_dir", type=Path)
    ap.add_argument("dataframe_library", type=Path,
                    help="the built libdftracer_utils_dataframe shared library")
    args = ap.parse_args()

    registered = registry_leaf_names(args.dataframe_library)
    every_leaf = set().union(*registered.values())

    failed = False
    for class_name, xml_name in CLASSES.items():
        xml_path = args.doxygen_xml_dir / xml_name
        if not xml_path.is_file():
            print(f"missing {xml_path}", file=sys.stderr)
            return 1
        names = public_methods(xml_path, class_name)
        reg = registered[REGISTRY_BUCKET[class_name]]
        folds = FOLDS[class_name]
        drifted = sorted(
            n for n in names
            if n not in reg
            and folds.get(n) not in every_leaf
            and n not in ALLOWLIST[class_name]
        )
        print(f"{class_name}: {len(names)} methods, {len(reg)} registered, "
              f"{len(folds)} folded, {len(ALLOWLIST[class_name])} allowlisted")
        if drifted:
            failed = True
            print(f"  DRIFT: {', '.join(drifted)}")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
