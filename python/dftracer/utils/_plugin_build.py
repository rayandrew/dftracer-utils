"""Shared compile backend for DFTracer plugins: resolve the ABI include dir,
select a compiler, and build a source file to a loadable shared object.

Both the ``@jit.plugin`` backend and the ``dftracer_plugin`` console-script use
this. Include-dir resolution order: (1) ``DFTRACER_PLUGIN_INCLUDE``, (2) headers
bundled in the installed package (``<pkg>/include``), (3) the source-tree parent
walk (dev checkout).
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


class PluginBuildError(Exception):
    """Include-dir resolution failed or the compiler rejected the source."""


def include_dir() -> str:
    """Resolve the directory that contains ``dftracer/utils/plugins/abi.h``."""
    env = os.environ.get("DFTRACER_PLUGIN_INCLUDE")
    if env:
        return env
    rel = Path("dftracer") / "utils" / "plugins" / "abi.h"
    bundled = Path(__file__).resolve().parent / "include"
    if (bundled / rel).is_file():
        return str(bundled)
    src_rel = Path("include") / rel
    for base in Path(__file__).resolve().parents:
        if (base / src_rel).is_file():
            return str(base / "include")
    raise PluginBuildError("cannot locate the plugin include dir; set DFTRACER_PLUGIN_INCLUDE")


def compiler() -> str:
    return os.environ.get("CXX") or shutil.which("c++") or shutil.which("clang++") or "c++"


def cflags() -> List[str]:
    """Compile flags a plugin needs: C++20, position-independent, shared."""
    flags = ["-std=c++20", "-fPIC", "-shared", f"-I{include_dir()}"]
    if sys.platform == "darwin":
        flags += ["-undefined", "dynamic_lookup"]
    return flags


def cache_dir() -> Path:
    env = os.environ.get("DFTRACER_JIT_CACHE")
    base = Path(env) if env else Path.home() / ".cache" / "dftracer-utils" / "jit"
    base.mkdir(parents=True, exist_ok=True)
    return base


def _abi_fingerprint(include: str) -> str:
    # A compiled plugin's struct layout depends on the ABI headers, so a change
    # to them must invalidate the cache - the emitted source text alone would
    # not, silently reusing a .so built against an incompatible layout.
    h = hashlib.sha256()
    base = Path(include) / "dftracer" / "utils"
    for rel in ("plugins/abi.h", "dataframe/abi.h"):
        try:
            h.update((base / rel).read_bytes())
        except OSError:
            h.update(b"\0")
    return h.hexdigest()[:16]


def source_digest(source: str, include: str, cxx: str) -> str:
    return hashlib.sha256(
        "\0".join(
            [source, include, cxx, sys.platform, _abi_fingerprint(include)]
        ).encode("utf-8")
    ).hexdigest()[:16]


def build_shared(src: str, out: str | None = None, name: str | None = None) -> str:
    """Compile the source file ``src`` to a loadable ``.so`` and return its path.

    With ``out`` unset, the output is a content-hashed path in the cache dir
    (``name`` sets its basename, defaulting to ``src``'s stem) so an unchanged
    source never recompiles. Raises :class:`PluginBuildError` on compile failure.
    """
    src_path = Path(src)
    cxx = compiler()
    inc = include_dir()
    if out is None:
        source = src_path.read_text(encoding="utf-8")
        digest = source_digest(source, inc, cxx)
        out_path = cache_dir() / f"{name or src_path.stem}.{digest}.so"
        if out_path.is_file():
            return str(out_path)
    else:
        out_path = Path(out)
    cmd = [cxx, *cflags(), "-o", str(out_path), str(src_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise PluginBuildError(proc.stderr)
    return str(out_path)
