"""jit.config: a runtime-configurable @jit.plugin. The threshold is bound at
construction from Plugins config and read in the body as self.threshold."""

import gzip
import shutil

import pytest

from dftracer.utils import jit
from dftracer.utils.plugins import Plugins

pa = pytest.importorskip("pyarrow")
_HAS_CXX = bool(shutil.which("c++") or shutil.which("clang++") or shutil.which("g++"))
_needs_cxx = pytest.mark.skipif(not _HAS_CXX, reason="no C++ compiler for the jit backend")


def _write(path, durs):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for i, d in enumerate(durs):
            f.write(
                f'{{"name":"read","cat":"POSIX","pid":1,"tid":1,'
                f'"ts":{1000 + i},"dur":{d},"ph":"X","args":{{}}}}\n'
            )


def _count(plugins, path):
    return int(
        pa.table(plugins.run(path).results["hits"])
        .column("value")
        .to_numpy(zero_copy_only=False)
        .sum()
    )


@_needs_cxx
def test_jit_config_threshold_drives_behavior(tmp_path):
    @jit.plugin
    class Slow:
        threshold = jit.config(jit.i64)
        hits = jit.map(key=(jit.i64,), value=jit.count())

        @jit.each_event
        def step(self, e):
            if e.dur > self.threshold:
                self.hits[(e.pid,)] += 1

    _write(str(tmp_path / "t.pfw.gz"), [10, 20, 30, 40, 50, 60, 70])

    hi_plugins = Plugins([Slow], config={"Slow": {"threshold": 45}})
    assert _count(hi_plugins, str(tmp_path)) == 3  # 50, 60, 70

    lo_plugins = Plugins([Slow], config={"Slow": {"threshold": 5}})
    assert _count(lo_plugins, str(tmp_path)) == 7  # all


class TestAuthoring:
    def test_string_config_rejected(self):
        with pytest.raises(jit.JitError):
            jit.config(jit.str_)
