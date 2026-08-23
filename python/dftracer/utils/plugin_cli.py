"""``dftracer_plugin`` console-script: scaffold and build native plugins."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from . import _plugin_build


def _ident(name: str) -> str:
    s = "".join(c if c.isalnum() else "_" for c in name)
    if not s or s[0].isdigit():
        s = "P" + s
    return s


def _c_template(name: str) -> str:
    return f"""#include <dftracer/utils/plugins/abi.h>

#include <stdint.h>
#include <stdlib.h>

static uint32_t needs(void* self) {{
    (void)self;
    return 0;
}}

static void* make_slice(void* self) {{
    (void)self;
    return calloc(1, 1);
}}

static dftu_task* on_batch(void* slice, const dftu_batch* b,
                          const dftu_host* host) {{
    const dftu_ext_map* map =
        (const dftu_ext_map*)host->get_extension(host->h, DFTU_EXT_MAP);
    static const dftu_type key_types[1] = {{DFTU_T_I64}};
    dftu_map* m;
    uint32_t i;
    (void)slice;
    if (!map || !map->map_new) return NULL;
    m = map->map_new(host->h, "{name}", key_types, 1, DFTU_MONOID_COUNTER);
    if (!m) return NULL;
    for (i = 0; i < b->count; ++i) {{
        const dftu_event* e = &b->events[i];
        int64_t key[1];
        key[0] = (int64_t)e->pid;
        map->map_add_u64(host->h, m, key, 1);
    }}
    return NULL;
}}

static void merge(void* into, void* other) {{
    (void)into;
    (void)other;
}}

static dftu_task* on_finalize(void* slice, const dftu_host* host) {{
    (void)slice;
    (void)host;
    return NULL;
}}

static void destroy_slice(void* slice) {{ free(slice); }}

static void destroy(void* self) {{ (void)self; }}

static dftu_plugin g_plugin;

#ifdef __cplusplus
extern "C"
#endif
dftu_plugin* dftracer_plugin(const dftu_value* config) {{
    (void)config;
    g_plugin.abi_version = DFTRACER_PLUGIN_ABI_VERSION;
    g_plugin.self = NULL;
    g_plugin.needs = needs;
    g_plugin.plan_query = NULL;
    g_plugin.make_slice = make_slice;
    g_plugin.on_batch = on_batch;
    g_plugin.merge = merge;
    g_plugin.on_finalize = on_finalize;
    g_plugin.destroy_slice = destroy_slice;
    g_plugin.destroy = destroy;
    return &g_plugin;
}}
"""


def _cpp_template(name: str) -> str:
    cls = _ident(name)
    return f"""#include <dftracer/utils/plugins/plugin.h>

#include <cstdint>

using namespace dftracer::utils::plugins;

struct {cls} {{
    explicit {cls}(const Config&) {{}}

    void step(const dftu_batch& b, Host host) {{
        dftu_map* m = host.map_new("{name}", {{DFTU_T_I64}}, DFTU_MONOID_COUNTER);
        if (!m) return;
        for (std::uint32_t i = 0; i < b.count; ++i) {{
            const dftu_event& e = b.events[i];
            host.map_add_u64(m, {{static_cast<std::int64_t>(e.pid)}}, 1);
        }}
    }}

    void merge({cls}&) {{}}
    void finalize(Host) {{}}
}};

extern "C" dftu_plugin* dftracer_plugin(const dftu_value* config) {{
    return make_plugin<{cls}>(config);
}}
"""


def _cmd_new(args: argparse.Namespace) -> int:
    ext = "cpp" if args.cpp else "c"
    out_dir = Path(args.output) if args.output else Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.name}.{ext}"
    source = _cpp_template(args.name) if args.cpp else _c_template(args.name)
    path.write_text(source, encoding="utf-8")
    print(str(path))
    return 0


def _cmd_build(args: argparse.Namespace) -> int:
    src = Path(args.src)
    if not src.is_file():
        print(f"dftracer_plugin: no such file: {src}", file=sys.stderr)
        return 1
    out = args.output or str(src.with_suffix(".so"))
    try:
        so = _plugin_build.build_shared(str(src), out=out)
    except _plugin_build.PluginBuildError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(so)
    return 0


def _cmd_cflags(args: argparse.Namespace) -> int:
    print(" ".join(_plugin_build.cflags()))
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="dftracer_plugin", description="Scaffold and build native DFTracer plugins."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_new = sub.add_parser("new", help="scaffold a template plugin source")
    p_new.add_argument("name", help="plugin (and result) name")
    p_new.add_argument("--cpp", action="store_true", help="C++ template against plugin.h")
    p_new.add_argument("-o", "--output", metavar="DIR", help="output directory (default: cwd)")
    p_new.set_defaults(func=_cmd_new)

    p_build = sub.add_parser("build", help="compile a plugin source to a loadable .so")
    p_build.add_argument("src", help="plugin source file (.c or .cpp)")
    p_build.add_argument("-o", "--output", metavar="OUT", help="output .so path")
    p_build.set_defaults(func=_cmd_build)

    p_cflags = sub.add_parser("cflags", help="print the plugin compile flags")
    p_cflags.set_defaults(func=_cmd_cflags)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
