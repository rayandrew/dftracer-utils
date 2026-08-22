"""Author a compose op (``dftu_op``) in Python and JIT-compile it to a native
plugin.

``@op`` turns a typed value transform into a compose leaf; ``|`` pipes them,
mirroring the C++ typed ``Op<In,Out>`` over the ``dftu_ext_compose`` ABI. Calling
an op JIT-compiles it and drives the ``dftu_op`` graph on a standalone compose
host (no plugin, no scan); the same op referenced from a ``@jit.each_event``
body is inlined into the plugin as a static C function instead.

Example::

    @op
    def double(x: jit.i64) -> jit.i64:
        return x * 2

    @op
    def plus10(x: jit.i64) -> jit.i64:
        return x + 10

    (double | plus10)(5)  # 5*2+10 = 20
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from typing import Any, Dict, List

from . import _plugin_build

__all__ = ["op", "compile_op", "run_op", "Op", "JitOpError"]

# POD value types that cross the compose ABI as bytes, with their C type + size.
_CTYPE = {
    "DFTU_T_I64": "int64_t",
    "DFTU_T_I32": "int32_t",
    "DFTU_T_I16": "int16_t",
    "DFTU_T_I8": "int8_t",
    "DFTU_T_U64": "uint64_t",
    "DFTU_T_U32": "uint32_t",
    "DFTU_T_U16": "uint16_t",
    "DFTU_T_U8": "uint8_t",
    "DFTU_T_F64": "double",
    "DFTU_T_F32": "float",
}
_SIZE = {
    "DFTU_T_I64": 8,
    "DFTU_T_I32": 4,
    "DFTU_T_I16": 2,
    "DFTU_T_I8": 1,
    "DFTU_T_U64": 8,
    "DFTU_T_U32": 4,
    "DFTU_T_U16": 2,
    "DFTU_T_U8": 1,
    "DFTU_T_F64": 8,
    "DFTU_T_F32": 4,
}
_FLOAT = {"DFTU_T_F64", "DFTU_T_F32"}
_ARITH = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.Mod: "%",
}


class JitOpError(Exception):
    """A compose-op body outside the supported subset (arithmetic on the input)."""


class Op:
    """A composable value transform, one or more leaf stages piped in order."""

    def __init__(self, stages: List[Dict[str, Any]]):
        self.stages = stages

    @property
    def in_type(self) -> str:
        return self.stages[0]["in"]

    @property
    def out_type(self) -> str:
        return self.stages[-1]["out"]

    def __or__(self, other: "Op") -> "Op":
        if not isinstance(other, Op):
            return NotImplemented
        if self.out_type != other.in_type:
            raise JitOpError(
                f"cannot pipe {self.out_type} -> {other.in_type}: the middle type must chain"
            )
        return Op(self.stages + other.stages)

    def __call__(self, value: Any) -> Any:
        """Run this op over `value` in plain Python: JIT-compile it and drive the
        graph on a standalone compose host, returning the output value.

        Inside a ``@jit.each_event`` body an op call is inlined into the plugin
        at compile time instead - the body is AST-compiled, never executed, so
        this Python path never runs there."""
        return run_op(self, value)


def _type_of(t: Any) -> str:
    dft = getattr(t, "dft", None)
    if dft not in _CTYPE:
        raise JitOpError(f"unsupported op value type {t!r}; use jit.i64/i32/u64/f64/...")
    return dft


def _lower(node: ast.expr, var: str) -> str:
    """Lower an arithmetic expression on the single input `var` to C, with the
    input bound to the local `x`."""
    if isinstance(node, ast.BinOp):
        sym = _ARITH.get(type(node.op))
        if sym is None:
            raise JitOpError("only + - * / % are allowed in an op body")
        return f"({_lower(node.left, var)} {sym} {_lower(node.right, var)})"
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        return f"(-{_lower(node.operand, var)})"
    if isinstance(node, ast.Name):
        if node.id != var:
            raise JitOpError(f"unknown name {node.id!r}; an op body uses only its input {var!r}")
        return "x"
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, (int, float))
        and not isinstance(node.value, bool)
    ):
        return repr(node.value)
    raise JitOpError("op body must be `return <arithmetic on the input>`")


def op(fn: Any) -> Op:
    """Turn a typed one-argument function into a compose leaf.

    The body must be a single ``return <arithmetic on the argument>``; anything
    else raises :class:`JitOpError` at decoration time.
    """
    ann = getattr(fn, "__annotations__", {})
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    func = tree.body[0]
    if not isinstance(func, ast.FunctionDef) or len(func.args.args) != 1:
        raise JitOpError("@op needs a function of exactly one typed argument")
    var = func.args.args[0].arg
    if var not in ann or "return" not in ann:
        raise JitOpError("@op needs annotations on the argument and the return")
    in_type = _type_of(ann[var])
    out_type = _type_of(ann["return"])
    # Skip a leading docstring; the rest must be a single return.
    body = [s for s in func.body if not isinstance(s, ast.Expr)]
    if len(body) != 1 or not isinstance(body[0], ast.Return) or body[0].value is None:
        raise JitOpError("@op body must be a single `return <expr>`")
    return Op([{"in": in_type, "out": out_type, "expr": _lower(body[0].value, var)}])


def _emit(op_obj: Op) -> str:
    """Emit a .so that builds `op_obj` as a dftu_op graph on a caller-provided
    host. The standalone runner (native OpRunner binding) supplies the host and
    drives the run; there is no plugin/scan here."""
    lines: List[str] = [
        "#include <dftracer/utils/plugins/abi.h>",
        "#include <stdint.h>",
        "",
    ]
    for i, st in enumerate(op_obj.stages):
        ict = _CTYPE[st["in"]]
        oct_ = _CTYPE[st["out"]]
        lines += [
            f"static dftu_task* leaf_{i}(void* s, const void* in, void* out, int* rc) {{",
            "    (void)s;",
            f"    {ict} x = *(const {ict}*)in;",
            f"    *({oct_}*)out = ({oct_})({st['expr']});",
            "    if (rc) *rc = 0;",
            "    return NULL;",
            "}",
            "",
        ]
    lines += [
        "#ifdef __cplusplus",
        'extern "C"',
        "#endif",
        "dftu_op* dftracer_build_op(const dftu_host* host) {",
        "    const dftu_ext_compose* c = (const dftu_ext_compose*)"
        "host->get_extension(host->h, DFTU_EXT_COMPOSE);",
        "    if (!c) return 0;",
    ]
    for i, st in enumerate(op_obj.stages):
        lines.append(
            f"    dftu_op* op_{i} = c->make_op(host->h, leaf_{i}, 0, 0, "
            f"{st['in']}, {_SIZE[st['in']]}, {st['out']}, {_SIZE[st['out']]});"
        )
    lines.append("    dftu_op* g = op_0;")
    for i in range(1, len(op_obj.stages)):
        lines.append(f"    g = c->then(host->h, g, op_{i});")
    lines += ["    return g;", "}", ""]
    return "\n".join(lines)


def emit_inline(op_obj: Op, c_name: str) -> str:
    """Emit `op_obj` as scalar static C functions callable as ``c_name(x)``.

    Used to inline an op referenced from a ``@jit.each_event`` body: each stage
    becomes ``c_name__s<i>`` and ``c_name`` chains them, so the plugin calls the
    op directly with no dftu_op graph or host round-trip."""
    lines: List[str] = []
    for i, st in enumerate(op_obj.stages):
        ict = _CTYPE[st["in"]]
        oct_ = _CTYPE[st["out"]]
        lines.append(f"static {oct_} {c_name}__s{i}({ict} x) {{ return ({oct_})({st['expr']}); }}")
    call = "x"
    for i in range(len(op_obj.stages)):
        call = f"{c_name}__s{i}({call})"
    lines.append(
        f"static {_CTYPE[op_obj.out_type]} {c_name}"
        f"({_CTYPE[op_obj.in_type]} x) {{ return {call}; }}"
    )
    return "\n".join(lines)


def is_float_type(dft: str) -> bool:
    """True if `dft` (a DFTU_T_* name) is a floating-point op value type."""
    return dft in _FLOAT


def c_type(dft: str) -> str:
    """C scalar type for a DFTU_T_* op value type."""
    return _CTYPE[dft]


def compile_op(op_obj: Op, *, name: str = "op") -> str:
    """Emit and build a .so exposing ``dftracer_build_op(host)`` for `op_obj`.
    Returns the cached ``.so`` path (run it via :func:`run_op`)."""
    source = _emit(op_obj)
    include = _plugin_build.include_dir()
    cxx = _plugin_build.compiler()
    digest = _plugin_build.source_digest(source, include, cxx)
    out = _plugin_build.cache_dir() / f"{name}.{digest}.so"
    if not out.is_file():
        src = out.with_suffix(".cpp")
        src.write_text(source, encoding="utf-8")
        _plugin_build.build_shared(str(src), out=str(out), name=name)
    return str(out)


def run_op(op_obj: Op, value: Any) -> Any:
    """Compile `op_obj`, then build and run its dftu_op graph over `value` on a
    standalone compose host (no plugin, no scan), returning the output value."""
    import struct

    from .dftracer_utils_ext import jit_run_op  # native standalone runner

    so = compile_op(op_obj)
    in_fmt = _STRUCT[op_obj.in_type]
    out_fmt = _STRUCT[op_obj.out_type]
    in_bytes = struct.pack("=" + in_fmt, value)
    out_bytes = jit_run_op(so, in_bytes, _SIZE[op_obj.out_type])
    return struct.unpack("=" + out_fmt, out_bytes)[0]


_STRUCT = {
    "DFTU_T_I64": "q",
    "DFTU_T_I32": "i",
    "DFTU_T_I16": "h",
    "DFTU_T_I8": "b",
    "DFTU_T_U64": "Q",
    "DFTU_T_U32": "I",
    "DFTU_T_U16": "H",
    "DFTU_T_U8": "B",
    "DFTU_T_F64": "d",
    "DFTU_T_F32": "f",
}
