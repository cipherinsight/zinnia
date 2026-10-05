"""What both Zinnia adapters need: compiling a case and reading its output contract."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from benchmarking.harness.adapters.base import StepFailure
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.kernels import compile_kernel
from benchmarking.harness.outputs import (LOGICAL, ZINNIA_EXPOSURE, DecodedPublics,
                                          decode_exposed, decode_with_layout)

_EXPOSE_KINDS = {"ExposePublicIIR": "i", "ExposePublicFIR": "f"}


def compile_case(case: PerfCase) -> Any:
    """Compile the case's kernel; a compiler rejection is the `unsupported` failure."""
    try:
        return compile_kernel(case.kernel_path, case.function)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise StepFailure("unsupported", f"{type(exc).__name__}: {exc}") from None


def ir_statements(compiled: Any) -> list[dict]:
    return compiled.get_ir_stmts()


def ir_sha256(statements: list[dict]) -> str:
    """A digest of the exported IR, used to check that compilation is deterministic."""
    canonical = json.dumps(statements, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def expose_kinds(statements: list[dict]) -> list[str]:
    """`i` or `f` for every public output statement, in exposure order."""
    kinds = []
    for stmt in statements:
        kind = _EXPOSE_KINDS.get(stmt["ir_instance"]["__class__"])
        if kind is not None:
            kinds.append(kind)
    return kinds


def elided_values(compiled: Any) -> list[int | float]:
    """Constant outputs the compiler left out of the public vector, in output order."""
    values: list[int | float] = []
    for entry in compiled.get_elided_publics():
        values.append(int(entry["Int"]) if "Int" in entry else float(entry["Float"]))
    return values


def output_layout(compiled: Any) -> list[dict] | None:
    """The compiler's output layout (`outputs.decode_with_layout`), once it exports one."""
    getter = getattr(compiled, "get_output_layout", None)
    return getter() if callable(getter) else None


def describe_compiled(compiled: Any) -> dict:
    statements = ir_statements(compiled)
    return {
        "ir_sha256": ir_sha256(statements),
        "ir_statements": len(statements),
        "expose_kinds": "".join(expose_kinds(statements)),
        "elided": elided_values(compiled),
        "output_layout": output_layout(compiled),
        "program_inputs": [pi.export() for pi in compiled.program_inputs],
    }


def zinnia_publics(fields: list[int], info: dict) -> DecodedPublics:
    """A Zinnia proof's public vector as the output check takes it.

    With the compiler's output layout the values are decoded into the kernel's
    output; without it they stay in exposure order (see `outputs`).
    """
    try:
        if info.get("output_layout") is not None:
            values, shapes = decode_with_layout(fields, info["output_layout"])
            return DecodedPublics(LOGICAL, values, shapes=shapes)
        return DecodedPublics(ZINNIA_EXPOSURE, decode_exposed(fields, list(info["expose_kinds"])),
                              info["elided"])
    except ValueError as exc:
        raise StepFailure("wrong_output", f"public vector: {exc}") from None


def check_same_build(compiled: Any, state: dict) -> None:
    """A recompilation in a later step must produce the IR setup produced (audit H-20)."""
    digest = ir_sha256(ir_statements(compiled))
    if digest != state["ir_sha256"]:
        raise StepFailure("nondeterministic_build",
                          f"IR digest {digest[:12]} differs from setup's {state['ir_sha256'][:12]}")
