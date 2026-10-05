#!/usr/bin/env python3
"""Turn DS-1000 NumPy reference solutions into Zinnia kernels.

For one DS-1000 record the kernel is built mechanically:

* Parameters: the names the reference solution reads that the exec_context
  prelude binds (usually from `test_input`), in prelude order, plus test inputs
  that nothing reads (DS-1000 passes them, so the signature keeps them).
* Types: each parameter's shape is its shape in the record's last DS-1000 test
  case; its dtype is the smallest of Boolean < Integer < Float < Complex that
  holds its value in every test case (a solution must accept every test input).
  A value with no Zinnia type (a string, an object or ragged array, a NaN or
  infinity, an empty array) is an `UntypeableInput`, never a guessed type.
* Body: the reference text. A solution DS-1000 wraps in `def f(...)` keeps its
  own `return`. A reference made of helper functions plus statements keeps the
  helpers at module level as `@zk_chip`. A reference that is one function the
  exec_context suffix calls with that function's own parameter names is inlined.
* Return value: what DS-1000 checks. If the exec_context suffix says
  `result = <expr>`, the kernel returns `<expr>`; otherwise it returns `result`.
  A solution with neither is a `GenerationError`.

No preconditions (`@requires`) are added.

The performance-set variant binds chosen parameters to their DS-1000 value as
compile-time constants (the author's decision for shape parameters). This tool
never writes into the performance set (`datasets/ds1000/`): it writes generated
kernels only under a directory it marked as its own, and `--clean` deletes only
files it wrote there. `check_perf_kernels.py` compares the performance-set
kernels with this tool's output in memory.
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ds1000_records as records  # noqa: E402

DATASETS = records.DATASETS
PERF_SET = DATASETS / "ds1000"
DEFAULT_OUT = DATASETS / "ds1000_numpy" / "generated"
MARKER = ".generated-by-generate_ds1000_kernels"
TOOL = "benchmarking/datasets/tools/generate_ds1000_kernels.py"

DTYPE_ORDER = ["Boolean", "Integer", "Float", "Complex"]


class GenerationError(Exception):
    """The reference solution does not fit one of the mechanical forms."""


class UntypeableInput(Exception):
    """A DS-1000 test input has no Zinnia type.

    `category` is one of: non_numeric (strings, bytes, objects, ragged
    sequences, datetimes, None), non_finite (NaN or infinity), empty (an array
    with a zero-length axis).
    """

    def __init__(self, name: str, category: str, detail: str) -> None:
        super().__init__(f"{name}: {category}: {detail}")
        self.name = name
        self.category = category
        self.detail = detail


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ParamType:
    dtype: str                         # one of DTYPE_ORDER
    shape: tuple[int, ...] | None      # None for a scalar

    def annotation(self) -> str:
        if self.shape is None:
            return self.dtype
        dims = " ".join(str(d) for d in self.shape)
        return f'{self.dtype}[Array, "{dims}"]'


def _check_finite(name: str, values: np.ndarray) -> None:
    if not np.all(np.isfinite(values)):
        raise UntypeableInput(name, "non_finite", "test input contains NaN or infinity")


def value_type(name: str, value: Any) -> ParamType:
    if isinstance(value, (bool, np.bool_)):
        return ParamType("Boolean", None)
    if isinstance(value, (int, np.integer)):
        return ParamType("Integer", None)
    if isinstance(value, (float, np.floating)):
        _check_finite(name, np.asarray(value))
        return ParamType("Float", None)
    if isinstance(value, (complex, np.complexfloating)):
        _check_finite(name, np.asarray(value))
        return ParamType("Complex", None)
    if isinstance(value, (str, bytes, np.str_, np.bytes_)):
        raise UntypeableInput(name, "non_numeric", f"test input is a {type(value).__name__}")
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise UntypeableInput(name, "non_numeric", f"test input is a {type(value).__name__}")
    try:
        array = np.asarray(value)
    except ValueError as exc:
        raise UntypeableInput(name, "non_numeric", f"ragged sequence ({exc})") from exc
    kind = array.dtype.kind
    if kind == "b":
        dtype = "Boolean"
    elif kind in "iu":
        dtype = "Integer"
    elif kind == "f":
        _check_finite(name, array)
        dtype = "Float"
    elif kind == "c":
        _check_finite(name, array)
        dtype = "Complex"
    elif kind in "US":
        raise UntypeableInput(name, "non_numeric", "array of strings")
    elif kind == "O":
        raise UntypeableInput(name, "non_numeric", "object array (e.g. arrays of different shapes)")
    else:
        raise UntypeableInput(name, "non_numeric", f"array of dtype {array.dtype}")
    if array.ndim == 0:
        return ParamType(dtype, None)
    if 0 in array.shape:
        raise UntypeableInput(name, "empty", f"array of shape {array.shape}")
    return ParamType(dtype, tuple(int(d) for d in array.shape))


def join_types(types: list[ParamType]) -> ParamType:
    """dtype: the widest over all test cases; shape: the last test case's."""
    dtype = max((t.dtype for t in types), key=DTYPE_ORDER.index)
    return ParamType(dtype, types[-1].shape)


# ---------------------------------------------------------------------------
# Kernel source
# ---------------------------------------------------------------------------


@dataclass
class GeneratedKernel:
    problem_id: int
    source: str
    params: list[tuple[str, ParamType]]          # kernel signature, in order
    constants: dict[str, Any]                    # performance variant only
    shape_case: int                              # DS-1000 test case the shapes come from
    form: str                                    # how the body was built
    notes: list[str] = field(default_factory=list)


def _lines(source: str, node: ast.AST) -> list[str]:
    all_lines = source.splitlines()
    return all_lines[node.lineno - 1:node.end_lineno]  # type: ignore[attr-defined]


def _result_expression(context: records.ExecContext) -> tuple[str | None, ast.expr | None]:
    """The expression DS-1000 checks, if the suffix assigns `result`."""
    found: tuple[str | None, ast.expr | None] = (None, None)
    for node in context.suffix:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "result"):
            found = (ast.get_source_segment(context.suffix_source, node.value), node.value)
        else:
            raise GenerationError(f"exec_context suffix has a statement other than `result = ...`: "
                                  f"{ast.unparse(node)}")
    return found


def _call_matches_params(call: ast.expr | None, func_name: str, params: list[str]) -> bool:
    return (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == func_name
            and not call.keywords and [a.id if isinstance(a, ast.Name) else None for a in call.args] == params)


def _function_params(fn: ast.FunctionDef) -> list[str]:
    return [a.arg for a in fn.args.args]


def _body(record: dict[str, Any], context: records.ExecContext) -> tuple[list[str], list[str], str]:
    """Returns (helper_lines, body_lines, form). Body lines are not indented."""
    reference = record["reference_code"]
    result_src, result_node = _result_expression(context)

    if context.function_header is not None:
        header = context.function_header
        if not _call_matches_params(result_node, header.name, _function_params(header)):
            raise GenerationError("exec_context calls the wrapping function with other names than its parameters")
        body = textwrap.dedent(reference).strip("\n").splitlines()
        if not any(isinstance(n, ast.Return) for n in ast.walk(ast.parse("\n".join(body)))):
            raise GenerationError("function-form solution has no return")
        return [], body, ("the reference function body as DS-1000 runs it inside `def f(...)`, with its own "
                          "return")

    text = textwrap.dedent(reference).strip("\n")
    tree = ast.parse(text)
    helpers = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    statements = [n for n in tree.body if not isinstance(n, ast.FunctionDef)]

    if helpers and not statements:
        if len(helpers) == 1 and _call_matches_params(result_node, helpers[0].name, _function_params(helpers[0])):
            fn = helpers[0]
            body = textwrap.dedent("\n".join(_lines(text, fn)[fn.body[0].lineno - fn.lineno:])).splitlines()
            return [], body, (f"the body of the reference function {fn.name}, inlined: DS-1000 calls it with its "
                              "own parameter names")
        raise GenerationError("reference defines functions but no statement uses them")

    helper_lines: list[str] = []
    helper_line_numbers: set[int] = set()
    for fn in helpers:
        for name in records.function_globals(ast.unparse(fn)):
            if name not in records.BUILTIN_NAMES:
                raise GenerationError(f"helper {fn.name} reads the outer name {name}; it cannot move to module level")
        start = fn.lineno - len(fn.decorator_list)
        helper_line_numbers.update(range(start, fn.end_lineno + 1))  # type: ignore[operator]
        helper_lines.extend(["@zk_chip", *_lines(text, fn), ""])
    body = [line for number, line in enumerate(text.splitlines(), start=1) if number not in helper_line_numbers]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()

    stored = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if result_src is not None:
        body.append(f"return {result_src}")
        returned = f"`{result_src}`, the value DS-1000 checks (exec_context: `result = {result_src}`)"
    elif "result" in stored:
        body.append("return result")
        returned = "`result`, the value DS-1000 checks"
    else:
        raise GenerationError("solution assigns no `result` and the suffix names no checked value")
    form = "the reference statements" + (" with their helper functions moved to module level as @zk_chip"
                                         if helpers else "")
    return helper_lines, body, f"{form}; returns {returned}"


def constant_literal(value: Any) -> str:
    if isinstance(value, (bool, np.bool_)):
        return repr(bool(value))
    if isinstance(value, (int, np.integer)):
        return repr(int(value))
    if isinstance(value, np.ndarray):
        return f"np.array({value.tolist()!r})"
    if isinstance(value, tuple):
        return repr(tuple(plain_value(v) for v in value))
    if isinstance(value, list):
        return repr([plain_value(v) for v in value])
    raise GenerationError(f"cannot write a constant of type {type(value).__name__}")


def plain_value(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def infer_types(record: dict[str, Any], names: list[str]) -> tuple[dict[str, ParamType], records.TestCase]:
    cases = [records.evaluate_test_case(record, k) for k in records.test_case_ids(record)]
    types: dict[str, ParamType] = {}
    for name in names:
        per_case = [value_type(name, case.env[name]) for case in cases if name in case.env]
        if not per_case:
            raise GenerationError(f"no test case binds {name}")
        types[name] = join_types(per_case)
    return types, cases[-1]


def generate(record: dict[str, Any], constants: Sequence[str] = ()) -> GeneratedKernel:
    """Build the kernel. Names in `constants` become compile-time constants
    bound to their value in the shape test case (performance-set variant)."""
    context = records.parse_exec_context(record["code_context"])
    inputs = records.program_inputs(record)
    unknown = [c for c in constants if c not in inputs.names]
    if unknown:
        raise GenerationError(f"constants {unknown} are not parameters of the program")
    types, shape_case = infer_types(record, inputs.names)
    helper_lines, body, form = _body(record, context)

    notes: list[str] = []
    if shape_case.pandas_used:
        notes.append("DS-1000 passes this input as a pandas Series; the kernel takes the NumPy array of its values")
    if inputs.dead_test_inputs:
        notes.append(f"unused DS-1000 test inputs kept as parameters: {', '.join(inputs.dead_test_inputs)}")

    constant_values = {name: shape_case.env[name] for name in constants}
    params = [(n, types[n]) for n in inputs.names if n not in constant_values]
    pid = record["problem_id"]
    origin = 291 + int(record["perturbation_origin_id"])
    header = [
        "from zinnia import *",
        "import numpy as np",
        "",
        f"# DS-1000 problem {pid} (NumPy problem {record['library_problem_id']}; "
        f"{record['perturbation_type']}{'' if origin == pid else f' perturbation of problem {origin}'}).",
        "# Generated from the DS-1000 reference solution by",
        f"# {TOOL}: shapes from DS-1000",
        f"# test case {shape_case.case_id}, each dtype wide enough for every DS-1000 test input.",
    ]
    if constant_values:
        verb = "is a compile-time constant" if len(constant_values) == 1 else "are compile-time constants"
        header.append(f"# Performance-set variant: {', '.join(constant_values)} {verb} "
                      f"(test case {shape_case.case_id} value).")
    header.append("")
    signature = ", ".join(f"{name}: {ptype.annotation()}" for name, ptype in params)
    constant_lines = [f"{name} = {constant_literal(value)}" for name, value in constant_values.items()]
    lines = header + helper_lines + ["@zk_circuit", f"def kernel({signature}):"]
    lines += ["    " + line if line.strip() else "" for line in constant_lines + body]
    return GeneratedKernel(pid, "\n".join(lines) + "\n", params, constant_values, shape_case.case_id, form, notes)


# ---------------------------------------------------------------------------
# Writing (acceptance kernels only)
# ---------------------------------------------------------------------------


def _check_output_dir(out: Path) -> None:
    resolved = out.resolve()
    perf = PERF_SET.resolve()
    if resolved == perf or perf in resolved.parents or resolved in perf.parents:
        raise SystemExit(f"refusing to write into or above the performance set {perf}")
    if resolved.exists() and any(resolved.iterdir()) and not (resolved / MARKER).exists():
        raise SystemExit(f"{resolved} is not empty and was not created by this tool (no {MARKER})")


def clean(out: Path) -> None:
    """Delete the files this tool wrote under `out` (and nothing else)."""
    _check_output_dir(out)
    if not (out / MARKER).exists():
        return
    for case_dir in out.glob("case[0-9][0-9][0-9][0-9]"):
        for name in ("kernel.py",):
            (case_dir / name).unlink(missing_ok=True)
        if case_dir.is_dir() and not any(case_dir.iterdir()):
            case_dir.rmdir()
    (out / "index.json").unlink(missing_ok=True)


def describe(kernel: GeneratedKernel) -> dict[str, Any]:
    return {
        "params": [{"name": n, "annotation": t.annotation()} for n, t in kernel.params],
        "shape_test_case": kernel.shape_case,
        "form": kernel.form,
        "notes": kernel.notes,
    }


def write_kernels(manifest: dict[int, dict[str, Any]], out: Path, problem_ids: Sequence[int]) -> dict[str, Any]:
    _check_output_dir(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(f"Kernels in this directory are written by {TOOL}.\n", encoding="utf-8")
    index: dict[str, Any] = {}
    for pid in problem_ids:
        key = f"case{pid:04d}"
        try:
            kernel = generate(manifest[pid])
        except (GenerationError, UntypeableInput, records.RecordError) as exc:
            index[key] = {"error": f"{type(exc).__name__}: {exc}"}
            continue
        case_dir = out / key
        case_dir.mkdir(exist_ok=True)
        (case_dir / "kernel.py").write_text(kernel.source, encoding="utf-8")
        index[key] = describe(kernel)
    (out / "index.json").write_text(json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, default=records.MANIFEST)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT,
                        help="output directory (never the performance set)")
    parser.add_argument("--clean", action="store_true", help="first delete the kernels this tool wrote in --out")
    parser.add_argument("--all", action="store_true",
                        help="generate every program, not only those the manifest marks in scope")
    parser.add_argument("--show", type=int, metavar="PROBLEM_ID", help="print one kernel and exit")
    args = parser.parse_args()

    manifest = records.load_manifest(args.manifest)
    if args.show is not None:
        print(generate(manifest[args.show]).source, end="")
        return
    if args.clean:
        clean(args.out)
    ids = sorted(pid for pid, row in manifest.items() if args.all or row["status"] == "in_scope")
    index = write_kernels(manifest, args.out, ids)
    errors = sum(1 for entry in index.values() if "error" in entry)
    print(json.dumps({"written": len(index) - errors, "errors": errors, "out": str(args.out)}, indent=2))


if __name__ == "__main__":
    main()
