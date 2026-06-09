#!/usr/bin/env python3
"""Generate one kernel.py folder per in-scope DS-1000 NumPy program."""

from __future__ import annotations

import argparse
import ast
import copy
import json
import re
import shutil
import textwrap
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_MANIFEST = ROOT / "ds1000_numpy" / "manifest.json"

SCALAR_NAMES = {
    "p",
    "n",
    "m",
    "N",
    "row",
    "col",
    "low",
    "high",
    "threshold",
    "power",
    "degree",
    "divide_number",
    "multiply_number",
    "bin_size",
    "patch_size",
    "size",
    "length",
    "pos",
    "element",
    "a",
    "b",
}

INTEGER_NAMES = {
    "n",
    "m",
    "N",
    "row",
    "col",
    "nrow",
    "ncol",
    "power",
    "degree",
    "divide_number",
    "multiply_number",
    "bin_size",
    "patch_size",
    "size",
    "length",
    "pos",
    "element",
    "low",
    "high",
}


def expr_names(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def load_test_input_names(code_context: str) -> list[str]:
    match = re.search(r"exec_context\s*=\s*r?\"\"\"(.*?)\"\"\"", code_context, re.S)
    if not match:
        return []
    prefix_lines: list[str] = []
    for line in match.group(1).splitlines():
        if "[insert]" in line:
            break
        if line.startswith((" ", "\t")):
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith(("def ", "class ")):
            continue
        prefix_lines.append(line)
    tree = ast.parse("\n".join(prefix_lines) or "pass")
    names: list[str] = []
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "test_input" for target in ast.walk(node.value)):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    if isinstance(elt, ast.Name):
                        names.append(elt.id)
        if names:
            return names
    return []


def assigned_result_expr(code_context: str) -> str | None:
    match = re.search(r"exec_context\s*=\s*r?\"\"\"(.*?)\"\"\"", code_context, re.S)
    if not match:
        return None
    suffix_lines: list[str] = []
    after = False
    for line in match.group(1).splitlines():
        if "[insert]" in line:
            after = True
            continue
        if not after:
            continue
        if line.startswith((" ", "\t")):
            continue
        stripped = line.strip()
        if not stripped or stripped.startswith("import ") or stripped.startswith("from "):
            continue
        if stripped.startswith(("def ", "class ")):
            continue
        suffix_lines.append(line)
    tree = ast.parse("\n".join(suffix_lines) or "pass")
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            continue
        if node.targets[0].id != "result":
            continue
        return ast.unparse(node.value)
    return None


def callable_from_reference(reference_code: str) -> ast.FunctionDef | None:
    try:
        tree = ast.parse(textwrap.dedent(reference_code))
    except SyntaxError:
        return None
    funcs = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if len(funcs) == 1:
        return funcs[0]
    return None


def split_helpers_and_body(reference_code: str) -> tuple[list[ast.FunctionDef], str]:
    """Pull top-level `def`s out of a multi-statement reference so they can
    be hoisted as module-level `@zk_chip` helpers.

    Returns `(helpers, body_source)` where `helpers` is the (possibly empty)
    list of FunctionDef nodes to render outside the circuit, and
    `body_source` is the textual remainder used as the circuit body.

    The single-fn reference case is handled separately by `build_body` (it
    inlines the function body) and is left untouched here.
    """
    dedented = textwrap.dedent(reference_code).strip("\n")
    if not dedented:
        return [], ""
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return [], dedented
    if len(tree.body) <= 1:
        return [], dedented
    helpers = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if not helpers:
        return [], dedented
    rest = [node for node in tree.body if not isinstance(node, ast.FunctionDef)]
    if not rest:
        # All top-level statements are defs (rare). Leaving them in the body
        # would be no different from the existing behaviour; return empty
        # so the caller falls back to `build_body`.
        return [], dedented
    body_source = "\n".join(ast.unparse(node) for node in rest)
    return helpers, body_source


def build_body(reference_code: str, result_alias: str | None) -> str:
    dedented = textwrap.dedent(reference_code).strip("\n")
    if not dedented:
        return "    pass\n"
    try:
        tree = ast.parse(dedented)
    except SyntaxError:
        return textwrap.indent(dedented + "\n", "    ")

    if len(tree.body) == 1 and isinstance(tree.body[0], ast.FunctionDef):
        fn = tree.body[0]
        body = [ast.unparse(stmt) for stmt in fn.body]
        if body and body[-1].strip().startswith("return "):
            return "\n".join("    " + line for stmt in body for line in stmt.splitlines()) + "\n"
        return "\n".join("    " + line for stmt in body for line in stmt.splitlines()) + "\n"

    assigned_names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                assigned_names.update(expr_names(target))
        elif isinstance(node, ast.AnnAssign):
            assigned_names.update(expr_names(node.target))

    lines = dedented.splitlines()
    if not any(re.match(r"\s*return\b", line) for line in lines):
        all_store_names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
        if "result" in all_store_names:
            lines.append("return result")
        elif result_alias and result_alias in assigned_names:
            lines.append(f"return {result_alias}")
    return textwrap.indent("\n".join(lines).rstrip() + "\n", "    ")


def infer_params(record: dict[str, Any]) -> list[str]:
    names = load_test_input_names(record["code_context"])
    fn = callable_from_reference(record["reference_code"])
    if fn is not None:
        fn_args = [arg.arg for arg in fn.args.args]
        if set(fn_args).issubset(set(names)) or not names:
            return fn_args
    if names:
        return names

    try:
        tree = ast.parse(textwrap.dedent(record["reference_code"]))
    except SyntaxError:
        return []
    assigned: set[str] = set()
    loaded: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            if isinstance(node.ctx, ast.Store):
                assigned.add(node.id)
            elif isinstance(node.ctx, ast.Load):
                loaded.add(node.id)
    builtin_names = {"np", "numpy", "result", "True", "False", "None"}
    return sorted(loaded - assigned - builtin_names)


class TestInputExtractor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.cases: list[ast.AST] = []

    def visit_If(self, node: ast.If) -> None:
        for child in ast.walk(node.test):
            if isinstance(child, ast.Name) and child.id == "test_case_id":
                self.cases.extend(copy.deepcopy(stmt) for stmt in node.body)
                for stmt in node.orelse:
                    if isinstance(stmt, ast.If):
                        self.visit_If(stmt)
                return
        self.generic_visit(node)


def sample_env(record: dict[str, Any]) -> dict[str, Any]:
    env: dict[str, Any] = {}
    try:
        tree = ast.parse(record["code_context"])
    except SyntaxError:
        return env
    extractor = TestInputExtractor()
    extractor.visit(tree)
    if not extractor.cases:
        return env
    module = ast.Module(body=extractor.cases, type_ignores=[])
    ast.fix_missing_locations(module)
    try:
        exec(compile(module, "<ds1000-test-inputs>", "exec"), {"np": __import__("numpy")}, env)
    except Exception:
        return env
    return env


def shape_from_value(value: Any) -> tuple[str, str]:
    import numpy as np

    if isinstance(value, (bool, np.bool_)):
        return "Bool", ""
    if isinstance(value, (int, np.integer)):
        return "Integer", ""
    if isinstance(value, (float, np.floating)):
        return "Float", ""
    try:
        arr = np.asarray(value)
    except ValueError:
        if isinstance(value, (list, tuple)):
            return "Integer", f"[Array, \"{max(1, len(value))}\"]"
        return "Integer", ""
    dtype = "Float" if arr.dtype.kind in {"f", "c"} else "Integer"
    if arr.ndim == 0:
        return dtype, ""
    dims = " ".join(str(max(1, int(dim))) for dim in arr.shape)
    return dtype, f"[Array, \"{dims}\"]"


def annotation_for(name: str, value: Any | None) -> str:
    if value is not None:
        dtype, shape = shape_from_value(value)
        return f"{dtype}{shape}"
    if name in INTEGER_NAMES:
        return "Integer"
    if name in SCALAR_NAMES:
        return "Float"
    # Fallback for parameters whose shape can't be inferred from
    # `sample_env`. The bound is intentionally tight: the previous
    # `1024` cap meant up to 1024³ ≈ 10⁹ cells per input, which the
    # compiler's per-cell input-materialization pass
    # (`src/ir_gen/helpers.rs::read_input_value` loop, ~1 GiB/s of IR
    # growth) blew past available RAM for the 9 fallback-using DS-1000
    # kernels (0398/0399/0423/0424/0440/0471/0489/0490/0498). 64³ ≈
    # 262 144 cells keeps compile memory in the kB range while still
    # exercising the relevant ops.
    return 'Float[Array, "N<=64 M<=64 K<=64"]'


def param_signature(record: dict[str, Any], params: list[str]) -> str:
    env = sample_env(record)
    return ", ".join(f"{name}: {annotation_for(name, env.get(name))}" for name in params)


def kernel_source(record: dict[str, Any]) -> str:
    params = infer_params(record)
    signature = param_signature(record, params)
    result_alias = assigned_result_expr(record["code_context"])

    # If the reference defines top-level helpers alongside other
    # statements, hoist each helper to module level as `@zk_chip` so it
    # isn't a nested-def inside the circuit body (which Zinnia rejects).
    # `@zk_chip` does not require annotations — call-site inference fills
    # them in. The single-fn reference case stays in `build_body` (which
    # inlines the function body directly).
    helpers, helper_body_source = split_helpers_and_body(record["reference_code"])
    if helpers:
        body = build_body(helper_body_source, result_alias)
    else:
        body = build_body(record["reference_code"], result_alias)

    if "return " not in body and params:
        body = body.rstrip() + f"\n    return {params[0]}\n"
    if body.strip() == "pass":
        body = "    # Empty DS-1000 reference snippet.\n    pass\n"
    header = [
        "from zinnia import *",
        "import numpy as np",
        "",
        "# DS-1000 NumPy acceptance-study kernel.",
        "# The body below is generated from the original DS-1000 reference solution.",
        "# Type annotations/preconditions are intentionally not inferred here.",
        "",
    ]
    for helper in helpers:
        header.append("@zk_chip")
        header.append(ast.unparse(helper))
        header.append("")
    header.extend(["@zk_circuit", f"def kernel({signature}):"])
    return "\n".join(header) + "\n" + body


def write_kernels(manifest: Path, root: Path, clean: bool) -> dict[str, Any]:
    rows = json.loads(manifest.read_text(encoding="utf-8"))["programs"]
    candidates = [row for row in rows if row["status"] == "acceptance_candidate"]
    if clean:
        for folder in root.glob("ds1000_case*"):
            if folder.is_dir():
                shutil.rmtree(folder)
    created: list[str] = []
    for row in candidates:
        folder = root / f"ds1000_case{row['problem_id']:04d}"
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "kernel.py").write_text(kernel_source(row), encoding="utf-8")
        (folder / "record.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        created.append(str(folder.relative_to(root)))
    return {"created": len(created), "folders": created}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--clean", action="store_true")
    args = parser.parse_args()
    payload = write_kernels(args.manifest, args.root, args.clean)
    print(json.dumps({"created": payload["created"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
