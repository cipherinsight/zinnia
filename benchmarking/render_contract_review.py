#!/usr/bin/env python3
"""Generate and render benchmark contract review artifacts.

The generated JSON files are intentionally heuristic. They are a review
starting point for adding `@zk_circuit` wrappers, type annotations, and
preconditions while keeping the original benchmark kernels unchanged.
"""

from __future__ import annotations

import argparse
import ast
import html
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
DEFAULT_CONTRACT_DIR = ROOT / "contracts"
DEFAULT_HTML = ROOT / "contracts_review.html"
DEFAULT_VALIDATION = ROOT / "compile_validation.json"


DIMENSION_NAMES = {
    "n",
    "m",
    "k",
    "N",
    "M",
    "K",
    "NR",
    "NM",
    "NP",
    "NQ",
    "NI",
    "NJ",
    "NK",
    "NL",
    "NX",
    "NY",
    "T",
    "TSTEPS",
    "nit",
    "Nt",
    "npt",
    "n_max",
    "max_iter",
}

FLOAT_SCALAR_NAMES = {
    "alpha",
    "beta",
    "dt",
    "dx",
    "dy",
    "dz",
    "eps",
    "epsilon",
    "rho",
    "nu",
    "G",
    "softening",
    "mu",
    "sigma",
}


def node_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = node_name(node.value)
        return f"{base}.{node.attr}" if base else node.attr
    if isinstance(node, ast.Subscript):
        return node_name(node.value)
    return ""


def call_name(node: ast.AST) -> str:
    if isinstance(node, (ast.Name, ast.Attribute)):
        return node_name(node)
    return ""


def index_rank(slice_node: ast.AST) -> int:
    if isinstance(slice_node, ast.Tuple):
        return len(slice_node.elts)
    return 1


def name_refs(node: ast.AST) -> set[str]:
    return {child.id for child in ast.walk(node) if isinstance(child, ast.Name)}


def expression_is_array_like(node: ast.AST, array_params: set[str]) -> bool:
    if isinstance(node, ast.Name):
        return node.id in array_params
    if isinstance(node, ast.Subscript):
        return True
    if isinstance(node, ast.BinOp):
        return expression_is_array_like(node.left, array_params) or expression_is_array_like(
            node.right, array_params
        )
    if isinstance(node, ast.Call):
        name = call_name(node.func)
        return name.startswith(("np.", "numpy.")) or any(
            expression_is_array_like(arg, array_params) for arg in node.args
        )
    return False


@dataclass
class FunctionFacts:
    name: str
    args: list[str]
    line_start: int
    line_end: int
    calls: Counter[str]
    attrs: Counter[str]
    ops: Counter[str]
    array_params: set[str]
    scalar_params: set[str]
    mutated_params: set[str]
    max_index_rank: dict[str, int]
    has_for: bool
    has_while: bool
    has_if: bool


def analyze_function(fn: ast.FunctionDef) -> FunctionFacts:
    args = [arg.arg for arg in fn.args.args]
    arg_set = set(args)
    calls: Counter[str] = Counter()
    attrs: Counter[str] = Counter()
    ops: Counter[str] = Counter()
    array_params: set[str] = set()
    mutated_params: set[str] = set()
    max_index_rank: dict[str, int] = {}
    has_for = False
    has_while = False
    has_if = False

    for node in ast.walk(fn):
        if isinstance(node, ast.For):
            has_for = True
        elif isinstance(node, ast.While):
            has_while = True
        elif isinstance(node, ast.If):
            has_if = True
        elif isinstance(node, ast.Call):
            name = call_name(node.func)
            if name:
                calls[name] += 1
            if name.startswith(("np.", "numpy.")):
                for arg in node.args:
                    for ref in name_refs(arg):
                        if ref in arg_set:
                            array_params.add(ref)
        elif isinstance(node, ast.Attribute):
            attrs[node.attr] += 1
            base = node_name(node.value).split(".")[0]
            if base in arg_set and node.attr in {"shape", "size", "ndim", "T", "real", "imag"}:
                array_params.add(base)
        elif isinstance(node, ast.BinOp):
            ops[type(node.op).__name__] += 1
            if isinstance(node.op, ast.MatMult):
                for ref in name_refs(node.left) | name_refs(node.right):
                    if ref in arg_set:
                        array_params.add(ref)
        elif isinstance(node, ast.Compare):
            for op in node.ops:
                ops[type(op).__name__] += 1
        elif isinstance(node, ast.Subscript):
            base = node_name(node.value).split(".")[0]
            if base in arg_set:
                array_params.add(base)
                max_index_rank[base] = max(max_index_rank.get(base, 0), index_rank(node.slice))
        elif isinstance(node, (ast.Assign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Subscript):
                    base = node_name(target.value).split(".")[0]
                    if base in arg_set:
                        array_params.add(base)
                        mutated_params.add(base)
                elif isinstance(target, ast.Name) and target.id in arg_set:
                    if expression_is_array_like(
                        node.value if isinstance(node, ast.Assign) else node.value, array_params
                    ):
                        array_params.add(target.id)
                        if isinstance(node, ast.AugAssign):
                            mutated_params.add(target.id)

    scalar_params = set(args) - array_params
    return FunctionFacts(
        name=fn.name,
        args=args,
        line_start=fn.lineno,
        line_end=getattr(fn, "end_lineno", fn.lineno) or fn.lineno,
        calls=calls,
        attrs=attrs,
        ops=ops,
        array_params=array_params,
        scalar_params=scalar_params,
        mutated_params=mutated_params,
        max_index_rank=max_index_rank,
        has_for=has_for,
        has_while=has_while,
        has_if=has_if,
    )


def read_status_from_filtering() -> dict[str, str]:
    filtering = ROOT / "FILTERING.md"
    if not filtering.exists():
        return {}
    status: dict[str, str] = {}
    for line in filtering.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| `"):
            continue
        parts = [p.strip() for p in line.strip("|").split("|")]
        if len(parts) >= 2:
            folder = parts[0].strip("`")
            status[folder] = parts[1].strip("`")
    return status


def find_entry_functions(functions: list[ast.FunctionDef]) -> set[str]:
    names = {fn.name for fn in functions}
    called: set[str] = set()
    for fn in functions:
        aliases: dict[str, str] = {}
        for node in ast.walk(fn):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Name) and node.value.id in names:
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        aliases[target.id] = node.value.id
                        called.add(node.value.id)
            if isinstance(node, ast.Call):
                name = call_name(node.func).split(".")[0]
                if name in names:
                    called.add(name)
                if name in aliases:
                    called.add(aliases[name])
    entries = names - called
    return entries or names


def infer_scalar_annotation(name: str) -> str:
    if name in DIMENSION_NAMES or name.lower().startswith(("n", "num")):
        return "Integer"
    if name in FLOAT_SCALAR_NAMES:
        return "Float"
    if len(name) == 1 and name.isupper():
        return "Float"
    if name.isupper() and len(name) <= 4:
        return "Integer"
    return "Float"


def dim_names(rank: int) -> list[str]:
    pool = ["N", "M", "K", "L", "R", "C", "D"]
    return pool[: max(rank, 1)]


def infer_array_annotation(param: str, rank: int, complex_program: bool, integer_hint: bool) -> str:
    dtype = "Complex" if complex_program else ("Integer" if integer_hint else "Float")
    default_bounds = {"N": 1024, "M": 1024, "K": 1024, "L": 256, "R": 256, "C": 256, "D": 64}
    dims = " ".join(f"{dim}<={default_bounds.get(dim, 1024)}" for dim in dim_names(rank))
    return f'{dtype}[Array, "{dims}"]'


def classify_contract(
    facts: list[FunctionFacts], source_text: str, entry_names: set[str]
) -> tuple[list[str], list[str], list[str], list[str], str]:
    calls = Counter()
    ops = Counter()
    for fact in facts:
        calls.update(fact.calls)
        ops.update(fact.ops)

    categories: list[str] = []
    preconditions: list[str] = []
    diagnostics: list[str] = []
    notes: list[str] = []

    has_for = any(f.has_for for f in facts)
    has_while = any(f.has_while for f in facts)
    has_mutation = any(f.mutated_params for f in facts)
    complex_program = any(
        token in source_text
        for token in ["complex", "1j", "np.complex", "cmath", ".conjugate", "np.conj"]
    )

    if has_for:
        categories.append("loop-bounded")
        preconditions.append("Loop trip counts must be finite and tied to scalar bounds or array dimensions.")
    if has_while:
        categories.append("data-dependent-loop")
        preconditions.append("Add an explicit iteration cap or classify as fail-closed until a bound is available.")
        diagnostics.append("while loops require a public/padded maximum iteration count.")
    if has_mutation:
        categories.append("in-place-mutation")
        preconditions.append("Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.")
    if "MatMult" in ops or any(name in calls for name in ["np.dot", "np.matmul"]) or any(
        name.endswith(".dot") for name in calls
    ):
        categories.append("matmul-dot")
        preconditions.append("Declare shared matrix/vector dimensions explicitly, e.g. A[N,K] with x[K].")
    if any(op in ops for op in ["Div", "FloorDiv", "Mod"]):
        categories.append("denominator-obligation")
        preconditions.append("Add nonzero or positive denominator facts at the exact division/modulo sites.")
    if any(any(token in name for token in ["sqrt", "log", "acos", "arccos"]) for name in calls):
        categories.append("domain-obligation")
        preconditions.append("Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].")
    if any(name.startswith("np.linalg") for name in calls):
        categories.append("linalg")
        preconditions.append("Record square/invertible/SPD requirements where the linalg call needs them.")
        diagnostics.append("np.linalg kernels may stay fail-closed until dynamic/static linalg support is selected.")
    if any(name in calls for name in ["np.histogram", "np.bincount"]):
        categories.append("histogram-binning")
        preconditions.append("Record bin count, finite radius/value ranges, and weighted/unweighted histogram semantics.")
        diagnostics.append("Histogram/binning is an operator-specific benchmark row unless support is added.")
    if any(name in calls for name in ["np.sort", "np.argsort", "sorted"]):
        categories.append("sort-permutation")
        preconditions.append("Record sortedness/permutation facts only if the kernel relies on them after sorting.")
        diagnostics.append("Sort/permutation kernels should remain explicit diagnostic rows until supported.")
    if any(name in calls for name in ["np.where", "np.nonzero", "np.argwhere"]):
        categories.append("mask-where")
        preconditions.append("Record mask shape equality and active-element bounds for where/nonzero-style indexing.")
    if any(name in calls for name in ["np.std", "np.var", "np.cov"]) or any(
        name.endswith(".std") or name.endswith(".var") for name in calls
    ):
        categories.append("statistics")
        preconditions.append("Record sample-size requirements such as N > 1 and nonzero variance denominators.")
    if complex_program:
        categories.append("complex")
        diagnostics.append("Complex kernels require Complex dtype support or a fail-closed classification.")
    if not categories:
        categories.append("shape-dtype-only")
        notes.append("Likely first-pass candidate: mostly needs dtype/rank/shape annotations.")

    outcome = "compile/prove candidate"
    if diagnostics:
        outcome = "diagnostic/fail-closed candidate"
    elif has_while:
        outcome = "requires explicit loop bound"

    return sorted(set(categories)), sorted(set(preconditions)), sorted(set(diagnostics)), notes, outcome


def build_contract(kernel_path: Path, filtering_status: dict[str, str]) -> dict[str, Any]:
    source = kernel_path.read_text(encoding="utf-8", errors="ignore")
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    entries = find_entry_functions(functions)
    facts = [analyze_function(fn) for fn in functions]
    categories, preconditions, diagnostics, notes, outcome = classify_contract(facts, source, entries)
    complex_program = "complex" in categories
    integer_hint = any(
        token in source
        for token in ["np.arange", "np.ones_like", "np.zeros_like", "range(", " // ", "& 1"]
    )

    function_nodes = {node.name: node for node in functions}
    function_rows = []
    for fact in facts:
        fn_node = function_nodes.get(fact.name)
        arg_nodes = {a.arg: a for a in fn_node.args.args} if fn_node is not None else {}
        params = []
        for arg in fact.args:
            arg_node = arg_nodes.get(arg)
            if arg in fact.array_params:
                rank = max(fact.max_index_rank.get(arg, 1), 1)
                annotation = (
                    ast.unparse(arg_node.annotation)
                    if arg_node is not None and arg_node.annotation is not None
                    else infer_array_annotation(arg, rank, complex_program, integer_hint and arg in {"n", "indices"})
                )
                role = "array"
            else:
                annotation = (
                    ast.unparse(arg_node.annotation)
                    if arg_node is not None and arg_node.annotation is not None
                    else infer_scalar_annotation(arg)
                )
                role = "scalar"
            params.append(
                {
                    "name": arg,
                    "role": role,
                    "proposed_annotation": annotation,
                    "rank_hint": fact.max_index_rank.get(arg),
                    "mutated": arg in fact.mutated_params,
                }
            )
        function_rows.append(
            {
                "name": fact.name,
                "entry_candidate": fact.name in entries,
                "line_start": fact.line_start,
                "line_end": fact.line_end,
                "parameters": params,
                "calls": sorted(fact.calls),
                "ops": sorted(fact.ops),
                "has_for": fact.has_for,
                "has_while": fact.has_while,
                "has_if": fact.has_if,
                "mutated_params": sorted(fact.mutated_params),
            }
        )

    return {
        "program": kernel_path.parent.name,
        "kernel_path": str(kernel_path.relative_to(ROOT)),
        "source_status": filtering_status.get(kernel_path.parent.name, "unknown"),
        "review_status": "draft",
        "expected_outcome": outcome,
        "categories": categories,
        "functions": function_rows,
        "preconditions": preconditions,
        "diagnostics": diagnostics,
        "notes": notes,
        "source_excerpt": "\n".join(source.splitlines()[:80]),
    }


def generate_contracts(contract_dir: Path) -> list[Path]:
    filtering_status = read_status_from_filtering()
    contract_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for kernel_path in sorted(ROOT.glob("*/kernel.py")):
        contract = build_contract(kernel_path, filtering_status)
        path = contract_dir / f"{contract['program']}.json"
        path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        written.append(path)
    return written


def badge(text: str) -> str:
    return f'<span class="badge">{html.escape(text)}</span>'


def load_contracts(contract_dir: Path) -> list[dict[str, Any]]:
    contracts = []
    for path in sorted(contract_dir.glob("*.json")):
        contracts.append(json.loads(path.read_text(encoding="utf-8")))
    return contracts


def load_validation(validation_path: Path = DEFAULT_VALIDATION) -> dict[str, Any] | None:
    if not validation_path.exists():
        return None
    return json.loads(validation_path.read_text(encoding="utf-8"))


def attach_validation(contracts: list[dict[str, Any]], validation: dict[str, Any] | None) -> Counter[str]:
    if validation is None:
        return Counter()
    by_key = {
        (row["program"], row["function"]): row
        for row in validation.get("results", [])
    }
    status_counts: Counter[str] = Counter()
    for contract in contracts:
        rows = []
        for fn in contract["functions"]:
            row = by_key.get((contract["program"], fn["name"]))
            if row is None:
                continue
            fn["validation"] = row
            rows.append(row)
        circuit_rows = [row for row in rows if row.get("role") == "circuit"]
        compile_ok_count = sum(1 for row in circuit_rows if row["status"] == "compile_ok")
        compile_timeout_count = sum(1 for row in circuit_rows if row["status"] == "compile_timeout")
        compile_error_count = sum(1 for row in circuit_rows if row["status"] == "compile_error")
        if any(row["status"] == "compile_ok" for row in circuit_rows):
            status = "has compiling entry"
        elif circuit_rows:
            status = "entry compile errors"
        elif rows:
            status = "helpers only"
        else:
            status = "not validated"
        contract["validation_status"] = status
        contract["validation_summary"] = {
            "entry_circuits": len(circuit_rows),
            "compile_ok": compile_ok_count,
            "compile_error": compile_error_count,
            "compile_timeout": compile_timeout_count,
        }
        status_counts[status] += 1
    return status_counts


def render_html(contract_dir: Path, output: Path) -> None:
    contracts = load_contracts(contract_dir)
    validation = load_validation()
    validation_counts = attach_validation(contracts, validation)
    category_counts = Counter(cat for c in contracts for cat in c["categories"])
    outcome_counts = Counter(c["expected_outcome"] for c in contracts)
    rows = []
    for contract in contracts:
        entry_params = []
        validation_lines = []
        for fn in contract["functions"]:
            if not fn["entry_candidate"]:
                continue
            bits = []
            for param in fn["parameters"]:
                marker = "*" if param["mutated"] else ""
                bits.append(
                    f"{html.escape(param['name'])}{marker}: "
                    f"{html.escape(param['proposed_annotation'])}"
                )
            entry_params.append(f"<b>{html.escape(fn['name'])}</b>({', '.join(bits)})")
            val = fn.get("validation")
            if val:
                detail = val["status"]
                if val.get("error_message"):
                    detail += ": " + val["error_message"]
                validation_lines.append(f"<b>{html.escape(fn['name'])}</b>: {html.escape(detail)}")
        validation_status = contract.get("validation_status", "not validated")
        validation_summary = contract.get("validation_summary", {})
        validation_summary_line = (
            f"{validation_summary.get('compile_ok', 0)} ok / "
            f"{validation_summary.get('compile_error', 0)} errors / "
            f"{validation_summary.get('compile_timeout', 0)} timeouts "
            f"across {validation_summary.get('entry_circuits', 0)} entry circuit(s)"
            if validation_summary else "No validation result loaded."
        )
        rows.append(
            f"""
            <article class="kernel" data-outcome="{html.escape(contract['expected_outcome'])}"
                     data-validation="{html.escape(validation_status)}"
                     data-categories="{html.escape(' '.join(contract['categories']))}">
              <header>
                <h3>{html.escape(contract['program'])}</h3>
                <span class="status">{html.escape(validation_status)}</span>
              </header>
              <div class="meta">
                {''.join(badge(cat) for cat in contract['categories'])}
                {badge(contract['expected_outcome'])}
              </div>
              <p class="path">{html.escape(contract['kernel_path'])}</p>
              <section>
                <h4>Entry Annotation Sketch</h4>
                <p>{'<br>'.join(entry_params) if entry_params else 'No entry function inferred.'}</p>
              </section>
              <section>
                <h4>Preconditions</h4>
                <ul>{''.join(f'<li>{html.escape(item)}</li>' for item in contract['preconditions']) or '<li>Shape/rank/dtype annotations only.</li>'}</ul>
              </section>
              <section>
                <h4>Diagnostics / Blockers</h4>
                <ul>{''.join(f'<li>{html.escape(item)}</li>' for item in contract['diagnostics']) or '<li>None identified by heuristic skim.</li>'}</ul>
              </section>
              <section>
                <h4>Validation</h4>
                <p>{html.escape(validation_summary_line)}</p>
                <ul>{''.join(f'<li>{line}</li>' for line in validation_lines) or '<li>No validation result loaded.</li>'}</ul>
              </section>
              <details>
                <summary>Source excerpt</summary>
                <pre>{html.escape(contract['source_excerpt'])}</pre>
              </details>
            </article>
            """
        )

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Zinnia Benchmark Contract Review</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --ink: #202124;
      --muted: #666b70;
      --line: #d8d8d2;
      --panel: #ffffff;
      --accent: #0f766e;
      --warn: #9f5f00;
      --bad: #9b1c1c;
      --ok: #166534;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header.page {{
      padding: 24px 28px 16px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
      position: sticky;
      top: 0;
      z-index: 2;
    }}
    h1 {{ margin: 0 0 8px; font-size: 24px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 10px;
      margin-top: 16px;
    }}
    .metric {{
      border: 1px solid var(--line);
      background: var(--panel);
      padding: 10px 12px;
      border-radius: 8px;
    }}
    .metric b {{ display: block; font-size: 20px; }}
    .controls {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 16px;
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 8px 10px;
      min-height: 36px;
      background: #fff;
    }}
    input {{ min-width: min(520px, 100%); flex: 1; }}
    main {{
      padding: 20px 28px 40px;
      display: grid;
      gap: 12px;
    }}
    .kernel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
    }}
    .kernel header {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: baseline;
    }}
    h3 {{ margin: 0; font-size: 16px; }}
    h4 {{ margin: 12px 0 4px; font-size: 13px; color: var(--muted); }}
    .status {{
      color: #fff;
      background: var(--accent);
      border-radius: 999px;
      padding: 3px 8px;
      white-space: nowrap;
      font-size: 12px;
    }}
    [data-validation="has compiling entry"] .status {{ background: var(--ok); }}
    [data-validation="entry compile errors"] .status {{ background: var(--bad); }}
    [data-validation="helpers only"] .status {{ background: var(--warn); }}
    .badge {{
      display: inline-block;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 2px 7px;
      margin: 8px 5px 0 0;
      color: var(--muted);
      background: #fbfbf8;
      font-size: 12px;
    }}
    .path {{ color: var(--muted); font-family: ui-monospace, SFMono-Regular, Menlo, monospace; }}
    ul {{ margin: 4px 0 0 18px; padding: 0; }}
    pre {{
      max-height: 320px;
      overflow: auto;
      padding: 10px;
      background: #f1f1ec;
      border-radius: 6px;
      border: 1px solid var(--line);
    }}
    .hidden {{ display: none; }}
  </style>
</head>
<body>
  <header class="page">
    <h1>Zinnia Benchmark Contract Review</h1>
    <div>Generated from per-kernel JSON contracts. These are heuristic review drafts, not final annotations.</div>
    <div class="summary">
      <div class="metric"><b>{len(contracts)}</b>kernels</div>
      {''.join(f'<div class="metric"><b>{count}</b>{html.escape(name)}</div>' for name, count in outcome_counts.items())}
      {''.join(f'<div class="metric"><b>{count}</b>{html.escape(name)}</div>' for name, count in validation_counts.items())}
    </div>
    <div class="controls">
      <input id="search" placeholder="Search program, category, annotation, or precondition">
      <select id="outcome">
        <option value="">All outcomes</option>
        {''.join(f'<option value="{html.escape(name)}">{html.escape(name)} ({count})</option>' for name, count in sorted(outcome_counts.items()))}
      </select>
      <select id="category">
        <option value="">All categories</option>
        {''.join(f'<option value="{html.escape(name)}">{html.escape(name)} ({count})</option>' for name, count in sorted(category_counts.items()))}
      </select>
      <select id="validation">
        <option value="">All validation states</option>
        {''.join(f'<option value="{html.escape(name)}">{html.escape(name)} ({count})</option>' for name, count in sorted(validation_counts.items()))}
      </select>
    </div>
  </header>
  <main id="kernels">
    {''.join(rows)}
  </main>
  <script>
    const search = document.getElementById('search');
    const outcome = document.getElementById('outcome');
    const category = document.getElementById('category');
    const validation = document.getElementById('validation');
    const cards = [...document.querySelectorAll('.kernel')];
    function applyFilters() {{
      const q = search.value.toLowerCase();
      const out = outcome.value;
      const cat = category.value;
      const val = validation.value;
      for (const card of cards) {{
        const okSearch = !q || card.innerText.toLowerCase().includes(q);
        const okOutcome = !out || card.dataset.outcome === out;
        const okCategory = !cat || card.dataset.categories.split(' ').includes(cat);
        const okValidation = !val || card.dataset.validation === val;
        card.classList.toggle('hidden', !(okSearch && okOutcome && okCategory && okValidation));
      }}
    }}
    search.addEventListener('input', applyFilters);
    outcome.addEventListener('change', applyFilters);
    category.addEventListener('change', applyFilters);
    validation.addEventListener('change', applyFilters);
  </script>
</body>
</html>
"""
    output.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-json", action="store_true", help="Regenerate per-kernel JSON contracts.")
    parser.add_argument("--contracts-dir", type=Path, default=DEFAULT_CONTRACT_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_HTML)
    args = parser.parse_args()
    if args.generate_json:
        written = generate_contracts(args.contracts_dir)
        print(f"Wrote {len(written)} JSON contract files to {args.contracts_dir}")
    render_html(args.contracts_dir, args.output)
    print(f"Wrote review page to {args.output}")


if __name__ == "__main__":
    main()
