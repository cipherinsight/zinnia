#!/usr/bin/env python3
"""Extract and scope the original DS-1000 NumPy slice.

This script intentionally does not decide whether Zinnia can compile a program.
It only separates structural out-of-scope programs from programs that should
enter the acceptance/coverage study. Unsupported NumPy operators remain in the
acceptance set.
"""

from __future__ import annotations

import argparse
import ast
import gzip
import json
import re
import textwrap
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_URL = "https://raw.githubusercontent.com/xlang-ai/DS-1000/main/data/ds1000.jsonl.gz"
DEFAULT_CACHE = ROOT / "datasets" / "ds1000_numpy" / "ds1000.jsonl.gz"
DEFAULT_OUT_DIR = ROOT / "datasets" / "ds1000_numpy"
DEFAULT_MANIFEST = DEFAULT_OUT_DIR / "manifest.json"
DEFAULT_REPORT = DEFAULT_OUT_DIR / "FILTERING.md"


OUT_OF_SCOPE_RULES = {
    "exception_control_flow": "try/except/finally control flow",
    "external_io": "file, path, network, stdin/stdout, or NumPy file-I/O dependence",
    "nondeterminism": "randomness, current time, or process/environment dependence in the submitted solution",
    "nonfinite_float_semantics": "NaN or Inf values / predicates / propagation semantics",
    "non_numeric_data_model": "string/bytes/object/datetime/structured-record data model",
    "string_operations": "string parsing or string-method operations",
    "excluded_numpy_family": "np.linalg, np.lib, np.polyfit, and their NumPy-qualified variants",
    "external_library_runtime": "runtime dependence on non-NumPy data-science libraries",
    "opaque_python_runtime": "eval/exec/reflection/import machinery or dynamic code loading",
    "interactive_or_plotting": "interactive display, plotting, widgets, or visualization output",
    "unbounded_runtime_behavior": "data-dependent unbounded while/generator behavior not statically bounded by input shape",
}

ATTEMPT_NOTE = (
    "accepted for coverage study; unsupported operators, dynamic output shape, "
    "or current compiler gaps must be recorded as acceptance failures rather than filtered"
)

EXTERNAL_IO_CALLS = {
    "open",
    "input",
    "print",
    "np.load",
    "np.save",
    "np.loadtxt",
    "np.savetxt",
    "np.genfromtxt",
    "np.fromfile",
    "np.memmap",
    "np.DataSource",
    "numpy.load",
    "numpy.save",
    "numpy.loadtxt",
    "numpy.savetxt",
    "numpy.genfromtxt",
    "numpy.fromfile",
    "numpy.memmap",
}

NONDET_CALL_PREFIXES = {
    "np.random",
    "numpy.random",
    "random",
    "time",
    "datetime",
    "uuid",
    "os.environ",
}

OPAQUE_CALLS = {
    "eval",
    "exec",
    "compile",
    "__import__",
    "getattr",
    "setattr",
    "globals",
    "locals",
}

EXTERNAL_LIBRARY_NAMES = {
    "pandas",
    "pd",
    "scipy",
    "sp",
    "sklearn",
    "torch",
    "tensorflow",
    "tf",
    "matplotlib",
    "plt",
    "cv2",
    "PIL",
}

EXTERNAL_LIBRARY_CALLS = {
    "rankdata",
    "MinMaxScaler",
    "interp2d",
}

EXTERNAL_LIBRARY_METHODS = {
    # DS-1000 NumPy rows sometimes convert framework tensors back to NumPy.
    # Those are not pure NumPy kernels, even when the receiver is not imported
    # in the reference snippet itself.
    "numpy",
}

NON_NUMERIC_NUMPY_NAMES = {
    "object",
    "str_",
    "string_",
    "bytes_",
    "unicode_",
    "datetime64",
    "timedelta64",
    "recarray",
    "record",
    "char",
}

PANDAS_ONLY_ATTRS = {
    # DataFrame/Series attribute-access surface that has no numpy ndarray
    # equivalent. Reference snippets that touch these are pandas solutions
    # even when the snippet itself imports nothing — the DS-1000 prompt
    # supplies a `pandas.DataFrame` via the exec_context prelude.
    "values",
    "iloc",
    "loc",
    "iat",
    "iteritems",
    "itertuples",
    "to_frame",
    "to_series",
    "reset_index",
    "set_index",
}

STRING_METHODS = {
    "replace",
    "split",
    "strip",
    "lstrip",
    "rstrip",
    "lower",
    "upper",
    "startswith",
    "endswith",
}

EXCLUDED_NUMPY_PREFIXES = {
    "np.linalg",
    "numpy.linalg",
    "np.lib",
    "numpy.lib",
}

EXCLUDED_NUMPY_CALLS = {
    "np.polyfit",
    "numpy.polyfit",
}

NONFINITE_NAMES = {
    "np.nan",
    "numpy.nan",
    "np.inf",
    "numpy.inf",
    "np.NAN",
    "numpy.NAN",
    "np.Inf",
    "numpy.Inf",
    "np.isinf",
    "numpy.isinf",
    "np.isnan",
    "numpy.isnan",
    "np.isfinite",
    "numpy.isfinite",
}


def dotted_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        if base:
            return f"{base}.{node.attr}"
    return None


@dataclass
class ScopeResult:
    status: str
    reasons: list[str]
    notes: list[str]


class ScopeVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.reasons: set[str] = set()
        self.notes: set[str] = set()

    def visit_Try(self, node: ast.Try) -> None:
        self.reasons.add("exception_control_flow")
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        self.reasons.add("unbounded_runtime_behavior")
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            root = alias.name.split(".")[0]
            asname = alias.asname or root
            if root in EXTERNAL_LIBRARY_NAMES or asname in EXTERNAL_LIBRARY_NAMES:
                self.reasons.add("external_library_runtime")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        root = (node.module or "").split(".")[0]
        if root in EXTERNAL_LIBRARY_NAMES:
            self.reasons.add("external_library_runtime")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = dotted_name(node.func)
        if name:
            if name in EXTERNAL_IO_CALLS:
                self.reasons.add("external_io")
            if name in OPAQUE_CALLS or name.startswith("importlib."):
                self.reasons.add("opaque_python_runtime")
            if any(name == prefix or name.startswith(prefix + ".") for prefix in NONDET_CALL_PREFIXES):
                self.reasons.add("nondeterminism")
            if any(name == lib or name.startswith(lib + ".") for lib in EXTERNAL_LIBRARY_NAMES):
                self.reasons.add("external_library_runtime")
            if name in EXTERNAL_LIBRARY_CALLS or name.split(".")[-1] in EXTERNAL_LIBRARY_CALLS:
                self.reasons.add("external_library_runtime")
            if name.split(".")[-1] in EXTERNAL_LIBRARY_METHODS and not name.startswith(("np.", "numpy.")):
                self.reasons.add("external_library_runtime")
            if name.startswith(("plt.", "matplotlib.")):
                self.reasons.add("interactive_or_plotting")
            if any(name in {f"np.{n}", f"numpy.{n}"} or name.startswith(f"np.{n}.") for n in NON_NUMERIC_NUMPY_NAMES):
                self.reasons.add("non_numeric_data_model")
            if name.split(".")[-1] in STRING_METHODS:
                self.reasons.add("string_operations")
            if name in EXCLUDED_NUMPY_CALLS or any(
                name == prefix or name.startswith(prefix + ".") for prefix in EXCLUDED_NUMPY_PREFIXES
            ):
                self.reasons.add("excluded_numpy_family")
            if name in NONFINITE_NAMES:
                self.reasons.add("nonfinite_float_semantics")
        for keyword in node.keywords:
            if keyword.arg == "dtype":
                dtype_name = dotted_name(keyword.value)
                if dtype_name and any(dtype_name.endswith("." + n) or dtype_name == n for n in NON_NUMERIC_NUMPY_NAMES):
                    self.reasons.add("non_numeric_data_model")
                if isinstance(keyword.value, ast.Constant) and keyword.value.value in {"object", "str", "string", "datetime64"}:
                    self.reasons.add("non_numeric_data_model")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        name = dotted_name(node)
        if name and any(name == f"np.{n}" or name == f"numpy.{n}" for n in NON_NUMERIC_NUMPY_NAMES):
            self.reasons.add("non_numeric_data_model")
        if name in NONFINITE_NAMES:
            self.reasons.add("nonfinite_float_semantics")
        if node.attr in STRING_METHODS or (name and ".str." in name):
            self.reasons.add("string_operations")
        # Pandas-only attribute access on something that isn't qualified
        # by `np.`/`numpy.`. DS-1000 sometimes labels rows as "Numpy" even
        # when the reference solution operates on a DataFrame supplied by
        # the prompt's exec_context.
        if node.attr in PANDAS_ONLY_ATTRS and not (
            name and (name.startswith("np.") or name.startswith("numpy."))
        ):
            self.reasons.add("external_library_runtime")
        self.generic_visit(node)


def ensure_dataset(path: Path, url: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=30) as response:
        path.write_bytes(response.read())


def load_records(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def _free_load_names(tree: ast.AST) -> set[str]:
    """Names that are loaded before any binding in this AST. These come
    from the surrounding scope (function params or exec_context prelude),
    not from the AST itself."""
    bound: set[str] = set()
    free: set[str] = set()

    class V(ast.NodeVisitor):
        def visit_Assign(self, node: ast.Assign) -> None:
            # Walk RHS first (before the target is in scope).
            self.visit(node.value)
            for target in node.targets:
                self._collect_targets(target)

        def visit_AugAssign(self, node: ast.AugAssign) -> None:
            self.visit(node.value)
            self._collect_targets(node.target)

        def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
            if node.value is not None:
                self.visit(node.value)
            self._collect_targets(node.target)

        def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
            # The def itself binds `node.name` in the enclosing scope; we
            # don't dive into the body for free-var analysis.
            bound.add(node.name)

        def visit_Name(self, node: ast.Name) -> None:
            if isinstance(node.ctx, ast.Load) and node.id not in bound:
                free.add(node.id)

        def _collect_targets(self, node: ast.AST) -> None:
            if isinstance(node, ast.Name):
                bound.add(node.id)
            elif isinstance(node, (ast.Tuple, ast.List)):
                for el in node.elts:
                    self._collect_targets(el)
            elif isinstance(node, ast.Starred):
                self._collect_targets(node.value)
            else:
                # Subscript/Attribute targets read the base; visiting the
                # subtree is enough.
                self.visit(node)

    V().visit(tree)
    return free


def _prelude_lines_for_names(prelude: str, free_names: set[str]) -> list[ast.stmt]:
    """Return the prelude top-level statements that bind any of
    `free_names`. Used to propagate scope reasons from a prelude binding
    site into the reference (e.g. `i = np.linalg.svd(...)[1]` bound by
    the prelude is consumed as `i` by the reference)."""
    try:
        tree = ast.parse(textwrap.dedent(prelude))
    except SyntaxError:
        return []
    hits: list[ast.stmt] = []
    for stmt in tree.body:
        targets: list[ast.AST] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign):
            targets = [stmt.target]
        elif isinstance(stmt, ast.AugAssign):
            targets = [stmt.target]
        if not targets:
            continue
        names: set[str] = set()
        for t in targets:
            for sub in ast.walk(t):
                if isinstance(sub, ast.Name):
                    names.add(sub.id)
        if names & free_names:
            hits.append(stmt)
    return hits


def _extract_prelude(code_context: str) -> str:
    match = re.search(r"exec_context\s*=\s*r?\"\"\"(.*?)\"\"\"", code_context, re.S)
    if not match:
        return ""
    block = match.group(1)
    head, sep, _ = block.partition("[insert]")
    return head if sep else ""


def scope_record(record: dict[str, Any]) -> ScopeResult:
    reference = textwrap.dedent(record["reference_code"])
    try:
        tree = ast.parse(reference)
    except SyntaxError:
        return ScopeResult("out_of_scope", ["syntax_not_standalone_python"], [])

    visitor = ScopeVisitor()
    visitor.visit(tree)

    # Prelude-dependency check. The `exec_context` block above `[insert]`
    # runs before the reference snippet, so any free names the reference
    # loads come from there. If the binding statement uses an excluded
    # family (e.g. np.linalg.svd), the case effectively depends on that
    # excluded op even though the reference snippet itself looks clean.
    # Without this propagation, cases like ds1000_case0477
    # (`i = np.diag(i)` with `U, i, V = np.linalg.svd(a, ...)` in the
    # prelude) leak into the acceptance set.
    free_names = _free_load_names(tree) - {"np", "numpy", "True", "False", "None"}
    if free_names:
        prelude = _extract_prelude(record.get("code_context", ""))
        for stmt in _prelude_lines_for_names(prelude, free_names):
            prelude_visitor = ScopeVisitor()
            prelude_visitor.visit(stmt)
            visitor.reasons.update(prelude_visitor.reasons)
            if prelude_visitor.reasons:
                visitor.notes.add(
                    "scope reason propagated from exec_context prelude binding"
                )

    prompt_and_context = f"{record.get('prompt', '')}\n{record.get('code_context', '')}"
    if re.search(r"\b(file|csv|read_csv|read from|path|directory|image file)\b", prompt_and_context, re.I):
        visitor.notes.add("prompt/context mentions external data source; keep only if solution itself is pure in-memory")
    if re.search(r"\b(string|character|datetime|date|timestamp|object dtype)\b", prompt_and_context, re.I):
        visitor.notes.add("prompt/context mentions possible non-numeric data; manual audit recommended")

    reasons = sorted(visitor.reasons)
    if reasons:
        return ScopeResult("out_of_scope", reasons, sorted(visitor.notes))
    return ScopeResult("acceptance_candidate", [], sorted([ATTEMPT_NOTE, *visitor.notes]))


def record_payload(record: dict[str, Any], scope: ScopeResult) -> dict[str, Any]:
    meta = record["metadata"]
    return {
        "problem_id": meta["problem_id"],
        "library_problem_id": meta["library_problem_id"],
        "library": meta["library"],
        "status": scope.status,
        "out_of_scope_reasons": scope.reasons,
        "notes": scope.notes,
        "reference_code": record["reference_code"],
        "prompt": record["prompt"],
        "code_context": record["code_context"],
        "test_case_cnt": meta.get("test_case_cnt"),
        "perturbation_type": meta.get("perturbation_type"),
        "perturbation_origin_id": meta.get("perturbation_origin_id"),
    }


def write_program_files(out_dir: Path, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        case_dir = out_dir / f"case{row['problem_id']:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        (case_dir / "reference.py").write_text(row["reference_code"].rstrip() + "\n", encoding="utf-8")
        (case_dir / "record.json").write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")


def write_report(path: Path, rows: list[dict[str, Any]], source_url: str) -> None:
    counts = Counter(row["status"] for row in rows)
    reason_counts = Counter(reason for row in rows for reason in row["out_of_scope_reasons"])
    candidate_ids = [row["problem_id"] for row in rows if row["status"] == "acceptance_candidate"]
    excluded_ids = [row["problem_id"] for row in rows if row["status"] == "out_of_scope"]

    lines = [
        "# DS-1000 NumPy Scope Inventory",
        "",
        "Generated from the original DS-1000 JSONL metadata and reference solutions.",
        "",
        f"- Source: `{source_url}`",
        f"- Original NumPy slice: `{len(rows)}` programs",
        f"- Acceptance-study candidates: `{counts['acceptance_candidate']}`",
        f"- Out-of-scope before acceptance study: `{counts['out_of_scope']}`",
        "",
        "## Filtering Policy",
        "",
        "The acceptance study should start from all DS-1000 programs labeled `Numpy`.",
        "A program is filtered out only when it is structurally outside Zinnia's",
        "intended benchmark claim: deterministic, in-memory, numeric NumPy-style",
        "array computation that can be considered for static or bounded circuit",
        "synthesis.",
        "",
        "Do **not** filter a program merely because Zinnia currently lacks an",
        "operator, shape rule, dynamic-output path, or solver fact. Those cases must",
        "remain in the acceptance study and be counted as rejection/coverage failures.",
        "",
        "Out-of-scope categories:",
        "",
    ]
    for key, desc in OUT_OF_SCOPE_RULES.items():
        lines.append(f"- `{key}`: {desc}.")
    lines.extend(["", "## Counts", "", "| Status | Count |", "| --- | ---: |"])
    for status, count in sorted(counts.items()):
        lines.append(f"| `{status}` | `{count}` |")
    lines.extend(["", "| Out-of-scope reason | Count |", "| --- | ---: |"])
    for reason, count in reason_counts.most_common():
        lines.append(f"| `{reason}` | `{count}` |")
    lines.extend(
        [
            "",
            "## Acceptance Candidate IDs",
            "",
            ", ".join(f"`{pid}`" for pid in candidate_ids) if candidate_ids else "_none_",
            "",
            "## Out-of-Scope IDs",
            "",
        ]
    )
    for row in rows:
        if row["status"] == "out_of_scope":
            reasons = ", ".join(f"`{r}`" for r in row["out_of_scope_reasons"])
            lines.append(f"- `case{row['problem_id']:04d}`: {reasons}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `manifest.json`: full metadata, original prompt/context, reference code, status, and reasons.",
            "- `caseNNNN/reference.py`: original DS-1000 reference solution text only.",
            "- `caseNNNN/record.json`: per-case metadata copied from the manifest.",
        ]
    )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-url", default=DEFAULT_DATASET_URL)
    parser.add_argument("--dataset-cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()

    if not args.no_download:
        ensure_dataset(args.dataset_cache, args.dataset_url)
    records = load_records(args.dataset_cache)
    numpy_records = [r for r in records if r.get("metadata", {}).get("library") == "Numpy"]
    rows = [record_payload(record, scope_record(record)) for record in numpy_records]
    rows.sort(key=lambda row: row["problem_id"])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps({"programs": rows}, indent=2, sort_keys=True), encoding="utf-8")
    write_program_files(args.out_dir, rows)
    write_report(args.report, rows, args.dataset_url)

    counts = Counter(row["status"] for row in rows)
    print(json.dumps({"total_numpy": len(rows), "counts": dict(counts)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
