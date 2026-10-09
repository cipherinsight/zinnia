"""Instantiate handwritten NPBench/Pythran baselines at materialized fixture sizes.

Only the literal values of annotated top-level size constants may change. This
is a source audit, not a compilation, soundness or output-validation result.
Use the integrated handwritten checkout as --baseline-root; its sizes.py is the
single formula implementation (including factorial). No timing jobs are run.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import re
import sys
from collections import Counter
from pathlib import Path
from types import ModuleType

SYSTEMS = {"halo2_hw": ("target", "rs"), "noir_hw": ("main", "nr")}


def load_sizes(root: Path) -> ModuleType:
    path = root / "benchmarking/baselines/hw_check/sizes.py"
    spec = importlib.util.spec_from_file_location("_scaling_npb_hw_sizes", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load size formulas from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolves annotations here
    spec.loader.exec_module(module)
    return module


def _literal_mask(text: str, sizes: ModuleType) -> str:
    """Preserve every byte except annotated declaration value spans."""
    result = []
    for line in text.splitlines(keepends=True):
        match = sizes.DECLARATION.match(line.rstrip("\r\n"))
        if match:
            line = line[:match.start("value")] + "<SIZE>" + line[match.end("value"):]
        result.append(line)
    return "".join(result)


def check_rewrite(source: str, candidate: str, original: dict, scaled: dict,
                  sizes: ModuleType) -> list[dict]:
    """Independently reject all changes except formula-justified size literals.

    Includes whitespace, comments, formulas and unchanged literal validation;
    unlike a token diff, it also catches changes to literal digits in strings.
    """
    if _literal_mask(source, sizes) != _literal_mask(candidate, sizes):
        raise ValueError("non-size source changed (only annotated literal values may differ)")
    old = sizes.size_constants(source.splitlines())
    new = sizes.size_constants(candidate.splitlines())
    if len(old) != len(new):
        raise ValueError("size declaration count changed")
    changes = []
    for before, after in zip(old, new):
        for constant, fixture in ((before, original), (after, scaled)):
            expected = sizes.evaluate(constant.formula, sizes.dimensions(fixture))
            if expected < 0 or expected != constant.value:
                raise ValueError(f"line {constant.line}: {constant.name} = {constant.value}, "
                                 f"but {constant.formula} = {expected}")
        # These declarations are unsigned size types. Reject an unrepresentable
        # literal before writing source that cannot compile.
        declaration = candidate.splitlines()[after.line - 1]
        width = re.search(r":\s*u(8|16|32|64|128)\s*=", declaration)
        if width and after.value >= 1 << int(width[1]):
            raise ValueError(f"line {after.line}: {after.name} does not fit u{width[1]}")
        changes.append({"line": before.line, "name": before.name,
                        "old": before.value, "new": after.value, "formula": before.formula})
    if not old and sizes.dimensions(original) != sizes.dimensions(scaled):
        raise ValueError("changed array shapes but no annotated size constants; manual review required")
    return changes


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def generate(data: Path, baseline_root: Path, *, write: bool = False,
             caps: set[int] | None = None) -> dict:
    """Audit every point; optionally write verified candidates beside scaled fixtures.

    Missing implementations and failures remain explicit rows. Size-like literal
    warnings are preserved for review, never silently certified as scalable.
    Existing baseline files may only be reused if byte-identical. Inputs and
    case-set metadata are not changed. Prover TOML must be regenerated from the
    compiled scaled ABI by the handwritten driver, not copied from CAP 16.
    """
    data, baseline_root = data.resolve(), baseline_root.resolve()
    sizes = load_sizes(baseline_root)
    bench = baseline_root / "benchmarking"
    originals = {c["case"]: c for c in json.loads(
        (bench / "datasets/case_sets/npbench_pythran_perf.json").read_text())["cases"]}
    reviews = json.loads(Path(__file__).with_name("handwritten_reviews.json").read_text())
    rows = []
    for point in json.loads((data / "points.json").read_text())["points"]:
        if caps is not None and point["cap"] not in caps:
            continue
        for system, (stem, suffix) in SYSTEMS.items():
            row = {"case": point["case"], "cap": point["cap"], "system": system}
            rows.append(row)
            if point["status"] != "scaled":
                row.update(status=point["status"], reason=point.get("reason"),
                           same_as=point.get("same_as"))
                continue
            original_case = point["case"].removesuffix(f"_cap{point['cap']}")
            raw = originals[original_case]
            filename = f"{stem}.{raw['function']}.{suffix}"
            source_path = bench / raw["path"] / filename
            row["source"] = str(source_path)
            if not source_path.is_file():
                row.update(status="missing", reason="original-size baseline is absent")
                continue
            folder = (data / point["folder"]).resolve()
            if not folder.is_relative_to(data):
                raise ValueError(f"scaled folder escapes DATA: {folder}")
            dest = folder / filename
            row["output"] = str(dest)
            try:
                original = json.loads((source_path.parent / (raw.get("fixture") or "fixture.json")).read_text())
                scaled = json.loads((folder / point["fixture"]).read_text())
                errors, warnings = sizes.check(source_path, original)
                row["warnings"] = warnings
                if errors:
                    raise ValueError("; ".join(errors))
                source = source_path.read_bytes().decode()
                candidate = sizes.instantiate(source, scaled)
                row["changes"] = check_rewrite(source, candidate, original, scaled, sizes)
                row.update(source_sha256=_digest(source), output_sha256=_digest(candidate),
                           body_sha256=_digest(_literal_mask(source, sizes)))
                if dest.exists() and dest.read_bytes() != candidate.encode():
                    raise ValueError("existing output differs; refusing to overwrite it")
                # Warnings can be harmless numerical constants, but this command
                # cannot decide that. Such candidates require an explicit review.
                review = reviews.get(f"{original_case}:{system}", {})
                fixed = review.get("fixed_dimensions", {})
                reviewed = (review.get("source_sha256") == row["source_sha256"]
                            and all(sizes.dimensions(original).get(k) == v
                                    and sizes.dimensions(scaled).get(k) == v for k, v in fixed.items()))
                if reviewed and warnings:
                    row["literal_review"] = review
                pending = bool(warnings) and not reviewed
                row["status"] = "needs_review" if pending else "rewritten"
                row["written"] = write and not pending
                if row["written"]:
                    dest.write_bytes(candidate.encode())
            except (ValueError, SyntaxError, ArithmeticError, OSError) as exc:
                row.update(status="not_scalable", reason=str(exc))
    counts = Counter(row["status"] for row in rows)
    return {"baseline_root": str(baseline_root), "data": str(data), "write": write,
            "validation": "source diff only; compile and fixture output checks remain required",
            "counts": dict(sorted(counts.items())), "points": rows}


def main(args) -> int:
    report = generate(args.data, args.baseline_root, write=args.write,
                      caps={int(c) for c in args.caps.split(",")} if args.caps else None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["counts"], sort_keys=True))
    # Missing/review rows are coverage, not failed invocations. Formula/body
    # mismatches are failures so automation cannot quietly omit them.
    return int(bool(report["counts"].get("not_scalable")))
