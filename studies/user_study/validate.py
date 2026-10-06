"""Compile every inventoried implementation and check private-input semantics.

Run from zinnia-src: .venv/bin/python -m studies.user_study.validate --output PATH
No expected failures or exclusions are permitted in the inventory gate.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import platform
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from .tasks import TASKS, fixtures, oracle, signatures, source


def environment():
    """Fingerprint the checkout and loaded extension without recording user data."""
    root = Path(__file__).resolve().parents[2]

    def git(*args):
        run = subprocess.run(["git", *args], cwd=root, capture_output=True, text=True)
        return run.stdout.strip() if run.returncode == 0 else None

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    extension = importlib.util.find_spec("zinnia._zinnia_core")
    # Include the complete working source, including unrelated local development
    # changes; HEAD alone does not identify this dirty development checkout.
    paths = [root / "Cargo.toml", root / "Cargo.lock", root / "pyproject.toml"]
    paths += sorted((root / "src").rglob("*.rs"))
    paths += sorted((root / "zinnia").rglob("*.py"))
    digest = hashlib.sha256()
    for path in paths:
        if path.is_file():
            digest.update(str(path.relative_to(root)).encode() + b"\0")
            digest.update(path.read_bytes() + b"\0")
    return {
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_head": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "working_source_sha256": digest.hexdigest(),
        "extension_sha256": sha(extension.origin) if extension and extension.origin else None,
        "task_inventory_sha256": sha(Path(__file__).with_name("tasks.py")),
        "validator_sha256": sha(__file__),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "platform": platform.platform(),
    }


def checked_source(text, output_shape):
    """The variant with an `expected` argument and a check of its result.

    Every `return v` of the variant's own body (not of helpers or lambdas
    defined inside it) becomes the checks of `v` against `expected`, followed by
    `return` of the checked value, so an early return still ends the function
    exactly where the variant does.
    """
    tree = ast.parse(text)
    fn = tree.body[0]
    shape = " ".join(map(str, output_shape))
    fn.args.args.append(ast.arg(arg="expected", annotation=ast.parse(f'Integer[Array, "{shape}"]', mode="eval").body))
    actual = "gate_checked_result"

    class CheckReturns(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Only the variant itself; a nested helper's returns are its own.
            return self.generic_visit(node) if node is fn else node

        def visit_Lambda(self, node):
            return node

        def visit_Return(self, node):
            checks = ast.parse(f"{actual} = np.asarray(None)").body
            checks[0].value.args[0] = node.value
            checks += ast.parse(f"assert {actual}.ndim == {len(output_shape)}").body
            for axis, size in enumerate(output_shape):
                checks += ast.parse(f"assert {actual}.shape[{axis}] == {size}").body
            checks += ast.parse(f"assert np.all({actual} == expected)").body
            checks += ast.parse(f"return {actual}").body
            return checks

    return ast.unparse(ast.fix_missing_locations(CheckReturns().visit(tree)))


def python_function(text):
    tree = ast.parse(text)
    for node in ast.walk(tree):
        if isinstance(node, ast.arg):
            node.annotation = None
        if isinstance(node, ast.FunctionDef):
            node.returns = None
    env = {"np": np}
    exec(compile(ast.fix_missing_locations(tree), "<study-variant>", "exec"), env)
    return env["kernel"]


def check_one(task, variant, profile_index, backend, fixture_index=None):
    from zinnia import ZKCircuit

    p = task.profiles[profile_index]
    text = source(task, variant, p)
    started = time.monotonic()
    row = {"task": task.name, "variant": variant, "profile": profile_index,
           "source_sha256": hashlib.sha256(text.encode()).hexdigest(), "backend": backend,
           "stage": "python_oracle", "status": "fail", "checked_inputs": 0}
    try:
        inputs = fixtures(task, p)
        if fixture_index is not None:
            inputs = [inputs[fixture_index]]
        expected = [oracle(task, p, values) for values in inputs]
        reference = python_function(text)
        for values, result in zip(inputs, expected):
            actual = np.asarray(reference(*copy.deepcopy(values)))
            np.testing.assert_array_equal(actual, result, strict=True)
        row["stage"] = "compile_original"
        ZKCircuit.from_source("kernel", text).compile()
        row["stage"] = "compile_checked"
        _, out_shape = signatures(task, p)
        circuit = ZKCircuit.from_source("kernel", checked_source(text, out_shape))
        circuit.compile()
        row["stage"] = "execute"
        for values, result in zip(inputs, expected):
            proof = circuit.prove(*copy.deepcopy(values), result, backend=backend)
            if backend == "mock":
                assert proof.proof_bytes_hex == "mock_satisfied", "correct answer rejected"
            else:
                assert circuit.verify(proof), "proof verification failed"
            row["checked_inputs"] += 1
        row["stage"] = "negative_control"
        wrong = expected[-1].copy()
        wrong.flat[0] += 1
        invalid = circuit.prove(*copy.deepcopy(inputs[-1]), wrong, backend="mock")
        assert invalid.proof_bytes_hex != "mock_satisfied", "incorrect output was accepted"
        row.update(status="pass", stage="complete")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.monotonic() - started, 3)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--task")
    parser.add_argument("--variant")
    parser.add_argument("--profile", type=int)
    parser.add_argument("--fixture", type=int)
    parser.add_argument("--backend", choices=("mock", "halo2"), default="mock")
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        task = next(t for t in TASKS if t.name == args.task)
        print(json.dumps(check_one(task, args.variant, args.profile, args.backend, args.fixture)))
        return
    fingerprint = environment()
    rows = []
    for task in TASKS:
        if args.task and task.name != args.task:
            continue
        for variant in task.variants:
            if args.variant and args.variant != variant:
                continue
            for profile in range(len(task.profiles)):
                if args.profile is not None and args.profile != profile:
                    continue
                command = [sys.executable, "-m", __package__ + ".validate", "--worker", "--task", task.name,
                           "--variant", variant, "--profile", str(profile), "--backend", args.backend]
                if args.fixture is not None:
                    command += ["--fixture", str(args.fixture)]
                try:
                    run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                    row = json.loads(run.stdout.strip().splitlines()[-1])
                except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
                    row = {"task": task.name, "variant": variant, "profile": profile, "status": "fail",
                           "stage": "worker", "error": str(exc)}
                rows.append(row)
                if row["status"] != "pass":
                    print(f'{task.name}/{variant}/{profile}: {row["stage"]}: {row.get("error", "")}', flush=True)
    report = {"inventory_only": True, "backend": args.backend,
              "environment": fingerprint,
              "selection": {"task": args.task, "variant": args.variant,
                            "profile": args.profile, "fixture": args.fixture,
                            "timeout_seconds": args.timeout},
              "rows": rows,
              "passed": sum(r["status"] == "pass" for r in rows), "total": len(rows)}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f'{report["passed"]}/{report["total"]} implementation/profile checks passed', flush=True)
    raise SystemExit(0 if rows and report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
