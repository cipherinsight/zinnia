"""Freeze and validate every submitted program without algorithm rewrites."""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from ..tasks import TASKS, fixtures, oracle, signatures
from ..validate import environment, python_function


ROUND = Path(__file__).with_name("round1")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def function_tree(text):
    tree = ast.parse(text)
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("submission interface requires one top-level kernel function")
    fn = tree.body[0]
    if fn.name != "kernel" or fn.decorator_list:
        raise ValueError("submission interface requires an undecorated kernel")
    return tree, fn


def normalized_body(text):
    """Count syntactically distinct submissions, ignoring comments/docstrings."""
    tree, fn = function_tree(text)
    if isinstance(fn.body[0], ast.Expr) and isinstance(fn.body[0].value, ast.Constant) and isinstance(fn.body[0].value.value, str):
        fn.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def freeze():
    path = ROUND / "frozen.json"
    if path.exists():
        raise ValueError("submissions are already frozen; do not overwrite the manifest")
    protocol = json.loads((ROUND / "protocol.json").read_text())
    programs = []
    for implementer, model in protocol["models"].items():
        for task in protocol["task_names"]:
            file = ROUND / "submissions" / implementer / f"{task}.py"
            raw = file.read_text()
            try:
                body_hash = sha(normalized_body(raw))
                interface_error = None
            except Exception as exc:
                body_hash = None
                interface_error = f"{type(exc).__name__}: {exc}"
            programs.append({"implementer": implementer, "model": model, "task": task,
                             "path": str(file.relative_to(ROUND)), "sha256": sha(raw),
                             "normalized_body_sha256": body_hash, "interface_error": interface_error})
    manifest = {"frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "brief_sha256": sha((ROUND / "brief.md").read_text()),
                "protocol_sha256": sha((ROUND / "protocol.json").read_text()),
                "environment": environment(), "programs": programs}
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Frozen {len(programs)} submissions before compiler feedback.")


def specialize(text, task, profile):
    tree, fn = function_tree(text)
    shapes, _ = signatures(task, profile)
    if [arg.arg for arg in fn.args.args] != list(shapes):
        raise ValueError("submission argument names do not match the provided interface")
    for arg in fn.args.args:
        dims = " ".join(map(str, shapes[arg.arg]))
        arg.annotation = ast.parse(f'Integer[Array, "{dims}"]', mode="eval").body
    if task.name == "smooth":
        statements = [node for node in fn.body if not (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))]
        config = statements[0]
        if not (isinstance(config, ast.Assign) and len(config.targets) == 1
                and isinstance(config.targets[0], ast.Name) and config.targets[0].id == "R"
                and isinstance(config.value, ast.Constant) and config.value.value == 3):
            raise ValueError("smoothing must start with the supplied public configuration R = 3")
        config.value = ast.Constant(profile["R"])
    return ast.unparse(ast.fix_missing_locations(tree))


def instrument(text, shape):
    tree, fn = function_tree(text)
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    used.update(node.arg for node in ast.walk(tree) if isinstance(node, ast.arg))

    def fresh(stem):
        name = stem
        while name in used:
            name += "_"
        used.add(name)
        return name

    expected, actual = fresh("pilot_expected"), fresh("pilot_actual")
    dims = " ".join(map(str, shape))
    fn.args.args.append(ast.arg(arg=expected, annotation=ast.parse(f'Integer[Array, "{dims}"]', mode="eval").body))

    class Returns(ast.NodeTransformer):
        def visit_FunctionDef(self, node):
            # Leave local helpers' return contracts intact.
            return node

        def visit_Return(self, node):
            result = ast.parse(f"{actual} = np.asarray(None)").body
            result[0].value.args[0] = node.value if node.value is not None else ast.Constant(None)
            result += ast.parse(f"assert {actual}.ndim == {len(shape)}").body
            for axis, size in enumerate(shape):
                result += ast.parse(f"assert {actual}.shape[{axis}] == {size}").body
            result += ast.parse(f"assert np.all({actual} == {expected})\nreturn").body
            return result

    transform = Returns()
    checked = []
    for node in fn.body:
        result = transform.visit(node)
        checked.extend(result if isinstance(result, list) else [result])
    fn.body = checked
    return ast.unparse(ast.fix_missing_locations(tree))


def inputs_for(task, profile_index):
    profile = task.profiles[profile_index]
    cases = fixtures(task, profile)
    shapes, _ = signatures(task, profile)
    seed = 913701 + 10000 * next(i for i, item in enumerate(TASKS) if item.name == task.name) + 100 * profile_index
    rng = np.random.default_rng(seed)
    for _ in range(12):
        cases.append([rng.integers(0 if task.name == "smooth" else -512, 4097, size=shape, dtype=np.int64)
                      for shape in shapes.values()])
    return cases


def check_one(program, profile_index, backend, fixture_index=None):
    from zinnia import ZKCircuit

    task = next(task for task in TASKS if task.name == program["task"])
    row = {"implementer": program["implementer"], "model": program["model"], "task": task.name,
           "profile": profile_index, "backend": backend, "submission_sha256": program["sha256"],
           "status": "fail", "stage": "prepare", "python_checked_inputs": 0,
           "checked_inputs": 0, "original_compiled": False, "checked_compiled": False,
           "negative_control_passed": False}
    started = time.monotonic()
    try:
        raw = (ROUND / program["path"]).read_text()
        assert sha(raw) == program["sha256"], "frozen submission was modified"
        text = specialize(raw, task, task.profiles[profile_index])
        row["specialized_source_sha256"] = sha(text)
        values = inputs_for(task, profile_index)
        if fixture_index is not None:
            values = [values[fixture_index]]
        expected = [oracle(task, task.profiles[profile_index], case) for case in values]
        row["stage"] = "python_oracle"
        try:
            reference = python_function(text)
            for case, answer in zip(values, expected):
                actual = np.asarray(reference(*copy.deepcopy(case)))
                assert actual.shape == answer.shape, "incorrect output shape"
                np.testing.assert_array_equal(actual, answer)
                row["python_checked_inputs"] += 1
            row["python_correct"] = True
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            row["python_correct"] = False
            row["python_error"] = f"{type(exc).__name__}: {exc}"
        # A reference failure must not prevent reporting whether it compiles.
        row["stage"] = "compile_original"
        ZKCircuit.from_source("kernel", text).compile()
        row["original_compiled"] = True
        row["stage"] = "compile_checked"
        _, shape = signatures(task, task.profiles[profile_index])
        checked = ZKCircuit.from_source("kernel", instrument(text, shape))
        checked.compile()
        row["checked_compiled"] = True
        row["stage"] = "execute"
        for case, answer in zip(values, expected):
            proof = checked.prove(*copy.deepcopy(case), answer, backend=backend)
            if backend == "mock":
                assert proof.proof_bytes_hex == "mock_satisfied", "correct answer rejected"
            else:
                assert checked.verify(proof), "proof failed verification"
            row["checked_inputs"] += 1
        row["stage"] = "negative_control"
        wrong = expected[-1].copy()
        wrong.flat[0] += 1
        proof = checked.prove(*copy.deepcopy(values[-1]), wrong, backend="mock")
        assert proof.proof_bytes_hex != "mock_satisfied", "incorrect answer accepted"
        row["negative_control_passed"] = True
        assert row["python_correct"], row.get("python_error")
        row.update(status="pass", stage="complete")
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.monotonic() - started, 3)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--implementer")
    parser.add_argument("--task")
    parser.add_argument("--profile", type=int)
    parser.add_argument("--fixture", type=int)
    parser.add_argument("--backend", choices=("mock", "halo2"), default="mock")
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.freeze:
        freeze()
        return
    manifest = json.loads((ROUND / "frozen.json").read_text())
    selected = [p for p in manifest["programs"] if (not args.implementer or p["implementer"] == args.implementer)
                and (not args.task or p["task"] == args.task)]
    if args.worker:
        assert len(selected) == 1
        print(json.dumps(check_one(selected[0], args.profile, args.backend, args.fixture)))
        return
    fingerprint = environment()
    rows = []
    for program in selected:
        task = next(t for t in TASKS if t.name == program["task"])
        for profile in range(len(task.profiles)):
            if args.profile is not None and profile != args.profile:
                continue
            command = [sys.executable, "-m", "studies.user_study.pilots.validate", "--worker",
                       "--implementer", program["implementer"], "--task", task.name,
                       "--profile", str(profile), "--backend", args.backend]
            if args.fixture is not None:
                command += ["--fixture", str(args.fixture)]
            try:
                run = subprocess.run(command, capture_output=True, text=True, timeout=args.timeout)
                row = json.loads(run.stdout.strip().splitlines()[-1])
                if run.returncode:
                    raise ValueError(f"worker exit {run.returncode}: {run.stderr[-2000:]}")
            except (subprocess.TimeoutExpired, ValueError, IndexError) as exc:
                row = {"implementer": program["implementer"], "model": program["model"], "task": task.name,
                       "profile": profile, "status": "fail", "stage": "worker", "error": str(exc)}
            rows.append(row)
            if row["status"] != "pass":
                print(f'{program["implementer"]}/{task.name}/{profile}: {row["stage"]}: {row.get("error")}', flush=True)
    report = {"environment": fingerprint, "pilot_validator_sha256": sha(Path(__file__).read_text()),
              "frozen_manifest_sha256": sha((ROUND / "frozen.json").read_text()), "backend": args.backend,
              "selection": {"implementer": args.implementer, "task": args.task, "profile": args.profile,
                            "fixture": args.fixture, "timeout_seconds": args.timeout},
              "passed": sum(r["status"] == "pass" for r in rows), "total": len(rows), "rows": rows}
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f'{report["passed"]}/{report["total"]} implementation/profile checks passed', flush=True)
    raise SystemExit(0 if rows and report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
