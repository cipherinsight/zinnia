#!/usr/bin/env python3
"""Walkthrough (Sec. 6) on the final compiler, halo2-base only, run on the remote container.

Usage (in the remote container, from this directory):
    python run_final.py            # both configurations, each in a child process
    python run_final.py MODE       # one configuration: default | local

MODE default = no ablation switch; local = ZINNIA_ABLATE_JOINT=1 (per-operator selection).
Writes MODE/{result.json, selection.jsonl, layer_input.txt, ir.json, ir.txt, layout.json}.
Cells = halo2_base_cell_stats advice + lookup, as in benchmarking/ablations/common.py.
"""
import hashlib, json, os, subprocess, sys
from pathlib import Path

D = Path(__file__).resolve().parent
WT = Path(os.environ.get("ZINNIA_WT", "/home/zhantong/zinnia-eval-run/zinnia-sizefix"))
sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["ZINNIA_SKIP_RELEASE_BUILD"] = "1"

if len(sys.argv) == 1:
    for mode in ["default", "local"]:
        env = {k: v for k, v in os.environ.items()
               if not k.startswith("ZINNIA_ABLATE_")
               and k not in ("ZINNIA_FORCE_CONSTRUCTION", "ZINNIA_SMT_ENABLE", "ZINNIA_REQ_DISABLE")}
        if mode == "local":
            env["ZINNIA_ABLATE_JOINT"] = "1"
        subprocess.run([str(WT / ".venv/bin/python"), str(D / "run_final.py"), mode],
                       env=env, cwd=D, check=True)
    sys.exit()

mode = sys.argv[1]
out = D / mode
out.mkdir(exist_ok=True)
for name in ("selection.jsonl", "layer_input.txt"):
    (out / name).unlink(missing_ok=True)
os.environ["ZINNIA_LAYER_REPORT"] = str(out / "selection.jsonl")
os.environ["ZINNIA_LAYER_INPUT"] = str(out / "layer_input.txt")
sys.path[:0] = [str(WT), str(WT / "benchmarking")]
from zinnia import ZKCircuit
from zinnia.config.zinnia_config import ZinniaConfig
from zinnia._zinnia_core import halo2_base_cell_stats
from benchmarking.harness.fixtures import load_fixture
from benchmarking.harness.acceptance.validate import _mock_publics
from benchmarking.harness.adapters.zinnia_common import describe_compiled, zinnia_publics
from benchmarking.harness.reference import run_kernel_with_arguments
from benchmarking.harness.outputs import reference_outputs, check_outputs
from benchmarking.harness.case_sets import Tolerance
from benchmarking.ablations.common import summarize_report

compiled = ZKCircuit("kernel", (D / "kernel.py").read_text(),
                     config=ZinniaConfig(backend="halo2")).compile()
stmts = compiled.get_ir_stmts()
(out / "ir.json").write_text(json.dumps(stmts, indent=2))
info = describe_compiled(compiled)
(out / "layout.json").write_text(json.dumps(info, indent=2))
fixture = load_fixture(D / "fixture.json")
checks = {}
for name in ["A", "B"]:
    args = fixture.arguments(name)
    mock = compiled.prove(*args, backend="mock")  # mock interpreter only
    values, satisfied = _mock_publics(mock)
    returned, after = run_kernel_with_arguments(D / "kernel.py", "kernel", args)
    expected, shapes = reference_outputs(returned, after, args, info["output_layout"])
    check = check_outputs(expected, zinnia_publics(values, info), Tolerance(0, 0, "exact integer"),
                          expected_shapes=shapes).to_json()
    checks[name] = {"satisfied": satisfied, "status": check["status"],
                    "numpy_result": returned.tolist(), "numpy_shape": list(returned.shape)}
    assert satisfied and check["status"] == "match", checks[name]
stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
result = {
    "program": "walkthrough/kernel", "mode": mode, "target": "halo2",
    "commit": subprocess.check_output(["git", "-C", str(WT), "rev-parse", "HEAD"], text=True).strip(),
    "dirty": subprocess.check_output(["git", "-C", str(WT), "status", "--porcelain",
                                      "--untracked-files=no"], text=True).strip(),
    "build": json.loads((WT / "target/zinnia-harness-build.json").read_text())["source"],
    "source_sha256": hashlib.sha256((D / "kernel.py").read_bytes()).hexdigest(),
    "host": os.uname().nodename,
    "cells": stats["advice"] + stats["lookup"], "stats": stats, "ir_stmts": len(stmts),
    "checks": checks,
    "layer": summarize_report((out / "selection.jsonl").read_text().splitlines()),
}
(out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
with open(out / "ir.txt", "w") as f:
    for s in stmts:
        ir = s["ir_instance"]
        f.write(f"{s['stmt_id']}: {ir['__class__']} {s['arguments']} {ir['ir_data']}\n")
print(mode, json.dumps({k: result[k] for k in ("commit", "dirty", "host", "cells", "ir_stmts")}),
      "advice", stats["advice"], "lookup", stats["lookup"], "k", stats["k"], flush=True)
