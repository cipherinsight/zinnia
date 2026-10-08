#!/usr/bin/env python3
"""Size sweeps of the array-fact constructions (`rules.py`), with and without each rule.

For every rule, size and configuration (the rule on; off through a measurement switch,
`docs/ablations.md`; for rules with alternatives, each construction forced), and for both
targets, one compile in a child process with the configuration's environment:

* halo2-base: `cells` (advice + lookup cells at the `k` the estimator picks), `k`, and
  `cells8` (the per-statement profile summed at `lookup_bits = 8`, comparable across
  sizes); the mock proof of the rule's input, whose outputs (decoded with the compiler's
  output layout) must equal NumPy's (the kernel run as NumPy, `harness/reference.py`);
* Noir: `gates` (`bb gates` of the emitted circuit, `noir_gates.gate_count`).

    python benchmarking/fact_constructions/run.py --out results.json \\
        [--rules range_compare,sorted_argmax] [--sizes 8,16,32] [--jobs 6] [--timeout 900]
    python benchmarking/fact_constructions/run.py --table results.json   # markdown summary

Exit status 1 when an output differs from NumPy's or a compile fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))


def measure(kernel: Path, inputs_path: Path, target: str) -> dict:
    """One compile and measurement in this process (the configuration is in os.environ)."""
    import numpy as np

    sys.path.insert(0, str(ROOT / "benchmarking"))
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    from benchmarking.harness.kernels import compile_kernel

    args = [np.array(a) if isinstance(a, list) else a for a in json.loads(inputs_path.read_text())]
    compiled = compile_kernel(kernel, "kernel", backend=target)
    record: dict = {"stmts": len(compiled.get_ir_stmts())}
    if target == "halo2":
        from zinnia._zinnia_core import halo2_base_cell_profile, halo2_base_cell_stats

        stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
        record.update(cells=stats["advice"] + stats["lookup"], k=stats["k"])
        record["cells8"] = sum(a + l for _, a, l in json.loads(halo2_base_cell_profile(compiled._compiled_ir, 8)))
        record["output"] = check_outputs(kernel, compiled, args)
    else:
        from noir_gates import NOIR_FPC, gate_count

        record.update(gate_count(compiled.emit_structural(*args, backend="noir"), "noir_fpc", NOIR_FPC))
    return record


def check_outputs(kernel: Path, compiled, args: list) -> str:
    """`match` when the mock proof's outputs equal NumPy's, else what differs."""
    from benchmarking.check_outputs import mock_exposed_values
    from benchmarking.harness.adapters.zinnia_common import describe_compiled, with_public_input_gaps, zinnia_publics
    from benchmarking.harness.case_sets import DEFAULT_TOLERANCE
    from benchmarking.harness.outputs import check_outputs as compare, reference_leaves
    from benchmarking.harness.reference import run_kernel_as_numpy

    expected, shapes = reference_leaves(run_kernel_as_numpy(kernel, "kernel", args))
    proof = compiled.prove(*args, backend="mock")
    if proof.proof_bytes_hex != "mock_satisfied":
        return "unsatisfied"
    info = describe_compiled(compiled)
    publics = zinnia_publics(with_public_input_gaps(mock_exposed_values(proof), info["public_inputs"]), info)
    check = compare(expected, publics, DEFAULT_TOLERANCE, expected_shapes=shapes)
    return "match" if check.matched else f"mismatch: {check.detail}"


def child(argv: list[str]) -> int:
    kernel, inputs, target = argv
    try:
        record = measure(Path(kernel), Path(inputs), target)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, KeyboardInterrupt):
            raise
        record = {"error": f"{type(exc).__name__}: {exc}"[:600]}
    print("RESULT " + json.dumps(record), flush=True)
    return 0


def run_one(rule, n: int, config: str, target: str, tmp: Path, timeout: float) -> dict:
    import numpy as np

    kernel = tmp / f"{rule.name}_{n}.py"
    if not kernel.exists():
        kernel.write_text(rule.source(n))
        values = rule.inputs(n, np.random.default_rng(n))
        (tmp / f"{rule.name}_{n}.json").write_text(json.dumps([v.tolist() if hasattr(v, "tolist") else v for v in values]))
    env = {**os.environ, **rule.configs[config], "ZINNIA_SKIP_RELEASE_BUILD": "1", "PYTHONPATH": str(ROOT)}
    cmd = [sys.executable, str(Path(__file__).resolve()), "--child", str(kernel), str(tmp / f"{rule.name}_{n}.json"), target]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=ROOT, env=env)
    except subprocess.TimeoutExpired:
        return {"error": f"timeout after {timeout:.0f} s"}
    lines = [line for line in proc.stdout.splitlines() if line.startswith("RESULT ")]
    if not lines:
        return {"error": (proc.stderr.strip().splitlines() or ["?"])[-1][:600]}
    return json.loads(lines[-1][len("RESULT "):])


def table(results: dict) -> str:
    """Per rule: cells8 and gates of each configuration by size, and the ratio to `with`."""
    from rules import RULES

    out = []
    for name, by_config in results.items():
        rule = RULES.get(name)
        out.append(f"### {name}\n\n{rule.description if rule else ''}\n")
        configs = list(by_config)
        sizes = sorted({int(n) for c in by_config.values() for n in c})
        head = "| n | " + " | ".join(f"{c} cells8 | {c} gates" for c in configs) + " |"
        out.append(head)
        out.append("|" + "---|" * (1 + 2 * len(configs)))
        for n in sizes:
            cells = []
            for c in configs:
                rec = by_config[c].get(str(n), {})
                h, z = rec.get("halo2", {}), rec.get("noir", {})
                cells.append(str(h.get("cells8", h.get("error", "-")[:30])))
                cells.append(str(z.get("gates", z.get("error", "-")[:30])))
            out.append(f"| {n} | " + " | ".join(cells) + " |")
        out.append("")
    return "\n".join(out)


def main() -> int:
    if len(sys.argv) > 1 and sys.argv[1] == "--child":
        return child(sys.argv[2:])
    from rules import RULES

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--rules", default=",".join(RULES))
    ap.add_argument("--sizes", help="comma-separated sizes (default: each rule's)")
    ap.add_argument("--targets", default="halo2,noir")
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--table", type=Path, help="print the markdown summary of a results file")
    args = ap.parse_args()
    if args.table:
        print(table(json.loads(args.table.read_text())))
        return 0
    results: dict = json.loads(args.out.read_text()) if args.out and args.out.exists() else {}
    tmp = Path(tempfile.mkdtemp(prefix="fact-constructions-"))
    jobs = []
    for name in args.rules.split(","):
        rule = RULES[name]
        sizes = [int(s) for s in args.sizes.split(",")] if args.sizes else list(rule.sizes)
        for n in sizes:
            for config in rule.configs:
                for target in args.targets.split(","):
                    jobs.append((rule, n, config, target))
    failed = False
    with ThreadPoolExecutor(args.jobs) as pool:
        for (rule, n, config, target), record in zip(jobs, pool.map(lambda j: run_one(*j, tmp, args.timeout), jobs)):
            results.setdefault(rule.name, {}).setdefault(config, {}).setdefault(str(n), {})[target] = record
            bad = "error" in record or record.get("output", "match") != "match"
            failed |= bad
            print(f"{rule.name} n={n} {config} {target}: {record if bad else {k: record.get(k) for k in ('cells8', 'cells', 'gates', 'output')}}",
                  flush=True)
            if args.out:
                args.out.write_text(json.dumps(results, indent=1, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
