"""Scaling series of the challenge constructions (port of
eval-redo/layer/layer-challenge-data/scale.py, paths as arguments).

Families: argsort (DS-1000 0381's kernel `np.argsort(a)[::-1][:len(a)]`, N floats),
matmul (`a @ b`, N x N floats), convolve (`np.convolve(a, v)`, two length-N float arrays).
Each point: compile, halo2-base cells (advice + lookup at the k estimate_params picks),
and with --time keygen on input A, 3 proofs on input B (median), each verified, a tampered
public value rejected, outputs compared with NumPy. `--force` selects a construction through
ZINNIA_FORCE_CONSTRUCTION (a measurement aid); `--force ''` is the compiler's own choice.

    python scale_challenge.py run --wt WT --out OUT --family argsort --sizes 16,64 \
        --force "sort: grand-product permutation" --label gp --time [--threads 8 --mem 30G]

Scaled kernels go to OUT/scaled/<family><NNNN>/ (same text and seeds as the local run);
records (one JSON line each) are appended to OUT/scale_<family>.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent

KERNELS = {
    "argsort": ('@zk_circuit\ndef kernel(a: Float[Array, "{n}"]):\n'
                '    result = np.argsort(a)[::-1][:len(a)]\n    return result\n'),
    "matmul": ('@zk_circuit\ndef kernel(a: Float[Array, "{n} {n}"], b: Float[Array, "{n} {n}"]):\n'
               '    return a @ b\n'),
    "convolve": ('@zk_circuit\ndef kernel(a: Float[Array, "{n}"], v: Float[Array, "{n}"]):\n'
                 '    return np.convolve(a, v)\n'),
}


def case_dir(out: Path, family: str, n: int) -> Path:
    return out / "scaled" / f"{family}{n:04d}"


def inputs(family: str, n: int, seed: int) -> dict:
    rng = np.random.RandomState(seed)
    if family == "argsort":
        return {"a": ((rng.rand(n) - 0.5) * 100).tolist()}
    if family == "matmul":
        return {"a": ((rng.rand(n, n) - 0.5) * 20).tolist(), "b": ((rng.rand(n, n) - 0.5) * 20).tolist()}
    return {"a": ((rng.rand(n) - 0.5) * 20).tolist(), "v": ((rng.rand(n) - 0.5) * 20).tolist()}


def gen(out: Path, family: str, n: int) -> Path:
    folder = case_dir(out, family, n)
    folder.mkdir(parents=True, exist_ok=True)
    header = (f"from zinnia import *\nimport numpy as np\n\n# Scaled kernel of the layer-challenge scaling runs "
              f"({family}, N = {n}).\n\n")
    (folder / "kernel.py").write_text(header + KERNELS[family].format(n=n))
    fixture = {"inputs": {"A": inputs(family, n, 1000 + n), "B": inputs(family, n, 2000 + n)}}
    (folder / "fixture.json").write_text(json.dumps(fixture))
    return folder


def numpy_reference(family: str, args: dict) -> np.ndarray:
    if family == "argsort":
        a = np.array(args["a"])
        return np.argsort(a)[::-1][:len(a)]
    if family == "matmul":
        return np.array(args["a"]) @ np.array(args["b"])
    return np.convolve(np.array(args["a"]), np.array(args["v"]))


def child(wt: Path, out: Path, family: str, n: int, timed: bool) -> dict:
    sys.path.insert(0, str(wt))
    from benchmarking.harness.kernels import compile_kernel
    from zinnia._zinnia_core import halo2_base_cell_stats

    folder = case_dir(out, family, n)
    fixture = json.loads((folder / "fixture.json").read_text())
    started = time.perf_counter()
    compiled = compile_kernel(folder / "kernel.py", "kernel")
    record: dict = {"family": family, "n": n, "compile_s": round(time.perf_counter() - started, 3)}
    stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
    record.update({k: stats.get(k) for k in ("k", "lookup_bits", "advice", "lookup", "second_advice", "challenges")})
    record["cells"] = stats["advice"] + stats["lookup"]
    if timed:
        from benchmarking.harness.adapters.zinnia_common import describe_compiled, zinnia_publics
        from benchmarking.harness.outputs import field_from_le_hex, field_to_le_hex

        args_b = list(fixture["inputs"]["B"].values())
        started = time.perf_counter()
        keys = compiled.setup()
        record["setup_s"] = round(time.perf_counter() - started, 3)
        walls, proofs = [], []
        for _ in range(3):
            started = time.perf_counter()
            proof = compiled.prove_with_keys(keys, *args_b)
            walls.append(time.perf_counter() - started)
            proofs.append(proof)
        record["prove_wall_s"] = [round(s, 3) for s in walls]
        record["prove_median_s"] = round(statistics.median(walls), 3)
        record["create_proof_s"] = [round(p["prove_seconds"], 3) for p in proofs]
        record["create_proof_median_s"] = round(statistics.median(p["prove_seconds"] for p in proofs), 3)
        record["witness_median_s"] = round(statistics.median(p["witness_seconds"] for p in proofs), 3)
        record["verified"] = all(compiled.verify_with_keys(keys, p) for p in proofs)
        tampered = dict(proofs[0])
        first = field_from_le_hex(tampered["public_values"][0])
        tampered["public_values"] = [field_to_le_hex(first + 1), *tampered["public_values"][1:]]
        record["tamper_rejected"] = not compiled.verify_with_keys(keys, tampered)
        fields = [field_from_le_hex(v) for v in proofs[0]["public_values"]]
        decoded = zinnia_publics(fields, describe_compiled(compiled))
        got = np.array(decoded.values, dtype=float).reshape(-1)
        ref = numpy_reference(family, fixture["inputs"]["B"]).reshape(-1)
        record["max_abs_error"] = float(np.max(np.abs(got - ref)))
    return record


def run(wt: Path, out: Path, family: str, sizes: list[int], force: str, label: str, timed: bool,
        threads: int, mem: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / f"scale_{family}.jsonl"
    py = str(wt / ".venv" / "bin" / "python")
    for n in sizes:
        gen(out, family, n)
        env = dict(os.environ, ZINNIA_FORCE_CONSTRUCTION=force, ZWT=str(wt),
                   RAYON_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                   HARDWARE_CONCURRENCY=str(threads))
        cmd = [py, str(HERE / "capped.py"), mem, "7200", py, __file__, "child", "--wt", str(wt),
               "--out", str(out), "--family", family, "--sizes", str(n)] + (["--time"] if timed else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        rec = json.loads(lines[-1]) if lines else {"family": family, "n": n, "error": proc.stderr[-2000:]}
        peak = [l for l in proc.stderr.splitlines() if l.startswith("capped: peak_rss_bytes=")]
        if peak:
            rec["peak_rss_bytes"] = int(peak[-1].split("=")[1])
        rec.update(label=label, force=force, threads=threads, memory_cap=mem, exit=proc.returncode)
        print(json.dumps(rec), flush=True)
        with out_file.open("a") as f:
            f.write(json.dumps(rec) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gen", "run", "child"])
    ap.add_argument("--wt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--family", default="argsort")
    ap.add_argument("--sizes", default="16")
    ap.add_argument("--force", default="")
    ap.add_argument("--label", default="")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", default="30G")
    a = ap.parse_args()
    wt, out = Path(a.wt), Path(a.out)
    sizes = [int(s) for s in a.sizes.split(",") if s]
    if a.what == "gen":
        for n in sizes:
            print(gen(out, a.family, n))
    elif a.what == "run":
        run(wt, out, a.family, sizes, a.force, a.label, a.time, a.threads, a.mem)
    else:
        print(json.dumps(child(wt, out, a.family, sizes[0], a.time)))


if __name__ == "__main__":
    main()
