"""Scaled copies of two perf cases with their handwritten halo2_hw baselines, Zinnia against
the baseline (port of eval-redo/headroom-data/scale.py and layer-challenge-data/scale_hw.py,
paths as arguments; checks added).

Cases (only the size changes in kernel.py, target.rs, main.nr and the fixture):
  0381 argsort of N floats, N = 16, 64, 256, 1024;
  0319 gather of the N planes of a: Float[N, 6, 8] by the inverse of a permutation, N = 8..64.

    python scale_hw.py gen    --wt WT [--data D]    # write ds1000_case<base>n<NNNN> into D's dataset
    python scale_hw.py zinnia --wt WT [--data D] --out OUT [--only CASES] [--time]   # Zinnia's own choice
    python scale_hw.py hw     --wt WT [--data D] --out OUT [--only CASES] [--time]   # D's halo2_hw build
                                                    # (build it after `gen`; it compiles every target.rs)

WT is the checkout whose build (Zinnia extension, venv, harness) is used; D (default WT) is a
checkout of the same commit that holds the scaled cases and the halo2_hw build that includes
them, so WT's dataset stays the original one.

Zinnia: cells (advice + lookup at the k estimate_params picks); with --time keygen on input A,
3 proofs on input B (wall = witness + create_proof, and create_proof alone), each verified,
a tampered public value rejected, outputs compared with NumPy. halo2_hw: `stats` on A; with
--time `setup` on A, `prove --repeats 3` on B (create_proof samples), `verify` of the proof,
and `verify --tamper-index 0` (must be rejected). Each child runs under capped.py (--mem) with
RAYON_NUM_THREADS / OMP_NUM_THREADS / HARDWARE_CONCURRENCY = --threads.
Records go to OUT/scale_zinnia.jsonl and OUT/scale_hw.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SIZES = {"0381": [16, 64, 256, 1024], "0319": [8, 16, 32, 64]}


def scaled_name(base: str, n: int) -> str:
    return f"ds1000_case{base}n{n:04d}"


def all_cases() -> list[tuple[str, int]]:
    return [(base, n) for base, ns in SIZES.items() for n in ns]


def gen_0381(src: Path, dst: Path, n: int) -> None:
    b = max(1, math.ceil(math.log2(n)))  # index bits
    kernel = (src / "kernel.py").read_text().replace('Float[Array, "100"]', f'Float[Array, "{n}"]')
    target = (src / "target.rs").read_text()
    target = target.replace("const N: usize = 100;", f"const N: usize = {n};")
    target = target.replace("const KEY_SHIFT: i128 = 1 << 70;", f"const KEY_SHIFT: i128 = 1 << {63 + b};")
    target = target.replace("const KEY_BITS: usize = 71;", f"const KEY_BITS: usize = {64 + b};")
    target = target.replace("Constant(fe(128))", f"Constant(fe({1 << b}))")
    target = target.replace("floor_div_const(ctx, shifted, 128)", f"floor_div_const(ctx, shifted, {1 << b})")
    noir = (src / "main.nr").read_text()
    noir = noir.replace("global N: u32 = 100;", f"global N: u32 = {n};")
    noir = noir.replace("array_index::<8>", f"array_index::<{b + 1}>")
    noir = noir.replace("(window(v) as Field) * 128 + i", f"(window(v) as Field) * {1 << b} + i")
    noir = noir.replace("assert_max_bit_size::<71>", f"assert_max_bit_size::<{64 + b}>")
    noir = noir.replace("// @outputs: int[100]", f"// @outputs: int[{n}]")
    assert target.count(str(1 << b)) >= 2 or n == 128
    fixture = json.loads((src / "fixture.json").read_text())
    fixture["case"] = dst.name
    fixture["parameters"][0]["shape"] = [n]
    inputs = {}
    for name, seed in (("A", 381001 + n), ("B", 381002 + n)):
        rng = np.random.RandomState(seed)
        a = (rng.rand(n) - 0.5) * 100
        assert len(set(a.tolist())) == n
        inputs[name] = {"a": a.tolist()}
    fixture["inputs"] = inputs
    fixture["notes"] = f"scaled copy of ds1000_case0381 (N = {n}) for the headroom scaling experiment"
    write_case(dst, kernel, target, noir, fixture)


def gen_0319(src: Path, dst: Path, n: int) -> None:
    kernel = (src / "kernel.py").read_text()
    kernel = kernel.replace('Float[Array, "8 6 8"]', f'Float[Array, "{n} 6 8"]')
    kernel = kernel.replace('Integer[Array, "8"]', f'Integer[Array, "{n}"]')
    target = (src / "target.rs").read_text().replace("const N: usize = 8;", f"const N: usize = {n};")
    noir = (src / "main.nr").read_text()
    noir = noir.replace("fn main(a: [[Field; 48]; 8], permutation: [Field; 8]) -> pub [Field; 384]",
                        f"fn main(a: [[Field; 48]; {n}], permutation: [Field; {n}]) -> pub [Field; {48 * n}]")
    noir = noir.replace("let mut out: [Field; 384] = [0; 384];", f"let mut out: [Field; {48 * n}] = [0; {48 * n}];")
    noir = noir.replace("for r in 0..8 {", f"for r in 0..{n} {{")
    noir = noir.replace("for j in 0..8 {", f"for j in 0..{n} {{")
    noir = noir.replace("for k in 0..8 {", f"for k in 0..{n} {{")
    noir = noir.replace("// @outputs: float[384]", f"// @outputs: float[{48 * n}]")
    fixture = json.loads((src / "fixture.json").read_text())
    fixture["case"] = dst.name
    fixture["parameters"][0]["shape"] = [n, 6, 8]
    fixture["parameters"][1]["shape"] = [n]
    inputs = {}
    for name, seed in (("A", 319001 + n), ("B", 319002 + n)):
        rng = np.random.RandomState(seed)
        perm = rng.permutation(n)
        while (perm == np.arange(n)).all():
            perm = rng.permutation(n)
        inputs[name] = {"a": rng.rand(n, 6, 8).tolist(), "permutation": perm.tolist()}
    fixture["inputs"] = inputs
    fixture["notes"] = f"scaled copy of ds1000_case0319 (N = {n}) for the headroom scaling experiment"
    write_case(dst, kernel, target, noir, fixture)


def write_case(dst: Path, kernel: str, target: str, noir: str, fixture: dict) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "kernel.py").write_text(kernel)
    (dst / "target.rs").write_text(target)
    (dst / "main.nr").write_text(noir)
    (dst / "fixture.json").write_text(json.dumps(fixture))


def reference(base: str, args: dict) -> np.ndarray:
    if base == "0381":
        a = np.array(args["a"])
        return np.argsort(a)[::-1][:len(a)]
    perm = np.array(args["permutation"])
    c = np.empty_like(perm)
    c[perm] = np.arange(len(perm))
    return np.array(args["a"])[c, :, :]


def cmd_gen(data: Path) -> None:
    root = data / "benchmarking" / "datasets" / "ds1000"
    for base, n in all_cases():
        src = root / f"ds1000_case{base}"
        dst = root / scaled_name(base, n)
        {"0381": gen_0381, "0319": gen_0319}[base](src, dst, n)
        print("wrote", dst)


def zinnia_child(wt: Path, data: Path, case: str, timed: bool) -> dict:
    sys.path.insert(0, str(wt))
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel
    from zinnia._zinnia_core import halo2_base_cell_stats

    folder = data / "benchmarking" / "datasets" / "ds1000" / case
    fixture = load_fixture(folder / "fixture.json")
    started = time.perf_counter()
    compiled = compile_kernel(folder / "kernel.py", "kernel")
    record = {"case": case, "compile_s": round(time.perf_counter() - started, 2)}
    stats = json.loads(halo2_base_cell_stats(compiled._compiled_ir))
    record.update({k: stats.get(k) for k in ("k", "lookup_bits", "advice", "lookup", "second_advice", "challenges")})
    record["cells"] = stats["advice"] + stats["lookup"]
    if timed:
        from benchmarking.harness.adapters.zinnia_common import describe_compiled, zinnia_publics
        from benchmarking.harness.outputs import field_from_le_hex, field_to_le_hex

        started = time.perf_counter()
        keys = compiled.setup()
        record["setup_s"] = round(time.perf_counter() - started, 3)
        walls, proofs = [], []
        for _ in range(3):
            started = time.perf_counter()
            proofs.append(compiled.prove_with_keys(keys, *fixture.arguments("B")))
            walls.append(time.perf_counter() - started)
        record["prove_wall_s"] = [round(s, 3) for s in walls]
        record["prove_median_s"] = round(statistics.median(walls), 3)
        record["create_proof_s"] = [round(p["prove_seconds"], 3) for p in proofs]
        record["create_proof_median_s"] = round(statistics.median(p["prove_seconds"] for p in proofs), 3)
        record["verified"] = all(compiled.verify_with_keys(keys, p) for p in proofs)
        tampered = dict(proofs[0])
        first = field_from_le_hex(tampered["public_values"][0])
        tampered["public_values"] = [field_to_le_hex(first + 1), *tampered["public_values"][1:]]
        record["tamper_rejected"] = not compiled.verify_with_keys(keys, tampered)
        fields = [field_from_le_hex(v) for v in proofs[0]["public_values"]]
        got = np.array(zinnia_publics(fields, describe_compiled(compiled)).values, dtype=float).reshape(-1)
        raw = json.loads((folder / "fixture.json").read_text())["inputs"]["B"]
        ref = reference(case[len("ds1000_case"):][:4], raw).reshape(-1)
        record["max_abs_error"] = float(np.max(np.abs(got - ref)))
    return record


def env_for(threads: int, wt: Path) -> dict:
    return dict(os.environ, ZWT=str(wt), RAYON_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads),
                HARDWARE_CONCURRENCY=str(threads))


def selected(only: list[str]) -> list[str]:
    cases = [scaled_name(b, n) for b, n in all_cases()]
    return [c for c in cases if not only or c in only or c[len("ds1000_case"):] in only]


def cmd_zinnia(wt: Path, data: Path, out: Path, timed: bool, only: list[str], threads: int, mem: str) -> None:
    py = str(wt / ".venv" / "bin" / "python")
    for case in selected(only):
        cmd = [py, str(HERE / "capped.py"), mem, "7200", py, __file__, "zinnia-child", "--wt", str(wt),
               "--data", str(data), "--out", str(out), "--case", case] + (["--time"] if timed else [])
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env_for(threads, wt))
        lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
        rec = json.loads(lines[-1]) if lines else {"case": case, "error": proc.stderr[-1500:]}
        rec.update(threads=threads, memory_cap=mem, exit=proc.returncode, peak_rss_bytes=peak_rss(proc.stderr))
        emit(out / "scale_zinnia.jsonl", rec)


def peak_rss(stderr: str) -> int | None:
    peaks = [l for l in stderr.splitlines() if l.startswith("capped: peak_rss_bytes=")]
    return int(peaks[-1].split("=")[1]) if peaks else None


def emit(path: Path, rec: dict) -> None:
    print(json.dumps(rec), flush=True)
    with path.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def cmd_hw(wt: Path, data: Path, out: Path, timed: bool, only: list[str], threads: int, mem: str) -> None:
    binary = str(data / "benchmarking" / "baselines" / "halo2_hw" / "target" / "release" / "halo2_hw")
    py = str(wt / ".venv" / "bin" / "python")
    env = env_for(threads, wt)
    for case in selected(only):
        folder = data / "benchmarking" / "datasets" / "ds1000" / case
        rec: dict = {"case": case, "threads": threads, "memory_cap": mem}
        with tempfile.TemporaryDirectory(prefix="hw-scale-") as work:
            base = [py, str(HERE / "capped.py"), mem, "7200", binary]
            common = ["--case", str(folder), "--workdir", work, "--fixture", str(folder / "fixture.json")]
            stats = run_json(base + ["stats", *common, "--input", "A"], env)
            cells = stats.get("cells") or {}
            rec.update({"k": stats.get("k"), "lookup_bits": stats.get("lookup_bits"), "cells_detail": cells,
                        "cells": (cells.get("advice", 0) + cells.get("lookup", 0)) if cells else None,
                        "error": stats.get("error")})
            if timed and stats.get("ok"):
                setup = run_json(base + ["setup", *common, "--input", "A"], env)
                prove = run_json(base + ["prove", *common, "--input", "B", "--repeats", "3"], env)
                verify = run_json(base + ["verify", "--case", str(folder), "--workdir", work], env)
                tamper = run_json(base + ["verify", "--case", str(folder), "--workdir", work, "--tamper-index", "0"], env)
                samples = (prove.get("timings_s") or {}).get("prove_samples_s") or []
                rec.update(setup=setup.get("timings_s"), prove=prove.get("timings_s"), prove_ok=prove.get("ok"),
                           create_proof_median_s=round(statistics.median(samples), 3) if samples else None,
                           verified=verify.get("ok"), tamper_rejected=(tamper.get("ok") is False and "verified" in tamper),
                           tamper_detail={k: tamper.get(k) for k in ("ok", "verified", "error")})
        emit(out / "scale_hw.jsonl", rec)


def run_json(cmd: list[str], env: dict) -> dict:
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    start = proc.stdout.find("{")
    if start < 0:
        return {"ok": False, "error": (proc.stdout + proc.stderr)[-1500:]}
    try:
        doc, _ = json.JSONDecoder().raw_decode(proc.stdout[start:])
        return doc
    except json.JSONDecodeError:
        return {"ok": False, "error": proc.stdout[-1500:]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["gen", "zinnia", "zinnia-child", "hw"])
    ap.add_argument("--wt", required=True)
    ap.add_argument("--data", help="checkout holding the scaled cases (default: --wt)")
    ap.add_argument("--out", default=".")
    ap.add_argument("--case")
    ap.add_argument("--only", default="")
    ap.add_argument("--time", action="store_true")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mem", default="30G")
    a = ap.parse_args()
    wt, out = Path(a.wt), Path(a.out)
    data = Path(a.data) if a.data else wt
    only = [c for c in a.only.split(",") if c]
    if a.what == "gen":
        cmd_gen(data)
    elif a.what == "zinnia":
        out.mkdir(parents=True, exist_ok=True)
        cmd_zinnia(wt, data, out, a.time, only, a.threads, a.mem)
    elif a.what == "zinnia-child":
        print(json.dumps(zinnia_child(wt, data, a.case, a.time)))
    else:
        out.mkdir(parents=True, exist_ok=True)
        cmd_hw(wt, data, out, a.time, only, a.threads, a.mem)


if __name__ == "__main__":
    main()
