#!/usr/bin/env python3
"""Gate counts (`bb gates` circuit size) of the performance-set Noir circuits.

Sources (one per run):
  --zinnia          emit each case's `main.nr` with the installed Zinnia build
                    (`emit_structural` on fixture input B; the circuit text does
                    not depend on the input values, only on their shapes) and
                    stage it with `noir_fpc`;
  --zinnia-dir DIR  pre-rendered Zinnia circuits `DIR/<case>.nr`, with `noir_fpc`;
  --baselines       the handwritten `datasets/ds1000/<case>/main.nr`, staged with
                    `baselines/noir_hw` exactly as `hw_check.noir_driver setup`
                    stages it (the number `hw_check.check` records as `gates`).

Both sides are compiled with `nargo compile` and sized with `bb gates
--scheme ultra_honk --verifier_target noir-recursive` (zero-knowledge
UltraHonk, the target of the harness and of Zinnia's Noir driver). Cases
default to every `perf` case of `datasets/case_sets/ds1000_perf.json`. Each
case runs in a child process with a timeout; `--wrap` prefixes the child
command (e.g. a memory cap); `--jobs` runs that many children at once.

    .venv/bin/python benchmarking/noir_gates.py --zinnia --out zn.json [--emit-dir DIR]
    .venv/bin/python benchmarking/noir_gates.py --baselines --out hw.json
    .venv/bin/python benchmarking/noir_gates.py --compare zn.json hw.json [--markdown]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DATASET = THIS_DIR / "datasets" / "ds1000"
CASE_SET = THIS_DIR / "datasets" / "case_sets" / "ds1000_perf.json"
NOIR_FPC = THIS_DIR / "noir_fpc"
NOIR_HW = THIS_DIR / "baselines" / "noir_hw"
PROVE_INPUT = "B"
os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
BB_FLAGS = ("--scheme", "ultra_honk", "--verifier_target", "noir-recursive")


def tool(env: str, home_rel: str, name: str) -> str:
    explicit = os.environ.get(env)
    if explicit:
        return explicit
    home = Path.home() / home_rel
    return str(home) if home.is_file() else (shutil.which(name) or name)


NARGO = tool("ZINNIA_NARGO_BIN", ".nargo/bin/nargo", "nargo")
BB = tool("ZINNIA_BB_BIN", ".bb/bb", "bb")


def case_name(spec: str) -> str:
    return spec if spec.startswith("ds1000_case") else f"ds1000_case{spec}"


def perf_cases() -> list[str]:
    """Every case whose status in the performance case set is `perf`."""
    entries = json.loads(CASE_SET.read_text())["cases"]
    return sorted(e["case"] for e in entries if e["status"] == "perf")


def load_case(case: str):
    """(compiled program, kernel arguments of `input_name`) for one perf case."""
    sys.path.insert(0, str(ROOT))
    from benchmarking.harness.fixtures import load_fixture
    from benchmarking.harness.kernels import compile_kernel

    directory = DATASET / case
    fixture = load_fixture(directory / "fixture.json")
    compiled = compile_kernel(directory / "kernel.py", fixture.function)
    return compiled, fixture


def zinnia_main_nr(case: str) -> str:
    compiled, fixture = load_case(case)
    return compiled.emit_structural(*fixture.arguments(PROVE_INPUT), backend="noir")


def gate_count(main_nr: str, library: str, library_dir: Path) -> dict:
    """`nargo compile` + `bb gates` of `main.nr` depending on one library."""
    with tempfile.TemporaryDirectory(prefix="noir-gates-") as tmp:
        pkg = Path(tmp)
        (pkg / "src").mkdir()
        (pkg / "src" / "main.nr").write_text(main_nr)
        shutil.copytree(library_dir, pkg / library, ignore=shutil.ignore_patterns("target"))
        (pkg / "Nargo.toml").write_text(
            '[package]\nname = "circuit"\ntype = "bin"\nauthors = [""]\n\n'
            f'[dependencies]\n{library} = {{ path = "./{library}" }}\n')
        proc = subprocess.run([NARGO, "compile", "--silence-warnings"], cwd=pkg,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": "nargo compile: " + (proc.stdout + proc.stderr)[-1500:]}
        proc = subprocess.run([BB, "gates", "-b", str(pkg / "target" / "circuit.json"), *BB_FLAGS],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": "bb gates: " + (proc.stdout + proc.stderr)[-1500:]}
        document, _ = json.JSONDecoder().raw_decode(proc.stdout[proc.stdout.find("{"):])
        fn = document["functions"][0]
        return {"gates": fn["circuit_size"], "acir_opcodes": fn["acir_opcodes"]}


def child(case: str, source: str, source_dir: str | None, emit_dir: str | None) -> dict:
    try:
        if source == "zinnia":
            main_nr = zinnia_main_nr(case)
            if emit_dir:
                Path(emit_dir).mkdir(parents=True, exist_ok=True)
                (Path(emit_dir) / f"{case}.nr").write_text(main_nr)
        elif source == "zinnia-dir":
            main_nr = (Path(source_dir) / f"{case}.nr").read_text()
        else:
            main_nr = (DATASET / case / "main.nr").read_text()
    except BaseException as e:  # noqa: BLE001 - PyO3 panics derive from BaseException
        return {"error": f"source: {type(e).__name__}: {e}"[:1500]}
    if source == "baselines":
        return gate_count(main_nr, "noir_hw", NOIR_HW)
    return gate_count(main_nr, "noir_fpc", NOIR_FPC)


def geomean(xs: list[float]) -> float:
    return math.exp(sum(math.log(x) for x in xs) / len(xs)) if xs else float("nan")


def compare(zinnia_json: Path, baseline_json: Path, markdown: bool) -> int:
    """Per-case Zinnia/handwritten ratio, worst first, and the geomean."""
    zn = json.loads(zinnia_json.read_text())
    hw = json.loads(baseline_json.read_text())
    rows = []
    for case in sorted(set(zn) | set(hw)):
        z, h = zn.get(case, {}).get("gates"), hw.get(case, {}).get("gates")
        rows.append((case, z, h, z / h if z and h else None))
    rows.sort(key=lambda r: (r[3] is None, -(r[3] or 0)))
    ratios = [r[3] for r in rows if r[3] is not None]
    if markdown:
        print("| case | Zinnia | handwritten | Zinnia/hw |\n|---|---|---|---|")
        for case, z, h, r in rows:
            print(f"| {case.removeprefix('ds1000_case')} | {z if z is not None else '-'} "
                  f"| {h if h is not None else '-'} | {f'{r:.3f}' if r else '-'} |")
    else:
        for case, z, h, r in rows:
            print(f"{case}  {z!s:>8}  {h!s:>8}  {f'{r:.3f}' if r else '-'}")
    print(f"pairs {len(ratios)}  geomean {geomean(ratios):.4f}  "
          f"Zinnia cheaper on {sum(r < 1 for r in ratios)}, "
          f"costlier on {sum(r > 1 for r in ratios)}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zinnia", action="store_true")
    src.add_argument("--zinnia-dir")
    src.add_argument("--baselines", action="store_true")
    src.add_argument("--compare", nargs=2, metavar=("ZINNIA_JSON", "BASELINE_JSON"))
    ap.add_argument("--markdown", action="store_true", help="--compare: a markdown table")
    ap.add_argument("--cases", default="all", help="comma-separated case numbers, or 'all'")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--wrap", default="", help="command prefix for each child (e.g. a memory cap)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--emit-dir", help="--zinnia: also write each emitted main.nr here")
    ap.add_argument("--out")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.compare:
        return compare(Path(a.compare[0]), Path(a.compare[1]), a.markdown)
    source = "zinnia" if a.zinnia else ("zinnia-dir" if a.zinnia_dir else "baselines")

    if a.child:
        print(json.dumps(child(a.child, source, a.zinnia_dir, a.emit_dir)))
        return 0
    if not a.out:
        ap.error("--out is required")

    cases = perf_cases() if a.cases == "all" else [case_name(c) for c in a.cases.split(",")]
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    todo = [c for c in cases if c not in results]

    def run(case: str) -> tuple[str, dict]:
        cmd = a.wrap.split() + [sys.executable, __file__, "--child", case]
        cmd += {"zinnia": ["--zinnia"], "zinnia-dir": ["--zinnia-dir", a.zinnia_dir or ""],
                "baselines": ["--baselines"]}[source]
        if a.emit_dir:
            cmd += ["--emit-dir", a.emit_dir]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(lines[-1]) if lines else {"error": "crash: " + proc.stderr[-800:]}
        except subprocess.TimeoutExpired:
            rec = {"error": "timeout"}
        return case, rec

    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as pool:
        for case, rec in pool.map(run, todo):
            results[case] = rec
            print(case, rec.get("gates", rec.get("error", "")[:160]), flush=True)
            out_path.write_text(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
