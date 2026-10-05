#!/usr/bin/env python3
"""Gate counts (`bb gates` circuit size) of performance-set Noir circuits.

Sources (one per run):
  --zinnia          emit each case's `main.nr` with the installed Zinnia build
                    (`emit_structural`; the circuit text does not depend on
                    the fixture values, only on its shapes);
  --zinnia-dir DIR  pre-rendered Zinnia circuits `DIR/<case>.nr` (for example
                    from an older build);
  --baselines       the handwritten `ds1000_case<NNNN>/main.nr`.

Every circuit is staged with the `noir_fpc` library given by `--lib` (default:
`benchmarking/noir_fpc/src/lib.nr`), compiled with `nargo compile` and sized
with `bb gates` (UltraHonk, the verifier target of the Noir driver). Each case
runs in a child process with a timeout; `--wrap` prefixes the child command
(e.g. a memory cap).

    .venv/bin/python benchmarking/noir_gates.py --zinnia --out zn_gates.json
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
DEFAULT_LIB = THIS_DIR / "noir_fpc" / "src" / "lib.nr"
os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
VERIFIER_TARGET = "noir-recursive-no-zk"  # as src/prove/noir/driver.rs


def tool(env: str, home_rel: str, name: str) -> str:
    explicit = os.environ.get(env)
    if explicit:
        return explicit
    home = Path.home() / home_rel
    return str(home) if home.is_file() else (shutil.which(name) or name)


NARGO = tool("ZINNIA_NARGO_BIN", ".nargo/bin/nargo", "nargo")
BB = tool("ZINNIA_BB_BIN", ".bb/bb", "bb")


def zinnia_main_nr(case: str) -> str:
    sys.path.insert(0, str(THIS_DIR))
    from noir_differential import PARAMS, load_case

    compiled, args = load_case(case)
    return compiled.emit_structural(*args, backend="noir", params=PARAMS)


def gate_count(main_nr: str, lib_nr: Path) -> dict:
    with tempfile.TemporaryDirectory(prefix="noir-gates-") as tmp:
        pkg = Path(tmp)
        (pkg / "src").mkdir()
        (pkg / "noir_fpc" / "src").mkdir(parents=True)
        (pkg / "Nargo.toml").write_text(
            '[package]\nname = "circuit"\ntype = "bin"\nauthors = [""]\n\n'
            '[dependencies]\nnoir_fpc = { path = "./noir_fpc" }\n'
        )
        (pkg / "src" / "main.nr").write_text(main_nr)
        (pkg / "noir_fpc" / "Nargo.toml").write_text(
            (THIS_DIR / "noir_fpc" / "Nargo.toml").read_text()
        )
        shutil.copy(lib_nr, pkg / "noir_fpc" / "src" / "lib.nr")
        proc = subprocess.run([NARGO, "compile", "--silence-warnings"], cwd=pkg,
                              capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": "nargo compile: " + (proc.stdout + proc.stderr)[-1500:]}
        proc = subprocess.run(
            [BB, "gates", "-b", str(pkg / "target" / "circuit.json"),
             "--scheme", "ultra_honk", "--verifier_target", VERIFIER_TARGET],
            capture_output=True, text=True)
        if proc.returncode != 0:
            return {"error": "bb gates: " + (proc.stdout + proc.stderr)[-1500:]}
        blob = proc.stdout[proc.stdout.find("{"):]
        fn = json.loads(blob)["functions"][0]
        return {"gates": fn["circuit_size"], "acir_opcodes": fn["acir_opcodes"]}


def child(case: str, source: str, source_dir: str | None, lib: Path) -> dict:
    try:
        if source == "zinnia":
            main_nr = zinnia_main_nr(case)
        elif source == "zinnia-dir":
            main_nr = (Path(source_dir) / f"{case}.nr").read_text()
        else:
            main_nr = (THIS_DIR / f"ds1000_case{case}" / "main.nr").read_text()
    except BaseException as e:  # noqa: BLE001
        return {"error": f"source: {type(e).__name__}: {e}"[:1500]}
    return gate_count(main_nr, lib)


def cases_for(source: str, source_dir: str | None) -> list[str]:
    if source == "baselines":
        return sorted(p.parent.name.removeprefix("ds1000_case")
                      for p in THIS_DIR.glob("ds1000_case*/main.nr"))
    if source == "zinnia-dir":
        return sorted(p.stem for p in Path(source_dir).glob("*.nr"))
    sys.path.insert(0, str(THIS_DIR))
    from run_halo2_baselines import INPUT_BUILDERS
    return sorted(INPUT_BUILDERS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--zinnia", action="store_true")
    src.add_argument("--zinnia-dir")
    src.add_argument("--baselines", action="store_true")
    ap.add_argument("--lib", default=str(DEFAULT_LIB))
    ap.add_argument("--cases", default="all")
    ap.add_argument("--timeout", type=int, default=1200)
    ap.add_argument("--wrap", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--child", help=argparse.SUPPRESS)
    a = ap.parse_args()
    source = "zinnia" if a.zinnia else ("zinnia-dir" if a.zinnia_dir else "baselines")

    if a.child:
        print(json.dumps(child(a.child, source, a.zinnia_dir, Path(a.lib))))
        return 0

    cases = cases_for(source, a.zinnia_dir) if a.cases == "all" else a.cases.split(",")
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for case in cases:
        if case in results:
            continue
        cmd = a.wrap.split() + [sys.executable, __file__, "--child", case, "--lib", a.lib,
                                "--out", a.out]
        cmd += {"zinnia": ["--zinnia"], "zinnia-dir": ["--zinnia-dir", a.zinnia_dir or ""],
                "baselines": ["--baselines"]}[source]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
            lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
            rec = json.loads(lines[-1]) if lines else {"error": "crash: " + proc.stderr[-800:]}
        except subprocess.TimeoutExpired:
            rec = {"error": "timeout"}
        results[case] = rec
        print(case, rec.get("gates", rec.get("error", "")[:160]), flush=True)
        out_path.write_text(json.dumps(results, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
