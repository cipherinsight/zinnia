"""Run Zinnia proving cases one subprocess at a time.

This is intentionally a crash-isolation harness around run_zinnia_halo2.py:
one bad backend/case can be killed without losing results for previous cases.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).parent
ROOT = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))
from ensure_release_build import ensure_zinnia_release_build  # noqa: E402
from run_halo2_baselines import INPUT_BUILDERS  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="halo2-base")
    ap.add_argument("--cases", default=None)
    ap.add_argument("--out", default=str(THIS_DIR / "zinnia_backend_probe.json"))
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--repeats", type=int, default=0)
    ap.add_argument("--k-min", type=int, default=None)
    args = ap.parse_args()
    k_min = args.k_min if args.k_min is not None else 4
    ensure_zinnia_release_build()

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(
            c
            for c in INPUT_BUILDERS
            if (THIS_DIR / "datasets" / "ds1000" / f"ds1000_case{c}" / "target.rs").exists()
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    out.write_text(json.dumps({"backend": args.backend, "results": results}, indent=2))

    print(f"cases={len(cases)} backend={args.backend} out={out}", flush=True)
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case}", flush=True)
        case_out = THIS_DIR / f".probe_{args.backend}_{case}.json"
        if case_out.exists():
            case_out.unlink()
        cmd = [
            str(ROOT / ".venv" / "bin" / "python"),
            str(THIS_DIR / "run_zinnia_halo2.py"),
            "--backend",
            args.backend,
            "--cases",
            case,
            "--repeats",
            str(args.repeats),
            "--k-min",
            str(k_min),
            "--out",
            str(case_out),
        ]
        t0 = time.time()
        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=args.timeout_s,
            )
            rec = {
                "case": case,
                "returncode": proc.returncode,
                "wall_s": time.time() - t0,
                "stdout_tail": proc.stdout[-4000:],
            }
            if case_out.exists():
                try:
                    rec.update(json.loads(case_out.read_text())["results"][0])
                except Exception as exc:  # noqa: BLE001
                    rec["parse_error"] = str(exc)
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout if isinstance(exc.stdout, str) else ""
            rec = {
                "case": case,
                "returncode": "timeout",
                "wall_s": time.time() - t0,
                "stdout_tail": stdout[-4000:],
            }

        results.append(rec)
        out.write_text(json.dumps({"backend": args.backend, "results": results}, indent=2))
        if rec.get("verify_ok"):
            status = "ok"
        elif rec["returncode"] == "timeout":
            status = "timeout"
        else:
            status = "err"
        err = str(rec.get("error") or rec.get("parse_error") or "")[:120]
        print(
            f"  -> {status} rc={rec['returncode']} wall={rec['wall_s']:.1f}s err={err}",
            flush=True,
        )


if __name__ == "__main__":
    main()
