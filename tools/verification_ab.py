#!/usr/bin/env python3
"""A/B comparison runner for the Zinnia R/E/Q machinery.

For each program (a Python file, typically a pytest module under
`zinnia-src/testing/lang/`), runs pytest twice:

  - A: ZINNIA_REQ_DISABLE unset           — machinery on
  - B: ZINNIA_REQ_DISABLE=1               — machinery off

Each run gets its own `ZINNIA_TELEMETRY_DIR=/tmp/ab-{A,B}-<hash>/`, then we
parse the emitted JSON-Lines events to count strategy fires, discharges, and
fact emits. The CSV row captures whether each run compiled (pytest exit==0)
and the A-side telemetry counts.

Phase 1 of the verification loop (`compiler.verification-ab-disable-harness`).
"""

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
ZINNIA_SRC = REPO_ROOT / "zinnia-src"


def _short_hash(p: str) -> str:
    return hashlib.sha1(p.encode()).hexdigest()[:8]


def _run_pytest(program: Path, env_extra: dict[str, str]) -> bool:
    """Run pytest on `program` with `env_extra` overlaid on the inherited env.

    Returns True iff pytest exited 0 (all tests passed). stderr/stdout are
    swallowed; the telemetry directory is the load-bearing artifact.
    """
    env = dict(os.environ)
    env.update(env_extra)
    rel = program.relative_to(ZINNIA_SRC) if program.is_relative_to(ZINNIA_SRC) else program
    cmd = ["/opt/homebrew/bin/python3.13", "-m", "pytest", str(rel), "-q", "--no-header"]
    result = subprocess.run(
        cmd,
        cwd=str(ZINNIA_SRC),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return result.returncode == 0


def _read_telemetry(dir_path: Path) -> list[dict]:
    """Concatenate every `.jsonl` line under `dir_path` into a flat list of
    dicts. Missing dirs / unparseable lines are silently skipped (telemetry
    is best-effort)."""
    events: list[dict] = []
    if not dir_path.exists():
        return events
    for f in sorted(dir_path.glob("*.jsonl")):
        with f.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def _summarise(events: list[dict]) -> dict:
    """Reduce a telemetry-event stream to the counts we put in the CSV."""
    strat_names: list[str] = []
    discharges_proved = 0
    facts_emitted = 0
    for ev in events:
        kind = ev.get("event")
        if kind == "strategy_dispatch":
            strat_names.append(ev.get("strategy", "<unnamed>"))
        elif kind == "discharge":
            if ev.get("outcome") == "Proved":
                discharges_proved += 1
        elif kind == "fact_emit":
            facts_emitted += 1
    return {
        "strategies_fired": len(strat_names),
        "strategy_names": strat_names,
        "discharges_proved": discharges_proved,
        "facts_emitted": facts_emitted,
    }


def _ab_one(program: Path, keep_dirs: bool) -> dict:
    h = _short_hash(str(program))
    base = Path(tempfile.gettempdir())
    a_dir = base / f"ab-A-{h}"
    b_dir = base / f"ab-B-{h}"
    shutil.rmtree(a_dir, ignore_errors=True)
    shutil.rmtree(b_dir, ignore_errors=True)

    a_compiles = _run_pytest(program, {
        "ZINNIA_TELEMETRY_DIR": str(a_dir),
        "ZINNIA_REQ_DISABLE": "",
    })
    b_compiles = _run_pytest(program, {
        "ZINNIA_TELEMETRY_DIR": str(b_dir),
        "ZINNIA_REQ_DISABLE": "1",
    })

    a_summary = _summarise(_read_telemetry(a_dir))

    if not keep_dirs:
        shutil.rmtree(a_dir, ignore_errors=True)
        shutil.rmtree(b_dir, ignore_errors=True)

    return {
        "program": program.name,
        "A_compiles": a_compiles,
        "B_compiles": b_compiles,
        "A_strategies_fired": a_summary["strategies_fired"],
        "A_strategy_names": ";".join(a_summary["strategy_names"]),
        "A_discharges_proved": a_summary["discharges_proved"],
        "A_facts_emitted": a_summary["facts_emitted"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "programs",
        nargs="+",
        help="program paths or glob patterns (relative or absolute)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="-",
        help="output CSV path; default '-' is stdout",
    )
    parser.add_argument(
        "--keep-telemetry-dirs",
        action="store_true",
        help="leave the /tmp/ab-{A,B}-<hash> dirs in place for inspection",
    )
    args = parser.parse_args()

    paths: list[Path] = []
    for pat in args.programs:
        matches = sorted(glob.glob(pat, recursive=True)) or [pat]
        for m in matches:
            p = Path(m).resolve()
            if p.exists():
                paths.append(p)
            else:
                print(f"warning: skipping non-existent path {p}", file=sys.stderr)

    if not paths:
        print("error: no program paths matched", file=sys.stderr)
        return 1

    out = sys.stdout if args.output == "-" else open(args.output, "w")
    try:
        cols = [
            "program",
            "A_compiles",
            "B_compiles",
            "A_strategies_fired",
            "A_strategy_names",
            "A_discharges_proved",
            "A_facts_emitted",
        ]
        out.write(",".join(cols) + "\n")
        for p in paths:
            row = _ab_one(p, args.keep_telemetry_dirs)
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, str) and ("," in v or '"' in v):
                    v = '"' + v.replace('"', '""') + '"'
                cells.append(str(v))
            out.write(",".join(cells) + "\n")
            out.flush()
    finally:
        if out is not sys.stdout:
            out.close()

    return 0


if __name__ == "__main__":
    sys.exit(main())
