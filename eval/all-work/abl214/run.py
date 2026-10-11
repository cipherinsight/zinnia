#!/usr/bin/env python3
"""Ablation sizes of the 214 RQ1 programs: every (variant, backend, program), no proving.

    python run.py --checkout WT --out OUT [--variants final,abl_joint,...] [--backends halo2,noir]
                  [--jobs 48] [--mem-cap 24G] [--timeout 3600] [--min-free 32G]
                  [--programs a,b,c] [--retry-errors] [--dry-run]

Run it with any Python >= 3.10; each measurement is a child process of the checkout's own
interpreter (WT/.venv/bin/python child.py ...) under WT/benchmarking/ablations/capped.py
(RSS cap + timeout, no systemd needed), with exactly the variant's environment variables set
and every other ZINNIA_ABLATE_* / ZINNIA_FORCE_* variable removed.

Results: OUT/results/<variant>/<backend>/<program>.json, one per measurement, written
atomically. A rerun skips every result that exists (resumable); --retry-errors also redoes
the ones that recorded an error (e.g. after raising --mem-cap). OUT/programs.json lists the
214 programs with their corpus and whether RQ2 compares them (the 172); OUT/manifest.json
records the checkout's commit, the extension's SHA-256 and the tool versions.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent

#: The variants (docs/ablations.md of the checkout). Each disables one part of the compiler.
VARIANTS: dict[str, dict[str, str]] = {
    "final": {},                                                  # the compiler as evaluated
    "abl_joint": {"ZINNIA_ABLATE_JOINT": "1"},                    # local selection per group
    "fix_delegate": {"ZINNIA_FORCE_CONSTRUCTION": "@delegate"},   # accesses via backend memory
    "fix_selects": {"ZINNIA_FORCE_CONSTRUCTION": "@selects"},     # select chains everywhere
    "abl_layer_facts": {"ZINNIA_ABLATE_LAYER_FACTS": "1"},        # layer: no facts, no checks
    "abl_array_facts": {"ZINNIA_ABLATE_ARRAY_FACTS": "1"},        # no array facts anywhere
    "abl_smt": {"ZINNIA_ABLATE_SMT": "1"},                        # every Z3 value query Unknown
    "abl_check": {"ZINNIA_ABLATE_CHECK_ELIM": "1"},               # no check-dropping passes
    # Opt-in (not in the default list): both fact switches of the lowering layer at once.
    "abl_all_layer_facts": {"ZINNIA_ABLATE_LAYER_FACTS": "1", "ZINNIA_ABLATE_ARRAY_FACTS": "1"},
    # Everything at once: one fixed construction, no joint selection, no facts, no SMT, no
    # check elimination. Approximates a compiler without the fact-driven lowering layer.
    "abl_combined": {"ZINNIA_FORCE_CONSTRUCTION": "@selects", "ZINNIA_ABLATE_JOINT": "1",
                     "ZINNIA_ABLATE_LAYER_FACTS": "1", "ZINNIA_ABLATE_ARRAY_FACTS": "1",
                     "ZINNIA_ABLATE_SMT": "1", "ZINNIA_ABLATE_CHECK_ELIM": "1"},
}
DEFAULT_VARIANTS = [v for v in VARIANTS if v not in ("abl_all_layer_facts", "abl_combined")]
BACKENDS = ("halo2", "noir")
#: Variables the child must not inherit from this process.
SCRUB_PREFIXES = ("ZINNIA_ABLATE_", "ZINNIA_FORCE_")
SCRUB = {"ZINNIA_SMT_ENABLE", "ZINNIA_REQ_DISABLE", "ZINNIA_LAYER_REPORT", "ZINNIA_LAYER_INPUT"}
#: RQ2's constant-output programs (fig/data/rq2-perf.py CONSTANT_OUTPUT).
CONSTANT_OUTPUT = {f"ds1000_case0{n}" for n in (291, 349, 373, 400, 401, 451)}
DEFAULT_ACCEPTED = Path("/home/zhantong/Projects/zinnia-project/zinnia-src/eval/all-work/rq1-ds-accepted.txt")


def corpus(case: str) -> str:
    return case.split("_", 1)[0]  # ds1000 / npbench / pythran


def size_bytes(text: str) -> int:
    units = {"K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}
    text = text.strip().upper()
    return int(float(text[:-1]) * units[text[-1]]) if text[-1] in units else int(text)


def mem_available() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 1 << 62


# ---------------------------------------------------------------------- programs


def load_programs(wt: Path, accepted_file: Path) -> list[dict]:
    """The 214: the RQ1-accepted DS-1000 perf cases + every NPBench/Pythran perf case."""
    sets = wt / "benchmarking" / "datasets" / "case_sets"
    accepted = set(accepted_file.read_text().split())
    ds = {e["case"]: e for e in json.loads((sets / "ds1000_perf.json").read_text())["cases"]
          if e["status"] == "perf"}
    missing = accepted - set(ds)
    if missing:
        raise SystemExit(f"accepted DS-1000 cases not in the perf set: {sorted(missing)}")
    npb = {e["case"]: e for e in json.loads((sets / "npbench_pythran_perf.json").read_text())["cases"]
           if e["status"] == "perf"}
    programs = []
    for case, entry in sorted({**{c: ds[c] for c in accepted}, **npb}.items()):
        copy_only = bool(entry.get("data_movement_only"))
        directory = (wt / "benchmarking" / entry["path"] if "path" in entry
                     else wt / "benchmarking" / "datasets" / "ds1000" / case)
        programs.append({
            "case": case, "corpus": corpus(case),
            "kernel": str(directory / "kernel.py"),
            "fixture": str(directory / (entry.get("fixture") or "fixture.json")),
            "function": entry.get("function", "kernel"),
            "copy_only": copy_only, "constant_output": case in CONSTANT_OUTPUT,
            "rq2": not copy_only and case not in CONSTANT_OUTPUT,
        })
    return programs


# ---------------------------------------------------------------------- one job


def summarize_layer(path: Path) -> dict | None:
    """Compact summary of the layer report (benchmarking/ablations/common.summarize_report)."""
    if not path.exists():
        return None
    chosen: dict[str, dict[str, int]] = {}
    native = fallback = groups = estimated = 0
    memory = ("memory: element by table read", "memory: native RAM", "memory: RAM trace")
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        round_ = json.loads(line)
        estimated += round_.get("estimated_cost", 0)
        forcing = set(round_.get("forcing", []))
        for g in round_["groups"]:
            groups += 1
            for name in g["chosen"]:
                per = chosen.setdefault(g["family"], {})
                per[name] = per.get(name, 0) + 1
            native += any(n in memory for n in g["chosen"])
            fallback += bool(g.get("fixed_family")) and not forcing & set(g["chosen"])
    return {"groups": groups, "chosen": chosen, "native_memory": native,
            "fallback": fallback, "estimated_cost": estimated}


def run_job(args, program: dict, variant: str, backend: str) -> dict:
    wt = args.checkout
    py = wt / ".venv" / "bin" / "python"
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(SCRUB_PREFIXES) and k not in SCRUB}
    env.update(VARIANTS[variant])
    env.update(ZINNIA_SKIP_RELEASE_BUILD="1", ZWT=str(wt), PYTHONHASHSEED="0")
    tmp_root = os.environ.get("TMPDIR")
    with tempfile.TemporaryDirectory(prefix="abl214-", dir=tmp_root) as tmp:
        report = Path(tmp) / "layer.jsonl"
        env["ZINNIA_LAYER_REPORT"] = str(report)
        env["TMPDIR"] = tmp  # nargo/bb scratch packages land here and are removed with it
        cmd = [str(py), str(wt / "benchmarking/ablations/capped.py"), args.mem_cap,
               str(args.timeout), str(py), str(HERE / "child.py"), str(wt),
               program["kernel"], program["fixture"], program["function"], backend]
        if args.nice:
            cmd = ["nice", "-n", str(args.nice), *cmd]
        started = time.monotonic()
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, cwd=wt, env=env,
                                  timeout=args.timeout + 120)
            out, err, code = proc.stdout, proc.stderr, proc.returncode
        except subprocess.TimeoutExpired:
            out, err, code = "", "runner timeout", 124
        wall = round(time.monotonic() - started, 2)
        record = None
        for line in reversed(out.strip().splitlines()):
            if line.startswith("{"):
                record = json.loads(line)
                break
        if record is None:
            reason = ("over the memory cap" if code == 137 else
                      "timeout" if code == 124 else f"exit {code}")
            tail = " | ".join(err.strip().splitlines()[-3:])
            record = {"error": f"{reason}: {tail}"[:1500]}
        peak = [l for l in err.splitlines() if l.startswith("capped: peak_rss_bytes=")]
        record.update(peak_rss=int(peak[-1].split("=")[1]) if peak else None, wall_s=wall,
                      exit=code, layer=summarize_layer(report))
    record.update(program=program["case"], corpus=program["corpus"], variant=variant,
                  env=VARIANTS[variant], backend=backend)
    return record


# ---------------------------------------------------------------------- scheduling


class Queue:
    """Pending jobs, handed out at dispatch time: every `final` job first (it is the
    reference and its wall time estimates the rest), then the other variants, the
    slowest (program, backend) first by the final's measured wall time, so the long
    compiles do not end up in the tail."""

    def __init__(self, jobs: list[tuple], out: Path):
        self.pending = list(jobs)
        self.out = out
        self.lock = threading.Lock()

    def estimate(self, job) -> float:
        program, _, backend = job
        path = self.out / "results" / "final" / backend / f"{program['case']}.json"
        try:
            return json.loads(path.read_text()).get("wall_s") or 0.0
        except (OSError, ValueError):
            return 0.0

    def pop(self):
        with self.lock:
            if not self.pending:
                return None
            finals = [i for i, j in enumerate(self.pending) if j[1] == "final"]
            if finals:
                return self.pending.pop(finals[0])
            best = max(range(len(self.pending)), key=lambda i: self.estimate(self.pending[i]))
            return self.pending.pop(best)


def result_path(out: Path, case: str, variant: str, backend: str) -> Path:
    return out / "results" / variant / backend / f"{case}.json"


def write_atomic(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".tmp{os.getpid()}.{threading.get_ident()}")
    tmp.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def tool_version(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.strip()[:300]
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def manifest(args) -> dict:
    wt = args.checkout
    git = lambda *a: subprocess.run(["git", "-C", str(wt), *a], capture_output=True,
                                    text=True).stdout.strip()
    build = wt / "target" / "zinnia-harness-build.json"
    ext = sorted((wt / "zinnia").glob("_zinnia_core*.so"))
    sha = subprocess.run(["sha256sum", str(ext[0])], capture_output=True, text=True).stdout.split()[:1] if ext else []
    return {
        "host": socket.gethostname(), "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "checkout": str(wt), "commit": git("rev-parse", "HEAD"),
        "dirty_tracked": bool(git("status", "--porcelain", "--untracked-files=no")),
        "extension_sha256": sha[0] if sha else None,
        "build_record": json.loads(build.read_text()) if build.exists() else None,
        "nargo": tool_version([os.environ.get("ZINNIA_NARGO_BIN", "nargo"), "--version"]),
        "bb": tool_version([os.environ.get("ZINNIA_BB_BIN", "bb"), "--version"]),
        "args": {k: str(v) for k, v in vars(args).items()},
        "variants": {v: VARIANTS[v] for v in args.variants},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkout", type=Path, required=True, help="built Zinnia checkout")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--accepted", type=Path, default=DEFAULT_ACCEPTED,
                    help="RQ1-accepted DS-1000 cases (default: eval-redo/all-work/rq1-ds-accepted.txt; "
                         "on the remote pass the copy next to this script)")
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS),
                    help=f"comma-separated, from {', '.join(VARIANTS)}")
    ap.add_argument("--backends", default=",".join(BACKENDS))
    ap.add_argument("--programs", help="comma-separated subset (case names or DS-1000 numbers)")
    ap.add_argument("--jobs", type=int, default=48, help="parallel children")
    ap.add_argument("--mem-cap", default="64G", help="RSS cap of one child (capped.py)")
    ap.add_argument("--min-free", default="40G",
                    help="start a child only while MemAvailable is above this")
    ap.add_argument("--timeout", type=int, default=3600, help="seconds per child")
    ap.add_argument("--nice", type=int, default=10, help="nice level of the children (0 = none)")
    ap.add_argument("--retry-errors", action="store_true", help="redo results that recorded an error")
    ap.add_argument("--dry-run", action="store_true", help="list the jobs and exit")
    args = ap.parse_args()
    args.checkout = args.checkout.resolve()
    args.variants = args.variants.split(",")
    args.backends = args.backends.split(",")
    for v in args.variants:
        if v not in VARIANTS:
            ap.error(f"unknown variant {v}")
    if not (args.checkout / ".venv" / "bin" / "python").exists():
        ap.error(f"{args.checkout} has no .venv")
    if not args.accepted.exists():
        alt = HERE / "rq1-ds-accepted.txt"
        if not alt.exists():
            ap.error(f"{args.accepted} not found")
        args.accepted = alt

    programs = load_programs(args.checkout, args.accepted)
    args.out.mkdir(parents=True, exist_ok=True)
    write_atomic(args.out / "programs.json", {"programs": [
        {k: p[k] for k in ("case", "corpus", "copy_only", "constant_output", "rq2")} for p in programs]})
    if args.programs:
        wanted = set(args.programs.split(","))
        programs = [p for p in programs
                    if p["case"] in wanted or p["case"].removeprefix("ds1000_case") in wanted]
    print(f"programs {len(programs)} (rq2 {sum(p['rq2'] for p in programs)}), "
          f"variants {args.variants}, backends {args.backends}", flush=True)

    jobs, skipped = [], 0
    for variant in args.variants:
        for program in programs:
            for backend in args.backends:
                path = result_path(args.out, program["case"], variant, backend)
                if path.exists():
                    try:
                        old = json.loads(path.read_text())
                    except ValueError:
                        old = {"error": "unreadable"}
                    if "error" not in old or not args.retry_errors:
                        skipped += 1
                        continue
                jobs.append((program, variant, backend))
    print(f"jobs {len(jobs)}, already done {skipped}", flush=True)
    if args.dry_run:
        for program, variant, backend in jobs:
            print(variant, backend, program["case"])
        return 0
    if not jobs:
        return 0
    write_atomic(args.out / f"manifest-{time.strftime('%Y%m%dT%H%M%S')}.json", manifest(args))

    queue = Queue(jobs, args.out)
    min_free = size_bytes(args.min_free)
    admit = threading.Lock()
    done = [0, 0]  # finished, errors
    count_lock = threading.Lock()
    t0 = time.monotonic()

    def worker(_):
        while (job := queue.pop()) is not None:
            program, variant, backend = job
            with admit:  # one child starts at a time, and only with memory to spare
                while mem_available() < min_free:
                    time.sleep(5)
                time.sleep(0.2)
            record = run_job(args, program, variant, backend)
            write_atomic(result_path(args.out, program["case"], variant, backend), record)
            with count_lock:
                done[0] += 1
                done[1] += "error" in record
                n, e = done
            print(f"[{n}/{len(jobs)} err {e} {time.monotonic() - t0:.0f}s] {variant} {backend} "
                  f"{program['case']} size={record.get('size')} wall={record['wall_s']}s "
                  f"rss={(record.get('peak_rss') or 0) >> 20}M {record.get('error', '')[:160]}",
                  flush=True)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        list(pool.map(worker, range(args.jobs)))
    print(f"finished {done[0]} jobs, {done[1]} errors, {time.monotonic() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

