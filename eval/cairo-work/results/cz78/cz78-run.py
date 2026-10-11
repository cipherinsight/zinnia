"""Cairo timing of the 78 NPBench/Pythran RQ2 programs on the remote (epoch cz78).

The zk42 sandwich, per program: Zinnia (halo2-base and Noir, 5 proofs each), Cairo (1 proof),
Zinnia again (5 proofs each), all pinned to the same 8 cores, so that every Cairo / Zinnia ratio
compares runs of one session on one core group.

Eight core groups (0-7, ..., 56-63). Programs are taken lightest first by their Cairo step count on
input B (from the porting reports). A program reserves the memory cap of its heaviest phase, and
the reservations of the programs that run at once stay within BUDGET_GB, so that the heaviest Cairo
proofs run with few or no neighbours. Results in $RUN/cz78/epochs, one harness epoch per phase.

Usage (in the container): RUN=... python cz78-run.py [case,case,...]
"""

import concurrent.futures
import datetime
import json
import os
import pathlib
import subprocess
import sys
import threading

run = pathlib.Path(os.environ["RUN"])
wt = run / "zinnia-cairo"
out = run / "cz78"
out.mkdir(exist_ok=True)
steps = json.loads((out / "steps-B.json").read_text())
ids = sys.argv[1].split(",") if len(sys.argv) > 1 else sorted(steps, key=steps.get)
tag = os.environ.get("CZ_TAG", "cz78")
BUDGET_GB = 236
GROUPS = 8

entries = {e["case"]: e for e in json.loads(
    (wt / "benchmarking/datasets/case_sets/npbench_pythran_perf.json").read_text())["cases"]}
assert all(entries[c]["status"] == "perf" for c in ids)
case_set = out / f"{tag}-cases.json"
case_set.write_text(json.dumps({"cases": [entries[c] for c in ids]}, indent=2))
subprocess.run(["git", "rev-parse", "HEAD"], cwd=wt, stdout=(out / f"{tag}-commit.txt").open("w"), check=True)


def caps(case: str) -> dict[str, int]:
    """Memory cap (GB) of each phase. Stwo needs about 15 GB up to 1M steps (README, zkvm_common)."""
    zinnia = 64 if "resnet" in case else 30
    n = steps[case]
    cairo = 30 if n <= 1_200_000 else 80 if n <= 7_000_000 else BUDGET_GB
    return {"zpre": zinnia, "zkvm": cairo, "zpost": zinnia}


lock = threading.Condition()
reserved = 0
pending = list(ids)
failures = []


def take() -> tuple[str, int] | None:
    """The next program whose reservation fits, waiting while one does not."""
    global reserved
    with lock:
        while pending:
            case = pending[0]
            need = max(caps(case).values())
            if reserved + need <= BUDGET_GB:
                pending.pop(0)
                reserved += need
                return case, need
            lock.wait()
        return None


def release(need: int) -> None:
    global reserved
    with lock:
        reserved -= need
        lock.notify_all()


def shard(j: int) -> None:
    cpus = f"{8 * j}-{8 * j + 7}"
    while (job := take()) is not None:
        case, need = job
        print("ASSIGN", case, cpus, f"reserve={need}G", datetime.datetime.now().isoformat(), flush=True)
        try:
            for phase, systems, reps in (("zpre", "zinnia_halo2,zinnia_noir", 5), ("zkvm", "cairo", 1),
                                         ("zpost", "zinnia_halo2,zinnia_noir", 5)):
                label = f"{tag}-{case}-{phase}"
                cmd = ["taskset", "-c", cpus, str(wt / ".venv/bin/python"), "-m", "benchmarking.harness", "run",
                       "--label", label, "--case-set", str(case_set), "--cases", case, "--systems", systems,
                       "--repeats", str(reps), "--threads", "8", "--memory-max", f"{caps(case)[phase]}G",
                       "--timeout", "28800", "--results-dir", str(out / "epochs")]
                with (out / f"{tag}-commands-s{j}.jsonl").open("a") as f:
                    f.write(json.dumps({"time": datetime.datetime.now().isoformat(), "cwd": str(wt), "argv": cmd}) + "\n")
                with (out / f"{label}.log").open("w") as log:
                    p = subprocess.run(cmd, cwd=wt, stdout=log, stderr=subprocess.STDOUT)
                print(label, "exit", p.returncode, datetime.datetime.now().isoformat(), flush=True)
                if p.returncode:
                    failures.append({"case": case, "phase": phase, "returncode": p.returncode})
                    break
        finally:
            release(need)


with concurrent.futures.ThreadPoolExecutor(max_workers=GROUPS) as pool:
    list(pool.map(shard, range(GROUPS)))
(out / f"{tag}-failures.json").write_text(json.dumps(failures, indent=2))
print("ALL_SHARDS_FINISHED", "failures", len(failures), flush=True)
