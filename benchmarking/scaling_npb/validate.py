"""Zinnia on every scaled point: compile for each system, mock-prove input B, compare with NumPy.

This is the acceptance check (`harness/acceptance/validate.check_program`: the
compile, the mock prover on fixture input B, the output decode by the
compiler's layout and the comparison with the NumPy reference within the
entry's tolerance) run once per timed Zinnia system: the compile targets
halo2-base and Noir (the target selects between equivalent lowerings), each as
the compiler chooses and with the backend's own array lowering (the delegate
systems, `ZINNIA_FORCE_CONSTRUCTION=@delegate`, set in the child's
environment as the harness's delegate adapters set it for their steps). Each
(case, system) runs in its own child process under a memory cap and a timeout
(`harness.isolation`);
the outcome uses the acceptance classes (`accepted_correct`,
`accepted_wrong_output`, `accepted_unsatisfied`, `rejected_<class>`, a
resource limit). A failure is a result: it is recorded, nothing is retried.

The NumPy reference itself ran on A and B when the fixture was written
(`acceptance_fixtures.build` refuses a fixture whose reference fails).
The zkVM programs are checked by `baselines/zkvm_common/check_outputs.py`
(`--dataset-dir DATA/cap<C> --systems native,sp1,risc0`, execute mode).
"""

from __future__ import annotations

import functools
import json
import socket
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

from benchmarking.harness import environment, paths
from benchmarking.harness.acceptance import validate as acceptance
from benchmarking.harness.adapters.zinnia_common import DELEGATE_FORCING
from benchmarking.harness.case_sets import load_perf_cases
from benchmarking.harness.isolation import run_child
from benchmarking.scaling_npb.generate import case_set_path

#: system -> (compile target, delegate forcing)
SYSTEMS = {"zinnia_halo2": ("halo2", False), "zinnia_halo2_delegate": ("halo2", True),
           "zinnia_noir": ("noir", False), "zinnia_noir_delegate": ("noir", True)}


def child(request_path: Path, response_path: Path) -> None:
    """Run the acceptance check with the kernel compiled for the request's target (the
    forcing, if any, is in this process's environment)."""
    from benchmarking.harness import kernels

    request = json.loads(request_path.read_text())
    kernels.compile_kernel = functools.partial(kernels.compile_kernel, backend=request["target"])
    response = acceptance.check_program(request)
    response_path.write_text(json.dumps(response))


def check(case, system: str, work: Path, memory: str, timeout: float) -> dict:
    target, delegate = SYSTEMS[system]
    tag = f"{case.case}.{system}"
    request, response = work / f"{tag}.request.json", work / f"{tag}.response.json"
    request.write_text(json.dumps({"kernel": str(case.kernel_path), "function": case.function,
                                   "fixture": str(case.fixture_path), "target": target,
                                   "tolerance": asdict(case.tolerance)}))
    env = environment.child_environment(threads=None)
    if delegate:
        name, value = DELEGATE_FORCING
        env[name] = value
    outcome = run_child([sys.executable, "-m", "benchmarking.scaling_npb.validate", str(request), str(response)],
                        env=env, cwd=paths.REPO_ROOT, timeout_s=timeout, memory_max=memory,
                        log_path=work / f"{tag}.log")
    if response.exists() and not outcome.timed_out:
        result = json.loads(response.read_text())
    elif outcome.timed_out or outcome.killed_by_sigkill or outcome.over_memory_cap:
        reason = "timeout" if outcome.timed_out else "out of memory"
        result = {"stage": "resource", "error": f"{reason} after {outcome.wall_s:.0f}s"}
    else:
        log = (work / f"{tag}.log").read_text()[-1500:]
        result = {"stage": "compile", "error": f"child exited {outcome.returncode} without a result: {log}"}
    verdict = acceptance.outcome_of(result, acceptance.AcceptanceConfig(timeout_s=timeout, memory_max=memory))
    return {"case": case.case, "system": system, "target": target, "outcome": verdict.outcome, "construct": verdict.construct,
            "detail": verdict.detail[:1500], "compile_s": result.get("compile_s"), "wall_s": round(outcome.wall_s, 1),
            "peak_rss_bytes": outcome.peak_rss_bytes}


def main(data: Path, cap: int, out: Path, jobs: int, memory: str, timeout: float, wanted: list[str],
         systems: list[str]) -> int:
    cases = [c for c in load_perf_cases(case_set_path(data, cap), wanted or None, None) if c.included]
    jobs_list = [(case, system) for case in cases for system in systems]
    records = []
    started = time.time()
    with tempfile.TemporaryDirectory(prefix=f"scaling-npb-validate-{cap}-") as tmp:
        work = Path(tmp)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for record in pool.map(lambda job: check(*job, work, memory, timeout), jobs_list):
                print(f"{record['case']:<72} {record['system']:<22} {record['outcome']} {record['detail'][:120]}",
                      flush=True)
                records.append(record)
    counts: dict[str, int] = {}
    for record in records:
        counts[record["outcome"]] = counts.get(record["outcome"], 0) + 1
    document = {"cap": cap, "host": socket.gethostname(), "systems": systems, "memory_max": memory,
                "timeout_s": timeout, "jobs": jobs,
                "wall_s": round(time.time() - started), "counts": counts, "records": records}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=1) + "\n")
    print(json.dumps(counts))
    return 0


if __name__ == "__main__":
    child(Path(sys.argv[1]), Path(sys.argv[2]))
