"""The acceptance runner: compile + mock-run + reference compare for every program.

Merges the old `validate_contracts.py`, `validate_contracts_parallel.py` and
`validate_ds1000_numpy.py` (which only compiled, audit H-17). For every program
of an acceptance case set, in a child process under the memory cap and the
timeout:

1. the scope rule (imports other than numpy and math) -> `rejected_out_of_scope`;
2. compile; a rejection is classified by `classify.classify_compile_error`;
   a `no_provable_bound` rejection is compiled again with a larger SMT budget,
   and becomes `rejected_solver_budget` if that succeeds;
3. with a fixture: run the mock backend on input B (input A if there is no B)
   and compare its public outputs, in order, with the NumPy reference.

Outcomes: `accepted_correct`, `accepted_wrong_output` (including a mock run
whose constraints fail on the honest input), `accepted_unchecked` (no fixture,
no reference, or an output layout the harness cannot map; the reason is kept),
and `rejected_<class>`.
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path

from benchmarking.harness import environment, paths
from benchmarking.harness.acceptance.classify import classify_compile_error
from benchmarking.harness.case_sets import AcceptanceProgram, Tolerance
from benchmarking.harness.isolation import run_child
from benchmarking.harness.kernels import out_of_scope_imports

LARGER_SMT_BUDGET_MS = 5000


@dataclass
class AcceptanceConfig:
    timeout_s: float = 300.0
    memory_max: str | None = "8G"
    retry_smt_budget_ms: int | None = LARGER_SMT_BUDGET_MS


# ---------------------------------------------------------------------------
# Child side
# ---------------------------------------------------------------------------


def _mock_publics(proof) -> tuple[list[int], bool]:
    """Mock public values (signed integers, Q32 for floats) in exposure order.

    The mock names each public after its `ExposePublic` statement id, so sorting
    by that id restores the exposure order a real backend uses.
    """
    entries = []
    for entry in proof.public_values:
        name, _, value = entry.partition("=")
        entries.append((int(name.rsplit("_", 1)[1]), int(value)))
    entries.sort()
    return [v for _, v in entries], proof.proof_bytes_hex == "mock_satisfied"


def check_program(request: dict) -> dict:
    """Compile, mock-run and compare one program (runs in the child)."""
    from benchmarking.harness.adapters.base import StepFailure
    from benchmarking.harness.adapters.zinnia_common import describe_compiled, zinnia_publics
    from benchmarking.harness.fixtures import PROVE_INPUT, SETUP_INPUT, load_fixture
    from benchmarking.harness.kernels import compile_kernel
    from benchmarking.harness.outputs import check_outputs, reference_leaves
    from benchmarking.harness.reference import run_kernel_as_numpy

    kernel, function = Path(request["kernel"]), request["function"]
    started = time.perf_counter()
    try:
        compiled = compile_kernel(kernel, function)
    except BaseException as exc:  # noqa: BLE001 - PyO3 panics derive from BaseException
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {"stage": "compile", "error": f"{type(exc).__name__}: {exc}"[:2000]}
    result = {"stage": "compiled", "compile_s": time.perf_counter() - started}
    if not request.get("fixture"):
        return {**result, "unchecked": "no fixture"}
    fixture = load_fixture(Path(request["fixture"]))
    input_name = PROVE_INPUT if PROVE_INPUT in fixture.inputs else SETUP_INPUT
    args = fixture.arguments(input_name)
    try:
        proof = compiled.prove(*args, backend="mock")
    except BaseException as exc:  # noqa: BLE001
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {**result, "stage": "mock", "mock_error": f"{type(exc).__name__}: {exc}"[:2000]}
    values, satisfied = _mock_publics(proof)
    if not satisfied:
        return {**result, "stage": "mock", "mock_error": "constraints unsatisfied on the "
                                                         "honest input"}
    try:
        expected, shapes = reference_leaves(run_kernel_as_numpy(kernel, function, args))
    except Exception as exc:  # noqa: BLE001
        return {**result, "unchecked": f"no reference: {type(exc).__name__}: {exc}"[:500]}
    try:
        publics = zinnia_publics(values, describe_compiled(compiled))
    except StepFailure as failure:
        return {**result, "stage": "mock", "mock_error": failure.detail}
    check = check_outputs(expected, publics, Tolerance(**request["tolerance"]),
                          expected_shapes=shapes)
    return {**result, "stage": "checked", "input": input_name, "output_check": check.to_json()}


def _child_main(request_path: Path, response_path: Path) -> int:
    request = json.loads(request_path.read_text())
    try:
        response = check_program(request)
    except Exception as exc:  # noqa: BLE001
        response = {"stage": "harness", "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(limit=6)}
    response_path.write_text(json.dumps(response))
    return 0


# ---------------------------------------------------------------------------
# Parent side
# ---------------------------------------------------------------------------


def _run_isolated(program: AcceptanceProgram, config: AcceptanceConfig, work: Path,
                  smt_budget_ms: int | None) -> dict:
    tag = program.program.replace("/", "_") + (f"-smt{smt_budget_ms}" if smt_budget_ms else "")
    request_path, response_path = work / f"{tag}.request.json", work / f"{tag}.response.json"
    request_path.write_text(json.dumps({
        "kernel": str(program.kernel_path), "function": program.function,
        "fixture": str(program.fixture_path) if program.fixture_path else None,
        "tolerance": asdict(program.tolerance)}))
    response_path.unlink(missing_ok=True)
    env = environment.child_environment(threads=None)
    if smt_budget_ms is not None:
        env["ZINNIA_SMT_QUERY_TIMEOUT_MS"] = str(smt_budget_ms)
    outcome = run_child([sys.executable, "-m", "benchmarking.harness.acceptance.validate",
                         "--child", str(request_path), str(response_path)],
                        env=env, cwd=paths.REPO_ROOT, timeout_s=config.timeout_s,
                        memory_max=config.memory_max, log_path=work / f"{tag}.log")
    if response_path.exists() and not outcome.timed_out:
        return json.loads(response_path.read_text())
    if outcome.timed_out or outcome.killed_by_sigkill:
        reason = "timeout" if outcome.timed_out else "out of memory"
        return {"stage": "resource", "error": f"{reason} after {outcome.wall_s:.0f}s"}
    return {"stage": "compile",
            "error": f"compiler process exited {outcome.returncode} without a result"}


def outcome_of(response: dict) -> tuple[str, str]:
    """(outcome, detail) for one child response."""
    stage = response["stage"]
    if stage == "resource":
        return "rejected_resource_limit", response["error"]
    if stage == "harness":
        return "harness_error", response["error"]
    if stage == "compile":
        return "rejected_" + classify_compile_error(response["error"]), response["error"]
    if stage == "mock":
        return "accepted_wrong_output", response["mock_error"]
    if "unchecked" in response:
        return "accepted_unchecked", response["unchecked"]
    check = response["output_check"]
    if check["status"] == "match":
        return "accepted_correct", check["detail"]
    if check["status"] == "mismatch":
        return "accepted_wrong_output", check["detail"]
    return "accepted_unchecked", check["detail"]


def validate_program(program: AcceptanceProgram, config: AcceptanceConfig, work: Path) -> dict:
    row = {"program": program.program, "dataset": program.dataset,
           "kernel": paths.relative_to_repo(program.kernel_path), "function": program.function}
    out_of_scope = out_of_scope_imports(program.kernel_path)
    if out_of_scope:
        return {**row, "outcome": "rejected_out_of_scope",
                "detail": "imports " + ", ".join(out_of_scope)}
    response = _run_isolated(program, config, work, None)
    outcome, detail = outcome_of(response)
    row.update(outcome=outcome, detail=detail[:2000], response=response)
    if outcome == "rejected_no_provable_bound" and config.retry_smt_budget_ms:
        retry = _run_isolated(program, config, work, config.retry_smt_budget_ms)
        row["retry_with_larger_smt_budget"] = {"budget_ms": config.retry_smt_budget_ms,
                                               "outcome": outcome_of(retry)[0]}
        if retry["stage"] not in ("compile", "resource", "harness"):
            row["outcome"] = "rejected_solver_budget"
    return row


def run_acceptance(programs: list[AcceptanceProgram], config: AcceptanceConfig,
                   out_dir: Path) -> Path:
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    rows = []
    for index, program in enumerate(programs, 1):
        row = validate_program(program, config, work)
        rows.append(row)
        print(f"[{index}/{len(programs)}] {program.program}::{program.function}: "
              f"{row['outcome']}", flush=True)
    path = out_dir / "results.json"
    path.write_text(json.dumps({"config": asdict(config), "results": rows}, indent=1))
    return path


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--child":
        sys.exit(_child_main(Path(sys.argv[2]), Path(sys.argv[3])))
    print("usage: python -m benchmarking.harness accept ...", file=sys.stderr)
    sys.exit(2)
