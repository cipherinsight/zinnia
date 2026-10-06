"""Child-process entry point: run one step of one adapter.

    python -m benchmarking.harness.step REQUEST.json RESPONSE.json

The parent (`protocol`) writes the request and reads the response. The
response is always written unless the process dies (timeout, out of memory,
crash), which the parent classifies from the exit status.

Request: {"system", "step": "setup"|"prove"|"verify"|"reference", "case": PerfCase
json, "input": "A"|"B", "work_dir", "proof_dir"?, "proof_path"?, "tamper"?, "options"?}
Response: {"ok": bool, "result": {...} | null, "failure": {"class", "detail"} | null}
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from benchmarking.harness.adapters import get_adapter
from benchmarking.harness.adapters.base import StepContext, StepFailure
from benchmarking.harness.case_sets import PerfCase
from benchmarking.harness.isolation import oom_kills_in_own_cgroup


def _optional_path(request: dict, key: str) -> Path | None:
    return Path(request[key]) if request.get(key) else None


def _context(request: dict) -> StepContext:
    return StepContext(case=PerfCase.from_json(request["case"]), input_name=request["input"],
                       work_dir=Path(request["work_dir"]),
                       proof_dir=_optional_path(request, "proof_dir"),
                       proof_path=_optional_path(request, "proof_path"),
                       tamper=bool(request.get("tamper")), options=request.get("options", {}))


def run_reference(ctx: StepContext) -> dict:
    """The NumPy reference outputs for one input: the returned value, or for a
    kernel that returns `None` the arguments the case set lists as written."""
    from benchmarking.harness.outputs import call_outputs, leaves_to_json
    from benchmarking.harness.reference import run_kernel_with_arguments

    try:
        arguments = ctx.arguments()
        result, after = run_kernel_with_arguments(ctx.case.kernel_path, ctx.case.function,
                                                  arguments)
        leaves, shapes = call_outputs(result, after, arguments, ctx.case.mutated_outputs)
    except Exception as exc:  # noqa: BLE001 - any failure means "no reference"
        raise StepFailure("reference_failed", f"{type(exc).__name__}: {exc}") from None
    return {"leaves": leaves_to_json(leaves), "shapes": shapes}


def run_step(request: dict) -> dict:
    ctx = _context(request)
    ctx.work_dir.mkdir(parents=True, exist_ok=True)
    step = request["step"]
    if step == "reference":
        return run_reference(ctx)
    adapter = get_adapter(request["system"])
    if step == "setup":
        return adapter.setup(ctx).to_json()
    if step == "prove":
        return adapter.prove(ctx).to_json()
    if step == "verify":
        return adapter.verify(ctx).to_json()
    raise ValueError(f"unknown step {step!r}")


def _failure(failure_class: str, detail: str) -> dict:
    """A failed step's response; an OOM kill in the step's scope overrides the class.

    A program that loses a child to the memory cap often reports only that the
    child failed; the scope's OOM counter says why.
    """
    oom_kills = oom_kills_in_own_cgroup()
    if oom_kills and failure_class != "out_of_memory":
        detail = f"{oom_kills} process(es) OOM-killed at the memory cap; reported: {detail}"
        failure_class = "out_of_memory"
    return {"ok": False, "result": None, "failure": {"class": failure_class, "detail": detail}}


def main(argv: list[str]) -> int:
    request_path, response_path = Path(argv[0]), Path(argv[1])
    request = json.loads(request_path.read_text())
    try:
        response = {"ok": True, "result": run_step(request), "failure": None}
    except StepFailure as failure:
        response = _failure(failure.failure_class, failure.detail)
    except Exception as exc:  # noqa: BLE001 - recorded as a harness error, never dropped
        response = _failure("harness_error", f"{type(exc).__name__}: {exc}\n"
                                             + traceback.format_exc(limit=8))
    response_path.write_text(json.dumps(response))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
