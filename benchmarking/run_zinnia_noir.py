"""Run the Zinnia Noir source-to-source backend on DS-1000 cases.

Defaults to the 19 pilot cases. `--cases all` runs every case in
hw_halo2_results.json (the canonical Zinnia-compilable working set).

Methodology mirrors `run_zinnia_halo2.py`:

  1. Import `ds1000_case<NNNN>/kernel.py`, compile via ZKCircuit.
  2. Map INPUT_BUILDERS to positional kernel args.
  3. Cold prove (the emitter+nargo execute is the cold cost; bb write_vk is
     also one-shot per call but excluded from the timed window inside the
     Rust driver). Record as `first_prove_s`.
  4. `repeats` more proves at the same shape. The Rust driver re-runs the
     full emit -> nargo -> bb prove pipeline each call (no in-memory
     artifact cache yet; the bb prove call alone is what's measured by
     `prove_seconds`).
  5. Verify (already done inside the driver via `bb verify`; we just check
     the artifact carries a non-empty proof_bytes).

Per-case wall-clock cap is configurable via `--per-case-timeout` (default
60 s for the full sweep, no cap for explicit `--cases NNNN,...`). The cap
is enforced by running each case in a subprocess.

Failure taxonomy on the output records:
  - `status="ok"`: prove + verify both succeeded.
  - `status="unsupported_op"`: emitter hit an `UnsupportedIR` for some op.
  - `status="timeout"`: case exceeded the per-case wall cap.
  - `status="fail"`: anything else (compile error, nargo failure, etc.).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

THIS_DIR = Path(__file__).parent
ZINNIA_ROOT = THIS_DIR.parent
DEFAULT_PRECISION_BITS = 32

# Pilot cases (handwritten Noir baselines exist for these).
PILOT_CASES = [
    "0291", "0309", "0323", "0329", "0330", "0334", "0335", "0336",
    "0373", "0387", "0416", "0418", "0428", "0436", "0437", "0452",
    "0454", "0469", "0470",
]

sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ZINNIA_ROOT))
from ensure_release_build import ensure_zinnia_release_build  # noqa: E402
from run_halo2_baselines import INPUT_BUILDERS  # noqa: E402


def _coerce_arg(value, dt_spec=None):
    import numpy as np
    target_shape = None
    target_dtype = None
    if isinstance(dt_spec, dict):
        if "NDArray" in dt_spec:
            target_shape = tuple(dt_spec["NDArray"].get("shape", []))
            target_dtype = dt_spec["NDArray"].get("dtype", None)
        elif "Scalar" in dt_spec:
            target_dtype = dt_spec["Scalar"].get("dtype", None)
    if isinstance(value, list):
        if not value:
            arr = np.array([], dtype=np.float64)
        else:
            first = value[0]
            if isinstance(first, str):
                arr = np.array([int(s) for s in value], dtype=np.int64)
            elif isinstance(first, bool):
                arr = np.array(value, dtype=np.int64)
            elif isinstance(first, int):
                arr = np.array(value, dtype=np.int64)
            elif isinstance(first, float):
                arr = np.array(value, dtype=np.float64)
            else:
                arr = np.asarray(value)
        if target_shape:
            arr = arr.reshape(target_shape)
        if target_dtype == "Float":
            arr = arr.astype(np.float64)
        elif target_dtype in ("Integer", "Boolean"):
            arr = arr.astype(np.int64)
        return arr
    if isinstance(value, str):
        if target_dtype == "Float":
            return float(value)
        return int(value)
    if target_dtype == "Float":
        return float(value)
    return value


@dataclass
class CaseResult:
    case: str
    first_prove_s: float | None = None
    warm_prove_times_s: list = field(default_factory=list)
    prove_s: float | None = None
    proof_bytes: int | None = None
    verify_ok: bool | None = None
    error: str | None = None
    panicked: bool = False
    no_builder: bool = False
    status: str = "pending"
    unsupported_op: str | None = None
    wall_seconds: float | None = None
    gate_count: int | None = None


def _bb_path() -> str:
    """Resolve the `bb` binary the same way `run_noir_baselines.py` does."""
    override = os.environ.get("ZINNIA_BB_BIN")
    if override:
        return override
    default = Path.home() / ".bb" / "bb"
    if default.exists():
        return str(default)
    found = shutil.which("bb")
    if found:
        return found
    raise FileNotFoundError("bb not found (set ZINNIA_BB_BIN or install to ~/.bb/bb)")


def _find_latest_staging_dir(after_mtime: float) -> Path | None:
    """Find the most recent `/tmp/zinnia-noir-*` dir created after the given
    monotonic-ish mtime. We rely on the Rust driver to stage there when
    `ZINNIA_NOIR_KEEP_TEMP=1` is set."""
    candidates = []
    for entry in Path("/tmp").iterdir():
        if entry.is_dir() and entry.name.startswith("zinnia-noir-"):
            try:
                mt = entry.stat().st_mtime
            except OSError:
                continue
            if mt >= after_mtime:
                candidates.append((mt, entry))
    if not candidates:
        return None
    candidates.sort(reverse=True)
    return candidates[0][1]


def run_case(case: str, repeats: int, precision_bits: int) -> CaseResult:
    if case not in INPUT_BUILDERS:
        return CaseResult(case=case, no_builder=True)

    input_dict = INPUT_BUILDERS[case]()
    try:
        from zinnia.api.zk_circuit import ZKCircuit  # type: ignore
        kernel_path = THIS_DIR / f"ds1000_case{case}" / "kernel.py"
        spec = importlib.util.spec_from_file_location(f"case{case}_{id(case)}", kernel_path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        compiled = ZKCircuit.from_method(mod.kernel).compile()

        def lookup(name: str):
            for key in (name, name.lower(), name.upper(),
                        f"{name}_val", f"{name.lower()}_val"):
                if key in input_dict:
                    return input_dict[key]
            raise KeyError(f"no builder key for kernel param {name!r}; "
                           f"available={list(input_dict.keys())}")

        inputs = [_coerce_arg(lookup(pi.name), pi.export().get("dt"))
                  for pi in compiled.program_inputs]
    except BaseException as e:  # noqa: BLE001
        return CaseResult(case=case,
                          error=f"setup: {type(e).__name__}: {e}",
                          panicked=True)

    cr = CaseResult(case=case)
    last_proof = None
    params = {"k": 0, "precision_bits": precision_bits, "lookup_bits": 8}

    # Stage once + warm-repeat methodology (matches run_noir_baselines.py).
    # The Rust driver stages a fresh temp dir per `.prove()` call by default,
    # so calling it `repeats` times means every call is cold-cache for the
    # bytecode / witness / vk files — unfair vs the hw runner which stages
    # once then loops `bb prove`. Here:
    #   1. First call: KEEP_TEMP=1 so the staging dir survives.
    #   2. Capture the staged dir.
    #   3. Call `bb prove` directly N-1 times against the same staged files
    #      so calls 2..N hit warm OS-file-cache, matching the hw repeat shape.
    prev_keep = os.environ.get("ZINNIA_NOIR_KEEP_TEMP")
    os.environ["ZINNIA_NOIR_KEEP_TEMP"] = "1"
    pre_call_mtime = time.time()
    try:
        proof = compiled.prove(*inputs, backend="noir", params=params)
        cr.first_prove_s = getattr(proof, "prove_seconds", 0.0)
        last_proof = proof
    except BaseException as e:  # noqa: BLE001
        cr.error = f"first prove: {type(e).__name__}: {e}"
        if prev_keep is None:
            os.environ.pop("ZINNIA_NOIR_KEEP_TEMP", None)
        else:
            os.environ["ZINNIA_NOIR_KEEP_TEMP"] = prev_keep
        return cr

    staged_dir = _find_latest_staging_dir(pre_call_mtime)
    warm_proves = max(0, repeats - 1)
    bb = None
    if staged_dir is not None:
        try:
            bb = _bb_path()
        except FileNotFoundError as e:
            cr.error = (cr.error or "") + f" bb setup: {e}"
            bb = None
    if bb is not None and staged_dir is not None:
        target = staged_dir / "target"
        # Capture circuit_size via `bb gates` (mirrors run_noir_baselines.py:857).
        gates_proc = subprocess.run(
            [bb, "gates",
             "-b", str(target / "zinnia_circuit.json"),
             "--scheme", "ultra_honk",
             "--verifier_target", "noir-recursive-no-zk"],
            capture_output=True, text=True,
        )
        if gates_proc.returncode == 0:
            try:
                blob = gates_proc.stdout.strip()
                jstart = blob.find("{")
                if jstart >= 0:
                    obj = json.loads(blob[jstart:])
                    fns = obj.get("functions") or []
                    if fns:
                        cr.gate_count = (fns[0].get("circuit_size")
                                         or fns[0].get("acir_opcodes"))
            except Exception:
                pass
    if bb is not None and staged_dir is not None and warm_proves > 0:
        target = staged_dir / "target"
        # Match the Rust driver's CLI exactly (driver.rs:115-123).
        cmd = [
            bb, "prove",
            "-b", str(target / "zinnia_circuit.json"),
            "-w", str(target / "zinnia_circuit.gz"),
            "-k", str(target / "vk"),
            "-o", str(target),
            "--scheme", "ultra_honk",
            "--verifier_target", "noir-recursive-no-zk",
        ]
        for _ in range(warm_proves):
            t0 = time.time()
            proc = subprocess.run(cmd, capture_output=True, text=True)
            dt = time.time() - t0
            if proc.returncode != 0:
                cr.error = (cr.error or "") + (
                    f" warm bb prove rc={proc.returncode}: "
                    + (proc.stdout + proc.stderr)[-400:]
                )
                break
            cr.warm_prove_times_s.append(dt)

    # Cleanup the staged dir (matches default behaviour when KEEP_TEMP unset).
    # Honour ZINNIA_NOIR_KEEP_TEMP=1 if the caller set it before invoking us —
    # useful for debugging the emitted main.nr.
    if staged_dir is not None and prev_keep != "1":
        try:
            shutil.rmtree(staged_dir, ignore_errors=True)
        except OSError:
            pass

    # Restore caller's KEEP_TEMP setting.
    if prev_keep is None:
        os.environ.pop("ZINNIA_NOIR_KEEP_TEMP", None)
    else:
        os.environ["ZINNIA_NOIR_KEEP_TEMP"] = prev_keep

    if cr.warm_prove_times_s:
        cr.prove_s = statistics.median(cr.warm_prove_times_s)
    else:
        cr.prove_s = cr.first_prove_s

    if last_proof is not None:
        try:
            cr.proof_bytes = len(last_proof.proof_bytes_hex) // 2
            cr.verify_ok = bool(compiled.verify(last_proof))
        except BaseException as e:  # noqa: BLE001
            cr.error = (cr.error or "") + f" verify: {type(e).__name__}: {e}"
            cr.verify_ok = False
    return cr


def _load_working_cases() -> list[str]:
    """Working = the cases in hw_halo2_results.json (cleaned of the 28
    halo2-incompatible kernels by upstream construction)."""
    p = THIS_DIR / "hw_halo2_results.json"
    if not p.exists():
        return []
    data = json.loads(p.read_text())
    return sorted({r["case"] for r in data.get("results", [])})


def _classify_status(cr: CaseResult) -> CaseResult:
    if cr.error:
        msg = cr.error
        if "UnsupportedIR" in msg or "UnsupportedOp" in msg or "noir-emit:" in msg:
            cr.status = "unsupported_op"
            # Extract op name from "noir-emit:<op>"
            for tok in msg.split():
                if "noir-emit:" in tok:
                    op = tok.split("noir-emit:")[-1].rstrip(",;\"' )")
                    cr.unsupported_op = op
                    break
        else:
            cr.status = "fail"
    elif cr.prove_s is not None and cr.verify_ok:
        cr.status = "ok"
    elif cr.prove_s is not None and cr.verify_ok is False:
        cr.status = "fail"
        cr.error = (cr.error or "") + " verify_ok=False"
    else:
        cr.status = "fail"
    return cr


def _run_case_subprocess(case: str, repeats: int, precision_bits: int,
                         timeout_s: float | None) -> CaseResult:
    """Spawn this script in `--child` mode so a single hung case can be killed."""
    import subprocess
    cmd = [
        sys.executable, __file__, "--child", case,
        "--repeats", str(repeats),
        "--precision-bits", str(precision_bits),
    ]
    t0 = time.time()
    try:
        out = subprocess.run(cmd, capture_output=True, text=True,
                             timeout=timeout_s)
    except subprocess.TimeoutExpired:
        wall = time.time() - t0
        return CaseResult(case=case, status="timeout", wall_seconds=wall,
                          error=f"per-case timeout after {timeout_s:.0f}s")
    wall = time.time() - t0
    if out.returncode != 0:
        return _classify_status(CaseResult(
            case=case, wall_seconds=wall, panicked=True,
            error=f"subprocess exit {out.returncode}: " + (out.stderr or "")[-300:],
        ))
    # Child prints the JSON-serialized CaseResult on the LAST line of stdout.
    try:
        last = out.stdout.strip().splitlines()[-1]
        data = json.loads(last)
        cr = CaseResult(**{k: v for k, v in data.items() if k in CaseResult.__dataclass_fields__})
        cr.wall_seconds = wall
        return _classify_status(cr)
    except Exception as e:  # noqa: BLE001
        return _classify_status(CaseResult(
            case=case, wall_seconds=wall, panicked=True,
            error=f"could not parse child output: {e}",
        ))


def _child_main(case: str, repeats: int, precision_bits: int):
    try:
        cr = run_case(case, repeats, precision_bits)
    except BaseException as e:  # noqa: BLE001
        cr = CaseResult(case=case, error=f"runner exception: {type(e).__name__}: {e}",
                        panicked=True)
    print(json.dumps(asdict(cr)))
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default=None,
                    help='comma-separated case ids; "all" = all working cases '
                         '(from hw_halo2_results.json); default = 19 pilot cases')
    ap.add_argument("--out", default=str(THIS_DIR / "zinnia_noir_results.json"))
    ap.add_argument("--precision-bits", type=int, default=DEFAULT_PRECISION_BITS)
    ap.add_argument("--nargo", default=None,
                    help="override path to `nargo` via ZINNIA_NARGO_BIN")
    ap.add_argument("--bb", default=None,
                    help="override path to `bb` via ZINNIA_BB_BIN")
    ap.add_argument("--per-case-timeout", type=float, default=None,
                    help="per-case wall-clock cap in seconds. Default is 60s "
                         "for `--cases all`, no cap for explicit case lists.")
    ap.add_argument("--child", default=None,
                    help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.nargo:
        os.environ["ZINNIA_NARGO_BIN"] = args.nargo
    if args.bb:
        os.environ["ZINNIA_BB_BIN"] = args.bb

    ensure_zinnia_release_build()

    if args.child:
        _child_main(args.child, args.repeats, args.precision_bits)
        return

    if args.cases is None:
        cases = list(PILOT_CASES)
        use_subprocess = False
    elif args.cases.strip().lower() == "all":
        cases = _load_working_cases()
        if not cases:
            print("error: could not load working case list (no hw_halo2_results.json)")
            sys.exit(2)
        use_subprocess = True
    else:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
        use_subprocess = False

    timeout_s = args.per_case_timeout
    if timeout_s is None and use_subprocess:
        timeout_s = 60.0

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for i, c in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] case {c} ", end="", flush=True)
        t0 = time.time()
        try:
            if use_subprocess:
                r = _run_case_subprocess(c, args.repeats, args.precision_bits, timeout_s)
            else:
                r = run_case(c, args.repeats, args.precision_bits)
                r = _classify_status(r)
        except BaseException as e:  # noqa: BLE001
            r = CaseResult(case=c, error=f"runner exception: {type(e).__name__}: {e}",
                           panicked=True)
            r = _classify_status(r)
            traceback.print_exc()
        wall = time.time() - t0
        if r.wall_seconds is None:
            r.wall_seconds = wall
        results.append(r)
        if r.status == "ok":
            print(f"prove={r.prove_s*1000:.0f}ms first={(r.first_prove_s or 0.0)*1000:.0f}ms "
                  f"proof={r.proof_bytes}B verify={r.verify_ok} (wall {wall:.1f}s)")
        elif r.status == "unsupported_op":
            print(f"UNSUPPORTED op={r.unsupported_op} (wall {wall:.1f}s)")
        elif r.status == "timeout":
            print(f"TIMEOUT (wall {wall:.1f}s)")
        else:
            print(f"FAIL: {(r.error or '')[:200]} (wall {wall:.1f}s)")
        out_path.write_text(json.dumps(
            {"results": [asdict(x) for x in results]},
            indent=2,
        ))

    # Final summary
    statuses = {}
    for r in results:
        statuses[r.status] = statuses.get(r.status, 0) + 1
    print()
    print(f"Wrote {out_path}")
    print(f"Summary: {statuses}")


if __name__ == "__main__":
    main()
