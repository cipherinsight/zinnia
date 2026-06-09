"""Run the Zinnia halo2 backend on every DS-1000 case for which a target.rs
also exists. Methodology v2: single long-lived Python process so the
in-memory proving-key cache stays warm across cases.

Per case:
  1. Import `ds1000_case<NNNN>/kernel.py`, compile.
  2. Map the INPUT_BUILDERS dict to positional kernel args (shape-aware
     coercion using each program input's exported dt).
  3. Bisect `k` upward from 10 until `prove()` returns.
  4. At the discovered min-k, do ONE warm-up prove (this populates the
     in-process pk cache — analogous to HW's on-disk data/*.pk) — record
     its time as `first_prove_s`.
  5. Do `repeats` more proves at the same k — record warm prove times.
  6. Compute:
       keygen_s   = first_prove_s − median(warm_prove_s)
       prove_s    = median(warm_prove_s)
  7. Capture proof bytes from the last proof; verify and time it.

The PERSISTENT IPA params cache (env `ZINNIA_HALO2_IPA_PARAMS_DIR`) is
enabled by default so that fresh-process cold-start time across runs is
also amortized.

Crash isolation: a try/except around each case catches Python-side
ZinniaException and prove/verify errors. PyO3 panics (BaseException) are
also caught here without killing the run.

Usage:
    python3 run_zinnia_halo2.py [--repeats N] [--cases 0309,...] [--out FILE]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path

THIS_DIR = Path(__file__).parent
ZINNIA_ROOT = THIS_DIR.parent
K_MIN = 4
K_MAX = 22
DEFAULT_LOOKUP_BITS = 8
DEFAULT_PRECISION_BITS = 32
DEFAULT_IPA_PARAMS_DIR = THIS_DIR / ".ipa_params_cache"

# Import INPUT_BUILDERS from the HW runner so both sides share the same fixtures.
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(ZINNIA_ROOT))
from ensure_release_build import ensure_zinnia_release_build  # noqa: E402
from run_halo2_baselines import INPUT_BUILDERS  # noqa: E402


def _coerce_arg(value, dt_spec=None):
    """Map a CircuitInput-style JSON value to the python value Zinnia wants."""
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
    min_k: int | None = None
    # First prove (cold; includes keygen) and subsequent warm proves.
    first_prove_s: float | None = None
    warm_prove_times_s: list = field(default_factory=list)
    keygen_s: float | None = None
    prove_s: float | None = None
    proof_bytes: int | None = None
    verify_ok: bool | None = None
    verify_detail: str | None = None
    verify_time_s: float | None = None
    error: str | None = None
    panicked: bool = False
    no_target: bool = False
    no_builder: bool = False


def run_case(case: str, repeats: int, precision_bits: int, lookup_bits: int,
             backend: str,
             k_min: int, k_max: int) -> CaseResult:
    if not (THIS_DIR / f"ds1000_case{case}" / "target.rs").exists():
        return CaseResult(case=case, no_target=True)
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

    # Bisect k. The first successful prove time is "first_prove_s" (cold).
    cr = CaseResult(case=case)
    last_proof = None
    if backend == "halo2-base" and k_min == K_MIN:
        # Try backend auto-sizing first, then fall back to the explicit k
        # search if the estimate is still too small for halo2-base packing.
        k_candidates = [None, *range(k_min, k_max + 1)]
    else:
        k_candidates = range(k_min, k_max + 1)
    for k in k_candidates:
        params = None if k is None else {"k": k, "precision_bits": precision_bits, "lookup_bits": lookup_bits}
        try:
            t0 = time.time()
            proof = compiled.prove(*inputs, backend=backend, params=params)
            # Prefer the apples-to-apples `prove_seconds` reported by the
            # Rust backend (just create_proof + transcript finalize). Falls
            # back to wall-clock on backends that don't fill it.
            cr.first_prove_s = getattr(proof, "prove_seconds", 0.0) or (time.time() - t0)
            cr.min_k = getattr(proof, "k", None) or k
            last_proof = proof
            break
        except BaseException as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            lower = msg.lower()
            if ("notenoughrowsavailable" in lower
                    or "not enough rows" in lower
                    or "not enough advice columns" in lower
                    or "lookup table is too large" in lower
                    or "k is too small" in lower):
                continue
            if k is None:
                cr.error = msg
                return cr
            cr.error = msg
            return cr

    if cr.min_k is None:
        cr.error = f"no viable k in [{k_min}..{k_max}]"
        return cr

    # Warm proves
    params = {"k": cr.min_k, "precision_bits": precision_bits, "lookup_bits": lookup_bits}
    for _ in range(repeats):
        try:
            t0 = time.time()
            proof = compiled.prove(*inputs, backend=backend, params=params)
            cr.warm_prove_times_s.append(
                getattr(proof, "prove_seconds", 0.0) or (time.time() - t0)
            )
            last_proof = proof
        except BaseException as e:  # noqa: BLE001
            cr.error = f"warm prove: {type(e).__name__}: {e}"
            break

    if cr.warm_prove_times_s:
        cr.prove_s = statistics.median(cr.warm_prove_times_s)
        # keygen = cold − warm (clamp to 0 to avoid noise-driven negatives).
        cr.keygen_s = max(0.0, (cr.first_prove_s or 0.0) - cr.prove_s)

    if last_proof is not None:
        try:
            from zinnia.compile._bridge import verify_proof_artifact

            cr.proof_bytes = len(last_proof.proof_bytes_hex) // 2
            t0 = time.time()
            verify_result = json.loads(verify_proof_artifact(last_proof.to_json()))
            cr.verify_time_s = time.time() - t0
            cr.verify_ok = bool(verify_result.get("valid"))
            cr.verify_detail = verify_result.get("error")
        except BaseException as e:  # noqa: BLE001
            cr.error = (cr.error or "") + f" verify: {type(e).__name__}: {e}"

    return cr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--cases", default=None,
                    help="comma-separated case ids; default = all installed builders")
    ap.add_argument("--out", default=str(THIS_DIR / "zinnia_halo2_results.json"))
    ap.add_argument("--k-min", type=int, default=None)
    ap.add_argument("--k-max", type=int, default=K_MAX)
    ap.add_argument("--precision-bits", type=int, default=DEFAULT_PRECISION_BITS)
    ap.add_argument("--lookup-bits", type=int, default=DEFAULT_LOOKUP_BITS)
    ap.add_argument("--backend", default="halo2",
                    help="Zinnia prover backend name (default: halo2; e.g. halo2-base)")
    ap.add_argument("--ipa-params-dir", default=str(DEFAULT_IPA_PARAMS_DIR),
                    help="dir for persistent IPA params (default: ./.ipa_params_cache)")
    args = ap.parse_args()
    ensure_zinnia_release_build()
    if args.k_min is None:
        args.k_min = K_MIN

    Path(args.ipa_params_dir).mkdir(parents=True, exist_ok=True)
    os.environ["ZINNIA_HALO2_IPA_PARAMS_DIR"] = args.ipa_params_dir
    print(f"IPA params persistent cache: {args.ipa_params_dir}")

    if args.cases:
        cases = [c.strip() for c in args.cases.split(",") if c.strip()]
    else:
        cases = sorted(
            c for c in INPUT_BUILDERS
            if (THIS_DIR / f"ds1000_case{c}" / "target.rs").exists()
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[CaseResult] = []
    for i, c in enumerate(cases):
        print(f"[{i+1}/{len(cases)}] case {c} ", end="", flush=True)
        t0 = time.time()
        try:
            r = run_case(c, args.repeats, args.precision_bits, args.lookup_bits,
                         args.backend,
                         args.k_min, args.k_max)
        except BaseException as e:  # noqa: BLE001
            r = CaseResult(case=c, error=f"runner exception: {type(e).__name__}: {e}",
                           panicked=True)
            traceback.print_exc()
        wall = time.time() - t0
        results.append(r)
        if r.prove_s is not None:
            print(f"k={r.min_k} keygen={r.keygen_s*1000:.0f}ms "
                  f"prove={r.prove_s*1000:.0f}ms "
                  f"proof={r.proof_bytes}B verify={r.verify_ok} "
                  f"(wall {wall:.1f}s)")
        elif r.error:
            print(f"ERROR: {r.error[:120]} (wall {wall:.1f}s)")
        elif r.no_target:
            print("(no target.rs — skipped)")
        elif r.no_builder:
            print("(no INPUT_BUILDER — skipped)")
        else:
            print(f"(wall {wall:.1f}s)")
        out_path.write_text(json.dumps(
            {"results": [asdict(x) for x in results]},
            indent=2,
        ))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
