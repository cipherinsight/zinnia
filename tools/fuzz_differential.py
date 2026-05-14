#!/usr/bin/env python3
"""Phase 3a — minimal viable differential fuzzer for Zinnia.

Generates small numpy-like programs from three shapes:

    A) scalar reduction over a 1-D integer array
    B) elementwise transcendental + scalar reduction over a 1-D float array
    C) matrix multiplication

For each generated program:

  1. numpy is the oracle — run it directly on random inputs to get the
     reference output value(s).
  2. Zinnia is the device-under-test — compile the same source as a
     `@zk_circuit`, then run it on the same inputs with a synthesized
     `assert <expr> ~= <ref>` tail. The `@zk_circuit` machinery returns
     a ``ZKExecResult`` whose ``.satisfied`` is True iff every assert
     inside the circuit held; that's the signal we compare against.

A divergence is `satisfied == False` on a program whose reference numpy
output is, by construction, the value the assertion was synthesized
against. A compile failure is any exception (incl. BaseException to
catch PyO3 panics) raised while compiling/running Zinnia.

Run:

    cd zinnia-src
    python3.13 tools/fuzz_differential.py --iterations 30

Each non-success iteration writes a JSON report under
``tools/fuzz_reports/<timestamp>/<index>.json``. Success cases are
counted but not written, to keep the report directory small.

Stdlib + numpy + zinnia only.
"""
from __future__ import annotations

import argparse
import json
import linecache
import math
import multiprocessing as mp
import os
import random
import sys
import time
import traceback
from pathlib import Path

REPO_SRC = Path(__file__).resolve().parent.parent
REPORT_ROOT = REPO_SRC / "tools" / "fuzz_reports"

# Make sure `import zinnia` works regardless of where the script was launched
# from (the impl card says to invoke from the repo root). zinnia-src is the
# package's parent directory.
if str(REPO_SRC) not in sys.path:
    sys.path.insert(0, str(REPO_SRC))

import numpy as _numpy  # noqa: E402 — must happen after sys.path tweak above.

SHAPE_SIZES = [2, 4, 8]
MATMUL_SIZES = [2, 4]
REDUCTIONS_INT = ["sum", "max", "min"]      # `mean` over ints is float in numpy; skip for shape A
REDUCTIONS_FLOAT = ["sum", "mean", "max", "min"]
ELEMENTWISE_FLOAT = ["sqrt", "exp", "log", "sin", "cos"]
INT_LO, INT_HI = -10, 10
FLOAT_LO, FLOAT_HI = -3.0, 3.0
FLOAT_TOL = 1e-3   # ZK fixed-point quantisation is ~2^-precision_bits; transcendentals
                    # compound rounding, so 1e-3 is the practical absolute tolerance for
                    # the differential signal. Tighter would yield spurious divergences.
INT_TOL = 0         # ints are exact
TIMEOUT_SEC = 30


# ----------------------------------------------------------------------------
# Generator
# ----------------------------------------------------------------------------

def _gen_shape_a(rng: random.Random) -> tuple[str, dict, str, object]:
    """Shape A: 1-D integer reduction."""
    size = rng.choice(SHAPE_SIZES)
    op = rng.choice(REDUCTIONS_INT)
    x = _numpy.array([rng.randint(INT_LO, INT_HI) for _ in range(size)], dtype=_numpy.int64)
    ref = getattr(_numpy, op)(x)
    ref_val = int(ref)
    source = (
        "from zinnia import zk_circuit\n"
        "from zinnia.lang.operator import NDArray, Integer, Float\n"
        "import numpy as np\n"
        "\n"
        "@zk_circuit\n"
        f"def f(x: NDArray[Integer, {size}]):\n"
        f"    out = np.{op}(x)\n"
        f"    assert out == {ref_val}\n"
    )
    return source, {"x": x.tolist()}, "A", ref_val


def _gen_shape_b(rng: random.Random) -> tuple[str, dict, str, object]:
    """Shape B: elementwise float + reduce."""
    size = rng.choice(SHAPE_SIZES)
    elem_op = rng.choice(ELEMENTWISE_FLOAT)
    red_op = rng.choice(REDUCTIONS_FLOAT)
    # `x*x + 1.0` is always > 0, so sqrt/log are safe; exp can blow up so cap x.
    x = _numpy.array(
        [rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(size)],
        dtype=_numpy.float64,
    )
    y = getattr(_numpy, elem_op)(x * x + 1.0)
    ref = float(getattr(_numpy, red_op)(y))
    # Use a unique-looking variable name for the literal to make minimisation easier later.
    source = (
        "from zinnia import zk_circuit\n"
        "from zinnia.lang.operator import NDArray, Integer, Float\n"
        "import numpy as np\n"
        "\n"
        "@zk_circuit\n"
        f"def f(x: NDArray[Float, {size}]):\n"
        f"    y = np.{elem_op}(x * x + 1.0)\n"
        f"    out = np.{red_op}(y)\n"
        f"    diff = out - ({ref!r})\n"
        f"    assert diff < {FLOAT_TOL!r}\n"
        f"    assert diff > {-FLOAT_TOL!r}\n"
    )
    return source, {"x": x.tolist()}, "B", ref


def _gen_shape_c(rng: random.Random) -> tuple[str, dict, str, object]:
    """Shape C: matmul on float matrices."""
    M = rng.choice(MATMUL_SIZES)
    K = rng.choice(MATMUL_SIZES)
    N = rng.choice(MATMUL_SIZES)
    a = _numpy.array(
        [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(K)] for _ in range(M)],
        dtype=_numpy.float64,
    )
    b = _numpy.array(
        [[rng.uniform(FLOAT_LO, FLOAT_HI) for _ in range(N)] for _ in range(K)],
        dtype=_numpy.float64,
    )
    c_ref = _numpy.matmul(a, b)
    assert_lines = []
    for i in range(M):
        for j in range(N):
            ref_ij = float(c_ref[i, j])
            assert_lines.append(f"    d = c[{i}, {j}] - ({ref_ij!r})")
            assert_lines.append(f"    assert d < {FLOAT_TOL!r}")
            assert_lines.append(f"    assert d > {-FLOAT_TOL!r}")
    assertions = "\n".join(assert_lines)
    source = (
        "from zinnia import zk_circuit\n"
        "from zinnia.lang.operator import NDArray, Integer, Float\n"
        "import numpy as np\n"
        "\n"
        "@zk_circuit\n"
        f"def f(a: NDArray[Float, {M}, {K}], b: NDArray[Float, {K}, {N}]):\n"
        f"    c = np.matmul(a, b)\n"
        f"{assertions}\n"
    )
    return source, {"a": a.tolist(), "b": b.tolist()}, "C", c_ref.tolist()


SHAPE_GENERATORS = {"A": _gen_shape_a, "B": _gen_shape_b, "C": _gen_shape_c}


def gen_program(shape: str, rng: random.Random):
    return SHAPE_GENERATORS[shape](rng)


# ----------------------------------------------------------------------------
# Runner — child process so a panic / hang doesn't kill the orchestrator
# ----------------------------------------------------------------------------

def _zinnia_child_worker(source: str, inputs_repr: dict, sys_path: list, sys_cwd: str, conn) -> None:
    """Run inside a `multiprocessing` child. Compiles + executes the
    circuit; reports back via ``conn``.

    The pipe payload is one of:
      ("ok", {"satisfied": bool})
      ("fail", {"kind": "compile_failure", "exception": str})
    """
    try:
        # spawn-method children start with a fresh sys.path; replay the
        # parent's so `import zinnia` (and any sibling packages on the
        # path) keeps working. Also inject the parent's cwd so that bare
        # `import zinnia` from the zinnia-src checkout resolves.
        import sys as _sys
        import os as _os
        try:
            _os.chdir(sys_cwd)
        except Exception:
            pass
        for p in reversed(sys_path):
            if p == "":
                p = sys_cwd
            if p and p not in _sys.path:
                _sys.path.insert(0, p)

        # Register the source with linecache so `inspect.getsource` (used by
        # @zk_circuit at decoration time) can recover the function body even
        # though it was injected via ``exec``.
        fake_path = "<fuzz_source>"
        linecache.cache[fake_path] = (
            len(source),
            None,
            source.splitlines(keepends=True),
            fake_path,
        )
        code = compile(source, fake_path, "exec")

        import numpy as np  # type: ignore
        from zinnia import zk_circuit  # type: ignore
        from zinnia.lang.operator import NDArray, Integer, Float  # type: ignore

        g: dict = {
            "__name__": "__fuzz__",
            "np": np,
            "zk_circuit": zk_circuit,
            "NDArray": NDArray,
            "Integer": Integer,
            "Float": Float,
        }
        exec(code, g)  # populates g["f"]
        f = g["f"]
        np_inputs = {k: np.asarray(v) for k, v in inputs_repr.items()}
        # Argument order: the generator only ever names args `x` or `a, b`.
        if "x" in np_inputs:
            result = f(np_inputs["x"])
        elif "a" in np_inputs and "b" in np_inputs:
            result = f(np_inputs["a"], np_inputs["b"])
        else:
            raise RuntimeError(f"unknown input shape: {list(np_inputs)}")
        satisfied = bool(getattr(result, "satisfied", False))
        conn.send(("ok", {"satisfied": satisfied}))
    except BaseException as e:  # noqa: BLE001 — PyO3 panics are BaseException
        tb = traceback.format_exc()
        conn.send((
            "fail",
            {
                "kind": "compile_failure",
                "exception": f"{type(e).__name__}: {e}",
                "traceback": tb[-2000:],
            },
        ))
    finally:
        try:
            conn.close()
        except Exception:
            pass


def run_zinnia(source: str, inputs_repr: dict, timeout: float = TIMEOUT_SEC) -> dict:
    """Run the Zinnia circuit in a watchdog-protected child process.

    Returns a dict with at least:
      {"status": "ok", "satisfied": bool}
      {"status": "compile_failure", "exception": str, "traceback": str}
      {"status": "timeout"}
      {"status": "crash", "exception": str}
    """
    ctx = mp.get_context("spawn")
    parent_conn, child_conn = ctx.Pipe(duplex=False)
    proc = ctx.Process(
        target=_zinnia_child_worker,
        args=(source, inputs_repr, list(sys.path), os.getcwd(), child_conn),
    )
    proc.start()
    child_conn.close()  # parent only reads
    proc.join(timeout)
    if proc.is_alive():
        proc.terminate()
        proc.join(5)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return {"status": "timeout"}
    # Drain the pipe.
    if parent_conn.poll():
        try:
            tag, payload = parent_conn.recv()
        except EOFError:
            return {"status": "crash", "exception": f"child died (exitcode={proc.exitcode})"}
        if tag == "ok":
            return {"status": "ok", "satisfied": payload["satisfied"]}
        elif tag == "fail":
            return {"status": "compile_failure", **payload}
        else:
            return {"status": "crash", "exception": f"unknown tag: {tag!r}"}
    return {"status": "crash", "exception": f"no message; exitcode={proc.exitcode}"}


# ----------------------------------------------------------------------------
# Comparator
# ----------------------------------------------------------------------------

def interpret_outcome(zinnia_result: dict) -> tuple[str, str | None]:
    """Map a Zinnia child outcome onto one of:
      ("success", None) — assertion held; ref matched
      ("divergence", "unsatisfied") — assertion failed; ref didn't match
      ("compile_failure", <exception_str>)
      ("timeout", None)
      ("crash", <exception_str>)
    """
    status = zinnia_result["status"]
    if status == "ok":
        if zinnia_result["satisfied"]:
            return ("success", None)
        return ("divergence", "assertion-unsatisfied")
    if status == "compile_failure":
        return ("compile_failure", zinnia_result.get("exception"))
    if status == "timeout":
        return ("timeout", None)
    return ("crash", zinnia_result.get("exception"))


# ----------------------------------------------------------------------------
# Reporter
# ----------------------------------------------------------------------------

def write_report(
    report_dir: Path,
    index: int,
    shape: str,
    source: str,
    inputs: dict,
    ref_output,
    zinnia_result: dict,
    outcome_kind: str,
    detail: str | None,
) -> Path:
    payload = {
        "index": index,
        "shape": shape,
        "kind": outcome_kind,
        "detail": detail,
        "source": source,
        "inputs": inputs,
        "ref_output": ref_output,
        "zinnia": {
            "status": zinnia_result.get("status"),
            "satisfied": zinnia_result.get("satisfied"),
            "exception": zinnia_result.get("exception"),
        },
    }
    if "traceback" in zinnia_result:
        payload["zinnia"]["traceback"] = zinnia_result["traceback"]
    out_path = report_dir / f"{index:04d}-{outcome_kind}-{shape}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=_json_default))
    return out_path


def _json_default(o):
    if isinstance(o, _numpy.ndarray):
        return o.tolist()
    if isinstance(o, (_numpy.integer,)):
        return int(o)
    if isinstance(o, (_numpy.floating,)):
        return float(o)
    return str(o)


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--shape", choices=["A", "B", "C"], default=None,
                        help="restrict generator to a single shape")
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SEC)
    args = parser.parse_args(argv)

    seed = args.seed if args.seed is not None else int(time.time())
    rng = random.Random(seed)

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    report_dir = REPORT_ROOT / timestamp
    report_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "success": 0,
        "divergence": 0,
        "compile_failure": 0,
        "timeout": 0,
        "crash": 0,
    }
    per_shape = {"A": dict(summary), "B": dict(summary), "C": dict(summary)}

    print(f"[fuzz] seed={seed} iterations={args.iterations} report_dir={report_dir}")
    start = time.time()

    for i in range(args.iterations):
        shape = args.shape or rng.choice(["A", "B", "C"])
        try:
            source, inputs, shape_tag, ref_output = gen_program(shape, rng)
        except Exception as e:
            print(f"[fuzz] iter={i} GEN-ERROR shape={shape} {type(e).__name__}: {e}")
            continue

        zinnia_result = run_zinnia(source, inputs, timeout=args.timeout)
        outcome_kind, detail = interpret_outcome(zinnia_result)

        summary[outcome_kind] += 1
        per_shape[shape_tag][outcome_kind] += 1

        print(
            f"[fuzz] iter={i:04d} shape={shape_tag} "
            f"outcome={outcome_kind}"
            + (f" detail={detail}" if detail else "")
        )

        if outcome_kind != "success":
            write_report(
                report_dir=report_dir,
                index=i,
                shape=shape_tag,
                source=source,
                inputs=inputs,
                ref_output=ref_output,
                zinnia_result=zinnia_result,
                outcome_kind=outcome_kind,
                detail=detail,
            )

    elapsed = time.time() - start

    print()
    print(f"[fuzz] done in {elapsed:.1f}s")
    print(f"[fuzz] summary: {summary}")
    for shp, sub in per_shape.items():
        print(f"[fuzz]   shape {shp}: {sub}")
    print(f"[fuzz] reports: {report_dir}")

    # Always write a summary.json so the report dir has something even on
    # an all-success run.
    (report_dir / "summary.json").write_text(json.dumps({
        "seed": seed,
        "iterations": args.iterations,
        "elapsed_sec": elapsed,
        "summary": summary,
        "per_shape": per_shape,
    }, indent=2))

    return 0


if __name__ == "__main__":
    sys.exit(main())
