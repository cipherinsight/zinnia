#!/usr/bin/env python3
"""Position-aware output check: Zinnia's circuit outputs vs NumPy.

For every DS-1000 performance-set case with an input fixture
(`run_halo2_baselines.INPUT_BUILDERS`):

1. NumPy reference: run the kernel body as plain NumPy on the fixture (the
   `@zk_circuit` decorator replaced by an identity). No per-case overrides: a
   kernel NumPy rejects is reported as `numpy_error`.
2. Circuit outputs: compile the kernel, prove it on the mock backend, and
   rebuild the full output sequence in return order: the exposed publics in
   statement (= exposure) order, with every constant the compiler elided from
   the public instance put back at its reported slot.
3. Comparison by position. A static result part must equal NumPy's flattened
   values one for one. A part with a runtime shape is exposed as metadata
   (runtime length, rank and the runtime dims that are not compile-time
   constants) followed by a payload padded to the envelope; its metadata must
   equal NumPy's size/ndim/shape, and the runtime elements must sit at their
   row-major positions in either the compact layout (flat prefix) or a padded
   layout (prefix rectangle of an envelope shape). Padding slots are not data
   and are not compared. The layout that matched is reported per part.
   Integers compare exactly; floats within FLOAT_ATOL + FLOAT_RTOL * |x|
   (the circuit computes in Q32 fixed point).
4. Tampered-output control, per case: the same comparison is run on a copy of
   the circuit outputs with one runtime value changed, and on a copy with two
   different runtime values swapped. Both must be rejected; otherwise the
   check has no power on that case and the case is flagged.

This replaces the multiset comparison of `audit_bindings.py` (audit finding
M9), which could not see wrong positions.

Usage:
    python check_outputs.py [--cases all | 0301,0415] [--output check_outputs.json]
                            [--timeout 600] [--memory 8G]

Each case runs in its own process under the memory cap (`eval-redo/tools/
memrun.sh` when present). Exit status 0 iff every case passes. Status
`pass_stable_ties` means the outputs equal NumPy's once `np.argsort`/
`np.sort` order equal elements stably (NumPy's default sort leaves their
order unspecified); `pass_empty_output` means the result has no elements
(so there is nothing for the controls to tamper with).
"""

from __future__ import annotations

import argparse
import importlib.util
import itertools
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent
MEMRUN = Path("/home/zhantong/Projects/zinnia-project/eval-redo/tools/memrun.sh")

Q32 = 1 << 32
FLOAT_ATOL = 1e-3
FLOAT_RTOL = 1e-4


# ── NumPy reference ──────────────────────────────────────────────────────

def _identity_decorator(method=None, **_kwargs):
    if method is None:
        return lambda m: m
    return method


def numpy_reference(case: str, args: list, stable_sort: bool = False):
    """Run the kernel body as plain NumPy; returns its return value.

    With `stable_sort`, `np.argsort`/`np.sort` default to `kind="stable"`.
    NumPy's default sort kind leaves the order of equal elements
    unspecified, so a circuit that orders ties stably computes a valid
    result that may differ from the default run."""
    import numpy as np
    import zinnia

    kernel_path = THIS_DIR / f"ds1000_case{case}" / "kernel.py"
    namespace = {name: getattr(zinnia, name) for name in dir(zinnia) if not name.startswith("_")}
    numpy = np
    if stable_sort:
        import types
        numpy = types.SimpleNamespace(**{k: getattr(np, k) for k in dir(np) if not k.startswith("__")})
        numpy.argsort = lambda a, axis=-1, kind=None, **kw: np.argsort(a, axis=axis, kind=kind or "stable", **kw)
        numpy.sort = lambda a, axis=-1, kind=None, **kw: np.sort(a, axis=axis, kind=kind or "stable", **kw)
    namespace.update(np=numpy, zk_circuit=_identity_decorator, zk_chip=_identity_decorator,
                     zk_external=_identity_decorator)
    source = kernel_path.read_text().replace("from zinnia import *", "").replace(
        "import numpy as np", "")
    exec(compile(source, str(kernel_path), "exec"), namespace)
    return namespace["kernel"](*[np.array(a, copy=True) if isinstance(a, np.ndarray) else a
                                 for a in args])


def result_parts(result) -> list:
    """The return value as a list of NumPy arrays, one per returned part."""
    import numpy as np

    if isinstance(result, tuple) or (isinstance(result, list)
                                     and any(isinstance(p, np.ndarray) for p in result)):
        parts = list(result)
    else:
        parts = [result]
    out = []
    for part in parts:
        arr = np.asarray(part)
        if arr.dtype.kind == "c":
            arr = np.stack([arr.real, arr.imag], axis=-1)  # exposed as (re, im) pairs
        out.append(arr)
    return out


# ── Circuit outputs ──────────────────────────────────────────────────────

def circuit_outputs(proof, elided: list) -> list[tuple[float, str]]:
    """Full output sequence: (value, kind) with kind in {"int", "float",
    "const"}; exposed publics in statement order, elided constants at their
    reported slots."""
    def statement_id(entry: str) -> int:
        return int(re.search(r"_(\d+)$", entry.partition("=")[0]).group(1))

    exposed = []
    for entry in sorted(proof.public_values, key=statement_id):
        label, _, raw = entry.partition("=")
        value = int(raw)
        exposed.append((value / Q32, "float") if label.startswith("public_f_") else (value, "int"))
    total = len(exposed) + len(elided)
    slots: list = [None] * total
    for e in elided:
        value = e["Int"] if "Int" in e else e["Float"]
        slots[e["slot"]] = (value, "const")
    it = iter(exposed)
    return [s if s is not None else next(it) for s in slots]


# ── Position-aware comparison ────────────────────────────────────────────

def close(actual: float, expected: float, is_float: bool) -> bool:
    if not is_float:
        return actual == expected
    return abs(actual - expected) <= FLOAT_ATOL + FLOAT_RTOL * abs(expected)


def padded_positions(shape: tuple, envelope: tuple) -> list[int]:
    """Row-major positions, in an `envelope`-shaped buffer, of the elements
    of a `shape`-shaped array stored in its prefix rectangle."""
    strides = [1] * len(envelope)
    for i in range(len(envelope) - 2, -1, -1):
        strides[i] = strides[i + 1] * envelope[i + 1]
    return [sum(c * s for c, s in zip(coord, strides))
            for coord in itertools.product(*[range(n) for n in shape])]


def envelope_shapes(shape: tuple, payload: int, static_axes: set[int]):
    """Envelope shapes with `prod == payload`, each axis >= the runtime size,
    and the axes whose size is a compile-time constant left unchanged."""
    def rec(axis, remaining):
        if axis == len(shape):
            if remaining == 1:
                yield ()
            return
        options = [shape[axis]] if axis in static_axes else range(max(shape[axis], 1), remaining + 1)
        for n in options:
            if n > 0 and remaining % n == 0:
                for rest in rec(axis + 1, remaining // n):
                    yield (n,) + rest
    yield from rec(0, payload)


def part_layouts(arr, outputs: list, pos: int, last: bool):
    """Every way the result part `arr` can be laid out at `outputs[pos:]`.
    Yields (layout description, consumed slot count, {slot: expected
    metadata value}, {slot: expected element value}). The last part must
    consume every remaining slot."""
    import numpy as np

    flat = [float(v) for v in np.asarray(arr, dtype=float).ravel()]
    n = len(flat)
    remaining = len(outputs) - pos
    if n <= remaining and (n == remaining or not last):
        yield "static", n, {}, {pos + i: v for i, v in enumerate(flat)}
    shape = tuple(int(d) for d in arr.shape)
    rank = len(shape)
    # Runtime-shaped part: [length] [rank] [dims that are not constants] payload.
    for has_length, has_rank in itertools.product((True, False), (False, True)):
        for dyn_axes in itertools.chain.from_iterable(
                itertools.combinations(range(rank), k) for k in range(rank + 1)):
            meta_values = (([n] if has_length else []) + ([rank] if has_rank else [])
                           + [shape[a] for a in dyn_axes])
            if not meta_values:
                continue  # indistinguishable from a static part
            meta = {pos + i: float(v) for i, v in enumerate(meta_values)}
            start = pos + len(meta_values)
            static_axes = set(range(rank)) - set(dyn_axes)
            desc = f"length={has_length},rank={has_rank},dims={list(dyn_axes)}"
            payloads = ([len(outputs) - start] if last
                        else range(max(n, 1), len(outputs) - start + 1))
            for payload in payloads:
                # The length is left out exactly when it is a constant equal
                # to the envelope size.
                if payload < n or (has_length == (payload == n) and not (has_length and dyn_axes)):
                    continue
                yield (f"runtime({desc},compact,payload={payload})", len(meta_values) + payload,
                       meta, {start + i: v for i, v in enumerate(flat)})
                for env in envelope_shapes(shape, payload, static_axes):
                    if env == shape:
                        continue
                    positions = padded_positions(shape, env)
                    yield (f"runtime({desc},padded,envelope={list(env)})",
                           len(meta_values) + payload, meta,
                           {start + p: v for p, v in zip(positions, flat)})


def match(parts: list, outputs: list, float_parts: list[bool], pos: int = 0, found=()):
    """First assignment of layouts to `parts` under which every expected
    value matches the output at its position and every slot is consumed.
    Returns a list of (layout, metadata slots, {element slot: value}), or
    None."""
    if not parts:
        return list(found) if pos == len(outputs) else None
    for desc, consumed, meta, data in part_layouts(parts[0], outputs, pos, len(parts) == 1):
        if pos + consumed > len(outputs):
            continue
        if not all(outputs[slot][0] == value for slot, value in meta.items()):
            continue
        if all(close(outputs[slot][0], value, float_parts[0]) for slot, value in data.items()):
            result = match(parts[1:], outputs, float_parts[1:], pos + consumed,
                           found + ((desc, meta, data),))
            if result is not None:
                return result
    return None


def holds(layouts, outputs: list, float_parts: list[bool]) -> bool:
    """`outputs` agree with the expected values under fixed `layouts`."""
    for (_, meta, data), is_float in zip(layouts, float_parts):
        if not all(outputs[slot][0] == value for slot, value in meta.items()):
            return False
        if not all(close(outputs[slot][0], value, is_float) for slot, value in data.items()):
            return False
    return True


def tampered_copies(outputs: list, element_slots: list[int]):
    """(name, tampered outputs) controls: one result element changed, and
    two different result elements swapped."""
    copies = []
    if element_slots:
        changed = list(outputs)
        value, kind = changed[element_slots[0]]
        changed[element_slots[0]] = (value + 1, kind)
        copies.append(("value_changed", changed))
    for i, j in itertools.combinations(element_slots, 2):
        a, b = outputs[i][0], outputs[j][0]
        # Values the float tolerance cannot tell apart would make a swap
        # undetectable by construction, not by a weak check.
        if abs(a - b) > 2 * (FLOAT_ATOL + FLOAT_RTOL * max(abs(a), abs(b))):
            swapped = list(outputs)
            swapped[i], swapped[j] = swapped[j], swapped[i]
            copies.append(("values_swapped", swapped))
            break
    return copies


# ── One case ─────────────────────────────────────────────────────────────

def check_case(case: str) -> dict:
    import numpy as np
    from run_halo2_baselines import INPUT_BUILDERS
    from run_zinnia_noir import _coerce_arg
    from zinnia.api.zk_circuit import ZKCircuit

    rec: dict = {"case": case, "status": "fail"}
    kernel_path = THIS_DIR / f"ds1000_case{case}" / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"check_case{case}", kernel_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    try:
        program = ZKCircuit.from_method(module.kernel).compile()
    except Exception as e:  # noqa: BLE001
        rec.update(status="compile_error", error=f"{type(e).__name__}: {str(e)[:300]}")
        return rec

    fixture = INPUT_BUILDERS[case]()

    def lookup(name: str):
        for key in (name, name.lower(), name.upper(), f"{name}_val", f"{name.lower()}_val"):
            if key in fixture:
                return fixture[key]
        raise KeyError(f"no fixture value for kernel parameter {name!r}")

    args = [_coerce_arg(lookup(p.name), p.export().get("dt")) for p in program.program_inputs]
    try:
        parts = result_parts(numpy_reference(case, args))
    except Exception as e:  # noqa: BLE001
        rec.update(status="numpy_error", error=f"{type(e).__name__}: {e}")
        return rec

    proof = program.prove(*args, backend="mock")
    if proof.proof_bytes_hex != "mock_satisfied":
        rec.update(status="unsatisfied", error=proof.proof_bytes_hex[:200])
        return rec
    outputs = circuit_outputs(proof, program.get_elided_publics())
    float_parts = [p.dtype.kind == "f" for p in parts]
    rec["outputs"] = len(outputs)
    rec["elided"] = sum(1 for _, kind in outputs if kind == "const")
    rec["expected_elements"] = int(sum(p.size for p in parts))

    status = "pass"
    layouts = match(parts, outputs, float_parts)
    if layouts is None:
        stable_parts = result_parts(numpy_reference(case, args, stable_sort=True))
        layouts = match(stable_parts, outputs, float_parts)
        if layouts is None:
            rec.update(status="mismatch",
                       expected_head=[float(v) for p in parts for v in np.asarray(p, float).ravel()][:12],
                       actual_head=[v for v, _ in outputs[:12]])
            return rec
        status = "pass_stable_ties"  # equal elements in stable order
    rec["layouts"] = [desc for desc, _, _ in layouts]
    element_slots = sorted(slot for _, _, data in layouts for slot in data)
    # Elements exposed as the other number kind (e.g. a float result as an
    # integer public): reported, not failed; the values matched.
    rec["kind_mismatches"] = sum(
        1 for (_, _, data), is_float in zip(layouts, float_parts) for slot in data
        if outputs[slot][1] != "const" and outputs[slot][1] != ("float" if is_float else "int"))

    # Controls: the layout is a property of the compiled circuit, so the
    # tampered outputs are judged under the layout the honest run matched.
    controls = {name: "ACCEPTED" if holds(layouts, tampered, float_parts) else "rejected"
                for name, tampered in tampered_copies(outputs, element_slots)}
    rec["controls"] = controls
    if not element_slots:
        status = "pass_empty_output"  # nothing to tamper with
    elif not controls or any(v != "rejected" for v in controls.values()):
        status = "weak_control"
    rec["status"] = status
    return rec


# ── Driver ───────────────────────────────────────────────────────────────

def run_isolated(case: str, timeout: float, memory: str) -> dict:
    cmd = [sys.executable, str(Path(__file__).resolve()), "--one", case]
    if MEMRUN.exists():
        cmd = [str(MEMRUN)] + cmd
    env = dict(os.environ, MEMRUN_MAX=memory, ZINNIA_SKIP_RELEASE_BUILD="1")
    started = time.monotonic()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, env=env)
    except subprocess.TimeoutExpired:
        return {"case": case, "status": "timeout", "seconds": timeout}
    line = next((l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None)
    rec = json.loads(line[7:]) if line else {
        "case": case, "status": "crashed",
        "error": (proc.stderr.strip().splitlines() or [f"exit {proc.returncode}"])[-1][:300]}
    rec["seconds"] = round(time.monotonic() - started, 1)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--cases", default="all", help="comma-separated case ids, or 'all'")
    ap.add_argument("--output", default=str(THIS_DIR / "check_outputs.json"))
    ap.add_argument("--timeout", type=float, default=600.0)
    ap.add_argument("--memory", default="8G", help="per-case memory cap (memrun.sh)")
    ap.add_argument("--one", help=argparse.SUPPRESS)
    args = ap.parse_args()

    sys.path.insert(0, str(THIS_DIR))
    sys.path.insert(0, str(ROOT))
    os.environ.setdefault("ZINNIA_SKIP_RELEASE_BUILD", "1")
    if args.one:
        print("RESULT " + json.dumps(check_case(args.one)), flush=True)
        return

    from run_halo2_baselines import INPUT_BUILDERS
    cases = sorted(INPUT_BUILDERS) if args.cases == "all" else [c.strip() for c in args.cases.split(",")]
    results = []
    for i, case in enumerate(cases, 1):
        rec = run_isolated(case, args.timeout, args.memory)
        results.append(rec)
        detail = rec.get("error") or ",".join(rec.get("layouts", [])) or ""
        print(f"[{i:3}/{len(cases)}] {case}: {rec['status']:13} {detail[:110]}", flush=True)
        Path(args.output).write_text(json.dumps(results, indent=1))

    counts: dict = {}
    for rec in results:
        counts[rec["status"]] = counts.get(rec["status"], 0) + 1
    print("\n" + ", ".join(f"{k}: {v}" for k, v in sorted(counts.items())))
    failing = [r["case"] for r in results if not r["status"].startswith("pass")]
    if failing:
        print("not passing:", " ".join(failing))
    sys.exit(1 if failing else 0)


if __name__ == "__main__":
    main()
