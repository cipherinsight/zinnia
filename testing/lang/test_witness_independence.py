"""Witness-independence regression test (provenance plan, Step 0a).

A compiled circuit's STRUCTURE must not depend on the prover's witness:
compile once, then render the circuit for two different (both valid)
witnesses — the Noir `main.nr` must be byte-identical and the halo2-base
cell statistics identical. Any drift means an emission decision consumed
witness data, the soundness-bug class fixed in commit 71e3048 (the
emitter's value_cache folded witness values into structural decisions and
specialized the circuit to the prover's inputs).

The five kernels cover the historically-bitten lowering families:
  0433 — dynamic row/col zeroing (select zero-path)
  0391 — dynamic 2-D slice (affine gathers)
  0415 — runtime reshape + reduction
  0442 — boolean-mask compaction
  0345 — negative/derived dynamic index writes
"""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

# The runtime-shape DS-1000 kernels (every DS-1000 input a runtime parameter).
# The performance-set kernels in datasets/ds1000/ bind shape parameters such as
# 0391's low/high and 0415's bin_size as compile-time constants, so they no
# longer exercise the runtime-shape lowering this test is about.
BENCH_DIR = Path(__file__).resolve().parents[2] / "benchmarking" / "datasets" / "ds1000_numpy" / "generated"

# Matches the benchmark runners' proving params (precision matters: the
# emitted fixed-point helpers encode precision_bits in the source text).
NOIR_PARAMS = {"k": 0, "precision_bits": 32, "lookup_bits": 8}


def _compile_case(case: str):
    from zinnia.api.zk_circuit import ZKCircuit

    kernel_path = BENCH_DIR / f"case{case}" / "kernel.py"
    spec = importlib.util.spec_from_file_location(f"wit_indep_case{case}", kernel_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return ZKCircuit.from_method(mod.kernel).compile()


def _floats_2d(rows, cols, salt):
    return np.asarray(
        [[salt + 0.1 * (i * cols + j + 1) for j in range(cols)] for i in range(rows)]
    )


# case -> (witness_a, witness_b); both must satisfy the kernel's contract,
# and they deliberately differ in every input INCLUDING the dynamic
# indices/shape scalars (that's what a specialized circuit leaks).
CASES = {
    "0433": (
        (np.arange(1, 17).reshape(4, 4), 1, 0),
        (np.arange(101, 117).reshape(4, 4), 2, 3),
    ),
    "0391": (
        (_floats_2d(20, 10, 10.0), 3, 7),
        (_floats_2d(20, 10, 55.0), 2, 9),
    ),
    "0415": (
        (np.asarray([1.0, 2.0, 4.0, 3.0, 6.0, 5.0, 7.0, 8.0]), 2),
        (np.asarray([8.0, 1.0, 5.0, 2.0, 9.0, 4.0, 3.0, 7.0]), 3),
    ),
    "0442": (
        (np.arange(20), np.asarray([3, 7, 13])),
        (np.asarray([5, 5, 1, 0, 2, 9, 9, 9, 4, 6, 8, 11, 13, 17, 19, 2, 3, 3, 0, 1]),
         np.asarray([0, 4, 19])),
    ),
    "0345": (
        (_floats_2d(8, 6, 0.0), 3, 2.0),
        (_floats_2d(8, 6, 7.0), 6, 5.0),
    ),
}


@pytest.fixture(scope="module")
def compiled_cases():
    return {case: _compile_case(case) for case in CASES}


@pytest.mark.parametrize("case", sorted(CASES))
def test_noir_emission_is_witness_independent(case, compiled_cases):
    compiled = compiled_cases[case]
    wit_a, wit_b = CASES[case]
    src_a = compiled.emit_structural(*wit_a, backend="noir", params=NOIR_PARAMS)
    src_b = compiled.emit_structural(*wit_b, backend="noir", params=NOIR_PARAMS)
    if src_a != src_b:
        diff = [
            f"  line {i}: {la!r} != {lb!r}"
            for i, (la, lb) in enumerate(zip(src_a.splitlines(), src_b.splitlines()))
            if la != lb
        ]
        pytest.fail(
            f"case {case}: emitted main.nr differs between witnesses "
            f"(witness-specialized circuit). First diffs:\n"
            + "\n".join(diff[:10])
            + (f"\n  (lengths: {len(src_a)} vs {len(src_b)} chars)")
        )


@pytest.mark.parametrize("case", sorted(CASES))
def test_halo2_base_cells_are_witness_independent(case, compiled_cases):
    compiled = compiled_cases[case]
    wit_a, wit_b = CASES[case]
    # Three syntheses: two concrete witnesses (Some(a), Some(b)) and the
    # witness-LESS keygen (None — exactly what the verifier re-runs, all
    # reads default to zero). A sound, witness-independent circuit yields
    # ONE fingerprint for all three; any drift between Some(a)/Some(b) is a
    # prove-time leak, and a drift to None is a verify-breaking leak.
    fp_a = compiled.emit_structural(*wit_a, backend="halo2-base")
    fp_b = compiled.emit_structural(*wit_b, backend="halo2-base")
    fp_none = compiled.emit_structural(*wit_a, backend="halo2-base-witnessless")
    assert fp_a == fp_b == fp_none, (
        f"case {case}: halo2-base cell statistics differ across "
        f"Some(a)/Some(b)/None (witness-specialized circuit):\n"
        f"  Some(a): {fp_a}\n  Some(b): {fp_b}\n  None:    {fp_none}"
    )


def test_halo2_base_is_bit_is_witness_independent():
    """Regression for code review C03: halo2-base `is_bit` must not decide
    gate emission from the witness sample's value.

    `is_bit` used to return true whenever a cell's runtime value was 0 or 1,
    so a `select`/logical condition that happened to be 0/1 for one witness
    skipped the booleanization gadget while a non-0/1 witness emitted it —
    witness-dependent gate counts, and a verify-breaker (prove-side keygen
    runs with the real witness, verify-side re-keygen with `None`, all zeros).

    Compile a circuit whose select condition is the *value* of a private
    input (so it is 5 for one witness, 0 for another, 1 for a third) and
    assert the keygen structural fingerprint is identical for all three.
    """
    from zinnia import Public, Integer
    from zinnia.api.zk_circuit import ZKCircuit

    def kernel(v: Public[Integer]) -> Integer:
        # `if v:` booleanizes the raw value v; under the old is_bit a 0/1
        # witness skipped the truthy_bit gadget, a non-0/1 witness emitted it.
        if v:
            return v * 2
        return v + 100

    compiled = ZKCircuit.from_method(kernel).compile()
    fp_nonbool = compiled.emit_structural(5, backend="halo2-base")
    fp_zero = compiled.emit_structural(0, backend="halo2-base")
    fp_one = compiled.emit_structural(1, backend="halo2-base")
    # The witness-less (verifier re-keygen) shape must also match: this is
    # the Some(a) vs Some(b) vs None assertion R06 calls for.
    fp_none = compiled.emit_structural(5, backend="halo2-base-witnessless")
    assert fp_nonbool == fp_zero == fp_one == fp_none, (
        "halo2-base is_bit leaked witness data into gate emission:\n"
        f"  cond=5: {fp_nonbool}\n  cond=0: {fp_zero}\n  cond=1: {fp_one}\n"
        f"  None:   {fp_none}"
    )
