"""Facts the compiler reasons with must mean what the circuit enforces (fact review).

Each case used to compile to a circuit whose facts and constraints disagreed:

- `np.divide` of two integer scalars lowered to `DivI`, exact division in the
  field: `np.divide(7, 2)` proved a huge integer instead of NumPy's `3.5`.
- A precondition's `/` of integers was enforced as that field division but
  stated to the SMT layer as integer division: `n / 2 >= 3` rejected the honest
  `n = 7` and accepted `n = 5`.
- A precondition's `//` and `%` were stated with SMT-LIB's Euclidean convention
  while the circuit enforces Python's floor convention. For a negative divisor
  the two differ; `n % -3 == -1` (true for `n = 2` in Python) became an
  unsatisfiable fact, from which `prove()` concluded anything, including that a
  later divisor `n - 2` is non-zero, so its check was dropped and `n = 2`
  divided by zero with a proof.
"""
import re

import numpy as np
import pytest

from zinnia import NDArray, Integer, requires, zk_circuit
from zinnia.api.zk_circuit import ZKCircuit


def prove(fn, *args):
    """(satisfied, publics in exposure order) on the mock backend."""
    proof = ZKCircuit.from_method(fn).compile().prove(*args, backend="mock")
    entries = sorted(proof.public_values, key=lambda e: int(re.search(r"_(\d+)=", e).group(1)))
    return proof.proof_bytes_hex == "mock_satisfied", [e.split("=")[0] for e in entries], \
        [int(e.split("=")[1]) for e in entries]


def ir_classes(fn):
    return [s["ir_instance"]["__class__"] for s in ZKCircuit.from_method(fn).compile().get_ir_stmts()]


Q32 = 2 ** 32


@zk_circuit
def divide_scalars(a: int, b: int):
    return np.divide(a, b)


@zk_circuit
def divide_arrays(a: NDArray[Integer, 3], b: NDArray[Integer, 3]):
    return np.divide(a, b)


def test_np_divide_of_integer_scalars_is_true_division():
    satisfied, names, values = prove(divide_scalars, 7, 2)
    assert satisfied
    assert names[0].startswith("public_f"), "np.divide of ints is a float"
    assert values == [round(np.divide(7, 2) * Q32)]
    assert "DivIIR" not in ir_classes(divide_scalars)


def test_np_divide_of_integer_scalars_by_zero_has_no_proof():
    satisfied, _, _ = prove(divide_scalars, 7, 0)
    assert not satisfied


def test_np_divide_of_integer_arrays_unchanged():
    satisfied, _, values = prove(divide_arrays, np.array([7, 9, 4]), np.array([2, 4, 3]))
    assert satisfied
    assert values == [round(v * Q32) for v in np.divide([7, 9, 4], [2, 4, 3])]


@zk_circuit
@requires(lambda n: n / 2 >= 3.0)
def half_at_least_three(n: int):
    return n + 1


@zk_circuit
@requires(lambda n: n / 2 >= 3)
def half_at_least_int_three(n: int):
    return n + 1


@pytest.mark.parametrize("n, holds", [(7, True), (6, True), (5, False), (-7, False)])
def test_precondition_true_division_is_enforced_as_python(n, holds):
    satisfied, _, values = prove(half_at_least_three, n)
    assert satisfied == holds
    if holds:
        assert values == [n + 1]


@pytest.mark.parametrize("n, holds", [(7, True), (5, False)])
def test_precondition_true_division_against_an_integer_literal(n, holds):
    # `n / 2` is a float; the integer literal is compared as a float.
    assert prove(half_at_least_int_three, n)[0] == holds


@zk_circuit
@requires(lambda n: n % -3 == -1)
def remainder_by_negative(n: int, x: int):
    return x // (n - 2)


@zk_circuit
@requires(lambda n: n // -2 == -1)
def floor_quotient_by_negative(n: int, x: int):
    return x // (n - 1)


def test_precondition_modulo_by_negative_follows_python():
    # 5 % -3 == -1 in Python: the precondition holds and the division is fine.
    satisfied, _, values = prove(remainder_by_negative, 5, 7)
    assert satisfied and values == [7 // 3]
    # 4 % -3 == -2: the precondition fails.
    assert not prove(remainder_by_negative, 4, 7)[0]


def test_divisor_check_is_kept_after_a_precondition_with_negative_modulus():
    # 2 % -3 == -1 holds, and 7 // (2 - 2) raises in Python: no proof.
    assert not prove(remainder_by_negative, 2, 7)[0]


def test_precondition_floor_division_by_negative_follows_python():
    # 2 // -2 == -1 and 1 // -2 == -1 in Python (Euclidean: -1 and 0).
    satisfied, _, values = prove(floor_quotient_by_negative, 2, 7)
    assert satisfied and values == [7 // 1]
    # n = 1: the precondition holds and 7 // (1 - 1) raises: no proof.
    assert not prove(floor_quotient_by_negative, 1, 7)[0]
    # n = 3: 3 // -2 == -2, the precondition fails.
    assert not prove(floor_quotient_by_negative, 3, 7)[0]


# A bounded-dynamic input's runtime extents are public inputs; the compiler
# assumes each lies in [0, its axis bound] (it bounds loops over them), so the
# circuit checks that where the extents are read. Only the product used to be
# checked: (7, 0) has product 0 <= 6, and a while loop over the first extent
# was cut after 6 iterations by that assumption.

from zinnia import DynamicNDArray  # noqa: E402


@zk_circuit
def count_rows(a: DynamicNDArray[Integer, 6, 2]):
    i = 0
    while i < a.shape[0]:
        i += 1
    return i


def _prove_with_runtime_shape(program, honest_arg, shape):
    import json
    from zinnia.compile._bridge import prove_circuit
    from zinnia.exec.input_parser import build_circuit_inputs
    from zinnia.exec.proof_result import ZKProofResult

    witness = build_circuit_inputs(program.program_inputs, (honest_arg,))

    def set_shape(node):
        if isinstance(node, dict):
            for key, value in node.items():
                if key == "runtime_shape":
                    node[key] = shape
                else:
                    set_shape(value)
        elif isinstance(node, list):
            for value in node:
                set_shape(value)

    set_shape(witness)
    result = ZKProofResult.from_json(prove_circuit(program._compiled_ir, json.dumps(witness), {}, "mock", None))
    return result.proof_bytes_hex == "mock_satisfied"


def _extent_checks(program):
    """The runtime-extent reads and, per read, the interval `[lo, hi]` the
    circuit's asserts bound it to. A bound is asserted either as two
    comparisons with constants (`x >= lo`, `x <= hi`) or, after the lowering
    merges them (`lower::checks::range_merges`), as one `AssertU128Lt(x, h)`,
    which enforces `0 <= x < h`."""
    stmts = {s["stmt_id"]: s for s in program.get_ir_stmts()}

    def constant(stmt_id):
        ir = stmts[stmt_id]["ir_instance"]
        return ir["ir_data"]["value"] if ir["__class__"] == "ConstantIntIR" else None

    shape_reads = {i for i, s in stmts.items() if s["ir_instance"]["__class__"] == "ReadIntegerIR"
                   and any("RuntimeShape" in seg for seg in s["ir_instance"]["ir_data"]["path"]["segments"])}
    lower = {read: None for read in shape_reads}
    upper = {read: None for read in shape_reads}

    def tighten(read, lo=None, hi=None):
        if lo is not None:
            lower[read] = lo if lower[read] is None else max(lower[read], lo)
        if hi is not None:
            upper[read] = hi if upper[read] is None else min(upper[read], hi)

    for s in stmts.values():
        kind = s["ir_instance"]["__class__"]
        if kind == "AssertU128LtIR":
            read, bound = s["arguments"]
            if read in shape_reads and constant(bound) is not None:
                tighten(read, lo=0, hi=constant(bound) - 1)
        elif kind == "AssertIR":
            cond = stmts[s["arguments"][0]]
            if len(cond["arguments"]) != 2 or cond["arguments"][0] not in shape_reads:
                continue
            read, bound = cond["arguments"][0], constant(cond["arguments"][1])
            if bound is None:
                continue
            if cond["ir_instance"]["__class__"] == "GreaterThanOrEqualIIR":
                tighten(read, lo=bound)
            elif cond["ir_instance"]["__class__"] == "LessThanOrEqualIIR":
                tighten(read, hi=bound)
    return shape_reads, {read: (lower[read], upper[read]) for read in shape_reads}


def test_dynamic_input_extents_are_checked_per_axis():
    program = ZKCircuit.from_method(count_rows).compile()
    shape_reads, bounds = _extent_checks(program)
    assert len(shape_reads) == 2
    for read in shape_reads:
        assert bounds[read] == (0, 6)
    honest = np.zeros((2, 3), dtype=int)
    assert _prove_with_runtime_shape(program, honest, [2, 3])
    assert _prove_with_runtime_shape(program, honest, [6, 1])
    assert not _prove_with_runtime_shape(program, honest, [7, 0])
    assert not _prove_with_runtime_shape(program, honest, [0, 7])


# `prove()` minted every value as a Z3 integer, floats included. A float is a
# point of the Q32 grid: as an integer, `x > -0.5` entailed `x >= 0`, so
# `sqrt`'s domain check was dropped and sqrt(-0.25) had a proof; and
# `-0.5 < x < 0.5` entailed `x == 0`, so `y / x` was rejected at compile time.

@zk_circuit
@requires(lambda x: x > -0.5)
def sqrt_above_minus_half(x: float) -> float:
    return np.sqrt(x)


@zk_circuit
@requires(lambda x: x > -0.5)
@requires(lambda x: x < 0.5)
def divide_by_small(x: float, y: float) -> float:
    return y / x


def test_float_precondition_does_not_imply_the_sqrt_domain():
    satisfied, _, values = prove(sqrt_above_minus_half, 0.25)
    assert satisfied and values == [round(0.5 * Q32)]
    assert not prove(sqrt_above_minus_half, -0.25)[0]


def test_float_precondition_does_not_force_a_divisor_to_zero():
    satisfied, _, values = prove(divide_by_small, 0.25, 1.0)
    assert satisfied and values == [round(4.0 * Q32)]
    assert not prove(divide_by_small, 0.0, 1.0)[0]


# The IR generator's SMT resolver encoded a float input as an integer symbol
# and `int(x)` as the identity on it, so under `2.5 <= x <= 3.5` it resolved
# `int(x)` to the constant 3: the circuit's output was the constant
# `arange(4).sum() == 6` for every x, and x = 2.7 (NumPy: 3) had a proof of 6.

@zk_circuit
@requires(lambda x: x >= 2.5)
@requires(lambda x: x <= 3.5)
def truncated_bound(x: float):
    n = int(x)
    return np.arange(n + 1).sum()


def test_float_preconditions_do_not_make_a_truncation_constant():
    # `int(x)` has no static value: the array size is rejected (no bound on a
    # float's truncation is derived), never compiled to a constant.
    with pytest.raises(Exception, match="compile-time constant"):
        ZKCircuit.from_method(truncated_bound).compile()
