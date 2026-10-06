"""NumPy dtype conversions: `astype`, `np.<type>(x)`, `np.array(x, dtype=)`, `dtype=`.

Before, every conversion into `bool` or a fixed-width integer type proved the
unconverted values (`x.astype(np.uint8)` of `[0, 2, -1, 300]` proved
`[0, 2, -1, 300]`, NumPy gives `[0, 2, 255, 44]`). Now they follow NumPy:
`bool` is `x != 0`, an integer type of `w` bits wraps modulo `2^w` (two's
complement when signed), a float is truncated toward zero first, and `uint64`
of a negative value has no proof.

Each conversion is checked as a claim the circuit must accept with NumPy's
result and reject with a wrong one, on the mock, on halo2-base (a proof that
verifies) and, for bool, uint8, int8 and uint64, on Noir. Fixed-width values then cannot
take part in integer arithmetic or be assigned into, because NumPy computes
those in the type and Zinnia in int64: such programs are rejected.
"""
import json

import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel
from testing.lang.test_cost_lowerings import HALO2_PARAMS, assert_backends_agree, ir_classes
from zinnia.debug.exception.base import ZinniaException

NOIR_PARAMS = {"k": 0, "precision_bits": 32, "lookup_bits": 8}

INTEGER_TYPES = ["np.uint8", "np.int8", "np.uint16", "np.int16", "np.uint32", "np.int32", "np.int64"]
# Values that wrap in every narrow type (and fit int64).
X_INT = np.array([0, 2, -1, 300, -129, 70000, -(2**40) - 3, 2**33 + 7])
X_FLOAT = np.array([0.5, -2.5, 300.7, 0.0, 255.9, -128.5, 70000.25, -0.75])


def numpy_type(spelling: str):
    return {"bool": bool, "int": int, "float": float}.get(spelling) or getattr(np, spelling[3:])


def converted(x, spelling):
    """NumPy's conversion, as int64 values (the claim a kernel checks)."""
    return x.astype(numpy_type(spelling)).astype(np.int64)


def check_source(spelling, element="Integer", n=8):
    return (f'def kernel(x: {element}[Array, "{n}"], expected: Integer[Array, "{n}"]):\n'
            f'    assert np.all(x.astype({spelling}) == expected)\n')


def mock_accepts(program, *args) -> bool:
    return program.prove(*args, backend="mock").proof_bytes_hex == "mock_satisfied"


def halo2_accepts(program, *args) -> bool:
    try:
        proof = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    except Exception:  # noqa: BLE001 - an unsatisfiable witness raises at proving
        return False
    return program.verify(proof)


def noir_accepts(program, *args) -> bool:
    """`nargo execute` of the emitted circuit: it fails on any unsatisfied constraint."""
    from zinnia._zinnia_core import execute_noir_circuit
    from zinnia.exec.input_parser import build_circuit_inputs

    witness = build_circuit_inputs(program.program_inputs, args)
    try:
        execute_noir_circuit(program._compiled_ir, json.dumps(witness), {}, json.dumps(NOIR_PARAMS))
    except Exception:  # noqa: BLE001 - nargo reports the failed assertion
        return False
    return True


def wrong_claim(x, good):
    """The unconverted values (what the circuit proved before), or for a
    conversion that changes nothing, one element off by one."""
    wrong = np.trunc(x).astype(np.int64)
    if np.array_equal(wrong, good):
        wrong = good.copy()
        wrong[1] += 1
    return wrong


# ── each dtype: NumPy's result accepted, a wrong one rejected ────────────

@pytest.mark.parametrize("spelling", ["bool"] + INTEGER_TYPES)
@pytest.mark.parametrize("element, x", [("Integer", X_INT), ("Float", X_FLOAT)])
def test_conversion_proves_numpys_result_on_mock_and_halo2(spelling, element, x):
    program = compile_kernel(check_source(spelling, element))
    good = converted(x, spelling)
    wrong = wrong_claim(x, good)
    assert mock_accepts(program, x, good)
    assert not mock_accepts(program, x, wrong)
    assert halo2_accepts(program, x, good)
    assert not halo2_accepts(program, x, wrong)


@pytest.mark.parametrize("spelling", ["bool", "np.uint8", "np.int8", "np.uint64"])
@pytest.mark.parametrize("element, x", [("Integer", X_INT), ("Float", X_FLOAT)])
def test_conversion_proves_numpys_result_on_noir(spelling, element, x):
    if spelling == "np.uint64":
        x = np.abs(x)
    program = compile_kernel(check_source(spelling, element))
    good = converted(x, spelling)
    assert noir_accepts(program, x, good)
    assert not noir_accepts(program, x, wrong_claim(x, good))


@pytest.mark.parametrize("spelling", ["bool", "np.uint8"])
def test_noir_proof_of_a_conversion_verifies(spelling):
    program = compile_kernel(check_source(spelling))
    proof = program.prove(X_INT, converted(X_INT, spelling), backend="noir", params=NOIR_PARAMS)
    assert program.verify(proof)


@pytest.mark.parametrize("spelling", ["bool", "np.uint8", "np.int8", "np.int16", "np.uint32", "int"])
def test_converted_outputs_equal_numpy_and_halo2_publics_equal_the_mock(spelling):
    source = f'def kernel(x: Integer[Array, "8"]):\n    return x.astype({spelling})\n'
    assert_matches_numpy(source, X_INT)
    assert_backends_agree(source, X_INT)


def test_uint64_of_a_negative_value_has_no_proof():
    # NumPy gives 2^64 + x, which Zinnia's int64 values cannot hold.
    program = compile_kernel(check_source("np.uint64", n=3))
    good = np.array([0, 5, 2**40])
    assert mock_accepts(program, good, good)
    assert halo2_accepts(program, good, good)
    negative = np.array([0, -5, 2**40])
    assert not mock_accepts(program, negative, negative)
    assert not halo2_accepts(program, negative, negative)
    assert noir_accepts(program, good, good)
    assert not noir_accepts(program, negative, negative)


def test_float_to_bool_and_complex_to_bool():
    assert_matches_numpy('def kernel(x: Float[Array, "4"]):\n    return x.astype(bool)\n',
                         np.array([1e-6, -0.0, 3.5, -2.0]))
    assert_matches_numpy('def kernel(x: Float):\n    return np.array([x, 0.0, 2j]).astype(bool)\n', 0.0)


# ── every way of naming a dtype ──────────────────────────────────────────

@pytest.mark.parametrize("expression", [
    "np.array(x, dtype=np.uint8)", "np.asarray(x, dtype='int16')", "np.array(x, np.int8)",
    "x.astype('u1')", "x.astype(dtype=np.uint16)", "np.array(x, dtype=bool)", "x.astype(np.bool_)",
    "np.int32(x)", "np.uint8(x)", "np.bool_(x)",
])
def test_dtype_spellings_convert(expression):
    assert_matches_numpy(f'def kernel(x: Integer[Array, "8"]):\n    return {expression}\n', X_INT)


@pytest.mark.parametrize("expression", ["np.uint8(x)", "np.int8(x) == -56", "np.bool_(x) + 1", "np.int16(x) > 0"])
def test_numpy_scalar_types_convert_scalars(expression):
    source = f'def kernel(x: Integer):\n    return {expression}\n'
    for x in (200, -129, 0):
        assert_matches_numpy(source, x)


@pytest.mark.parametrize("expression", [
    "np.zeros(4, dtype=bool)", "~np.zeros(4, dtype=bool)", "~np.ones((2, 2), dtype=bool)",
    "np.full(3, 3.7, dtype=np.int8)", "np.full(3, True)", "~np.full(3, False)",
    "np.arange(250, 260, dtype=np.uint8)", "np.linspace(0, 300, 4, dtype=np.uint8)",
    "np.zeros(3, dtype=np.int32) == 0", "np.eye(3, dtype=bool)", "np.zeros_like(x, dtype=np.uint8)",
    "np.ones(4, dtype='uint16')",
])
def test_constructor_dtypes(expression):
    assert_matches_numpy(f'def kernel(x: Integer[Array, "4"]):\n    return {expression}\n',
                         np.array([3, 0, -2, 9]))


@pytest.mark.parametrize("expression", [
    "y.dtype == np.uint8", "y.dtype == np.int8", "y.dtype == 'uint8'", "np.zeros(2, dtype=y.dtype)",
])
def test_dtype_attribute_of_converted_arrays(expression):
    source = f'def kernel(x: Integer[Array, "4"]):\n    y = x.astype(np.uint8)\n    return {expression}\n'
    assert_matches_numpy(source, np.array([3, 0, -2, 9]))


# ── no work when the value already fits ──────────────────────────────────

@pytest.mark.parametrize("expression", [
    "np.clip(x, 0, 255).astype(np.uint8)", "(x > 0).astype(np.int8)", "np.clip(x, -128, 127).astype(np.int8)",
    "np.maximum(x, 0).astype(np.uint64)", "np.minimum(np.maximum(x, -5), 5).astype(np.int8)",
])
def test_values_in_range_convert_without_a_remainder(expression):
    # The facts, the interval analysis and the min/max structure of the
    # operand prove it in the type's range: no constraint is added.
    source = f'def kernel(x: Integer[Array, "8"]):\n    return {expression}\n'
    classes = ir_classes(compile_kernel(source))
    assert "ModIIR" not in classes and "AssertNonNegIIR" not in classes
    assert_matches_numpy(source, X_INT)


def test_values_out_of_range_use_one_remainder_per_element():
    source = 'def kernel(x: Integer[Array, "8"]):\n    return x.astype(np.uint8)\n'
    assert ir_classes(compile_kernel(source)).count("ModIIR") == 8


# ── what a fixed-width value can and cannot do afterwards ────────────────

X4 = np.array([0, 2, -1, 300])


@pytest.mark.parametrize("body", [
    "return y[1:3]", "return y.reshape(2, 2).T", "return np.sort(y)", "return np.max(y), np.argmax(y)",
    "return np.where(y > 100, y, y[0])", "return y == 44, y < 3", "return y.astype(int) + 1",
    "return int(y[3]) * 10", "return y * 0.5", "return np.concatenate([y, y])", "return y.astype(np.int8)",
    "return x[y[1]]", "z = y.copy()\n    z[0] = y[3]\n    return z", "return np.where(x > 0, y, x)",
])
def test_fixed_width_values_compare_select_and_convert(body):
    assert_matches_numpy(f'def kernel(x: Integer[Array, "4"]):\n    y = x.astype(np.uint8)\n    {body}\n', X4)


ARITHMETIC = r"on a value of (NumPy type uint8|a NumPy fixed-width integer type): NumPy computes it"
ASSIGNMENT = r"stored into an array of NumPy type (uint8|int32)"


@pytest.mark.parametrize("body, message", [
    ("return y + 1", ARITHMETIC),
    ("return y * y", ARITHMETIC),
    ("return y.sum()", ARITHMETIC),
    ("return -y", ARITHMETIC),
    ("return ~y", ARITHMETIC),
    ("y += 1\n    return y", ARITHMETIC),
    ("s = 0\n    for v in y:\n        s = s + v\n    return s", ARITHMETIC),
    ("z = list(y)\n    return z[0] + z[1]", ARITHMETIC),
    ("z = np.concatenate([y, x])\n    return z + 1", ARITHMETIC),
    ("y[0] = 300\n    return y", ASSIGNMENT),
    ("y[k] = x[3]\n    return y", ASSIGNMENT),
    ("y[1:3] = x[2:4]\n    return y", ASSIGNMENT),
    ("y[y > 1] = 300\n    return y", ASSIGNMENT),
    ("z = y.reshape(2, 2)\n    z[k, 0] = 300\n    return y", ASSIGNMENT),
    ("np.copyto(y, x)\n    return y", ASSIGNMENT),
    ("np.add(x, x, out=y)\n    return y", ASSIGNMENT),
])
def test_arithmetic_and_assignment_on_fixed_width_values_are_rejected(body, message):
    # NumPy computes these in uint8 (wrapping) and converts assigned values
    # into uint8; Zinnia would compute them in int64.
    source = f'def kernel(x: Integer[Array, "4"], k: Integer):\n    y = x.astype(np.uint8)\n    {body}\n'
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(source)


@pytest.mark.parametrize("source", [
    'def kernel(x: Integer):\n    return np.uint8(x) * 2\n',
    'def kernel(x: Integer[Array, "4"]):\n    return np.zeros(4, dtype=np.int8) - 1\n',
    'def kernel(x: Integer[Array, "4"]):\n    y = np.zeros(4, dtype=np.int32)\n'
    '    for i in range(4):\n        y[i] = x[i] * 2\n    return y\n',
])
def test_fixed_width_scalars_and_constructors_are_tracked_too(source):
    with pytest.raises(ZinniaException, match=f"{ARITHMETIC}|{ASSIGNMENT}|NumPy type int8"):
        compile_kernel(source)


def test_sum_with_a_bool_dtype_is_rejected():
    with pytest.raises(ZinniaException, match=r"dtype=bool\) is not supported"):
        compile_kernel('def kernel(x: Integer[Array, "4"]):\n    return np.sum(x, dtype=bool)\n')


def test_eye_with_a_rectangular_shape_or_offset_is_rejected():
    # np.eye(N, M, k) used to return the N x N identity whatever M and k were.
    for call in ["np.eye(3, 4)", "np.eye(3, k=1)"]:
        with pytest.raises(ZinniaException, match=r"only the square identity"):
            compile_kernel(f'def kernel(x: Integer):\n    return {call}\n')


# ── arithmetic on bool arrays ────────────────────────────────────────────

BOOL_EXPRESSIONS = [
    # `+` of two bool arrays is a logical or (it was an integer sum: the
    # first one proved 5, NumPy gives 3).
    ("Float", np.array([3.0, 1.0, 2.0]), "((x > 1.5) + (x > 0.5)).sum()"),
    ("Integer", np.array([0, 2, -1, 300]), "(x > 0) + (x > 1)"),
    ("Integer", np.array([0, 2, -1, 300]), "np.add(x > 0, x < 1)"),
    ("Integer", np.array([0, 2, -1, 300]), "(x > 0) + True"),
    ("Integer", np.array([0, 2, -1, 300]), "x[(x > 1) + (x < 0)].sum()"),
    # `~` of a bool array is the logical not, also when its cells are
    # integers (a product of two bool arrays; a write into a bool array).
    ("Integer", np.array([0, 2, -1, 300]), "~((x > 0) * (x > 1))"),
    ("Integer", np.array([0, 2, -1, 300]), "~((x > 0) + (x > 1))"),
]


@pytest.mark.parametrize("element, x, expression", BOOL_EXPRESSIONS)
def test_bool_array_arithmetic_follows_numpy(element, x, expression):
    n = len(x)
    good = np.asarray(eval(expression, {"np": np, "x": x}))
    shape = '"' + " ".join(map(str, good.shape)) + '"' if good.shape else None
    claim = f"Integer[Array, {shape}]" if shape else "Integer"
    source = (f'def kernel(x: {element}[Array, "{n}"], expected: {claim}):\n'
              f'    assert np.all(({expression}) == expected)\n')
    program = compile_kernel(source)
    good = good.astype(np.int64) if good.shape else int(good)
    wrong = good + 1
    assert mock_accepts(program, x, good)
    assert not mock_accepts(program, x, wrong)
    assert halo2_accepts(program, x, good)
    assert not halo2_accepts(program, x, wrong)
    assert noir_accepts(program, x, good)
    assert not noir_accepts(program, x, wrong)


def test_bool_array_writes_keep_the_bool_dtype():
    source = ('def kernel(x: Integer[Array, "4"]):\n'
              '    m = np.zeros(4, dtype=bool)\n'
              '    m[0] = True\n'
              '    m[1] = x[1] > 0\n'
              '    return ~m\n')
    assert_matches_numpy(source, np.array([0, 2, -1, 300]))


@pytest.mark.parametrize("expression, message", [
    ("(x > 0) - (x > 1)", "numpy boolean subtract"),
    ("-(x > 0)", "numpy boolean negative"),
])
def test_bool_array_subtraction_and_negation_are_numpys_type_error(expression, message):
    with pytest.raises(ZinniaException, match=message):
        compile_kernel(f'def kernel(x: Integer[Array, "4"]):\n    return {expression}\n')
