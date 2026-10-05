"""Every input element is checked to its dtype's domain.

A circuit input is a field element of the prover's choice. Each element is
now constrained when it is read: integers to the signed 64-bit window, floats
to the Q32 window (codes in the same signed 64-bit window, i.e. values in
[-2^31, 2^31)), booleans to {0, 1}. So
a proof means "some valid NumPy input gives these outputs". The malicious
witnesses below put a raw field element (the witness format's `Hex` node) in
an input cell; before the check they were accepted (the kernels only add and
compare), now the mock and halo2-base reject them.
"""
import json

import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel
from zinnia._zinnia_core import halo2_base_prove, halo2_base_verify
from zinnia.compile._bridge import prove_circuit
from zinnia.exec.input_parser import build_circuit_inputs

P = 21888242871839275222246405745257275088548364400416034343698204186575808495617
HALO2 = {"k": 12, "lookup_bits": 8, "precision_bits": 32}
I64_MIN, I64_MAX = -(2**63), 2**63 - 1
FLOAT_MAX = 2.0**31 - 2.0**-22  # the largest double below 2^31
# The witness JSON parser reads FLOAT_MAX one ULP high (as 2^31, outside the
# window; serde_json without `float_roundtrip`, a core handoff), so the honest
# extreme used here is the next double down.
FLOAT_HIGH = 2.0**31 - 2 * 2.0**-22

SOURCE = '''
def kernel(a: NDArray[int, 2], f: Float, b: Boolean, m: NDArray[bool, 2]):
    return a[0] + a[1], f + 1.0, b, m
'''


def field_hex(n: int) -> dict:
    """A raw field element as a witness leaf."""
    return {"Hex": (n % P).to_bytes(32, "little").hex()}


def witness(program, args, patch=None):
    w = build_circuit_inputs(program.program_inputs, args)
    if patch:
        patch(w["params"])
    return json.dumps(w)


def mock_satisfied(program, witness_json) -> bool:
    out = json.loads(prove_circuit(program._compiled_ir, witness_json, {}, "mock", None))
    return out["proof_bytes"] == "mock_satisfied"


def halo2_accepts(program, keys, witness_json) -> bool:
    try:
        proof = json.loads(halo2_base_prove(program._compiled_ir, keys, witness_json, {}))
    except (RuntimeError, ValueError):
        return False
    return json.loads(halo2_base_verify(keys, proof["proof"], proof["public_values"]))["valid"]


HONEST = (np.array([I64_MIN, I64_MAX]), FLOAT_HIGH, True, np.array([False, True]))


def set_array_element(index, node):
    return lambda params: params[0]["value"]["Array"]["elements"].__setitem__(index, node)


def set_scalar(position, node):
    return lambda params: params[position].__setitem__("value", node)


MALICIOUS = {
    "integer 2^63": set_array_element(1, field_hex(2**63)),
    "integer -2^63 - 1": set_array_element(0, field_hex(-(2**63) - 1)),
    "integer 2^100": set_array_element(1, field_hex(2**100)),
    "float code 2^63": set_scalar(1, field_hex(2**63)),
    "float code -2^63 - 1": set_scalar(1, field_hex(-(2**63) - 1)),
    "boolean 2": set_scalar(2, field_hex(2)),
    "boolean -1": set_scalar(2, field_hex(-1)),
    "boolean array element 2": lambda params: params[3]["value"]["Array"]["elements"]
    .__setitem__(0, field_hex(2)),
}


def test_extreme_valid_inputs_are_accepted():
    program = compile_kernel(SOURCE)
    assert mock_satisfied(program, witness(program, HONEST))
    assert mock_satisfied(program, witness(program, (np.array([0, -5]), -(2.0**31), False,
                                                     np.array([True, True]))))


@pytest.mark.xfail(strict=True, reason="the witness JSON parser reads the largest double "
                   "below 2^31 as 2^31 (serde_json without float_roundtrip)")
def test_the_largest_float_input_is_accepted():
    program = compile_kernel("def kernel(f: Float):\n    return f\n")
    assert program.prove(FLOAT_MAX, backend="mock").proof_bytes_hex == "mock_satisfied"


@pytest.mark.parametrize("name", sorted(MALICIOUS))
def test_out_of_domain_inputs_are_rejected_by_the_mock(name):
    program = compile_kernel(SOURCE)
    assert not mock_satisfied(program, witness(program, HONEST, MALICIOUS[name]))


def test_out_of_domain_inputs_are_rejected_by_halo2_base():
    program = compile_kernel(SOURCE)
    keys = program.setup(params=HALO2)
    assert halo2_accepts(program, keys, witness(program, HONEST))
    for name, patch in MALICIOUS.items():
        assert not halo2_accepts(program, keys, witness(program, HONEST, patch)), name


def test_every_input_element_gets_one_check():
    program = compile_kernel(SOURCE)
    kinds = [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]
    # two integers + one float (its Q32 code): a lower and an upper
    # comparison each; three booleans: one AssertBit each.
    assert kinds.count("GreaterThanOrEqualIIR") == 3
    assert kinds.count("LessThanOrEqualIIR") == 3
    assert kinds.count("AssertBitIR") == 3


def test_inputs_nothing_reads_get_no_check():
    """An unused parameter is not part of the proven statement (any valid value
    gives the same outputs); its read and check are removed, as the read alone
    was before (case 0402 declares a 36,864-cell input it never reads)."""
    program = compile_kernel('''
def kernel(unused: NDArray[float, 64], used: NDArray[int, 2]):
    return used[0] + 1
''')
    kinds = [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]
    assert kinds.count("ReadFloatIR") == 0 and kinds.count("ReadIntegerIR") == 1
    assert kinds.count("GreaterThanOrEqualIIR") == 1 and kinds.count("LessThanOrEqualIIR") == 1
