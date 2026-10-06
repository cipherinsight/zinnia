"""The output layout: for every public value, which output element it is.

`ZKCompiledProgram.get_output_layout()` lists the output parts (tuple items,
arrays, scalars) with, for each cell, its public-vector index or the constant
the compiler left out of the public vector; runtime-shaped results carry their
runtime length, rank and dims. Decoding a proof's public vector with it must
give NumPy's result element by element (values and shapes), and a proof whose
public outputs were changed must be rejected.
"""
import json

import numpy as np
import pytest

from benchmarking.harness.outputs import decode_with_layout
from testing.lang.mock_outputs import compile_kernel, mock_publics, numpy_outputs, run_mock_layout
from zinnia import ZKCircuit
from zinnia.api.zk_compiled_program import ZKCompiledProgram

BN254 = 21888242871839275222246405745257275088548364400416034343698204186575808495617
HALO2 = {"k": 12, "lookup_bits": 8, "precision_bits": 32}


def assert_layout_matches_numpy(source, *args, atol=1e-6):
    satisfied, values, shapes = run_mock_layout(compile_kernel(source), *args)
    expected, expected_shapes = numpy_outputs(source, *args)
    assert satisfied, "honest witness rejected"
    assert shapes == expected_shapes
    np.testing.assert_allclose(np.asarray(values, float), np.asarray(expected, float), atol=atol)
    return values


MIXED = '''
def kernel(a: NDArray[float, 2, 3], n: Integer):
    total = a.sum()
    return a * 2.0, (n + 1, 7), [total, 1.5], np.zeros((2, 2), dtype=int)
'''


def test_tuple_parts_scalars_constants_and_lists_decode_to_numpy():
    a = np.array([[1.0, -2.5, 3.0], [0.5, 4.0, -1.0]])
    assert_layout_matches_numpy(MIXED, a, 4)
    layout = compile_kernel(MIXED).get_output_layout()
    assert [part["output"] for part in layout] == [{"return": [i]} for i in range(4)]
    assert [part["shape"] for part in layout] == [[2, 3], [2], [2], [2, 2]]
    # Constants (the 7, the 1.5 and every zero) are not in the public vector.
    assert layout[1]["cells"][1] == {"constant": 7}
    assert all(cell == {"constant": 0} for cell in layout[3]["cells"])
    publics = [cell["public"] for part in layout for cell in part["cells"] if "public" in cell]
    assert publics == list(range(8))


def test_runtime_shaped_results_decode_at_their_runtime_shape():
    filtered = '''
def kernel(x: NDArray[int, 6]):
    return x[x > 2]
'''
    for x in ([1, 5, 3, 0, 9, 2], [0, 0, 0, 0, 0, 0], [3, 4, 5, 6, 7, 8]):
        assert_layout_matches_numpy(filtered, np.array(x))
    part = compile_kernel(filtered).get_output_layout()[0]
    assert "dynamic" in part and len(part["dynamic"]["cells"]) == 6

    sliced = '''
def kernel(a: NDArray[float, 4, 5], lo: Integer, hi: Integer):
    return a[:, lo:hi]
'''
    a = np.arange(20, dtype=float).reshape(4, 5)
    for lo, hi in ((1, 3), (0, 5), (2, 2)):
        assert_layout_matches_numpy(sliced, a, lo, hi)


def test_complex_results_have_two_cells_per_element():
    source = '''
def kernel(a: NDArray[float, 3], b: NDArray[float, 3]):
    return (a + 1j * b) * 2
'''
    assert_layout_matches_numpy(source, np.array([1.0, 0.0, 3.0]), np.array([2.0, -0.5, 0.0]))
    assert compile_kernel(source).get_output_layout()[0]["complex"] is True


def test_public_inputs_are_listed_and_skipped_by_the_decode():
    source = '''
def kernel(x: Public[Integer], a: NDArray[int, 2]):
    return a + x
'''
    program = compile_kernel(source)
    assert program.get_public_inputs() == [{"public": 0, "input": "x", "kind": "i"}]
    assert [c["public"] for c in program.get_output_layout()[0]["cells"]] == [1, 2]
    satisfied, values, shapes = run_mock_layout(program, 5, np.array([1, 2]))
    assert satisfied and values == [6, 7] and shapes == [[2]]
    described = program.describe_publics()
    assert described[0] == {"input": "x"}
    assert described[2] == {"output": {"return": []}, "element": 1}


def halo2_outputs(program, keys, proof):
    fields = [int.from_bytes(bytes.fromhex(v), "little") for v in proof["public_values"]]
    return decode_with_layout(fields, program.get_output_layout(), program.get_public_inputs())


def test_halo2_public_vector_decodes_and_tampered_outputs_are_rejected():
    program = ZKCircuit.from_source("kernel", MIXED).compile()
    keys = program.setup(params=HALO2)
    a = np.array([[1.0, -2.5, 3.0], [0.5, 4.0, -1.0]])
    proof = program.prove_with_keys(keys, a, 4)
    assert program.verify_with_keys(keys, proof)
    values, shapes = halo2_outputs(program, keys, proof)
    expected, expected_shapes = numpy_outputs(MIXED, a, 4)
    assert shapes == expected_shapes
    np.testing.assert_allclose(values, expected, atol=1e-6)
    # Claiming a different value for any exposed output element fails.
    for index in (0, 6, 7):
        tampered = dict(proof, public_values=list(proof["public_values"]))
        changed = (int.from_bytes(bytes.fromhex(tampered["public_values"][index]), "little") + 1) % BN254
        tampered["public_values"][index] = changed.to_bytes(32, "little").hex()
        assert not program.verify_with_keys(keys, tampered)


def test_every_public_value_is_described_exactly_once():
    program = compile_kernel(MIXED)
    described = program.describe_publics()
    assert len(described) == len(mock_publics(program.prove(np.ones((2, 3)), 1, backend="mock")))
    assert described[6] == {"output": {"return": [1]}, "element": 0}
    with pytest.raises(ValueError, match="exactly once"):
        decode_with_layout(list(range(9)), program.get_output_layout())


def test_layout_survives_serialization():
    program = compile_kernel(MIXED)
    restored = ZKCompiledProgram.deserialize(program.serialize())
    assert restored.get_output_layout() == program.get_output_layout()
    assert restored.get_public_inputs() == program.get_public_inputs()
    assert json.loads(program.serialize())["public_layout"]["outputs"] == program.get_output_layout()
