"""Zero-default compaction and early scalar identities preserve public values."""
import numpy as np
import pytest

from zinnia import ZKCircuit
from testing.lang.mock_outputs import run_mock_layout, mock_publics


@pytest.mark.parametrize("mask", [[False] * 4, [True] * 4, [False, True, False, True]])
def test_complex_compaction_shares_network_and_keeps_exact_lanes(mask):
    source = '''def kernel(x: Complex[Array, "4"], mask: Boolean[Array, "4"]):
    return x[mask]
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    x = (np.arange(4) - 3.25).astype(complex)
    x.imag = np.arange(4) / 8 - 0.5
    mask = np.array(mask)
    satisfied, values, shapes = run_mock_layout(program, x, mask)
    assert satisfied
    expected = x[mask]
    assert shapes == [list(expected.shape)]
    np.testing.assert_array_equal(values, np.stack([expected.real, expected.imag], axis=-1).ravel())
    # Products of one-hot bits and payload codes, never rescaled float products.
    ops = [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]
    assert "MulIIR" in ops
    assert "MulFIR" not in ops


@pytest.mark.parametrize("x", [-(2**63), -19, 0, 23, 2**63 - 1])
def test_integer_division_by_one_and_zero_mask(x):
    program = ZKCircuit.from_source("kernel", '''def kernel(x: Integer, expected: Integer):
    y = x // 1
    z = x if x != 0 else 0
    assert y == expected
    assert z == expected
    return y, z
''').compile()
    proof = program.prove(x, x, backend="mock")
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert mock_publics(proof) == [x, x]
    assert program.prove(x, x + (1 if x < 2**63 - 1 else -1), backend="mock").proof_bytes_hex != "mock_satisfied"
    assert not any("FloorDiv" in s["ir_instance"]["__class__"] for s in program.get_ir_stmts())


def test_inferred_column_dimension_shares_public_count():
    program = ZKCircuit.from_source("kernel", '''def kernel(x: Integer[Array, "5"]):
    return x[x != 0].reshape(-1, 1)
''').compile()
    layout = program.get_output_layout()[0]["dynamic"]
    assert layout["length"] == layout["dims"][0]
    for x in [[0] * 5, [-3, 0, 2, 0, -1], [1, 2, 3, 4, 5]]:
        satisfied, values, shapes = run_mock_layout(program, np.array(x))
        assert satisfied
        expected = np.array(x)[np.array(x) != 0].reshape(-1, 1)
        assert shapes == [list(expected.shape)]
        np.testing.assert_array_equal(values, expected.ravel())


@pytest.mark.parametrize("imag", [[0, 0, 0, 0], [1, -2, 3, -4], [0, -2, 0, 4]])
def test_complex_nonzero_compaction_padding_and_wrong_claim(imag):
    source = '''def kernel(x: Complex[Array, "4"], expected: Float):
    y = x[x.imag != 0]
    assert np.sum(y.imag) == expected
    return y
'''
    program = ZKCircuit.from_source("kernel", source).compile()
    x = np.array([1.25, -2.5, 3.75, -4.0], dtype=complex)
    x.imag = imag
    expected = x[x.imag != 0]
    total = float(expected.imag.sum())
    satisfied, values, shapes = run_mock_layout(program, x, total)
    assert satisfied and shapes == [list(expected.shape)]
    np.testing.assert_array_equal(values, np.stack([expected.real, expected.imag], axis=-1).ravel())
    assert program.prove(x, total + 1.0, backend="mock").proof_bytes_hex != "mock_satisfied"
    # The public buffer retains exact zero padding after the logical payload.
    proof = program.prove(x, total, backend="mock")
    publics = mock_publics(proof)
    assert publics[1 + 2 * len(expected):] == [0] * (2 * (4 - len(expected)))


@pytest.mark.parametrize("dtype", ["Integer", "Float"])
@pytest.mark.parametrize("expression", ["x if x != 0 else 0", "(x != 0) * x"])
def test_scalar_zero_mask_is_an_identity(dtype, expression):
    source = f'def kernel(x: {dtype}):\n    return {expression}\n'
    program = ZKCircuit.from_source("kernel", source).compile()
    ops = [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]
    assert not set(ops) & {"SelectIIR", "SelectFIR", "MulIIR", "MulFIR"}
    for x in [-3, 0, 5]:
        arg = float(x) if dtype == "Float" else x
        satisfied, values, shapes = run_mock_layout(program, arg)
        assert satisfied and values == [arg] and shapes == [[]]


@pytest.mark.parametrize("x", [2**62, -(2**62) - 1])
def test_division_identity_retains_quotient_window(x):
    program = ZKCircuit.from_source("kernel", '''def kernel(x: Integer):
    return (x + x) // 1
''').compile()
    assert program.prove(x, backend="mock").proof_bytes_hex != "mock_satisfied"
    satisfied, values, _ = run_mock_layout(program, x // 2)
    assert satisfied and values == [2 * (x // 2)]
