"""Exact square symmetry and constant diagonals, without a real prover."""
import numpy as np
import pytest

from testing.lang.mock_outputs import compile_kernel, run_mock_layout
from testing.lang.all_backends import noir_field_publics, nargo_available, mock_field_publics


@pytest.mark.parametrize("rows,columns", [(2, 1), (3, 2), (5, 4)])
@pytest.mark.parametrize("upper", [False, True])
def test_norm_square_symmetry_and_constant_output_layout(rows, columns, upper):
    expr = "np.linalg.norm(a - a[:, None], axis=-1)"
    if upper:
        expr = f"np.triu({expr})"
    source = f'def kernel(a: Float[Array, "{rows} {columns}"]):\n    return {expr}\n'
    program = compile_kernel(source)
    sums = [s for s in program.get_ir_stmts()
            if s["ir_instance"]["__class__"] == "ProductSumFIR"]
    assert len(sums) == rows * (rows - 1) // 2
    a = (np.arange(rows * columns).reshape(rows, columns) - rows) / 4.0
    a[-1] = a[0]  # zero differences as well as both orientations
    satisfied, values, shapes = run_mock_layout(program, a)
    expected = np.linalg.norm(a - a[:, None], axis=-1)
    if upper:
        expected = np.triu(expected)
    assert satisfied
    assert shapes == [[rows, rows]]
    np.testing.assert_allclose(np.array(values).reshape(expected.shape), expected, atol=1e-8, rtol=0)
    assert len(program.get_elided_publics()) == (rows * (rows + 1) // 2 if upper else rows)
    # The same optimized program can execute on Noir without generating keys.
    if nargo_available():
        assert noir_field_publics(program, a) == mock_field_publics(program, a)


@pytest.mark.parametrize("precision", [8, 16, 32])
def test_zero_norm_is_constant_at_every_precision(precision):
    program = compile_kernel('def kernel(a: Float[Array, "4"]):\n    return np.linalg.norm(a - a)\n')
    proof = program.prove(np.array([-2.0, 0.0, 1.25, 3.0]), backend="mock",
                          params={"k": 12, "precision_bits": precision, "lookup_bits": 8})
    assert proof.proof_bytes_hex == "mock_satisfied"
    assert program.get_elided_publics() == [{"slot": 0, "Float": 0.0}]
