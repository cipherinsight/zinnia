"""Cheaper lowerings must compute NumPy's result (perf-ops, 2026-10-06).

Each section covers one lowering that replaced a costlier construction. The
tests compare outputs with NumPy position by position on the mock backend,
check that the circuit backends agree with the mock, and pin the IR shape
that makes the lowering cheaper (so a regression to the old construction
fails here, not only in a cost table).
"""
import numpy as np
import pytest

from testing.lang.mock_outputs import assert_matches_numpy, compile_kernel, numpy_kernel, run_mock

BN254_P = 0x30644E72E131A029B85045B68181585D2833E84879B9709143E1F593F0000001
HALO2_PARAMS = {"k": 12, "precision_bits": 32, "lookup_bits": 8}


def ir_classes(program):
    return [s["ir_instance"]["__class__"] for s in program.get_ir_stmts()]


def assert_backends_agree(source, *args):
    """The halo2-base proof's public values equal the mock's, field element
    by field element."""
    program = compile_kernel(source)
    mock = program.prove(*args, backend="mock")
    circuit = program.prove(*args, backend="halo2-base", params=HALO2_PARAMS)
    mock_values = sorted(e for e in mock.public_values)
    by_statement = {int(e.split("=")[0].rsplit("_", 1)[1]): int(e.split("=")[1]) % BN254_P
                    for e in mock_values}
    expected = [by_statement[k] for k in sorted(by_statement)]
    got = [int.from_bytes(bytes.fromhex(h), "little") for h in circuit.public_values]
    assert got == expected


# ── products over a zero-length axis ─────────────────────────────────────

@pytest.mark.parametrize("source, args", [
    ('def kernel(a: Float[Array, "3"], b: Float[Array, "3"]):\n'
     '    return np.dot(a[:0], b[:0])\n', (np.ones(3), np.ones(3))),
    ('def kernel(a: Integer[Array, "3"], b: Integer[Array, "3"]):\n'
     '    return np.dot(a[:0], b[:0]) + 1\n', (np.ones(3, dtype=int), np.ones(3, dtype=int))),
    ('def kernel(a: Float[Array, "2 3"], b: Float[Array, "3 2"]):\n'
     '    return np.dot(a[:, :0], b[:0, :])\n', (np.ones((2, 3)), np.ones((3, 2)))),
    ('def kernel(a: Float[Array, "3"], b: Float[Array, "3 2"]):\n'
     '    return np.dot(a[:0], b[:0])\n', (np.ones(3), np.ones((3, 2)))),
    # NPBench cholesky: the first inner step is a dot product of empty rows.
    ('def kernel(A: Float[Array, "3 3"]):\n'
     '    for i in range(1, 3):\n'
     '        for j in range(i):\n'
     '            A[i, j] -= np.dot(A[i, :j], A[j, :j])\n'
     '    return A\n', (np.arange(9.0).reshape(3, 3),)),
])
def test_product_over_empty_axis_matches_numpy(source, args):
    assert_matches_numpy(source, *args)
