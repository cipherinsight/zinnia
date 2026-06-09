"""Common array operator coverage package B: np.linalg verification helpers."""

import pytest

from zinnia import *
from zinnia.api.zk_circuit import ZKCircuit


def test_np_linalg_verify_inverse_static_witness():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 0], [0, 1]])
        y = np.asarray([[1, 0], [0, 1]])
        inv = np.linalg.verify_inverse(a, y)
        assert inv[0, 0] == 1
        assert inv[1, 1] == 1

    assert foo()


def test_np_linalg_verify_solve_static_vector_and_matrix_rhs():
    @zk_circuit
    def foo():
        a = np.asarray([[1, 0], [0, 1]])
        x = np.asarray([3, 4])
        b = np.asarray([3, 4])
        y = np.linalg.verify_solve(a, x, b)
        assert y[1] == 4

        xm = np.asarray([[3, 4], [5, 6]])
        bm = np.asarray([[3, 4], [5, 6]])
        ym = np.linalg.verify_solve(a, xm, bm)
        assert ym[1, 0] == 5

    assert foo()


def test_np_linalg_verify_cholesky_static_witness_reconstruction_boundary():
    @zk_circuit
    def foo():
        a = np.asarray([[4, 2], [2, 5]])
        l = np.asarray([[2, 0], [1, 2]])
        out = np.linalg.verify_cholesky(a, l)
        assert out[1, 1] == 2

    assert foo()


def test_np_linalg_standard_inv_rejects_with_verify_helper_hint():
    source = """
def linalg_inv_deferred():
    a = np.asarray([[1, 0], [0, 1]])
    y = np.linalg.inv(a)
    _zinnia_result = y[0, 0]
"""

    with pytest.raises(Exception, match="verify_inverse/verify_solve/verify_cholesky"):
        ZKCircuit.from_source("linalg_inv_deferred", source).compile()


def test_np_linalg_verify_solve_rejects_shape_mismatch_before_matmul():
    source = """
def linalg_solve_bad_shape():
    a = np.asarray([[1, 0], [0, 1]])
    x = np.asarray([3, 4])
    b = np.asarray([3, 4, 5])
    y = np.linalg.verify_solve(a, x, b)
    _zinnia_result = y[0]
"""

    with pytest.raises(Exception, match="expected x and b to both be shape"):
        ZKCircuit.from_source("linalg_solve_bad_shape", source).compile()


def test_np_linalg_verify_inverse_rejects_non_square():
    source = """
def linalg_inverse_bad_shape():
    a = np.asarray([[1, 0, 0], [0, 1, 0]])
    y = np.asarray([[1, 0], [0, 1]])
    inv = np.linalg.verify_inverse(a, y)
    _zinnia_result = inv[0, 0]
"""

    with pytest.raises(Exception, match="A must be a square 2-D static array"):
        ZKCircuit.from_source("linalg_inverse_bad_shape", source).compile()


def test_np_linalg_verify_cholesky_rejects_complex_static_array():
    source = """
def linalg_cholesky_complex_deferred():
    a = np.asarray([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])
    l = np.asarray([[1 + 0j, 0 + 0j], [0 + 0j, 1 + 0j]])
    out = np.linalg.verify_cholesky(a, l)
    _zinnia_result = out[0, 0].real
"""

    with pytest.raises(Exception, match="Complex arrays are deferred"):
        ZKCircuit.from_source("linalg_cholesky_complex_deferred", source).compile()
