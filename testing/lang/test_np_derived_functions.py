"""NumPy functions computed from other array operations or cell by cell:
np.trace, np.diagonal, np.ptp, np.count_nonzero, np.average, np.ravel,
np.ediff1d, np.kron, np.vdot, np.cross, np.tri, np.vander, np.polyval,
np.trapezoid, np.heaviside and np.linalg.det.

Each kernel's outputs equal NumPy's on the mock, and halo2-base and Noir prove
the mock's values (``assert_all_backends_match_numpy``). Calls NumPy rejects
are compile errors.
"""
import numpy as np
import pytest

from testing.lang.all_backends import assert_all_backends_match_numpy
from testing.lang.mock_outputs import compile_kernel

INTS = np.array([[3, -1, 0, 2], [0, 5, -4, 1], [2, 2, 0, -3]], dtype=np.int64)
FLOATS = np.array([[1.5, -2.25, 0.0, 3.0], [0.5, 4.0, -1.0, 2.5], [-3.5, 1.25, 2.0, -0.75]])
VECTOR = np.array([1.5, -2.0, 0.0, 3.25, -0.5, 1.0])
SIGNATURE = 'a: Integer[Array, "3 4"], f: Float[Array, "3 4"], v: Float[Array, "6"]'


def check(body: str, atol: float = 1e-6):
    """``body`` returns from ``kernel(a, f, v)``: an integer and a float
    matrix and a float vector, each with negative values and zeros."""
    source = f"def kernel({SIGNATURE}):\n" + "".join(f"    {line}\n" for line in body.splitlines())
    assert_all_backends_match_numpy(source, INTS, FLOATS, VECTOR, atol=atol)


def rejected(body: str):
    source = f"def kernel({SIGNATURE}):\n    return {body}\n"
    with pytest.raises(Exception):
        compile_kernel(source)


@pytest.mark.parametrize("body", [
    "return np.trace(a), np.trace(a, 1), np.trace(a, -2), np.trace(a, 9)",
    "return np.trace(f), np.trace(f, axis1=1, axis2=0), np.trace(a > 0), np.trace(a, dtype=float)",
    "return np.trace(np.arange(24).reshape(2, 3, 4), -1, 1, 2)",
])
def test_trace(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.diagonal(a), np.diagonal(a, 2), np.diagonal(a, -1), np.diagonal(a, -5)",
    "return np.diagonal(f.T), np.diagonal(a > 0)",
    "return np.diagonal(np.arange(24).reshape(2, 3, 4), 1, 2, 0)",
])
def test_diagonal(body):
    check(body)


def test_diagonal_is_read_only():
    source = f"def kernel({SIGNATURE}):\n    d = np.diagonal(a)\n    d[0] = 7\n    return d\n"
    with pytest.raises(Exception):
        compile_kernel(source)


@pytest.mark.parametrize("body", [
    "return np.ptp(a), np.ptp(a, axis=0), np.ptp(a, 1), np.ptp(a, axis=1, keepdims=True)",
    "return np.ptp(f), np.ptp(f, axis=0), np.ptp(v[:1])",
    "return np.ptp(v[v > 0])",
])
def test_ptp(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.count_nonzero(a), np.count_nonzero(a, axis=0), np.count_nonzero(a, axis=(0, 1))",
    "return np.count_nonzero(f, axis=1, keepdims=True), np.count_nonzero(a > 0), np.count_nonzero(v[:0])",
    "return np.count_nonzero(v[v > 0]), np.count_nonzero(3), np.count_nonzero(0)",
])
def test_count_nonzero(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.average(a), np.average(f, axis=0), np.average(a, axis=1, keepdims=True)",
    "return np.average(a, weights=a + 5), np.average(f, axis=1, weights=np.array([1.0, 2.0, 0.5, 0.25]))",
    "return np.average(a, axis=0, weights=np.array([1, 2, 3]), returned=True)",
    "return np.average(v, returned=True), np.average(a, axis=1, returned=True), np.average(v[:1])",
    "return np.average(v[v > 0])",
])
def test_average(body):
    check(body)


def test_average_with_zero_weight_sum_has_no_proof():
    source = 'def kernel(x: Float[Array, "3"], w: Float[Array, "3"]):\n    return np.average(x, weights=w)\n'
    program = compile_kernel(source)
    weights = np.array([1.0, -2.0, 1.0])
    assert program.prove(np.array([1.0, 2.0, 3.0]), weights, backend="mock").proof_bytes_hex != "mock_satisfied"


@pytest.mark.parametrize("body", [
    "return np.ravel(a), np.ravel(f.T), np.ravel(a[:, ::2]), np.ravel(5)",
    "return np.ravel(v[v > 0])",
])
def test_ravel(body):
    check(body)


def test_ravel_is_a_view_of_a_contiguous_array():
    check("r = np.ravel(a)\nr[0] = 9\nreturn a")


@pytest.mark.parametrize("body", [
    "return np.ediff1d(a), np.ediff1d(f), np.ediff1d(v[:1]), np.ediff1d(v[:0])",
    "return np.ediff1d(a, to_begin=[7, 8], to_end=9), np.ediff1d(v, to_begin=np.array([1, 2]))",
    "return np.ediff1d(v[:0], to_end=[1.0])",
])
def test_ediff1d(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.kron(a, a[0]), np.kron(a[0], a), np.kron(a, 2)",
    "return np.kron(f[:2, :2], a[:2]), np.kron(v[:0], v), np.kron(v[:1], v[:1])",
])
def test_kron(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.vdot(a, a), np.vdot(a, f), np.vdot(v, v), np.vdot(v[:0], v[:0]), np.vdot(v[:1], v[2:3])",
])
def test_vdot(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.cross(a[:, :3], a[:, 1:]), np.cross(f[:, :3], v[:3])",
    "return np.cross(a[:, :2], a[0, :3]), np.cross(v[:3], a[:, 2:])",
    "return np.cross(a[:2, :2], a[1:, :2]), np.cross(v[:2], v[2:4])",
])
def test_cross(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.tri(4), np.tri(3, 5, 1, dtype=int), np.tri(3, k=-1), np.tri(3, 4, -5)",
    "return np.tri(0), np.tri(2, 0), np.tri(1), np.tri(2, dtype=bool)",
])
def test_tri(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.vander(v[:4]), np.vander(v[:3], 4, increasing=True), np.vander(a[0])",
    "return np.vander(v[:1]), np.vander(v[:3], 0), np.vander(v[:3], 1), np.vander(v[:0])",
])
def test_vander(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.polyval(a[0], v), np.polyval(v[:3], a), np.polyval([2, -1, 3], 2)",
    "return np.polyval(np.array([]), v), np.polyval([4], v), np.polyval(f[0], f[1, 0])",
    "return np.polyval(np.array([1.0, 2.0]), v[v > 0])",
])
def test_polyval(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.trapezoid(v), np.trapezoid(a), np.trapezoid(a, axis=0), np.trapezoid(f, dx=0.5)",
    "return np.trapezoid(a, dx=2), np.trapezoid(f, f + 0.5), np.trapezoid(f, np.array([0.0, 0.5, 1.5, 3.0]))",
    "return np.trapezoid(v[:1]), np.trapezoid(v[:0]), np.trapezoid(a, v[:3], axis=0)",
])
def test_trapezoid(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.heaviside(f, 0.5), np.heaviside(a, f), np.heaviside(v, a[0, 0])",
    "return np.heaviside(a > 0, a < 0), np.heaviside(f[0, 2], f[1, 2]), np.heaviside(-v[:1], 3)",
])
def test_heaviside(body):
    check(body)


@pytest.mark.parametrize("body", [
    "return np.linalg.det(f[:, :3]), np.linalg.det(a[:, 1:]), np.linalg.det(a[:2, :2] > 0)",
    "return np.linalg.det(np.eye(4)), np.linalg.det(np.ones((2, 1, 1))), np.linalg.det(np.zeros((0, 0)))",
    "return np.linalg.det(np.stack([f[:, :3], f[:, 1:]])), np.linalg.det(a[:1, :1])",
])
def test_linalg_det(body):
    check(body)


def test_linalg_det_of_a_4x4_float_matrix():
    source = 'def kernel(m: Float[Array, "4 4"]):\n    return np.linalg.det(m)\n'
    m = np.array([[2.5, -1.0, 0.5, 3.0], [1.25, 4.0, -2.0, 0.0], [0.0, 1.5, 3.5, -1.0], [2.0, 0.25, -0.5, 1.0]])
    assert_all_backends_match_numpy(source, m, atol=1e-6)


@pytest.mark.parametrize("body", [
    "np.ptp(a > 0)",                       # NumPy: subtract on bools raises
    "np.ediff1d(a, to_begin=[0.5])",       # float to_begin for an int array
    "np.ediff1d(a > 0)",
    "np.cross(a, a)",                      # vectors of 4 components
    "np.cross(a[:, :3], a[:2, :3])",       # vector stacks do not broadcast
    "np.cross(a[:, :3], a[:, :3], axisa=0)",
    "np.vdot(a, v)",                       # 12 and 6 elements
    "np.diagonal(v)",
    "np.diagonal(a, axis1=1, axis2=-1)",
    "np.trace(v)",
    "np.vander(a)",
    "np.polyval(a, v)",                    # 2-D coefficients
    "np.trapezoid(a, v)",                  # 6 sample points for rows of 4
    "np.linalg.det(a)",                    # not square
    "np.tri(-1)",
    "np.average(a, weights=v)",            # weights of another shape, no axis
    "np.average(a, axis=1, weights=v[:3])",
    "np.ravel(a, order='F')",
    "np.count_nonzero(a, 0, True)",        # keepdims is keyword-only
])
def test_numpy_rejections_are_compile_errors(body):
    rejected(body)
