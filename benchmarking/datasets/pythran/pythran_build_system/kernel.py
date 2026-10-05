# pythran program build_system.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def thin_plate_spline(r, xp):
    return xp.where(r == 0, 0, r ** 2 * xp.log(r))

@zk_chip
def _kernel_matrix_impl(x, y, kernel_func, xp):
    """Evaluate RBFs, with centerx at `y`, at `x`."""
    return kernel_func(xp.linalg.norm(x[None, :, :] - y[:, None, :], axis=-1), xp)

@zk_circuit
def build_system(x, y, powers, xp):
    """Construct the coefficients needed to evaluate
    the RBF.

    Parameters
    ----------
    x : (Q, N) float ndarray
        Evaluation point coordinates.
    y : (P, N) float ndarray
        Data point coordinates.
    powers : (R, N) int ndarray
        The exponents for each monomial in the polynomial.
    xp

    Returns
    -------
    (Q, P + R) float ndarray

    """
    kernel_func = thin_plate_spline
    vec = xp.concatenate([_kernel_matrix_impl(y, x, kernel_func, xp), xp.prod(x[:, None, :] ** powers, axis=-1)], axis=-1)
    return vec
