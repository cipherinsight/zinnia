from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: domain-obligation, linalg
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Record square/invertible/SPD requirements where the linalg call needs them.
# Diagnostic: np.linalg kernels may stay fail-closed until support is selected.
import numpy as np

@zk_chip
def thin_plate_spline(r: Integer, xp: Float):
    return xp.where(r == 0, 0, r ** 2 * xp.log(r))

@zk_chip
def _kernel_matrix_impl(x: Float[Array, 'N<=1024 M<=1024 K<=1024'], y: Float[Array, 'N<=1024 M<=1024 K<=1024'], kernel_func: Float, xp: Float):
    """Evaluate RBFs, with centerx at `y`, at `x`."""
    return kernel_func(xp.linalg.norm(x[None, :, :] - y[:, None, :], axis=-1), xp)

@zk_circuit
def build_system(x: Float[Array, 'N<=1024 M<=1024 K<=1024'], y: Float, powers: Float, xp: Float):
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
