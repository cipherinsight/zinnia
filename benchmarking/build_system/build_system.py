# Source: Pythran tests/cases/build_system.py
# Original #pythran export: build_system(float[:,:], float[:,:], int[:,:], numpy pkg)
# Migration notes: 'numpy pkg' parameter dropped; uses module np directly.
from zinnia import *

Q = 8
P = 8
R = 4
N = 4


def thin_plate_spline(r, xp):
    return xp.where(r == 0, 0, r ** 2 * xp.log(r))


def _kernel_matrix_impl(x, y, kernel_func, xp):
    return kernel_func(
        xp.linalg.norm(x[None, :, :] - y[:, None, :], axis=-1), xp
    )


@zk_circuit
def build_system(x: NDArray[Float, 8, 4], y: NDArray[Float, 8, 4], powers: NDArray[Integer, 4, 4], xp):
    kernel_func = thin_plate_spline

    vec = xp.concatenate(
        [
            _kernel_matrix_impl(y, x, kernel_func, xp),
            xp.prod(x[:, None, :] ** powers, axis=-1)
        ], axis=-1
    )

    _zinnia_result = vec
