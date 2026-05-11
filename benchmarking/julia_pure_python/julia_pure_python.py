# Source: Pythran tests/cases/julia_pure_python.py
# Original #pythran export: compute_julia(float, float, int, float?, float?, int?)
from zinnia import *
from time import time


def kernel(zr, zi, cr, ci, lim, cutoff):
    count = 0
    while ((zr * zr + zi * zi) < (lim * lim)) and count < cutoff:
        zr, zi = zr * zr - zi * zi + cr, 2 * zr * zi + ci
        count += 1
    return count


@zk_circuit
def compute_julia(cr: float, ci: float, N: int, bound: float = 1.5, lim: float = 1000., cutoff: int = 1000000):
    julia = np.empty((N, N), np.uint32)
    grid_x = np.linspace(-bound, bound, N)
    t0 = time()
    for i, x in enumerate(grid_x):
        for j, y in enumerate(grid_x):
            julia[i, j] = kernel(x, y, cr, ci, lim, cutoff)
    _zinnia_result = julia, time() - t0
