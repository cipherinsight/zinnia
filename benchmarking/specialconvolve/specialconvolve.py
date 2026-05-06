# Source: Pythran tests/cases/specialconvolve.py
# Original #pythran export: specialconvolve(uint32[][])
from zinnia import *

M = 16
N = 16


@zk_circuit
def specialconvolve(a: NDArray[Integer, 16, 16]):
    rowconvol = a[1:-1, :] + a[:-2, :] + a[2:, :]
    colconvol = rowconvol[:, 1:-1] + rowconvol[:, :-2] + rowconvol[:, 2:] - 9 * a[1:-1, 1:-1]
    _zinnia_result = colconvol
