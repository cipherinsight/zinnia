# Source: Pythran tests/cases/wdist.py
# Original #pythran export: slow_wdist(float64[][], float64[][], float64[][])
from zinnia import *

K = 4
M = 8
N = 8


@zk_circuit
def slow_wdist(A: NDArray[Float, 4, 8], B: NDArray[Float, 4, 8], W: NDArray[Float, 4, 8]):
    k, m = A.shape
    _, n = B.shape
    D = np.zeros((m, n))

    for ii in range(m):
        for jj in range(n):
            wdiff = (A[:, ii] - B[:, jj]) / W[:, ii]
            D[ii, jj] = np.sqrt((wdiff ** 2).sum())
    _zinnia_result = D
