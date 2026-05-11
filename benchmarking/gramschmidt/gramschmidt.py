# Source: NPBench polybench/gramschmidt (gramschmidt_numpy.py)
# Original signature: kernel(A) where A is (M, N) float.
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - A.shape[0]/A.shape[1] replaced with M / N for static loop bounds.
from zinnia import *

M = 70
N = 60


@zk_circuit
def gramschmidt(A: NDArray[Float, 70, 60]):
    Q = np.zeros_like(A)
    R = np.zeros((N, N), dtype=A.dtype)

    for k in range(N):
        nrm = np.dot(A[:, k], A[:, k])
        R[k, k] = np.sqrt(nrm)
        Q[:, k] = A[:, k] / R[k, k]
        for j in range(k + 1, N):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] -= Q[:, k] * R[k, j]

    _zinnia_result = Q, R
