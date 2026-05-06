# Source: NPBench polybench/cholesky2 (cholesky2_numpy.py)
# Original signature: kernel(A) where A is NxN float (symmetric positive definite).
# Migration notes:
#   - N hoisted as a module-level constant.
#   - Body uses np.linalg.cholesky / np.triu which are likely unsupported,
#     but recipe says copy verbatim (failures are interesting data).
from zinnia import *

N = 8


@zk_circuit
def cholesky2(A: NDArray[Float, 8, 8]):
    A[:] = np.linalg.cholesky(A) + np.triu(A, k=1)
