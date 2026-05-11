# Source: NPBench polybench/durbin (durbin_numpy.py)
# Original signature: kernel(r) where r is (N,) float.
# Migration notes:
#   - N hoisted as module-level constant.
#   - r.shape[0] replaced with N for static loop bound.
from zinnia import *

N = 1000


@zk_circuit
def durbin(r: NDArray[Float, 1000]):
    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]

    for k in range(1, N):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha

    _zinnia_result = y
