# Source: NPBench polybench/covariance (covariance_numpy.py)
# Original signature: kernel(M, float_n, data) where data is (N, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - float_n is a data-valued scalar -> kept as float param.
from zinnia import *

M = 500
N = 600


@zk_circuit
def covariance(float_n: float, data: NDArray[Float, 600, 500]):
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((M, M), dtype=data.dtype)
    for i in range(M):
        cov[i:M, i] = cov[i, i:M] = data[:, i] @ data[:, i:M] / (float_n - 1.0)

    _zinnia_result = cov
