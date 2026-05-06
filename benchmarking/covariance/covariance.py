# Source: NPBench polybench/covariance (covariance_numpy.py)
# Original signature: kernel(M, float_n, data) where data is (N, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - float_n is a data-valued scalar -> kept as float param.
from zinnia import *

M = 8
N = 8


@zk_circuit
def covariance(float_n: float, data: NDArray[Float, 8, 8]):
    mean = np.mean(data, axis=0)
    data -= mean
    cov = np.zeros((8, 8), dtype=data.dtype)
    for i in range(8):
        cov[i:8, i] = cov[i, i:8] = data[:, i] @ data[:, i:8] / (float_n - 1.0)

    _zinnia_result = cov
