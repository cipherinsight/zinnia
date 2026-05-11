# Source: NPBench polybench/correlation (correlation_numpy.py)
# Original signature: kernel(M, float_n, data) where data is (N, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - float_n is a data-valued scalar -> kept as float param.
#   - Boolean fancy assignment (stddev[stddev <= 0.1] = 1.0) likely unsupported.
from zinnia import *

M = 500
N = 600


@zk_circuit
def correlation(float_n: float, data: NDArray[Float, 600, 500]):
    mean = np.mean(data, axis=0)
    stddev = np.std(data, axis=0)
    stddev[stddev <= 0.1] = 1.0
    data -= mean
    data /= np.sqrt(float_n) * stddev
    corr = np.eye(M, dtype=data.dtype)
    for i in range(M - 1):
        corr[i + 1:M, i] = corr[i, i + 1:M] = data[:, i] @ data[:, i + 1:M]

    _zinnia_result = corr
