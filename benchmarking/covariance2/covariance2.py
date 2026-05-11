# Source: NPBench polybench/covariance2 (covariance2_numpy.py)
# Original signature: kernel(M, float_n, data) where data is (N, M).
# Migration notes:
#   - M, N hoisted as module-level shape constants.
#   - float_n kept as float param.
#   - Body uses np.cov which is likely unsupported but kept verbatim.
from zinnia import *

M = 500
N = 600


@zk_circuit
def covariance2(float_n: float, data: NDArray[Float, 600, 500]):
    _zinnia_result = np.cov(np.transpose(data))
