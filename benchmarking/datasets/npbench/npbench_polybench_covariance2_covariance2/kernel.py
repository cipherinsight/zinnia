# npbench program polybench/covariance2/covariance2_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
@requires(lambda M, float_n, data: 0 <= M <= 1024)
def kernel(M: Integer, float_n: Float, data: Float[Array, '600 500']):
    return np.cov(np.transpose(data))
