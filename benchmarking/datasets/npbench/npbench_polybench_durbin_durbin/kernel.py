# npbench program polybench/durbin/durbin_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(r: Float[Array, '16']):
    y = np.empty_like(r)
    alpha = -r[0]
    beta = 1.0
    y[0] = -r[0]
    for k in range(1, r.shape[0]):
        beta *= 1.0 - alpha * alpha
        alpha = -(r[k] + np.dot(np.flip(r[:k]), y[:k])) / beta
        y[:k] += alpha * np.flip(y[:k])
        y[k] = alpha
    return y
