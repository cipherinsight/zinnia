# npbench program polybench/gemver/gemver_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, '1000 1000'], u1: Float[Array, '1000'], v1: Float[Array, '1000'], u2: Float[Array, '1000'], v2: Float[Array, '1000'], w: Float[Array, '1000'], x: Float[Array, '1000'], y: Float[Array, '1000'], z: Float[Array, '1000']):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
