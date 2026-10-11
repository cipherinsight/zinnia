# npbench program polybench/gemver/gemver_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, '16 16'], u1: Float[Array, '16'], v1: Float[Array, '16'], u2: Float[Array, '16'], v2: Float[Array, '16'], w: Float[Array, '16'], x: Float[Array, '16'], y: Float[Array, '16'], z: Float[Array, '16']):
    A += np.outer(u1, v1) + np.outer(u2, v2)
    x += beta * y @ A + z
    w += alpha * A @ x
