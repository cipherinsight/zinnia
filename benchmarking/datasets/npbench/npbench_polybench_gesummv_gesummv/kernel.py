# npbench program polybench/gesummv/gesummv_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, '16 16'], B: Float[Array, '16 16'], x: Float[Array, '16']):
    return alpha * A @ x + beta * B @ x
