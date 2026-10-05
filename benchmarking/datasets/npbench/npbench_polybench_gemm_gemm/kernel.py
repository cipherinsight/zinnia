# npbench program polybench/gemm/gemm_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, C: Float[Array, '1000 1100'], A: Float[Array, '1000 1200'], B: Float[Array, '1200 1100']):
    C[:] = alpha * A @ B + beta * C
