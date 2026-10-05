# npbench program polybench/gemm/gemm_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, C: Float[Array, '14 15'], A: Float[Array, '14 16'], B: Float[Array, '16 15']):
    C[:] = alpha * A @ B + beta * C
