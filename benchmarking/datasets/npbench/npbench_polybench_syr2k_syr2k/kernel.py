# npbench program polybench/syr2k/syr2k_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, C: Float[Array, '13 13'], A: Float[Array, '13 9'], B: Float[Array, '13 9']):
    for i in range(A.shape[0]):
        C[i, :i + 1] *= beta
        for k in range(A.shape[1]):
            C[i, :i + 1] += A[:i + 1, k] * alpha * B[i, k] + B[:i + 1, k] * alpha * A[i, k]
