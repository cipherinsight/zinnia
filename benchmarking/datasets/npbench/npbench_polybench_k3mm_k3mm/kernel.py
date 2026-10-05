# npbench program polybench/k3mm/k3mm_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(A: Float[Array, '13 15'], B: Float[Array, '15 14'], C: Float[Array, '14 16'], D: Float[Array, '16 16']):
    return A @ B @ C @ D
