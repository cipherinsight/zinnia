# npbench program polybench/k2mm/k2mm_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(alpha: Float, beta: Float, A: Float[Array, '800 900'], B: Float[Array, '900 850'], C: Float[Array, '850 950'], D: Float[Array, '800 950']):
    D[:] = alpha * A @ B @ C + beta * D
