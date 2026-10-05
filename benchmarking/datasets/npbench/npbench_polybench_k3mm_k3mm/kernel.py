# npbench program polybench/k3mm/k3mm_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(A: Float[Array, '800 900'], B: Float[Array, '900 850'], C: Float[Array, '850 1000'], D: Float[Array, '1000 950']):
    return A @ B @ C @ D
