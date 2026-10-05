# npbench program polybench/mvt/mvt_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def kernel(x1: Float[Array, '16'], x2: Float[Array, '16'], y_1: Float[Array, '16'], y_2: Float[Array, '16'], A: Float[Array, '16 16']):
    x1 += A @ y_1
    x2 += y_2 @ A
