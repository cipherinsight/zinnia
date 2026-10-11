# npbench program compute/compute_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def compute(array_1: Integer[Array, '16 16'], array_2: Integer[Array, '16 16'], a: Integer, b: Integer, c: Integer):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c
