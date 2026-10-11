# pythran program reverse_cumsum.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def reverse_cumsum(x: Float[Array, '10']):
    return np.cumsum(x[::-1])[::-1]
