# pythran program l2norm.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def l2_norm(x: Float[Array, '16 16']):
    return np.sqrt(np.sum(np.abs(x) ** 2, 1))
