# pythran program euclidean_distance_square.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def euclidean_distance_square(x1: Float[Array, '1 3'], x2: Float[Array, '5 3']):
    return -2 * np.dot(x1, x2.T) + np.sum(np.square(x1), axis=1)[:, np.newaxis] + np.sum(np.square(x2), axis=1)
