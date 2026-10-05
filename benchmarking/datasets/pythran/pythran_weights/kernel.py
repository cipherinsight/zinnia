# pythran program weights.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def weights(input_data: Integer[Array, '5 2'], threshold: Float=0.3):
    n_seq, length = input_data.shape
    weights = np.zeros(n_seq, dtype=np.float32)
    for i in range(n_seq):
        vector = input_data[i, None, :]
        count_matches = np.sum(vector == input_data, axis=1)
        over_threshold = count_matches > threshold * length
        total = np.sum(over_threshold)
        weights[i] = 1 / total
    return weights
