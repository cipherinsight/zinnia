# pythran program pythagorean_triples.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def next_pythagorean_triples(previous: Integer[Array, '1 3']):
    matrices = np.array([[-1, 2, 2], [-2, 1, 2], [-2, 2, 3], [1, 2, 2], [2, 1, 2], [2, 2, 3], [1, -2, 2], [2, -1, 2], [2, -2, 3]])
    next_triples = np.transpose(np.dot(matrices, np.transpose(previous)))
    next_triples = next_triples.reshape((3 * previous.shape[0], previous.shape[1]))
    return next_triples
