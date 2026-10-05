# pythran program allpairs_distances.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def dists(X, Y):
    return np.array([[np.sum((x - y) ** 2) for x in X] for y in Y])

@zk_circuit
def allpairs_distances(d: Integer):
    X = np.arange(600 * d).reshape((600, d))
    Y = np.arange(200 * d).reshape((200, d))
    return dists(X, Y)
