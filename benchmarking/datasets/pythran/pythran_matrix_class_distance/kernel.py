# pythran program matrix_class_distance.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def matrix_class_distance(dat: Float[Array, '200 10'], dat_filter: Integer[Array, '200'], dat_points: Float[Array, '200 10'], iterations: Integer):
    aggregation = 0
    for i in range(iterations):
        aggregation += np.sum(np.linalg.norm(dat[dat_filter == i] - dat_points[i], axis=1))
    return aggregation
