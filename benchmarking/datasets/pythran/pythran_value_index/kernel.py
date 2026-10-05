# pythran program value_index.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def tri(score: Integer[Array, '10 10'], score_t: Integer[Array, '10 10']):
    score_t[0, 0] = score[0, 0]
    score_t = score[np.argsort(score[:, 1])]
    return score_t
