# pythran program repeating.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def repeating(x: Float[Array, '10'], nvar_y: Integer):
    nvar_x = x.shape[0]
    y = np.empty(nvar_x * (1 + nvar_y))
    y[0:nvar_x] = x[0:nvar_x]
    y[nvar_x:] = np.repeat(x, nvar_y)
    return y
