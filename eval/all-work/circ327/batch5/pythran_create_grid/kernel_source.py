# pythran program create_grid.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def create_grid(x: Float[Array, '16']):
    N = x.shape[0]
    z = np.zeros((N, N, 3))
    z[:, :, 0] = x.reshape(-1, 1)
    z[:, :, 1] = x
    fast_grid = z.reshape(N * N, 3)
    return fast_grid
