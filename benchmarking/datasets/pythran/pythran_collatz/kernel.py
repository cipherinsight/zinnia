# pythran program collatz.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
@requires(lambda n_max: 0 <= n_max <= 1024)
def collatz(n_max: Integer):
    n = np.arange(3, n_max + 1)
    count = np.ones_like(n)
    indices = n != 1
    while np.any(indices):
        count[indices] += 1
        x = n[indices]
        n[indices] = np.where(x & 1, 3 * x + 1, x // 2)
        indices = n != 1
    max_i = np.argmax(count)
    return (max_i + 3, count[max_i])
