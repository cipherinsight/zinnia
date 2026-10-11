# pythran program ln.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def ln(X: Float[Array, '16'], Y: Float[Array, '16']):
    Y[:] = X - 1 - (X - 1) ** 2 / 2 + (X - 1) ** 3 / 3 - (X - 1) ** 4 / 4 + (X - 1) ** 5 / 5 - (X - 1) ** 6 / 6 + (X - 1) ** 7 / 7 - (X - 1) ** 8 / 8 + (X - 1) ** 9 / 9
    return Y
