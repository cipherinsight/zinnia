# pythran program zero.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda n, m: 0 <= n <= 1024)
@requires(lambda n, m: 0 <= m <= 1024)
def zero(n: Integer, m: Integer):
    return [[0 for row in range(n)] for col in range(m)]
