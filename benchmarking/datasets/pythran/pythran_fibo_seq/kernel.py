# pythran program fibo_seq.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def fibo(n: Integer):
    """ fibonaccie compuation. """
    a, b = (1, 1)
    for _ in range(n):
        a, b = (a + b, a)
    return a
