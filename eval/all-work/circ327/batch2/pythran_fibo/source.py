# pythran program fibo.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def rfibo(n):
    if n < 2:
        return n
    else:
        n_1 = rfibo(n - 1)
        n_2 = rfibo(n - 2)
        return n_1 + n_2

@zk_chip
def fibo(n):
    if n < 10:
        return rfibo(n)
    else:
        n_1 = 0
        'omp task shared(n,n_1)'
        n_1 = fibo(n - 1)
        n_2 = fibo(n - 2)
        'omp taskwait'
        return n_1 + n_2

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def test(n: Integer):
    """omp parallel"""
    'omp single'
    f = fibo(n)
    return f
