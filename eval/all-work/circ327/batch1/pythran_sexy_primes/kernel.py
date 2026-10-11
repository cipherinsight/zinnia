# pythran program sexy_primes.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def is_prime(n):
    return all((n % j > 0 for j in range(2, n)))

@zk_circuit
def primes_below(x: Integer):
    return [[j - 6, j] for j in range(9, x + 1) if is_prime(j) and is_prime(j - 6)]
