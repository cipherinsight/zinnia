# pythran program primes_sieve.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def primes_sieve(limit: Integer):
    a = [True] * limit
    a[0] = a[1] = False
    primes = list()
    for i, isprime in enumerate(a):
        if isprime:
            primes.append(i)
            for n in range(i * i, limit, i):
                a[n] = False
    return primes
