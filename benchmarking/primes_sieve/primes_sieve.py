# Source: Pythran tests/cases/primes_sieve.py
# Original #pythran export: primes_sieve(int)
from zinnia import *


@zk_circuit
def primes_sieve(limit: int):
    a = [True] * limit
    a[0] = a[1] = False
    primes = list()

    for (i, isprime) in enumerate(a):
        if isprime:
            primes.append(i)
            for n in range(i * i, limit, i):
                a[n] = False

    _zinnia_result = primes
