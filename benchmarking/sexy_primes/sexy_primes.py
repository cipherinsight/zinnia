# Source: Pythran tests/cases/sexy_primes.py
# Original #pythran export: primes_below(int)
from zinnia import *


@zk_chip
def is_prime(n) -> Boolean:
    return all((n % j > 0) for j in range(2, n))


@zk_circuit
def primes_below(x: int):
    _zinnia_result = [[j - 6, j] for j in range(9, x + 1) if is_prime(j) and is_prime(j - 6)]
