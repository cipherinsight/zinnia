# Source: Pythran tests/cases/sum_primes.py
# Original #pythran export: sum_primes(int)
from zinnia import *
import math


def isprime(n):
    if n < 2:
        return False
    if n == 2:
        return True
    max = int(math.ceil(math.sqrt(n)))
    i = 2
    while i <= max:
        if n % i == 0:
            return False
        i += 1
    return True


@zk_circuit
def sum_primes(n: int):
    _zinnia_result = sum([x for x in range(2, n) if isprime(x)])
