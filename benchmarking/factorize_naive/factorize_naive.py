# Source: Pythran tests/cases/factorize_naive.py
# Original #pythran export: factorize_naive(int)
from zinnia import *


@zk_circuit
def factorize_naive(n: int):
    if n < 2:
        _zinnia_result = []
    factors = []
    p = 2

    while True:
        if n == 1:
            _zinnia_result = factors

        r = n % p
        if r == 0:
            factors.append(p)
            n = n / p
        elif p * p >= n:
            factors.append(n)
            _zinnia_result = factors
        elif p > 2:
            p += 2
        else:
            p += 1
    assert False, "unreachable"
