# Source: Pythran tests/cases/primes_sieve2.py
# Original #pythran export: get_primes7(int)
from zinnia import *


@zk_circuit
def get_primes7(n: int):
    if n < 2:
        _zinnia_result = []
    if n == 2:
        _zinnia_result = [2]
    s = list(range(3, n + 1, 2))
    mroot = n ** 0.5
    half = len(s)
    i = 0
    m = 3
    while m <= mroot:
        if s[i]:
            j = (m * m - 3) // 2
            s[j] = 0
            while j < half:
                s[j] = 0
                j += m
        i = i + 1
        m = 2 * i + 3
    _zinnia_result = [2] + [x for x in s if x]
