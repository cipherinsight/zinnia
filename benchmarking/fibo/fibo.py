# Source: Pythran tests/cases/fibo.py
# Original #pythran export: test(int)
from zinnia import *


@zk_chip
def rfibo(n) -> Integer:
    if n < 2:
        return n
    else:
        n_1 = rfibo(n - 1)
        n_2 = rfibo(n - 2)
        return n_1 + n_2


@zk_chip
def fibo(n) -> Integer:
    if n < 10:
        return rfibo(n)
    else:
        n_1 = 0
        n_1 = fibo(n - 1)
        n_2 = fibo(n - 2)
        return n_1 + n_2


@zk_circuit
def test(n: int):
    f = fibo(n)
    _zinnia_result = f
