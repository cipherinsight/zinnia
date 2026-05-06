# Source: Pythran tests/cases/fibo_seq.py
# Original #pythran export: fibo(int)
from zinnia import *


@zk_circuit
def fibo(n: int):
    a, b = 1, 1
    for _ in range(n):
        a, b = a + b, a
    _zinnia_result = a
