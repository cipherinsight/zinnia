# Source: Pythran tests/cases/zero.py
# Original #pythran export: zero(int, int)
from zinnia import *


@zk_circuit
def zero(n: int, m: int):
    _zinnia_result = [[0 for row in range(n)] for col in range(m)]
