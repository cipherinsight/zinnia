# Source: Pythran tests/cases/pivot.py
# Original #pythran export: pivot(int, int, complex list list, complex list)
# Migration notes: complex types and nested list-of-lists likely unsupported.
from zinnia import *


@zk_circuit
def pivot(n: int, i: int, a: list, b: list):
    i0 = i
    amp0 = abs(a[i - 1][i - 1])
    for j in range(i + 1, n + 1):
        amp = abs(a[i - 1][j - 1])
        if amp > amp0:
            i0 = j
            amp0 = amp
    if i == i0:
        pass
    temp = b[i - 1]
    b[i - 1] = b[i0 - 1]
    b[i0 - 1] = temp
    for j in range(i, n + 1):
        temp = a[j - 1][i - 1]
        a[j - 1][i - 1] = a[j - 1][i0 - 1]
        a[j - 1][i0 - 1] = temp
    _zinnia_result = a, b
