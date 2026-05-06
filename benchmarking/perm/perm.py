# Source: Pythran tests/cases/perm.py
# Original #pythran export: permutations(int list)
from zinnia import *


@zk_circuit
def permutations(iterable: NDArray[Integer, 64]):
    out = []
    pool = tuple(iterable)
    n = len(pool)
    r = n
    indices = list(range(n))
    cycles = list(range(n - r + 1, n + 1))[::-1]
    out.append(tuple([pool[i] for i in indices[:r]]))
    while 1:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i + 1:] + indices[i:i + 1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                out.append(tuple([pool[i] for i in indices[:r]]))
                break
        else:
            _zinnia_result = out
