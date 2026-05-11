# Source: Pythran tests/cases/nqueens.py
# Original #pythran export: n_queens(int)
# Migration notes: uses generator (yield); may be unsupported.
from zinnia import *


def permutations(iterable, r=None):
    """permutations(range(3), 2) --> (0,1) (0,2) (1,0) (1,2) (2,0) (2,1)"""
    pool = tuple(iterable)
    n = len(pool)
    if r is None:
        r = n
    indices = list(range(n))
    cycles = list(range(n-r+1, n+1))[::-1]
    yield tuple(pool[i] for i in indices[:r])
    while n:
        for i in reversed(range(r)):
            cycles[i] -= 1
            if cycles[i] == 0:
                indices[i:] = indices[i+1:] + indices[i:i+1]
                cycles[i] = n - i
            else:
                j = cycles[i]
                indices[i], indices[-j] = indices[-j], indices[i]
                yield tuple(pool[i] for i in indices[:r])
                break
        else:
            return


@zk_circuit
def n_queens(queen_count: int):
    out = list()
    cols = list(range(queen_count))
    for vec in permutations(cols, None):
        if (queen_count == len(set(vec[i] + i for i in cols))
                == len(set(vec[i] - i for i in cols))):
            out.append(vec)
    _zinnia_result = out
