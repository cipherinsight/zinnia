# Source: Pythran tests/cases/queens_numba.py
# Original #pythran export: solve(int)
from zinnia import *


@zk_chip
def hits(x1, y1, x2, y2) -> Boolean:
    return x1 == x2 or y1 == y2 or abs(x1 - x2) == abs(y1 - y2)


@zk_chip
def hitsany(x, y, queens_x, queens_y) -> Boolean:
    for i in range(len(queens_x)):
        if hits(x, y, queens_x[i], queens_y[i]):
            return True
    return False


@zk_chip
def _solve(n, queens_x, queens_y) -> Boolean:
    if n == 0:
        return True

    for x in range(1, 9):
        for y in range(1, 9):
            if not hitsany(x, y, queens_x, queens_y):
                queens_x.append(x)
                queens_y.append(y)

                if _solve(n - 1, queens_x, queens_y):
                    return True

                queens_x.pop()
                queens_y.pop()

    return False


@zk_circuit
def solve(n: int):
    queens_x = []
    queens_y = []

    if _solve(n, queens_x, queens_y):
        _zinnia_result = queens_x, queens_y
    else:
        _zinnia_result = None
