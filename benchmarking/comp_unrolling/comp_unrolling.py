# Source: Pythran tests/cases/comp_unrolling.py
# Original #pythran export: list_comp(int list list)
from zinnia import *


@zk_chip
def foo(cc, x, y) -> Boolean:
    for a in cc:
        if a:
            return True
        return False


@zk_circuit
def list_comp(cc: NDArray[Integer, 16, 16]):
    _zinnia_result = [(x, y) for x in range(1) for y in range(2) if foo(cc, x, y)]
