# Source: Pythran tests/cases/check_mask.py
# Original #pythran export: check_mask(bool[][], bool[])
from zinnia import *

M = 8
N = 4


@zk_circuit
def check_mask(db: NDArray[Integer, 8, 4], out: NDArray[Integer, 8], mask: tuple = (1, 0, 1)):
    for idx, line in enumerate(db):
        target, vector = line[0], line[1:]
        if (mask == np.bitwise_and(mask, vector)).all():
            if target == 1:
                out[idx] = 1
    _zinnia_result = out
