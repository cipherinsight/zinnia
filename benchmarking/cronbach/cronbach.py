# Source: Pythran tests/cases/cronbach.py
# Original #pythran export: cronbach(float[][])
from zinnia import *

M = 16
N = 16


@zk_circuit
def cronbach(itemscores: NDArray[Float, 16, 16]):
    itemvars = itemscores.var(1, None, None, 1)
    tscores = itemscores.sum(0)
    nitems = len(itemscores)
    _zinnia_result = nitems / (nitems - 1) * (1 - itemvars.sum() / tscores.var(None, None, None, 1))
