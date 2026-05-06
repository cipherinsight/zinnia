# Source: Pythran tests/cases/reverse_cumsum.py
# Original #pythran export: reverse_cumsum(float[])
from zinnia import *

N = 64


@zk_circuit
def reverse_cumsum(x: NDArray[Float, 64]):
    _zinnia_result = np.cumsum(x[::-1])[::-1]
