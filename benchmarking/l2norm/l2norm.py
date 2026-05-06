# Source: Pythran tests/cases/l2norm.py
# Original #pythran export: l2_norm(float64[][])
from zinnia import *

M = 16
N = 16


@zk_circuit
def l2_norm(x: NDArray[Float, 16, 16]):
    _zinnia_result = np.sqrt(np.sum(np.abs(x) ** 2, 1))
