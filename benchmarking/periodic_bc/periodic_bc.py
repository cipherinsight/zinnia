# Source: Pythran tests/cases/periodic_bc.py
# Original #pythran export: periodic_bc(float[][][])
from zinnia import *

A = 8
B = 8
C = 8


@zk_circuit
def periodic_bc(f: NDArray[Float, 8, 8, 8]):
    f[:, 0, :] = f[:, -2, :]
    _zinnia_result = f
