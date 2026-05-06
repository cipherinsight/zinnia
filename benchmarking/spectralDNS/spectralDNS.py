# Source: Pythran tests/cases/spectralDNS.py
# Original #pythran export: cross1(float[:,:,:,:], float[:,:,:,:], float[:,:,:,:])
# Migration notes: picked the first export (cross1).
from zinnia import *

A = 3
B = 2
C = 5
D = 7


@zk_circuit
def cross1(c: NDArray[Float, 3, 2, 5, 7], a: NDArray[Float, 3, 2, 5, 7], b: NDArray[Float, 3, 2, 5, 7]):
    c[0] = a[0] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    _zinnia_result = c
