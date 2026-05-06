# Source: Pythran tests/cases/crotg.py
# Original #pythran export: CROTG(complex, complex, float, complex)
# Migration notes: complex types likely unsupported.
from zinnia import *
import math


@zk_circuit
def CROTG(CA: complex, CB: complex, C: float = 0, S: complex = 0):
    if (abs(CA) == 0.):
        C = 0.
        S = complex(1., 0.)
        CA = CB
    else:
        SCALE = abs(CA) + abs(CB)
        NORM = SCALE * math.sqrt((abs(CA / SCALE)) ** 2 + (abs(CB / SCALE)) ** 2)
        ALPHA = CA / abs(CA)
        C = abs(CA) / NORM
        S = ALPHA * (CB.conjugate()) / NORM
        CA = ALPHA * NORM
    _zinnia_result = (CA, CB, C, S)
