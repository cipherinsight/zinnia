# Source: Pythran tests/cases/ramsurf.py
# Original #pythran export: deriv(int, float, float, complex list, complex list, complex list, float list list, float)
# Migration notes: complex types and nested list-of-lists likely unsupported.
from zinnia import *


@zk_circuit
def deriv(n: int, sig: float, alp: float, dg: list, dh1: list, dh3: list, bin: NDArray[Float, 16, 16], nu: float):
    dh2 = [complex(0, 0) for n in dh1]
    ci = complex(0.0, 1.0)
    dh1[0] = complex(0.5, 0) * ci * complex(sig, 0)
    exp1 = complex(-0.5, 0)
    dh2[0] = alp
    exp2 = complex(-1.0, 0)
    dh3[0] = -2.0 * nu
    exp3 = complex(-1.0, 0)
    for i in range(1, n):
        dh1[i] = dh1[i - 1] * exp1
        exp1 = exp1 - 1.0
        dh2[i] = dh2[i - 1] * exp2
        exp2 = exp2 - 1.0
        dh3[i] = -nu * dh3[i - 1] * exp3
        exp3 = exp3 - 1.0
    dg[0] = 1.0
    dg[1] = dh1[0] + dh2[0] + dh3[0]
    for i in range(2, n + 1):
        dg[i] = dh1[i - 1] + dh2[i - 1] + dh3[i - 1]
        for j in range(1, i):
            dg[i] = dg[i] + bin[j - 1][i - 1] * (dh1[j - 1] + dh2[j - 1] + dh3[j - 1]) * dg[i - j]
