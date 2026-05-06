# Source: Pythran tests/cases/make_decision.py
# Original #pythran export: md(complex128[], complex128[])
# Migration notes: complex types likely unsupported in Zinnia.
from zinnia import *

L = 64
M = 32


@zk_circuit
def md(E: NDArray[Float, 64], symbols: NDArray[Float, 32]):
    L = E.shape[0]
    M = symbols.shape[0]
    syms_out = np.zeros(L, dtype=E.dtype)
    for i in range(L):
        im = np.argmin(abs(E[i] - symbols) ** 2)
        syms_out[i] = symbols[im]
    _zinnia_result = syms_out
