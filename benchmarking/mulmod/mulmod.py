# Source: Pythran tests/cases/mulmod.py
# Original #pythran export: gf2mulmod(int, int, int)
from zinnia import *


@zk_circuit
def gf2mulmod(x: int, y: int, m: int):
    z = 0
    while x > 0:
        if (x & 1) != 0:
            z ^= y
        y <<= 1
        y2 = y ^ m
        if y2 < y:
            y = y2
        x >>= 1
    _zinnia_result = z
