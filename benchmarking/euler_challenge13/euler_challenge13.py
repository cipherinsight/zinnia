# Source: Pythran tests/cases/euler_challenge13.py
# Original #pythran export: solve(int)
from zinnia import *


@zk_circuit
def solve(v: int):
    t = (
        37107,
        46376,
        74324,
        91942,
        23067,
        89261,
        28112,
        44274,
        47451,
        70386,
    )
    _zinnia_result = str(sum(t) + v)[0:10]
