# Source: Pythran tests/cases/pi_buffon.py
# Original #pythran export: pi_estimate(int, float list, int)
from zinnia import *
from math import sqrt, pow


@zk_circuit
def pi_estimate(DARTS: int, rand: NDArray[Float, 64], randsize: int):
    hits = 0
    for i in range(0, DARTS):
        x = rand[i % randsize]
        y = rand[(randsize - i) % randsize]
        dist = sqrt(pow(x, 2) + pow(y, 2))
        if dist <= 1.0:
            hits += 1.0
    pi = 4 * (hits / DARTS)
    _zinnia_result = pi
