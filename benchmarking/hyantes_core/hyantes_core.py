# Source: Pythran tests/cases/hyantes_core.py
# Original #pythran export: run(float, float, float, float, float, float, int, int, float list list)
from zinnia import *
import math


@zk_circuit
def run(xmin: float, ymin: float, xmax: float, ymax: float, step: float, range_: float,
        range_x: int, range_y: int, t: NDArray[Float, 16, 16]):
    pt = [[0] * range_y for _ in range(range_x)]
    for i in range(range_x):
        for j in range(range_y):
            s = 0
            for k in t:
                tmp = 6368. * math.acos(math.cos(xmin + step * i) * math.cos(k[0]) *
                                        math.cos((ymin + step * j) - k[1]) + math.sin(xmin + step * i) * math.sin(k[0]))
                if tmp < range_:
                    s += k[2] / (1 + tmp)
            pt[i][j] = s
    _zinnia_result = pt
