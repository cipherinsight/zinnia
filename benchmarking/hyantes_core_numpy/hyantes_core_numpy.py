# Source: Pythran tests/cases/hyantes_core_numpy.py
# Original #pythran export: run(float, float, float, float, float, float, int, int, float[][])
from zinnia import *

T = 64


@zk_circuit
def run(xmin: float, ymin: float, xmax: float, ymax: float, step: float, range_: float,
        range_x: int, range_y: int, t: NDArray[Float, 64, 3]):
    X, Y = t.shape
    pt = np.zeros((X, Y))
    for i in range(X):
        for j in range(Y):
            for k in t:
                tmp = 6368. * np.arccos(np.cos(xmin + step * i) * np.cos(k[0])
                                        * np.cos((ymin + step * j) - k[1])
                                        + np.sin(xmin + step * i)
                                        * np.sin(k[0]))
                if tmp < range_:
                    pt[i, j] += k[2] / (1 + tmp)
    _zinnia_result = pt
