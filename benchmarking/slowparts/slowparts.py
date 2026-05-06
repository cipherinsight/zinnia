# Source: Pythran tests/cases/slowparts.py
# Original #pythran export: slowparts(int, int, float[][][], float[][][], float[][], float[][], float[][][], float[][][], int)
from zinnia import *

D = 4
RE = 5
TWOD = 8


@zk_circuit
def slowparts(d: int, re: int,
              preDz: NDArray[Float, 8, 5, 5], preWz: NDArray[Float, 4, 5, 5],
              SRW: NDArray[Float, 4, 8], RSW: NDArray[Float, 4, 8],
              yxV: NDArray[Float, 5, 5, 4], xyU: NDArray[Float, 5, 5, 4],
              resid: int):
    fprime = lambda x: 1 - np.power(np.tanh(x), 2)

    partialDU = np.zeros((d + 1, re, 2 * d, d))
    for k in range(2 * d):
        for i in range(d):
            partialDU[:, :, k, i] = fprime(preDz[k]) * fprime(preWz[i]) * (SRW[i, k] + RSW[i, k]) * yxV[:, :, i]

    _zinnia_result = partialDU
