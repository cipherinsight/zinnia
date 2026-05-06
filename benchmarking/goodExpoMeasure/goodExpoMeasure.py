# Source: Pythran tests/cases/goodExpoMeasure.py
# Original #pythran export: goodExpoMeasure(uint8[][][], float)
from zinnia import *

A = 3
B = 16
C = 16


@zk_circuit
def goodExpoMeasure(inRGB: NDArray[Integer, 3, 16, 16], sigma: float):
    R = inRGB[0, :, :].astype(np.float64)
    G = inRGB[1, :, :].astype(np.float64)
    B = inRGB[2, :, :].astype(np.float64)
    goodExpoR = np.exp(- ((R - 128) ** 2) / sigma)
    goodExpoG = np.exp(- ((G - 128) ** 2) / sigma)
    goodExpoB = np.exp(- ((B - 128) ** 2) / sigma)
    goodExpo = goodExpoR * goodExpoG * goodExpoB
    goodExpo = (np.round(goodExpo, 2) * (2 ** 8 - 1)).astype(np.uint8)

    _zinnia_result = goodExpo
