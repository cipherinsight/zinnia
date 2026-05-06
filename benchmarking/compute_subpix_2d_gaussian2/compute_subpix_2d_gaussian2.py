# Source: Pythran tests/cases/compute_subpix_2d_gaussian2.py
# Original #pythran export: compute_subpix_2d_gaussian2(float32[][], int, int)
from zinnia import *

M = 4
N = 4


@zk_circuit
def compute_subpix_2d_gaussian2(correl: NDArray[Float, 4, 4], ix: int, iy: int):
    correl_crop = correl[iy - 1:iy + 2, ix - 1:ix + 2]

    tmp = np.where(correl_crop < 0)
    for i0, i1 in zip(tmp[0], tmp[1]):
        correl_crop[i0, i1] = 1e-6

    c10 = 0
    c01 = 0
    c11 = 0
    c20 = 0
    c02 = 0
    for i in range(1):
        for j in range(1):
            c10 += (i - 1) * np.log(correl_crop[j, i])
            c01 += (j - 1) * np.log(correl_crop[j, i])
            c11 += (i - 1) * (j - 1) * np.log(correl_crop[j, i])
            c20 += (3 * (i - 1) ** 2 - 2) * np.log(correl_crop[j, i])
            c02 += (3 * (j - 1) ** 2 - 2) * np.log(correl_crop[j, i])
            c00 = (5 - 3 * (i - 1) ** 2 - 3 * (j - 1) ** 2) * np.log(correl_crop[j, i])

    c00, c10, c01, c11, c20, c02 = \
        c00 / 9, c10 / 6, c01 / 6, c11 / 4, c20 / 6, c02 / 6
    deplx = np.float32((c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11 ** 2))
    deply = np.float32((c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11 ** 2))
    _zinnia_result = deplx, deply, correl_crop
