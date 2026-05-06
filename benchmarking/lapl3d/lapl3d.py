# Source: Pythran tests/cases/lapl3d.py
# Original #pythran export: laplacien(float64[][][3])
from zinnia import *

A = 16
B = 16


@zk_circuit
def laplacien(image: NDArray[Float, 16, 16, 3]):
    out_image = np.abs(4 * image[1:-1, 1:-1] -
                       image[0:-2, 1:-1] - image[2:, 1:-1] -
                       image[1:-1, 0:-2] - image[1:-1, 2:])
    valmax = np.max(out_image)
    valmax = max(1., valmax) + 1.E-9
    out_image /= valmax
    _zinnia_result = out_image
