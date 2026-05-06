# Source: Pythran tests/cases/diffusion_pure_python.py
# Original #pythran export: diffusePurePython(float[][], float[][], int)
from zinnia import *

M = 16
N = 16


@zk_circuit
def diffusePurePython(u: NDArray[Float, 16, 16], tempU: NDArray[Float, 16, 16], iterNum: int):
    mu = .1
    row = u.shape[0]
    col = u.shape[1]

    for n in range(iterNum):
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                tempU[i, j] = u[i, j] + mu * (
                    u[i + 1, j] - 2 * u[i, j] + u[i - 1, j] +
                    u[i, j + 1] - 2 * u[i, j] + u[i, j - 1])
        for i in range(1, row - 1):
            for j in range(1, col - 1):
                u[i, j] = tempU[i, j]
                tempU[i, j] = 0.0
