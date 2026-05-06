# Source: Pythran tests/cases/diffusion_numpy.py
# Original #pythran export: diffuseNumpy(float[][], float[][], int)
from zinnia import *

M = 16
N = 16


@zk_circuit
def diffuseNumpy(u: NDArray[Float, 16, 16], tempU: NDArray[Float, 16, 16], iterNum: int):
    mu = .1

    for n in range(iterNum):
        tempU[1:-1, 1:-1] = u[1:-1, 1:-1] + mu * (
            u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1] +
            u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
        u[:, :] = tempU[:, :]
        tempU[:, :] = 0.0
