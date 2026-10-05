# pythran program diffusion_numpy.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def diffuseNumpy(u: Float[Array, '128 128'], tempU: Float[Array, '128 128'], iterNum: Integer):
    """
    Apply Numpy matrix for the Forward-Euler Approximation
    """
    mu = 0.1
    for n in range(iterNum):
        tempU[1:-1, 1:-1] = u[1:-1, 1:-1] + mu * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1] + u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2])
        u[:, :] = tempU[:, :]
        tempU[:, :] = 0.0
