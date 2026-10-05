# pythran program periodic_dist.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
def dist(x: Float[Array, '16'], y: Float[Array, '16'], z: Float[Array, '16'], L: Integer, periodicX: Boolean, periodicY: Boolean, periodicZ: Boolean):
    """ Computes distances between all particles and places the result in a matrix such that the ij th matrix entry corresponds to the distance between particle i and j """
    N = len(x)
    xtemp = np.tile(x, (N, 1))
    dx = xtemp - xtemp.T
    ytemp = np.tile(y, (N, 1))
    dy = ytemp - ytemp.T
    ztemp = np.tile(z, (N, 1))
    dz = ztemp - ztemp.T
    if periodicX:
        dx[dx > L / 2] = dx[dx > L / 2] - L
        dx[dx < -L / 2] = dx[dx < -L / 2] + L
    if periodicY:
        dy[dy > L / 2] = dy[dy > L / 2] - L
        dy[dy < -L / 2] = dy[dy < -L / 2] + L
    if periodicZ:
        dz[dz > L / 2] = dz[dz > L / 2] - L
        dz[dz < -L / 2] = dz[dz < -L / 2] + L
    d = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)
    d[d == 0] = -1
    return (d, dx, dy, dz)
