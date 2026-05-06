# Source: NPBench azimint_naive (azimint_naive_numpy.py)
# Original signature: azimint_naive(data, radius, npt) — data, radius are length-N float arrays; npt is bin count.
# Migration notes:
#   - npt hoisted to a module-level constant (ZK shapes / loop bounds must be static).
#   - N picked from the NPBench "S" preset (400000) shrunk to 16; npt shrunk to 8.
from zinnia import *

N = 16
NPT = 8


@zk_circuit
def azimint_naive(data: NDArray[Float, 16], radius: NDArray[Float, 16]):
    rmax = radius.max()
    res = np.zeros(8, dtype=np.float64)
    for i in range(8):
        r1 = rmax * i / 8
        r2 = rmax * (i + 1) / 8
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    _zinnia_result = res
