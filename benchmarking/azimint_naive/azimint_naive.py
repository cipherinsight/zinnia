# Source: NPBench azimint_naive (azimint_naive_numpy.py)
# Original signature: azimint_naive(data, radius, npt) — data, radius are length-N float arrays; npt is bin count.
# Migration notes:
#   - npt hoisted to a module-level constant (ZK shapes / loop bounds must be static).
#   - N picked from the NPBench "S" preset (400000); npt set to 1000.
from zinnia import *

N = 400000
NPT = 1000


@zk_circuit
def azimint_naive(data: NDArray[Float, 400000], radius: NDArray[Float, 400000]):
    rmax = radius.max()
    res = np.zeros(NPT, dtype=np.float64)
    for i in range(NPT):
        r1 = rmax * i / NPT
        r2 = rmax * (i + 1) / NPT
        mask_r12 = np.logical_and((r1 <= radius), (radius < r2))
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    _zinnia_result = res
