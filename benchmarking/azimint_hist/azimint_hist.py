# Source: NPBench azimint_hist (azimint_hist_numpy.py)
# Original signature: azimint_hist(data, radius, npt) — data, radius are length-N float arrays; npt is bin count.
# Migration notes:
#   - npt hoisted to a module-level constant (ZK shapes / loop bounds must be static).
#   - N picked from the NPBench "S" preset (400000); npt set to 1000.
from zinnia import *

N = 400000
NPT = 1000


@zk_circuit
def azimint_hist(data: NDArray[Float, 400000], radius: NDArray[Float, 400000]):
    histu = np.histogram(radius, NPT)[0]
    histw = np.histogram(radius, NPT, weights=data)[0]
    _zinnia_result = histw / histu
