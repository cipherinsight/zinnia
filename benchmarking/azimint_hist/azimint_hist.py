# Source: NPBench azimint_hist (azimint_hist_numpy.py)
# Original signature: azimint_hist(data, radius, npt) — data, radius are length-N float arrays; npt is bin count.
# Migration notes:
#   - npt hoisted to a module-level constant (ZK shapes / loop bounds must be static).
#   - N picked from the NPBench "S" preset (400000) shrunk to 16 for tractable circuits; npt set to 8.
from zinnia import *

N = 16
NPT = 8


@zk_circuit
def azimint_hist(data: NDArray[Float, 16], radius: NDArray[Float, 16]):
    histu = np.histogram(radius, 8)[0]
    histw = np.histogram(radius, 8, weights=data)[0]
    _zinnia_result = histw / histu
