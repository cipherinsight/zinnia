# npbench program azimint_naive/azimint_naive_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
@requires(lambda data, radius, npt: npt != 0)
def azimint_naive(data: Float[Array, '400000'], radius: Float[Array, '400000'], npt: Integer):
    rmax = radius.max()
    res = np.zeros(npt, dtype=np.float64)
    for i in range(npt):
        r1 = rmax * i / npt
        r2 = rmax * (i + 1) / npt
        mask_r12 = np.logical_and(r1 <= radius, radius < r2)
        values_r12 = data[mask_r12]
        res[i] = values_r12.mean()
    return res
