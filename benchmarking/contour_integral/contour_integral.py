# Source: NPBench contour_integral (contour_integral_numpy.py)
# Original signature: contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y) — Ham (slab_per_bc+1, NR, NR) complex,
#   int_pts (num_int_pts,) complex, Y (NR, NM) complex.
# Migration notes:
#   - NR, NM, SLAB_PER_BC, NUM_INT_PTS hoisted to module-level constants from the "S" preset, shrunk to NR=NM=8.
#   - Uses complex arrays + np.linalg.inv / np.linalg.solve which are likely unsupported; left in place per migration policy.
from zinnia import *

NR = 8
NM = 8
SLAB_PER_BC = 2
NUM_INT_PTS = 8


@zk_circuit
def contour_integral(Ham: NDArray[Float, 3, 8, 8],
                     int_pts: NDArray[Float, 8],
                     Y: NDArray[Float, 8, 8]):
    P0 = np.zeros((8, 8), dtype=np.complex128)
    P1 = np.zeros((8, 8), dtype=np.complex128)
    for z in int_pts:
        Tz = np.zeros((8, 8), dtype=np.complex128)
        for n in range(2 + 1):
            zz = np.power(z, 2 / 2 - n)
            Tz += zz * Ham[n]
        if 8 == 8:
            X = np.linalg.inv(Tz)
        else:
            X = np.linalg.solve(Tz, Y)
        if abs(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X

    _zinnia_result = P0, P1
