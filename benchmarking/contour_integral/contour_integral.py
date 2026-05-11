# Source: NPBench contour_integral (contour_integral_numpy.py)
# Original signature: contour_integral(NR, NM, slab_per_bc, Ham, int_pts, Y) — Ham (slab_per_bc+1, NR, NR) complex,
#   int_pts (num_int_pts,) complex, Y (NR, NM) complex.
# Migration notes:
#   - NR, NM, SLAB_PER_BC, NUM_INT_PTS hoisted to module-level constants from the "S" preset.
#   - Uses complex arrays + np.linalg.inv / np.linalg.solve which are likely unsupported; left in place per migration policy.
from zinnia import *

NR = 50
NM = 150
SLAB_PER_BC = 2
NUM_INT_PTS = 32


@zk_circuit
def contour_integral(Ham: NDArray[Float, 3, 50, 50],
                     int_pts: NDArray[Float, 32],
                     Y: NDArray[Float, 50, 150]):
    P0 = np.zeros((NR, NR), dtype=np.complex128)
    P1 = np.zeros((NR, NR), dtype=np.complex128)
    for z in int_pts:
        Tz = np.zeros((NR, NR), dtype=np.complex128)
        for n in range(SLAB_PER_BC + 1):
            zz = np.power(z, SLAB_PER_BC / 2 - n)
            Tz += zz * Ham[n]
        if NR == NM:
            X = np.linalg.inv(Tz)
        else:
            X = np.linalg.solve(Tz, Y)
        if abs(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X

    _zinnia_result = P0, P1
