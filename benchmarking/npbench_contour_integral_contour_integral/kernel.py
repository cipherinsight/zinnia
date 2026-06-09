from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, denominator-obligation, in-place-mutation, linalg, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Record square/invertible/SPD requirements where the linalg call needs them.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
# Diagnostic: np.linalg kernels may stay fail-closed until support is selected.
import numpy as np

@zk_circuit
def contour_integral(NR: Integer, NM: Integer, slab_per_bc: Integer, Ham: Complex[Array, 'S<=1024 NR<=1024 NR<=1024'], int_pts: Complex[Array, 'Z<=1024'], Y: Complex[Array, 'NR<=1024 NM<=1024']):
    P0 = np.zeros((NR, NM), dtype=np.complex128)
    P1 = np.zeros((NR, NM), dtype=np.complex128)
    for z in int_pts:
        Tz = np.zeros((NR, NR), dtype=np.complex128)
        for n in range(slab_per_bc + 1):
            zz = np.power(z, slab_per_bc / 2 - n)
            Tz += zz * Ham[n]
        if NR == NM:
            X = np.linalg.inv(Tz)
        else:
            X = np.linalg.solve(Tz, Y)
        if abs(z) < 1.0:
            X = -X
        P0 += X
        P1 += z * X
    return (P0, P1)
