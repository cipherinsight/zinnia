from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded, mask-where
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Record mask shape equality and active-element bounds for where/nonzero-style indexing.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def compute_subpix_2d_gaussian2(correl: Float[Array, 'N<=1024 M<=1024'], ix: Float, iy: Float):
    correl_crop = correl[iy - 1:iy + 2, ix - 1:ix + 2]
    tmp = np.where(correl_crop < 0)
    for i0, i1 in zip(tmp[0], tmp[1]):
        correl_crop[i0, i1] = 1e-06
    c10 = 0
    c01 = 0
    c11 = 0
    c20 = 0
    c02 = 0
    for i in range(1):
        for j in range(1):
            c10 += (i - 1) * np.log(correl_crop[j, i])
            c01 += (j - 1) * np.log(correl_crop[j, i])
            c11 += (i - 1) * (j - 1) * np.log(correl_crop[j, i])
            c20 += (3 * (i - 1) ** 2 - 2) * np.log(correl_crop[j, i])
            c02 += (3 * (j - 1) ** 2 - 2) * np.log(correl_crop[j, i])
            c00 = (5 - 3 * (i - 1) ** 2 - 3 * (j - 1) ** 2) * np.log(correl_crop[j, i])
    c00, c10, c01, c11, c20, c02 = (c00 / 9, c10 / 6, c01 / 6, c11 / 4, c20 / 6, c02 / 6)
    deplx = np.float32((c11 * c01 - 2 * c10 * c02) / (4 * c20 * c02 - c11 ** 2))
    deply = np.float32((c11 * c10 - 2 * c01 * c20) / (4 * c20 * c02 - c11 ** 2))
    return (deplx, deply, correl_crop)
