from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
import numpy as np

@zk_circuit
def tax(d: Float[Array, 'N<=1024']):
    tax_seg1 = d[d > 256303] * 0.45 - 16164.53
    tax_seg2 = d[(d > 54057) & (d <= 256303)] * 0.42 - 8475.44
    seg3 = d[(d > 13769) & (d <= 54057)] - 13769
    seg4 = d[(d > 8820) & (d <= 13769)] - 8820
    prog_seg3 = seg3 * 2.2376e-06 + 0.2397
    prog_seg4 = seg4 * 1.00727e-05 + 0.14
    return (np.sum(tax_seg1) + np.sum(tax_seg2) + np.sum(seg3 * prog_seg3 + 939.57) + np.sum(seg4 * prog_seg4)) / np.sum(d)
