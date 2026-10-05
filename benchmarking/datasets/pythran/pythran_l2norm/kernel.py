from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: domain-obligation
# Expected outcome: compile/prove candidate
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
import numpy as np

@zk_circuit
def l2_norm(x: Float[Array, 'N<=1024 M<=1024']):
    return np.sqrt(np.sum(np.abs(x) ** 2, 1))
