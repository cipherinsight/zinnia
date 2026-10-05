from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
import numpy as np

@zk_circuit
def arc_distance(theta_1: Float[Array, 'N<=1024'], phi_1: Float[Array, 'N<=1024'], theta_2: Float[Array, 'N<=1024'], phi_2: Float[Array, 'N<=1024']):
    """
    Calculates the pairwise arc distance between all points in vector a and b.
    """
    temp = np.sin((theta_2 - theta_1) / 2) ** 2 + np.cos(theta_1) * np.cos(theta_2) * np.sin((phi_2 - phi_1) / 2) ** 2
    distance_matrix = 2 * np.arctan2(np.sqrt(temp), np.sqrt(1 - temp))
    return distance_matrix
