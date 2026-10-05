from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, domain-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Add operation-specific domains: sqrt arg >= 0, log arg > 0, acos arg in [-1, 1].
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def summation(pos: Float[Array, 'N<=1024 M<=1024'], weights: Float[Array, 'N<=1024'], points: Float[Array, 'N<=1024 M<=1024']):
    n_points = len(points)
    n_weights = len(weights)
    sum_array3d = np.zeros((n_points, 3))

    @zk_chip
    def compute(i: Integer):
        pxi = points[i, 0]
        pyi = points[i, 1]
        pzi = points[i, 2]
        total = 0.0
        for j in range(n_weights):
            weight_j = weights[j]
            xj = pos[j, 0]
            yj = pos[j, 1]
            zj = pos[j, 2]
            dx = pxi - pos[j, 0]
            dy = pyi - pos[j, 1]
            dz = pzi - pos[j, 2]
            dr = 1.0 / np.sqrt(dx * dx + dy * dy + dz * dz)
            total += weight_j * dr
            sum_array3d[i, 0] += weight_j * dx
            sum_array3d[i, 1] += weight_j * dy
            sum_array3d[i, 2] += weight_j * dz
        return total
    sum_array = np.array([compute(i) for i in range(n_points)])
    return (sum_array, sum_array3d)
