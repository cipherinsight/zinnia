from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, linalg, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Record square/invertible/SPD requirements where the linalg call needs them.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: np.linalg kernels may stay fail-closed until support is selected.
import numpy as np

@zk_circuit
def matrix_class_distance(dat: Float[Array, 'N<=1024 M<=1024'], dat_filter: Integer[Array, 'N<=1024'], dat_points: Float[Array, 'N<=1024 M<=1024'], iterations: Integer):
    aggregation = 0
    for i in range(iterations):
        aggregation += np.sum(np.linalg.norm(dat[dat_filter == i] - dat_points[i], axis=1))
    return aggregation
