from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.
import numpy as np

@zk_circuit
@requires(lambda M, float_n, data: 0 <= M <= 1024)
def kernel(M: Integer, float_n: Float, data: Float[Array, 'N<=1024']):
    return np.cov(np.transpose(data))
