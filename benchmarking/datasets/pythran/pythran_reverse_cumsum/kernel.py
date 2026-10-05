from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.
import numpy as np

@zk_circuit
def reverse_cumsum(x: Float[Array, 'N<=1024']):
    return np.cumsum(x[::-1])[::-1]
