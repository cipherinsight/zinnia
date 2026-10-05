from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.
import numpy as np

@zk_circuit
def compute(array_1: Float[Array, 'N<=1024'], array_2: Float, a: Float, b: Float, c: Integer):
    return np.clip(array_1, 2, 10) * a + array_2 * b + c
