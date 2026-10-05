from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: shape-dtype-only
# Expected outcome: compile/prove candidate
# Precondition: Shape/rank/dtype annotations only.
import numpy as np

@zk_circuit
def harris(I: Float[Array, 'N<=1024 M<=1024']):
    m, n = I.shape
    dx = (I[1:, :] - I[:m - 1, :])[:, 1:]
    dy = (I[:, 1:] - I[:, :n - 1])[1:, :]
    A = dx * dx
    B = dy * dy
    C = dx * dy
    tr = A + B
    det = A * B - C * C
    k = 0.05
    return det - k * tr * tr
