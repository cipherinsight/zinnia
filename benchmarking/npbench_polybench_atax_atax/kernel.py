from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: matmul-dot
# Expected outcome: compile/prove candidate
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
import numpy as np

@zk_circuit
def kernel(A: Float[Array, 'N<=1024 M<=1024'], x: Float[Array, 'M<=1024']):
    return A @ x @ A
