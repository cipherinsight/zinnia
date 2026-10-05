from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(Y: Float[Array, "N<=64 M<=64 K<=64"]):
    X = np.zeros([Y.shape[1], Y.shape[0]])
    for i, mat in enumerate(Y):
        diag = np.sqrt(np.diag(mat))
        X[:, i] += diag
    return X
