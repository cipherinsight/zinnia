from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(x_dists: Float[Array, "3 4"], y_dists: Float[Array, "3 4"]):
    dists = np.vstack(([x_dists.T], [y_dists.T])).T
    return dists
