from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_circuit
def kernel(number: Integer):
    deg = np.sin(np.deg2rad(number))
    rad = np.sin(number)
    result = int(rad > deg)
    return result
