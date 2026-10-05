from zinnia import *
import numpy as np

# DS-1000 NumPy acceptance-study kernel.
# The body below is generated from the original DS-1000 reference solution.
# Type annotations/preconditions are intentionally not inferred here.

@zk_chip
def ecdf_result(x):
    xs = np.sort(x)
    ys = np.arange(1, len(xs) + 1) / float(len(xs))
    return (xs, ys)

@zk_circuit
def kernel(grades: Float[Array, "50"], eval: Integer[Array, "5"]):
    resultx, resulty = ecdf_result(grades)
    result = np.zeros_like(eval, dtype=float)
    for i, element in enumerate(eval):
        if element < resultx[0]:
            result[i] = 0
        elif element >= resultx[-1]:
            result[i] = 1
        else:
            result[i] = resulty[(resultx > element).argmax() - 1]
    return result
