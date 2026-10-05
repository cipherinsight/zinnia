from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation
# Expected outcome: compile/prove candidate
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def laplacien(image: Float[Array, 'N<=1024 M<=1024']):
    out_image = np.abs(4 * image[1:-1, 1:-1] - image[0:-2, 1:-1] - image[2:, 1:-1] - image[1:-1, 0:-2] - image[1:-1, 2:])
    valmax = np.max(out_image)
    valmax = max(1.0, valmax) + 1e-09
    out_image /= valmax
    return out_image
