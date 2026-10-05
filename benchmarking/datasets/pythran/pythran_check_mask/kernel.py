from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def check_mask(db: Bool[Array, 'N<=1024 M<=1024'], out: Bool[Array, 'N<=1024'], mask: Integer[Array, 'N<=1024']=(1, 0, 1)):
    for idx, line in enumerate(db):
        target, vector = (line[0], line[1:])
        if (mask == np.bitwise_and(mask, vector)).all():
            if target == 1:
                out[idx] = 1
    return out
