from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
import numpy as np

@zk_circuit
def weights(input_data: Integer[Array, 'N<=1024 M<=1024'], threshold: Float=0.3):
    n_seq, length = input_data.shape
    weights = np.zeros(n_seq, dtype=np.float32)
    for i in range(n_seq):
        vector = input_data[i, None, :]
        count_matches = np.sum(vector == input_data, axis=1)
        over_threshold = count_matches > threshold * length
        total = np.sum(over_threshold)
        weights[i] = 1 / total
    return weights
