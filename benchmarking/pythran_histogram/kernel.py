from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
def histogram(data: Float[Array, 'N<=1024'], bin_width: Integer):
    lower_bound, upper_bound = (min(data), max(data))
    out_data = [0] * (1 + bin_width)
    for i in data:
        out_data[int(bin_width * (i - lower_bound) / (upper_bound - lower_bound))] += 1
    out_data[-2] += out_data[-1]
    out_data.pop()
    return out_data
