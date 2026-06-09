from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_chip
def next_num(num: Integer):
    if num & 1:
        return 3 * num + 1
    else:
        return num // 2

@zk_chip
def series_length(num: Integer, lengths: Float[Array, 'N<=1024']):
    if num in lengths:
        return lengths[num]
    else:
        num2 = next_num(num)
        result = 1 + series_length(num2, lengths)
        lengths[num] = result
        return result

@zk_circuit
def euler14(MAX_NUM: Integer):
    num_with_max_length = 1
    max_length = 0
    lengths = {1: 0}
    for ii in range(1, MAX_NUM):
        length = series_length(ii, lengths)
        if length > max_length:
            max_length = length
            num_with_max_length = ii
    return (num_with_max_length, max_length)
