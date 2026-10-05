from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, in-place-mutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_circuit
@requires(lambda x, y, m: 0 <= m <= 1024)
def gf2mulmod(x: Float, y: Float, m: Integer):
    z = 0
    while x > 0:
        if x & 1 != 0:
            z ^= y
        y <<= 1
        y2 = y ^ m
        if y2 < y:
            y = y2
        x >>= 1
    return z
