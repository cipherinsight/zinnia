from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, in-place-mutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_circuit
def collatz_modified(target: Integer):
    start = 1
    while True:
        i = start
        steps = 0
        while True:
            if i == 1:
                break
            if i % 2 == 0:
                i = i // 2
            else:
                i = 3 * i + 1
            steps += 1
        if steps == target:
            return start
        start += 1
