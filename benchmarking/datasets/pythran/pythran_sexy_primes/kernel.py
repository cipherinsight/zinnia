from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: denominator-obligation
# Expected outcome: compile/prove candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.

@zk_chip
def is_prime(n: Integer):
    return all((n % j > 0 for j in range(2, n)))

@zk_circuit
def primes_below(x: Integer):
    return [[j - 6, j] for j in range(9, x + 1) if is_prime(j) and is_prime(j - 6)]
