from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: in-place-mutation, loop-bounded
# Expected outcome: compile/prove candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.

@zk_circuit
def primes_sieve(limit: Integer):
    a = [True] * limit
    a[0] = a[1] = False
    primes = list()
    for i, isprime in enumerate(a):
        if isprime:
            primes.append(i)
            for n in range(i * i, limit, i):
                a[n] = False
    return primes
