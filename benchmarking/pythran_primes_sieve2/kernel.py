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
@requires(lambda n: 0 <= n <= 1024)
def get_primes7(n: Integer):
    """
	standard optimized sieve algorithm to get a list of prime numbers
	--- this is the function to compare your functions against! ---
	"""
    if n < 2:
        return []
    if n == 2:
        return [2]
    s = list(range(3, n + 1, 2))
    mroot = n ** 0.5
    half = len(s)
    i = 0
    m = 3
    while m <= mroot:
        if s[i]:
            j = (m * m - 3) // 2
            s[j] = 0
            while j < half:
                s[j] = 0
                j += m
        i = i + 1
        m = 2 * i + 3
    return [2] + [x for x in s if x]
