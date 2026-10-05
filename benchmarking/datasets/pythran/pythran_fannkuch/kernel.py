from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, in-place-mutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_chip
def fannkuch(n: Integer):
    count = list(range(1, n + 1))
    max_flips = 0
    m = n - 1
    r = n
    check = 0
    perm1 = list(range(n))
    perm = list(range(n))
    while 1:
        if check < 30:
            check += 1
        while r != 1:
            count[r - 1] = r
            r -= 1
        if perm1[0] != 0 and perm1[m] != m:
            perm = perm1[:]
            flips_count = 0
            k = perm[0]
            while k:
                perm[:k + 1] = perm[k::-1]
                flips_count += 1
                k = perm[0]
            if flips_count > max_flips:
                max_flips = flips_count
        while r != n:
            perm1.insert(r, perm1.pop(0))
            count[r] -= 1
            if count[r] > 0:
                break
            r += 1
        else:
            return max_flips

@zk_circuit
@requires(lambda n: 0 <= n <= 1024)
def fannkuch2(n: Integer):
    fannkuch(n)
