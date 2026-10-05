from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.

@zk_circuit
@requires(lambda n, i, a, b: 0 <= n <= 1024)
def pivot(n: Integer, i: Integer, a: Complex[Array, 'N<=1024 M<=1024'], b: Complex[Array, 'N<=1024']):
    i0 = i
    amp0 = abs(a[i - 1][i - 1])
    for j in range(i + 1, n + 1):
        amp = abs(a[i - 1][j - 1])
        if amp > amp0:
            i0 = j
            amp0 = amp
    if i == i0:
        return
    temp = b[i - 1]
    b[i - 1] = b[i0 - 1]
    b[i0 - 1] = temp
    for j in range(i, n + 1):
        temp = a[j - 1][i - 1]
        a[j - 1][i - 1] = a[j - 1][i0 - 1]
        a[j - 1][i0 - 1] = temp
    return (a, b)
