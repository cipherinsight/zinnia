from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, data-dependent-loop, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
# Diagnostic: while loops require a public/padded maximum iteration count.

@zk_circuit
@requires(lambda a, n, z, err, nter: 0 <= n <= 1024)
def guerre(a: Complex[Array, 'N<=1024'], n: Integer, z: Complex, err: Float, nter: Integer):
    az = [complex(0, 0) for i in range(50)]
    azz = [complex(0, 0) for i in range(50)]
    ci = complex(0.0, 1.0)
    eps = 1e-20
    for i in range(1, n + 1):
        az[i - 1] = float(i) * a[i]
    for i in range(1, n):
        azz[i - 1] = float(i) * az[i]
    dz = err + 1
    itera = 0
    jter = 0
    while abs(dz) > err and itera < nter:
        p = a[n - 1] + a[n] * z
        for i in range(n - 1, 0, -1):
            p = a[i - 1] + z * p
        if abs(p) < eps:
            return z
        pz = az[n - 2] + az[n - 1] * z
        for i in range(n - 2, 0, -1):
            pz = az[i - 1] + z * pz
        pzz = azz[n - 3] + azz[n - 2] * z
        for i in range(n - 3, 0, -1):
            pzz = azz[i - 1] + z * pzz
        f = pz / p
        g = f ** 2 - pzz / p
        h = n * g
        amp1 = abs(f + h)
        amp2 = abs(f - h)
        if amp1 > amp2:
            dz = float(-n) / (f + h)
        else:
            dz = float(-n) / (f - h)
        itera = itera + 1
        jter = jter + 1
        if jter == 10:
            jter = 1
            dz = dz * ci
        z = z + dz
        if jter == 100:
            raise RuntimeError('Laguerre method not converging')
    return z
