# pythran program guerre.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda a, n, z, err, nter: 0 <= n <= 1024)
def guerre(a: Complex[Array, '4'], n: Integer, z: Complex, err: Float, nter: Integer):
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
