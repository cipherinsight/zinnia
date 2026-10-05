# pythran program deriv.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda n, sig, alp, dg, dh1, dh3, bin, nu: 0 <= n <= 1024)
def deriv(n: Integer, sig: Float, alp: Complex, dg: Complex[Array, '4'], dh1: Complex[Array, '4'], dh3: Complex[Array, '4'], bin: Complex[Array, '5 4'], nu: Float):
    dh2 = [complex(0, 0) for _ in dh1]
    ci = complex(0.0, 1.0)
    dh1[0] = complex(0.5, 0) * ci * complex(sig, 0)
    exp1 = complex(-0.5, 0)
    dh2[0] = alp
    exp2 = complex(-1.0, 0)
    dh3[0] = -2.0 * nu
    exp3 = complex(-1.0, 0)
    for i in range(1, n):
        dh1[i] = dh1[i - 1] * exp1
        exp1 = exp1 - 1.0
        dh2[i] = dh2[i - 1] * exp2
        exp2 = exp2 - 1.0
        dh3[i] = -nu * dh3[i - 1] * exp3
        exp3 = exp3 - 1.0
    dg[0] = complex(1.0)
    dg[1] = dh1[0] + dh2[0] + dh3[0]
    for i in range(2, n + 1):
        dg[i] = dh1[i - 1] + dh2[i - 1] + dh3[i - 1]
        for j in range(1, i):
            dg[i] = dg[i] + bin[j - 1][i - 1] * (dh1[j - 1] + dh2[j - 1] + dh3[j - 1]) * dg[i - j]
    return dg
