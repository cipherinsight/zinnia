# pythran program gauss.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def pivot(n, i, a, b):
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

@zk_circuit
@requires(lambda n, a, b: 0 <= n <= 1024)
def gauss(n: Integer, a: Float[Array, '4 4'], b: Float[Array, '4']):
    for i in range(1, n + 1):
        if i < n:
            pivot(n, i, a, b)
        a[i - 1][i - 1] = 1.0 / a[i - 1][i - 1]
        b[i - 1] = b[i - 1] * a[i - 1][i - 1]
        if i < n:
            for j in range(i + 1, n + 1):
                a[j - 1][i - 1] = a[j - 1][i - 1] * a[i - 1][i - 1]
            for k in range(i + 1, n + 1):
                b[k - 1] = b[k - 1] - a[i - 1][k - 1] * b[i - 1]
                for j in range(i + 1, n + 1):
                    a[j - 1][k - 1] = a[j - 1][k - 1] - a[i - 1][k - 1] * a[j - 1][i - 1]
    for i in range(n - 1, 0, -1):
        for j in range(i, n):
            b[i - 1] = b[i - 1] - a[j][i - 1] * b[j]
    return b
