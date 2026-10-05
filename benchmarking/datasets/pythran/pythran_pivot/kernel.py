# pythran program pivot.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
@requires(lambda n, i, a, b: 0 <= n <= 1024)
def pivot(n: Integer, i: Integer, a: Complex[Array, '2 2'], b: Complex[Array, '2']):
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
