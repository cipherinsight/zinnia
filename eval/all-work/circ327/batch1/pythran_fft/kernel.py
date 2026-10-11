# pythran program fft.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import math, numpy as np

@zk_circuit
def fft(x: Complex[Array, '16']):
    N = x.shape[0]
    if N == 1:
        return np.array(x)
    e = fft(x[::2])
    o = fft(x[1::2])
    M = N // 2
    l = [e[k] + o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
    r = [e[k] - o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
    return np.array(l + r)
