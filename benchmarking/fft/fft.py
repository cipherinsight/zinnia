# Source: Pythran tests/cases/fft.py
# Original #pythran export: fft(complex[])
from zinnia import *
import math

N = 64


@zk_circuit
def fft(x: NDArray[Complex, 64]):
    N = x.shape[0]
    if N == 1:
        _zinnia_result = np.array(x)
    e = fft(x[::2])
    o = fft(x[1::2])
    M = N // 2
    l = [e[k] + o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
    r = [e[k] - o[k] * math.e ** (-2j * math.pi * k / N) for k in range(M)]
    _zinnia_result = np.array(l + r)
