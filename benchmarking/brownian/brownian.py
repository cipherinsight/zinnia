# Source: Pythran tests/cases/brownian.py
# Original #pythran export: brownian_bridge(int, int, float, float, int)
from zinnia import *
from math import sqrt


def linspace(begin, end, nbsteps):
    assert begin < end
    return [begin + i * (end - begin) / nbsteps for i in range(nbsteps)]


def zeros(n):
    return [0.] * n


def norm(m, u):
    return ((m * u + 0.15) % 1)


def moy(t1, t2, b1, b2, t):
    return (1. * (t2 * b1 - t1 * b2) + t * (b2 - b1)) / (t2 - t1)


def p(t):
    t = 1


def var(t1, t2, b1, b2, t):
    return (1. * t - t1) * (t2 - t) / (t2 - t1)


@zk_circuit
def brownian_bridge(ti: int, tf: int, bi: float, bf: float, n: int):
    n = int(n * (tf - ti))
    T = linspace(ti, tf, n)
    pas = (tf - ti) / (n - 1.)
    B = zeros(n)
    B[0] = bi
    B[n - 1] = bf
    t1 = ti
    for k in range(1, n - 1):
        m = moy(t1, tf, B[k - 1], bf, t1 + pas)
        v = var(t1, tf, B[k - 1], bf, t1 + pas)
        B[k] = m + sqrt(v) * norm(0, 1)
        t1 += pas
    _zinnia_result = T, B
