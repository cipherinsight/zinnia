from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: data-dependent-loop, denominator-obligation, in-place-mutation, loop-bounded
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add an explicit iteration cap or classify as fail-closed until a bound is available.
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: while loops require a public/padded maximum iteration count.
from math import sin as DSIN, cos as DCOS, atan as DATAN, log as DLOG, exp as DEXP, sqrt as DSQRT

@zk_circuit
def whetstone(loopstart: Integer):
    T = 0.499975
    T1 = 0.50025
    T2 = 2.0
    LOOP = loopstart
    II = 1
    JJ = 1
    while JJ <= II:
        N1 = 0
        N2 = 12 * LOOP
        N3 = 14 * LOOP
        N4 = 345 * LOOP
        N6 = 210 * LOOP
        N7 = 32 * LOOP
        N8 = 899 * LOOP
        N9 = 616 * LOOP
        N10 = 0
        N11 = 93 * LOOP
        X1 = 1.0
        X2 = -1.0
        X3 = -1.0
        X4 = -1.0
        for I in range(1, N1 + 1):
            X1 = (X1 + X2 + X3 - X4) * T
            X2 = (X1 + X2 - X3 + X4) * T
            X3 = (X1 - X2 + X3 + X4) * T
            X4 = (-X1 + X2 + X3 + X4) * T
        E1 = [1.0, -1.0, -1.0, -1.0]
        for I in range(1, N2 + 1):
            E1[0] = (E1[0] + E1[1] + E1[2] - E1[3]) * T
            E1[1] = (E1[0] + E1[1] - E1[2] + E1[3]) * T
            E1[2] = (E1[0] - E1[1] + E1[2] + E1[3]) * T
            E1[3] = (-E1[0] + E1[1] + E1[2] + E1[3]) * T
        for I in range(1, N3 + 1):
            PA(E1, T, T2)
        J = 1
        for I in range(1, N4 + 1):
            if J == 1:
                J = 2
            else:
                J = 3
            if J > 2:
                J = 0
            else:
                J = 1
            if J < 1:
                J = 1
            else:
                J = 0
        J = 1
        K = 2
        L = 3
        for I in range(1, N6 + 1):
            J = J * (K - J) * (L - K)
            K = L * K - (L - J) * K
            L = (L - K) * (K + J)
            E1[L - 2] = J + K + L
            E1[K - 2] = J * K * L
        X = 0.5
        Y = 0.5
        for I in range(1, N7 + 1):
            X = T * DATAN(T2 * DSIN(X) * DCOS(X) / (DCOS(X + Y) + DCOS(X - Y) - 1.0))
            Y = T * DATAN(T2 * DSIN(Y) * DCOS(Y) / (DCOS(X + Y) + DCOS(X - Y) - 1.0))
        X = 1.0
        Y = 1.0
        Z = 1.0
        for I in range(1, N8 + 1):
            Z = P3(X, Y, T, T2)
        J = 1
        K = 2
        L = 3
        E1[0] = 1.0
        E1[1] = 2.0
        E1[2] = 3.0
        for I in range(1, N9 + 1):
            P0(E1, J, K, L)
        J = 2
        K = 3
        for I in range(1, N10 + 1):
            J = J + K
            K = J + K
            J = K - J
            K = K - J - J
        X = 0.75
        for I in range(1, N11 + 1):
            X = DSQRT(DEXP(DLOG(X) / T1))
        JJ += 1
    KIP = 100.0 * LOOP * II
    return KIP

@zk_chip
def PA(E: Float[Array, 'N<=1024'], T: Float, T2: Float):
    J = 0
    while J < 6:
        E[0] = (E[0] + E[1] + E[2] - E[3]) * T
        E[1] = (E[0] + E[1] - E[2] + E[3]) * T
        E[2] = (E[0] - E[1] + E[2] + E[3]) * T
        E[3] = (-E[0] + E[1] + E[2] + E[3]) / T2
        J += 1

@zk_chip
def P0(E1: Float[Array, 'N<=1024'], J: Integer, K: Integer, L: Integer):
    E1[J - 1] = E1[K - 1]
    E1[K - 1] = E1[L - 1]
    E1[L - 1] = E1[J - 1]

@zk_chip
def P3(X: Float, Y: Float, T: Float, T2: Float):
    X1 = X
    Y1 = Y
    X1 = T * (X1 + Y1)
    Y1 = T * (X1 + Y1)
    return (X1 + Y1) / T2
