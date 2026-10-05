# pythran program spectralDNS.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_chip
def cross1(c, a, b):
    c[0] = a[0] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

@zk_circuit
def cross2(c: Complex[Array, '3 2 5 7'], a: Float[Array, '3 2 5 7'], b: Complex[Array, '3 2 5 7']):
    cross1(c, a, b)
    c *= 1j
    return c

@zk_circuit
@requires(lambda dU, U_hat, K2, K, P_hat, K_over_K2, nu: 0 <= K <= 1024)
def add_pressure_diffusion(dU: Complex[Array, '3 2 5 7'], U_hat: Complex[Array, '3 2 5 7'], K2: Float[Array, '2 5 7'], K: Float[Array, '3 2 5 7'], P_hat: Complex[Array, '2 5 7'], K_over_K2: Float[Array, '3 2 5 7'], nu: Float):
    du0, du1, du2 = (dU[0], dU[1], dU[2])
    k_0, k_1, k_2 = (K_over_K2[0], K_over_K2[1], K_over_K2[2])
    P_hat[:] = du0 * k_0 + du1 * k_1 + du2 * k_2
    dU[:] = dU - P_hat * K - nu * K2 * U_hat
    return P_hat
