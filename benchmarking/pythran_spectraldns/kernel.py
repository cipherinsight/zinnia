from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, in-place-mutation
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.

@zk_chip
def cross1(c: Complex[Array, 'N<=1024'], a: Complex[Array, 'N<=1024'], b: Complex[Array, 'N<=1024']):
    c[0] = a[0] * b[2] - a[2] * b[1]
    c[1] = a[2] * b[0] - a[0] * b[2]
    c[2] = a[0] * b[1] - a[1] * b[0]
    return c

@zk_circuit
def cross2(c: Integer, a: Float, b: Float):
    cross1(c, a, b)
    c *= 1j
    return c

@zk_circuit
@requires(lambda dU, U_hat, K2, K, P_hat, K_over_K2, nu: 0 <= K <= 1024)
def add_pressure_diffusion(dU: Complex[Array, 'N<=1024'], U_hat: Float, K2: Float, K: Integer, P_hat: Complex[Array, 'N<=1024'], K_over_K2: Complex[Array, 'N<=1024'], nu: Float):
    du0, du1, du2 = (dU[0], dU[1], dU[2])
    k_0, k_1, k_2 = (K_over_K2[0], K_over_K2[1], K_over_K2[2])
    P_hat[:] = du0 * k_0 + du1 * k_1 + du2 * k_2
    dU[:] = dU - P_hat * K - nu * K2 * U_hat
    return P_hat
