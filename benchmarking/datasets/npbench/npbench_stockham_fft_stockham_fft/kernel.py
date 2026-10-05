from zinnia import *
# Zinnia benchmark contract draft: generated for review.
# Keep kernel body faithful; refine TODO contracts before final benchmark runs.
# Categories: complex, denominator-obligation, in-place-mutation, loop-bounded, matmul-dot
# Expected outcome: diagnostic/fail-closed candidate
# Precondition: Add nonzero or positive denominator facts at exact division/modulo sites when expressible.
# Precondition: Declare shared matrix/vector dimensions explicitly; when not expressible as @requires, encode with matching symbolic annotation names and keep as review obligation.
# Precondition: Loop trip counts must be finite and tied to scalar bounds or array dimensions.
# Precondition: Use writable fixed/bounded array shapes and record an aliasing policy for mutated inputs.
# Diagnostic: Complex kernels require Complex dtype support or a fail-closed classification.
import numpy as np

@zk_circuit
@requires(lambda N, R, K, x, y: R != 0)
def stockham_fft(N: Integer, R: Integer, K: Integer, x: Complex[Array, 'N<=1024'], y: Complex[Array, 'N<=1024']):
    i_coord, j_coord = np.mgrid[0:R, 0:R]
    dft_mat = np.empty((R, R), dtype=np.complex128)
    dft_mat = np.exp(-2j * np.pi * i_coord * j_coord / R)
    y[:] = x[:]
    ii_coord, jj_coord = np.mgrid[0:R, 0:R ** K]
    for i in range(K):
        yv = np.reshape(y, (R ** i, R, R ** (K - i - 1)))
        tmp_perm = np.transpose(yv, axes=(1, 0, 2))
        D = np.empty((R, R ** i, R ** (K - i - 1)), dtype=np.complex128)
        tmp = np.exp(-2j * np.pi * ii_coord[:, :R ** i] * jj_coord[:, :R ** i] / R ** (i + 1))
        D[:] = np.repeat(np.reshape(tmp, (R, R ** i, 1)), R ** (K - i - 1), axis=2)
        tmp_twid = np.reshape(tmp_perm, (N,)) * np.reshape(D, (N,))
        y[:] = np.reshape(dft_mat @ np.reshape(tmp_twid, (R, R ** (K - 1))), (N,))
