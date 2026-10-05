# npbench program stockham_fft/stockham_fft_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
@requires(lambda N, R, K, x, y: R != 0)
def stockham_fft(N: Integer, R: Integer, K: Integer, x: Complex[Array, '32768'], y: Complex[Array, '32768']):
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
