# npbench program mandelbrot2/mandelbrot2_numpy.py (spcl/npbench npbench/benchmarks).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_circuit
@requires(lambda xmin, xmax, ymin, ymax, xn, yn, itermax, horizon: 0 <= itermax <= 1024)
def mandelbrot(xmin: Float, xmax: Float, ymin: Float, ymax: Float, xn: Integer, yn: Integer, itermax: Integer, horizon: Float=2.0):
    Xi, Yi = np.mgrid[0:xn, 0:yn]
    X = np.linspace(xmin, xmax, xn, dtype=np.float64)[Xi]
    Y = np.linspace(ymin, ymax, yn, dtype=np.float64)[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi.shape = Yi.shape = C.shape = xn * yn
    Z = np.zeros(C.shape, np.complex128)
    for i in range(itermax):
        if not len(Z):
            break
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]
        np.logical_not(I, I)
        Z = Z[I]
        Xi, Yi = (Xi[I], Yi[I])
        C = C[I]
    return (Z_.T, N_.T)
