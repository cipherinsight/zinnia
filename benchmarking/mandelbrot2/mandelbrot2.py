# Source: NPBench mandelbrot2 (mandelbrot2_numpy.py)
# Original signature: mandelbrot(xmin, xmax, ymin, ymax, xn, yn, itermax, horizon=2.0) — no array inputs.
# Migration notes:
#   - xn, yn, itermax hoisted to module-level constants; values from the "S" preset (XN=YN=200, itermax=40).
#   - Heavy reliance on dynamic boolean-mask indexing (Z = Z[I]) and complex numbers; likely unsupported, left in place.
from zinnia import *

XN = 200
YN = 200
ITERMAX = 40


@zk_circuit
def mandelbrot2(xmin: float, xmax: float, ymin: float, ymax: float,
                horizon: float):
    # Adapted from
    # https://thesamovar.wordpress.com/2009/03/22/fast-fractals-with-python-and-numpy/
    Xi, Yi = np.mgrid[0:XN, 0:YN]
    X = np.linspace(xmin, xmax, XN, dtype=np.float64)[Xi]
    Y = np.linspace(ymin, ymax, YN, dtype=np.float64)[Yi]
    C = X + Y * 1j
    N_ = np.zeros(C.shape, dtype=np.int64)
    Z_ = np.zeros(C.shape, dtype=np.complex128)
    Xi.shape = Yi.shape = C.shape = XN * YN

    Z = np.zeros(C.shape, np.complex128)
    for i in range(ITERMAX):
        if not len(Z):
            break

        # Compute for relevant points only
        np.multiply(Z, Z, Z)
        np.add(Z, C, Z)

        # Failed convergence
        I = abs(Z) > horizon
        N_[Xi[I], Yi[I]] = i + 1
        Z_[Xi[I], Yi[I]] = Z[I]

        # Keep going with those who have not diverged yet
        np.logical_not(I, I)  # np.negative(I, I) not working any longer
        Z = Z[I]
        Xi, Yi = Xi[I], Yi[I]
        C = C[I]
    _zinnia_result = Z_.T, N_.T
