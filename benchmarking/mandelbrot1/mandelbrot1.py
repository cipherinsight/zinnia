# Source: NPBench mandelbrot1 (mandelbrot1_numpy.py)
# Original signature: mandelbrot(xmin, xmax, ymin, ymax, xn, yn, maxiter, horizon=2.0) — no array inputs (constants only).
# Migration notes:
#   - xn, yn, maxiter hoisted to module-level constants; values from the "S" preset (XN=YN=125) shrunk to 16; maxiter=8.
#   - Uses complex numbers / boolean-mask indexing; likely unsupported in Zinnia, left in place per policy.
from zinnia import *

XN = 16
YN = 16
MAXITER = 8


@zk_circuit
def mandelbrot1(xmin: float, xmax: float, ymin: float, ymax: float,
                horizon: float):
    # Adapted from https://www.ibm.com/developerworks/community/blogs/jfp/...
    #              .../entry/How_To_Compute_Mandelbrodt_Set_Quickly?lang=en
    X = np.linspace(xmin, xmax, 16, dtype=np.float64)
    Y = np.linspace(ymin, ymax, 16, dtype=np.float64)
    C = X + Y[:, None] * 1j
    N = np.zeros(C.shape, dtype=np.int64)
    Z = np.zeros(C.shape, dtype=np.complex128)
    for n in range(8):
        I = np.less(abs(Z), horizon)
        N[I] = n
        Z[I] = Z[I]**2 + C[I]
    N[N == 8 - 1] = 0
    _zinnia_result = Z, N
