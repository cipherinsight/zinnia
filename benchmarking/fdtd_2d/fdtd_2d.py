# Source: NPBench polybench/fdtd_2d (fdtd_2d_numpy.py)
# Original signature: kernel(TMAX, ex, ey, hz, _fict_) where ex, ey, hz are (NX, NY)
#   and _fict_ is (TMAX,).
# Migration notes:
#   - TMAX, NX, NY hoisted as module-level constants.
from zinnia import *

TMAX = 20
NX = 200
NY = 220


@zk_circuit
def fdtd_2d(ex: NDArray[Float, 200, 220],
            ey: NDArray[Float, 200, 220],
            hz: NDArray[Float, 200, 220],
            _fict_: NDArray[Float, 20]):
    for t in range(TMAX):
        ey[0, :] = _fict_[t]
        ey[1:, :] -= 0.5 * (hz[1:, :] - hz[:-1, :])
        ex[:, 1:] -= 0.5 * (hz[:, 1:] - hz[:, :-1])
        hz[:-1, :-1] -= 0.7 * (ex[:-1, 1:] - ex[:-1, :-1] + ey[1:, :-1] -
                               ey[:-1, :-1])
