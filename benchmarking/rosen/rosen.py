# Source: Pythran tests/cases/rosen.py
# Original `#pythran export` directives: rosen(int[]) and rosen(float[]).
# Migration notes:
#   - Picked the float[] signature; chose a small const length N=64.
from zinnia import *

N = 64


@zk_circuit
def rosen(x: NDArray[Float, 64]):
    t0 = 100 * (x[1:] - x[:-1] ** 2) ** 2
    t1 = (1 - x[:-1]) ** 2
    _zinnia_result = np.sum(t0 + t1)
