# Source: Pythran tests/cases/empirical.py
# Original #pythran export: empirical(float[:], float, float)
from zinnia import *

N = 64


def find_first(seq, pred):
    for i, x in enumerate(seq):
        print(i, x, pred(x))
        if pred(x):
            return i
    return None


@zk_circuit
def empirical(ds: NDArray[Float, 64], alpha: float, x: float):
    sds = np.sort(ds)
    ds_to_the_alpha = sds ** alpha
    fractions = ds_to_the_alpha
    thresholds = np.cumsum(fractions)
    thresholds /= thresholds[-1]
    i = find_first(thresholds, lambda u: x < u)
    _zinnia_result = i
