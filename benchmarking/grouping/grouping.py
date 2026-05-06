# Source: Pythran tests/cases/grouping.py
# Original #pythran export: grouping(uint32[])
from zinnia import *

N = 64


@zk_circuit
def grouping(values: NDArray[Integer, 64]):
    diff = np.concatenate(([1], np.diff(values)))
    idx = np.concatenate((np.where(diff)[0], [len(values)]))
    _zinnia_result = values[idx[:-1]], np.diff(idx)
