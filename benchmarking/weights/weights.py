# Source: Pythran tests/cases/weights.py
# Original #pythran export: weights(uint8[:,:],float?)
# Migration notes: chose small const dimensions; threshold has default value.
from zinnia import *

M = 16
N = 16


@zk_circuit
def weights(input_data: NDArray[Integer, 16, 16], threshold: float = 0.3):
    n_seq, length = input_data.shape
    weights = np.zeros(n_seq, dtype=np.float32)

    for i in range(n_seq):
        vector = input_data[i, None, :]
        count_matches = np.sum(vector == input_data, axis=1)
        over_threshold = count_matches > (threshold * length)
        total = np.sum(over_threshold)
        weights[i] = 1 / total

    _zinnia_result = weights
