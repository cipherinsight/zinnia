# Source: Pythran tests/cases/_histogram.py
# Original #pythran export: histogram_neq_edges_weights(int32[][], int32[], int32[][])
from zinnia import *

M = 8
N = 4
B = 4


def _histogram_neq_edges_weights(data, bin_edges, weights):
    _BLOCK = 65536

    bin_edges_length = (len(bin_edges) - 1)

    hist = np.zeros(len(bin_edges), weights.dtype)

    for j in range(0, len(data), _BLOCK):
        tmp_hist = np.zeros(len(bin_edges), weights.dtype)
        tmp_data = data[j:j + _BLOCK]
        tmp_weights = weights[j:j + _BLOCK]

        for i in range(0, len(tmp_data)):
            value = tmp_data[i]

            if np.isnan(value) or not (bin_edges[0] <= value <= bin_edges[-1]):
                continue

            bin_idx = 0
            while bin_idx < bin_edges_length and bin_edges[bin_idx + 1] <= value:
                bin_idx += 1

            tmp_hist[bin_idx] += tmp_weights[i]

        hist += tmp_hist

    hist[-2] += hist[-1]

    return hist[:-1], bin_edges


@zk_circuit
def histogram_neq_edges_weights(data: NDArray[Integer, 8, 4], bin_edges: NDArray[Integer, 4],
                                weights: NDArray[Integer, 8, 4]):
    if weights.shape != data.shape:
        raise ValueError('weights should have the same shape as data.')
    weights = weights.ravel()

    data = data.ravel()
    _zinnia_result = _histogram_neq_edges_weights(data, bin_edges, weights)
