# pythran program lttb_pt.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *
import numpy as np

@zk_chip
def _areas_of_triangles(a, bs, c):
    bs_minus_a = bs - a
    a_minus_bs = a - bs
    return 0.5 * abs((a[0] - c[0]) * bs_minus_a[:, 1] - a_minus_bs[:, 0] * (c[1] - a[1]))

@zk_circuit
@requires(lambda data, n_out: 0 <= n_out <= 1024)
def downsample(data: Float[Array, '16 2'], n_out: Integer):
    if n_out > data.shape[0]:
        raise ValueError('n_out must be <= number of rows in data')
    if n_out == data.shape[0]:
        return data
    if n_out < 3:
        raise ValueError('Can only downsample to a minimum of 3 points')
    n_bins = n_out - 2
    data_bins = np.array_split(data[1:len(data) - 1], n_bins)
    out = np.zeros((n_out, 2))
    out[0] = data[0]
    out[len(out) - 1] = data[len(data) - 1]
    for i in range(len(data_bins)):
        this_bin = data_bins[i]
        if i < n_bins - 1:
            next_bin = data_bins[i + 1]
        else:
            next_bin = data[len(data) - 1:]
        a = out[i]
        bs = this_bin
        c = next_bin.mean(axis=0)
        areas = _areas_of_triangles(a, bs, c)
        out[i + 1] = bs[np.argmax(areas)]
    return out
