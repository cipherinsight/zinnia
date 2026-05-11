# Source: Pythran tests/cases/lttb_pt.py
# Original #pythran export: downsample(float[:, :] order(C), int)
from zinnia import *

N = 64


@zk_chip
def _areas_of_triangles(a, bs, c) -> NDArray[Float, 64]:
    bs_minus_a = bs - a
    a_minus_bs = a - bs
    return 0.5 * abs(
        (a[0] - c[0]) * (bs_minus_a[:, 1]) - (a_minus_bs[:, 0]) * (c[1] - a[1])
    )


@zk_circuit
def downsample(data: NDArray[Float, 64, 2], n_out: int):
    if n_out > data.shape[0]:
        raise ValueError("n_out must be <= number of rows in data")

    if n_out == data.shape[0]:
        _zinnia_result = data

    if n_out < 3:
        raise ValueError("Can only downsample to a minimum of 3 points")

    n_bins = n_out - 2
    data_bins = np.array_split(data[1: len(data) - 1], n_bins)

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

    _zinnia_result = out
