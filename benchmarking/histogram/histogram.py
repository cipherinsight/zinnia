# Source: Pythran tests/cases/histogram.py
# Original #pythran export: histogram(float list, int)
from zinnia import *


@zk_circuit
def histogram(data: NDArray[Float, 64], bin_width: int):
    lower_bound, upper_bound = min(data), max(data)
    out_data = [0] * (1 + bin_width)
    for i in data:
        out_data[int(bin_width * (i - lower_bound) / (upper_bound - lower_bound))] += 1
    out_data[-2] += out_data[-1]
    out_data.pop()
    _zinnia_result = out_data
