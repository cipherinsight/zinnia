# pythran program histogram.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def histogram(data: Float[Array, '10000'], bin_width: Integer):
    lower_bound, upper_bound = (min(data), max(data))
    out_data = [0] * (1 + bin_width)
    for i in data:
        out_data[int(bin_width * (i - lower_bound) / (upper_bound - lower_bound))] += 1
    out_data[-2] += out_data[-1]
    out_data.pop()
    return out_data
