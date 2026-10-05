# pythran program smoothing.py (serge-sans-paille/pythran pythran/tests/cases).
# Function bodies are the upstream text. Entry-function annotations are derived from upstream
# types by benchmarking/datasets/tools/acceptance_case_set.py; see datasets/case_sets/acceptance.json.

from zinnia import *

@zk_circuit
def smoothing(x: Float[Array, '100'], alpha: Float):
    """
  Exponential smoothing of a time series
  For x = 10**6 floats
  - Python runtime: 9 seconds
  - Parakeet runtime: .01 seconds
  """
    s = x.copy()
    for i in range(1, len(x)):
        s[i] = alpha * x[i] + (1 - alpha) * s[i - 1]
    return s
